"""Accelerator pedal, gear schedule and engine speed (spec section 4, NFR-15)."""

from dataclasses import dataclass

import numpy as np
import pytest
from driven import DrivenTrip, drawn, drive_with_steering
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.catalogue import catalogue
from simulator.config_files import van_model
from simulator.constants import RAD_S_TO_RPM
from simulator.model.brakes import (
    BrakeParameters,
    brake_pedal_pressed,
    held_steps,
    hold_force,
    master_cylinder_pressure,
)
from simulator.model.longitudinal import LongitudinalParameters, traction_force_limit
from simulator.model.powertrain import (
    PowertrainParameters,
    accelerator_pedal,
    engine_speed,
    gear_engine_speed,
    pedal_demand,
    select_gear,
)
from simulator.model.wheels import WheelParameters, axle_slip
from simulator.profile_config import drive_profile

RADIUS = 0.35  # m, nominal tyre radius (van_model.yaml)


def params() -> PowertrainParameters:
    return PowertrainParameters.from_model(van_model())


def one(value: float) -> np.ndarray:
    return np.array([value])


@dataclass(frozen=True)
class Powertrain:
    """Powertrain signals of a driven trip, one entry per step."""

    demand: np.ndarray  # unclipped pedal demand
    pedal: np.ndarray  # %
    previous_gear: np.ndarray  # gear before the step's selection (1 at the trip start)
    gear: np.ndarray  # gear selected for the step
    engine_speed: np.ndarray  # rpm
    slip_rear: np.ndarray
    radius: float  # m, the van's drawn tyre radius
    pressure: np.ndarray  # Pa, master cylinder
    held: np.ndarray  # bool


def powertrain(driven: DrivenTrip, seed: int, number: int) -> Powertrain:
    """Run the pedal, the gear schedule and the engine speed over a trip of VAN-<number>."""
    model, p = van_model(), params()
    trace = driven.trace
    mass = np.full(len(trace.speed), driven.mass)
    _, slip_rear = axle_slip(
        trace.engine_force, trace.brake_force, mass, WheelParameters.from_model(model)
    )
    radius = drawn(seed, number).tyre_rolling_radius
    demand = pedal_demand(trace.engine_force, trace.speed, p)
    gears = np.empty(len(trace.speed), dtype=int)
    gear = np.ones(1, dtype=int)
    for k in range(len(trace.speed)):
        s = slice(k, k + 1)
        gear = select_gear(gear, trace.speed[s], slip_rear[s], one(radius), p)
        gears[k] = gear[0]
    held = held_steps(trace.speed, np.append(trace.speed[1:], 0.0))
    brakes = BrakeParameters.from_configs(model, drive_profile())
    pressure = master_cylinder_pressure(
        trace.brake_force, held, trace.at_delivery, hold_force(mass, trace.grade), brakes
    )
    return Powertrain(
        demand=demand,
        pedal=accelerator_pedal(demand),
        previous_gear=np.concatenate([[1], gears[:-1]]),
        gear=gears,
        engine_speed=engine_speed(gears, trace.speed, slip_rear, np.full(len(gears), radius), p),
        slip_rear=slip_rear,
        radius=radius,
        pressure=pressure,
        held=held,
    )


def test_pedal_by_hand() -> None:
    """P_max 100 kW, speed floor 3 m/s: 5000 N at 10 m/s is 50 %; at 1 m/s the floor gives
    5000 * 3 / 100000 = 15 %; the traction limit at 20 m/s (5000 N) is exactly 100 %; engine drag
    (-1320 N, laden van) is 0 %."""
    p = params()
    longitudinal = LongitudinalParameters.from_model(van_model())
    force = np.array([5000.0, 5000.0, traction_force_limit(one(20.0), longitudinal)[0], -1320.0])
    speed = np.array([10.0, 1.0, 20.0, 10.0])
    demand = pedal_demand(force, speed, p)
    assert demand == pytest.approx([0.5, 0.15, 1.0, 0.0], rel=1e-12)
    assert demand[2] <= 1 + 1e-12
    assert accelerator_pedal(demand) == pytest.approx([50.0, 15.0, 100.0, 0.0], rel=1e-12)
    assert accelerator_pedal(np.array([1.5, -0.2])).tolist() == [100.0, 0.0]


def test_engine_speed_by_hand() -> None:
    """25 m/s in sixth (3.042), nominal radius, no slip: 25 / 0.35 * 3.042 * 60 / (2 pi) = 2075 rpm.
    A van with 1 % larger tyres and 2 % rear slip: times 1.02 / 1.01. At rest: idle (750 rpm).
    1 m/s in first: 478.8 rpm without the clutch, so the idle floor gives 750 rpm."""
    p = params()
    nominal = engine_speed(one(6), one(25.0), one(0.0), one(RADIUS), p)[0]
    assert nominal == pytest.approx(25 / RADIUS * 3.042 * RAD_S_TO_RPM, rel=1e-12)
    assert nominal == pytest.approx(2075, abs=0.5)
    biased = engine_speed(one(6), one(25.0), one(0.02), one(1.01 * RADIUS), p)[0]
    assert biased == pytest.approx(nominal * 1.02 / 1.01, rel=1e-12)
    assert engine_speed(one(3), one(0.0), one(0.0), one(RADIUS), p)[0] == 750
    assert gear_engine_speed(one(1), one(1.0), one(0.0), one(RADIUS), p)[0] == pytest.approx(
        478.8, abs=0.05
    )
    assert engine_speed(one(1), one(1.0), one(0.0), one(RADIUS), p)[0] == 750


def speed_at(rpm: float, gear: int) -> float:
    """Speed, m/s, at which the engine turns at `rpm` in `gear` (nominal radius, no slip)."""
    return rpm / (params().ratios[gear - 1] * RAD_S_TO_RPM / RADIUS)


@pytest.mark.parametrize(
    ("gear", "rpm", "expected"),
    [
        (1, 2501, 2),  # above upshift_speed: second, at 2501 * 10.14 / 17.55 = 1445 rpm
        (1, 2499, 1),
        (2, 1199, 1),  # below downshift_speed: first
        (2, 1201, 2),
        (6, 3000, 6),  # no gear above sixth
        (1, 700, 1),  # no gear below first
        (3, 3500, 4),  # one shift per step, even far above the threshold
    ],
)
def test_gear_selection_at_the_thresholds(gear: int, rpm: float, expected: int) -> None:
    p = params()
    v = one(speed_at(rpm, gear))
    new = select_gear(one(gear).astype(int), v, one(0.0), one(RADIUS), p)
    assert new[0] == expected
    if (gear, expected) == (1, 2):
        n = engine_speed(new, v, one(0.0), one(RADIUS), p)[0]
        assert n == pytest.approx(2501 * 10.14 / 17.55, rel=1e-12)


def test_first_gear_at_rest() -> None:
    gears = select_gear(
        np.array([1, 2, 4, 6]), np.zeros(4), np.zeros(4), np.full(4, RADIUS), params()
    )
    assert gears.tolist() == [1, 1, 1, 1]


def test_long_trips_shift_both_ways() -> None:
    """Eight 20-minute trips (seed 7, trip 1): the vans shift up and down through the gears (167
    upshifts and 167 downshifts while moving), up to sixth gear (van 5, the fastest rural van)."""
    upshifts = downshifts = 0
    highest = 1
    for number, driven in enumerate(drive_with_steering(7, 1, 1200.0, 10.0, vans=8), start=1):
        signals = powertrain(driven, 7, number)
        moving = driven.trace.speed > 0
        change = (signals.gear - signals.previous_gear)[moving]
        upshifts += int(np.sum(change == 1))
        downshifts += int(np.sum(change == -1))
        highest = max(highest, int(signals.gear.max()))
    assert upshifts > 100 and downshifts > 100
    assert highest == params().top_gear


@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    duration=st.floats(min_value=60.0, max_value=300.0),
    ambient=st.floats(min_value=-5.0, max_value=30.0),
)
def test_driven_powertrain_respects_the_invariants(
    seed: int, trip: int, duration: float, ambient: float
) -> None:
    model, p = van_model(), params()
    margin = model.value("shift_margin")
    speed_max = model.value("engine_speed_max")
    rpm_low, rpm_high = catalogue().signal("engine_speed").range
    switch = BrakeParameters.from_configs(model, drive_profile())
    for number, driven in enumerate(drive_with_steering(seed, trip, duration, ambient), start=1):
        trace = driven.trace
        signals = powertrain(driven, seed, number)
        # Pedal: the clip only absorbs rounding (traction power at most P_max), range.
        assert np.all(signals.demand <= 1 + 1e-12)
        assert np.all((signals.pedal >= 0) & (signals.pedal <= 100))
        # Accelerator and brake never applied together: wherever the pedal is pressed the master
        # cylinder pressure is exactly 0 (so the brake pedal is not pressed); the pedal is
        # released on held steps, deliveries included.
        pressed = signals.pedal > 0
        assert np.all(signals.pressure[pressed] == 0)
        assert not np.any(pressed & brake_pedal_pressed(signals.pressure, switch))
        assert np.all(signals.pedal[signals.held] == 0)
        # At rest: first gear, engine at idle.
        rest = trace.speed == 0
        assert np.all(signals.gear[rest] == 1)
        assert np.all(signals.engine_speed[rest] == p.idle)
        # No gear hunting. While moving: at most one shift per step; an upshift only above
        # upshift_speed and a downshift only below downshift_speed (engine speed in the previous
        # gear); right after a shift the engine speed in the new gear is at least shift_margin
        # inside the opposite threshold, so the next step does not shift back.
        moving = ~rest
        radius = np.full(len(trace.speed), signals.radius)
        before, after = (
            gear_engine_speed(gear, trace.speed, signals.slip_rear, radius, p)
            for gear in (signals.previous_gear, signals.gear)
        )
        change = signals.gear - signals.previous_gear
        assert np.all(np.abs(change[moving]) <= 1)
        up, down = moving & (change == 1), moving & (change == -1)
        assert np.all(before[up] > p.upshift) and np.all(after[up] >= p.downshift + margin)
        assert np.all(before[down] < p.downshift) and np.all(after[down] <= p.upshift - margin)
        assert np.all(after[moving & (signals.gear < p.top_gear)] <= p.upshift)
        assert np.all(after[moving & (signals.gear > 1)] >= p.downshift)
        # Engine speed: idle floor (binding in first gear only), governed maximum, catalogue.
        n = signals.engine_speed
        assert np.all(n >= p.idle) and np.all(n <= speed_max)
        assert np.all((rpm_low <= n) & (n <= rpm_high))
        floored = moving & (after < p.idle)
        assert np.all(signals.gear[floored] == 1)
        assert np.array_equal(n[moving], np.maximum(p.idle, after[moving]))
