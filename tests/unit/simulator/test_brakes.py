"""Brake pressure, brake switch and disc heating power (spec section 4, NFR-15)."""

import math

import numpy as np
import pytest
from driven import drive_with_steering
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.catalogue import catalogue
from simulator.config_files import plausibility, van_model
from simulator.constants import GRAVITY, PHYSICS_DT
from simulator.model.brakes import (
    BrakeParameters,
    brake_pedal_pressed,
    disc_heating_power,
    held_steps,
    hold_force,
    master_cylinder_pressure,
)
from simulator.model.longitudinal import LongitudinalParameters, split_forces
from simulator.profile.road import StopKind
from simulator.profile_config import drive_profile

BAR = 1e5
NO = np.zeros(1, dtype=bool)
YES = np.ones(1, dtype=bool)
ZERO = np.zeros(1)


def params() -> BrakeParameters:
    return BrakeParameters.from_configs(van_model(), drive_profile())


def test_pressure_follows_the_friction_brake_force() -> None:
    """Laden van at -4 m/s2, flat road, 10 m/s (test_wheels): F_brake = 11478.38 N, so
    p = 10 bar/(m/s2) * 11478.38 N / 2800 kg = 40.99 bar. No brake force, no pressure."""
    longitudinal = LongitudinalParameters.from_model(van_model())
    forces = split_forces(
        np.array([-4.0]), np.array([10.0]), np.zeros(1), np.array([3300.0]), 1.2, longitudinal
    )
    pressure = master_cylinder_pressure(forces.brake, NO, NO, ZERO, params())
    assert pressure[0] / BAR == pytest.approx(10 * 11478.38055 / 2800, rel=1e-9)
    assert pressure[0] / BAR == pytest.approx(40.99, abs=0.005)
    assert master_cylinder_pressure(np.zeros(1), NO, NO, ZERO, params())[0] == 0


def test_brake_switch_threshold() -> None:
    """Pressed exactly above brake_switch_pressure (2 bar); at 2 bar it is not pressed."""
    pressure = np.array([0.0, 1.99, 2.0, 2.01, 10.0]) * BAR
    assert brake_pedal_pressed(pressure, params()).tolist() == [False, False, False, True, True]


def test_pressure_at_rest() -> None:
    """Held step (v = 0 at both ends): the driver holds 10 bar (pedal pressed), even where only
    a small force holds the van on a slope (246 N on 0.8 %); delivery stop: pedal released."""
    p = params()
    held = held_steps(np.array([0.0, 0.0, 0.0, 0.5]), np.array([0.0, 0.0, 0.4, 0.0]))
    assert held.tolist() == [True, True, False, False]
    holding = hold_force(np.array([3300.0]), np.array([0.0076]))
    assert holding[0] == pytest.approx(246, abs=1)
    on_slope = master_cylinder_pressure(np.zeros(1), YES, NO, holding, p)
    delivery = master_cylinder_pressure(np.zeros(1), YES, YES, holding, p)
    assert on_slope[0] == 10 * BAR and brake_pedal_pressed(on_slope, p)[0]
    assert delivery[0] == 0 and not brake_pedal_pressed(delivery, p)[0]


@pytest.mark.parametrize("grade", [-0.10, 0.10])
def test_pressure_holds_a_laden_van_on_a_ten_percent_grade(grade: float) -> None:
    """Final stop on a 10 % grade: 3300 * g * sin(atan 0.1) = 3220.3 N needs
    10 * 3220.3 / 2800 = 11.50 bar, above the 10 bar standstill pressure."""
    holding = hold_force(np.array([3300.0]), np.array([grade]))
    assert holding[0] == pytest.approx(3300 * GRAVITY * math.sin(math.atan(0.1)), rel=1e-12)
    pressure = master_cylinder_pressure(np.zeros(1), YES, NO, holding, params())
    assert pressure[0] / BAR == pytest.approx(10 * holding[0] / 2800, rel=1e-12)
    assert pressure[0] / BAR == pytest.approx(11.50, abs=0.005)


def test_a_delivery_stop_must_be_held() -> None:
    with pytest.raises(ValueError, match="at rest over the whole step"):
        master_cylinder_pressure(np.zeros(1), NO, YES, ZERO, params())


def test_disc_power_on_the_step_that_comes_to_rest() -> None:
    """The van moves 0.0505 m in its last step (speed 1.0 -> 0.01 m/s, then set to 0): the discs
    get energy_to_disc * F_brake * 0.0505 m, whatever the speed at the end of the step."""
    p = params()
    force, distance = np.array([3000.0]), np.array([0.0505])
    power = disc_heating_power(force, distance, NO, p)
    assert power.sum() * PHYSICS_DT == pytest.approx(0.9 * 3000.0 * 0.0505, rel=1e-12)
    assert power[0, 0] == power[0, 1] == pytest.approx(power.sum() * 0.35, rel=1e-12)
    assert power[0, 2] == power[0, 3] == pytest.approx(power.sum() * 0.15, rel=1e-12)


def test_no_disc_power_on_a_held_step_that_moves() -> None:
    """Seed 11, trip 3: on one step a stop cancels the pull-away the speed plan commanded from
    rest. The model still moves the van about 6 mm on that step (spec section 8), but the brakes
    hold it by static friction with F_hold and the discs get no energy."""
    p = params()
    moving_holds = 0
    for driven in drive_with_steering(11, 3, 300.0, 10.0):
        trace = driven.trace
        held = held_steps(trace.speed, np.append(trace.speed[1:], 0.0))
        distance = np.diff(driven.positions)
        power = disc_heating_power(trace.brake_force, distance, held, p)
        assert np.all(power[held] == 0)
        moving_holds += int(np.sum(held & (distance > 0) & (trace.brake_force > 0)))
    assert moving_holds > 0


@pytest.mark.parametrize("force", [-1.0, np.nan])
def test_negative_or_nan_brake_force_is_rejected(force: float) -> None:
    with pytest.raises(ValueError, match="non-negative and not NaN"):
        master_cylinder_pressure(np.array([force]), NO, NO, ZERO, params())
    with pytest.raises(ValueError, match="non-negative and not NaN"):
        disc_heating_power(np.array([force]), np.array([0.1]), NO, params())


def test_worst_case_pressure_stays_in_the_catalogue_range() -> None:
    """Laden van braking at approach_decel + tolerance on the steepest descent, engine drag
    deducted, rolling resistance and air drag ignored (they would lower it): 3300 * (1.7
    + g sin(atan 0.10) - 0.4) = 7511 N, i.e. 26.8 bar."""
    model, profile = van_model(), drive_profile()
    mass = model.si("mass_empty") + model.si("payload_max")
    grade = math.sin(math.atan(profile.si("downhill_grade_max")))
    decel = profile.si("approach_decel") + plausibility().si("approach_decel_tolerance")
    force = mass * (decel + GRAVITY * grade - model.si("engine_braking_decel"))
    pressure = master_cylinder_pressure(np.array([force]), NO, NO, ZERO, params())[0] / BAR
    assert pressure == pytest.approx(26.8, abs=0.05)
    low, high = catalogue().signal("master_cyl_pressure").range
    assert low <= pressure <= high


def test_long_trips_hold_traffic_and_delivery_stops() -> None:
    """Eight 20-minute trips with fixed inputs reach both kinds of stop: 10 bar and the pedal
    pressed at traffic stops, 0 bar and the pedal released at delivery stops. No step at rest
    records traction, including the unmarked rest steps before a stop captures the van."""
    p = params()
    traffic_steps = delivery_steps = unmarked_steps = 0
    for driven in drive_with_steering(7, 1, 1200.0, 10.0, vans=8):
        trace = driven.trace
        traffic = trace.stopped & ~trace.at_delivery
        held = held_steps(trace.speed, np.append(trace.speed[1:], 0.0))
        holding = hold_force(np.full(len(trace.speed), driven.mass), trace.grade)
        pressure = master_cylinder_pressure(trace.brake_force, held, trace.at_delivery, holding, p)
        assert np.all(pressure[traffic] == 10 * BAR)
        assert np.all(pressure[trace.at_delivery] == 0)
        assert not np.any(trace.at_delivery & ~trace.stopped)
        assert not np.any(held & (trace.engine_force > 0))
        traffic_steps += int(traffic.sum())
        delivery_steps += int(trace.at_delivery.sum())
        unmarked_steps += int((held & ~trace.stopped).sum())
    assert traffic_steps > 1000 and delivery_steps > 1000 and unmarked_steps > 0


@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    duration=st.floats(min_value=60.0, max_value=300.0),
    ambient=st.floats(min_value=-5.0, max_value=30.0),
)
def test_driven_braking_respects_the_invariants(
    seed: int, trip: int, duration: float, ambient: float
) -> None:
    p, profile = params(), drive_profile()
    high = catalogue().signal("master_cyl_pressure").range[1] * BAR
    for driven in drive_with_steering(seed, trip, duration, ambient):
        trace = driven.trace
        traffic = trace.stopped & ~trace.at_delivery
        next_speed = np.append(trace.speed[1:], 0.0)  # 0 after the last step: ignition off
        held = held_steps(trace.speed, next_speed)
        holding = hold_force(np.full(len(trace.speed), driven.mass), trace.grade)
        pressure = master_cylinder_pressure(trace.brake_force, held, trace.at_delivery, holding, p)
        # Range, and the brake switch from the pressure.
        assert np.all(pressure >= 0) and np.all(pressure <= high)
        pressed = brake_pedal_pressed(pressure, p)
        assert np.array_equal(pressed, pressure > p.switch_pressure)
        # Traction and friction braking never at the same time.
        assert not np.any((trace.engine_force > 0) & (trace.brake_force > 0))
        # At rest the brakes hold the van (the parking brake at a delivery): no engine force, and
        # the recorded friction brake force is F_hold. The pressure above comes from `holding`,
        # not from this force.
        assert np.all(trace.engine_force[held] == 0)
        assert np.array_equal(trace.brake_force[held], holding[held])
        # Disc energy over each step that is not held: energy_to_disc * F_brake * the distance
        # moved.
        distance = np.diff(driven.positions)
        power = disc_heating_power(trace.brake_force, distance, held, p)
        energy = power.sum(axis=1) * PHYSICS_DT
        expected = p.energy_to_disc * trace.brake_force * distance
        assert np.allclose(energy[~held], expected[~held], rtol=1e-12)
        # Held steps (v = 0 at both ends) heat nothing. They are the stops, plus one step at rest
        # when the exact approach law brings the van to 0 before the stop captures it: that step
        # is followed by the stop, or is the trip's last step (at the final stop, before
        # ignition off, possibly on a steep descent). Held outside a delivery, the driver holds
        # at least 10 bar and enough for the slope (pedal pressed), so the pedal does not drop
        # out between the approach and the stop.
        assert np.all(power[held] == 0)
        hold = held & ~trace.at_delivery
        needed = np.maximum(p.standstill_pressure, p.pressure_gain * holding / p.reference_mass)
        assert np.array_equal(pressure[hold], needed[hold]) and np.all(pressed[hold])
        assert np.all(held[trace.stopped])
        waiting = np.flatnonzero(held & ~trace.stopped)
        last = len(trace.speed) - 1
        assert all(k == last or trace.stopped[k + 1] for k in waiting)
        # Traffic stops lie inside urban segments, where the held pressure must hold the van;
        # a delivery stop sits at an urban segment's end and may take the next segment's grade,
        # the van then held by the parking brake (not modelled, spec section 8).
        assert np.all(np.abs(trace.grade[traffic]) <= profile.si("urban_grade_max"))
        capture = profile.si("stop_speed_threshold") * PHYSICS_DT
        final = driven.road.final_stop
        assert final is not None
        if last in waiting:  # the stop captures the van where the step leaves it
            assert abs(trace.next_position[last] - final.position) <= capture
        assert np.all(pressure[trace.at_delivery] == 0) and not np.any(pressed[trace.at_delivery])
        # Each delivery hold is at a delivery stop of the road, each traffic hold at a traffic
        # stop (or at the final stop a stop became when the trip time ran out there).
        for k in np.flatnonzero(trace.stopped):
            stop = min(driven.road.stops, key=lambda s: abs(s.position - trace.position[k]))
            assert abs(stop.position - trace.position[k]) <= capture
            expected = StopKind.DELIVERY if trace.at_delivery[k] else StopKind.TRAFFIC
            assert stop.kind in (expected, StopKind.FINAL)
