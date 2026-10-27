"""Brake disc and tyre temperatures, tyre pressures (spec section 4, NFR-15)."""

import math

import numpy as np
import pytest
from driven import DrivenTrip, drawn, drive_with_steering
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.catalogue import catalogue
from simulator.config_files import plausibility, van_model
from simulator.constants import PHYSICS_DT
from simulator.model.brakes import BrakeParameters, held_steps
from simulator.model.forces import (
    air_density,
    drag_force,
    engine_braking_force,
    grade_force,
    rolling_resistance_force,
)
from simulator.model.thermal import (
    DiscParameters,
    TyreParameters,
    disc_cooling_tau,
    disc_step,
    tyre_pressure,
    tyre_step,
    tyre_target_temperature,
)
from simulator.plausibility import disc_rise_ratio
from simulator.profile_config import drive_profile, duty_cycle

BAR = 1e5
ATMOSPHERIC = 1.013 * BAR  # duty_cycle.yaml
AMBIENT = 10.0
NOMINAL = np.array([[3700.0, 3700.0, 2800.0, 2800.0]])  # J/K, van_model.yaml
SHARES = np.array([0.35, 0.35, 0.15, 0.15])  # front_brake_share 0.70 split per wheel


def params() -> DiscParameters:
    return DiscParameters.from_model(van_model())


def brakes() -> BrakeParameters:
    return BrakeParameters.from_configs(van_model(), drive_profile())


def one(value: float) -> np.ndarray:
    return np.array([value])


def tyres() -> TyreParameters:
    return TyreParameters.from_model(van_model())


def capacities(seed: int, number: int) -> np.ndarray:
    """Disc heat capacities drawn for VAN-<number>, shape (1, 4)."""
    return np.array([drawn(seed, number).disc_heat_capacity])


def tyre_temperatures(driven: DrivenTrip, ambient: float) -> np.ndarray:
    """Tyre temperatures of a trip starting at ambient, (steps + 1, 4): one row per step
    boundary."""
    trace = driven.trace
    next_speed = np.append(trace.speed[1:], 0.0)  # 0 after the last step: ignition off
    p = tyres()
    temp = np.full((1, 4), ambient)
    rows = [temp[0]]
    for k in range(len(trace.speed)):
        s = slice(k, k + 1)
        temp = tyre_step(temp, trace.speed[s], next_speed[s], ambient, p)
        rows.append(temp[0])
    return np.array(rows)


def disc_temperatures(
    driven: DrivenTrip, ambient: float, capacity: np.ndarray, p: DiscParameters | None = None
) -> np.ndarray:
    """Disc temperatures of a trip starting at ambient, (steps + 1, 4): one row per step
    boundary."""
    trace = driven.trace
    next_speed = np.append(trace.speed[1:], 0.0)  # 0 after the last step: ignition off
    distance = np.diff(driven.positions)
    disc, brake = p or params(), brakes()
    temp = np.full((1, 4), ambient)
    rows = [temp[0]]
    for k in range(len(trace.speed)):
        s = slice(k, k + 1)
        temp = disc_step(
            temp,
            trace.brake_force[s],
            trace.speed[s],
            next_speed[s],
            distance[s],
            ambient,
            capacity,
            disc,
            brake,
        )
        rows.append(temp[0])
    return np.array(rows)


def test_cooling_time_constant_is_affine_in_speed() -> None:
    """1/tau = (1/600 s)(1 + v / v_c), v_c = 20 / (600 / 150 - 1) = 6.667 m/s from the config:
    600 s at rest, 300 s at v_c, 150 s at the 20 m/s reference speed."""
    p = params()
    assert p.cooling_speed == pytest.approx(20 / 3, rel=1e-12)
    tau = disc_cooling_tau(np.array([0.0, 20 / 3, 20.0]), p)
    assert tau == pytest.approx([600.0, 300.0, 150.0], rel=1e-12)


def test_one_step_follows_the_exact_first_order_update() -> None:
    """Discs at ambient + 50 K; the van moves 0.0505 m while coming to rest (1.0 -> 0 m/s) with
    3000 N of friction brake force. Mean speed 0.5 m/s: tau = 600 / (1 + 0.5 / 6.667) = 558.14 s.
    P = 0.9 * 3000 N * 0.0505 m / 0.1 s = 1363.5 W, i.e. 477.2 W per front and 204.5 W per rear
    disc: T' = 10 + 50 e^(-0.1/tau) + (P_i / C_i) tau (1 - e^(-0.1/tau)), 60.003939 degC front,
    59.998346 degC rear (heating beats cooling on the front discs only)."""
    temp = disc_step(
        np.full((1, 4), AMBIENT + 50),
        one(3000.0),
        one(1.0),
        one(0.0),
        one(0.0505),
        AMBIENT,
        NOMINAL,
        params(),
        brakes(),
    )[0]
    tau = 600 / (1 + 0.5 / (20 / 3))
    decay = 1 - math.exp(-0.1 / tau)
    power = 0.9 * 3000 * 0.0505 / 0.1 * SHARES
    expected = AMBIENT + 50 * (1 - decay) + power / NOMINAL[0] * tau * decay
    assert temp == pytest.approx(expected, rel=1e-12)
    assert temp == pytest.approx([60.003939, 60.003939, 59.998346, 59.998346], abs=1e-6)


def test_constant_braking_reaches_the_steady_state() -> None:
    """500 N at a constant 20 m/s (2 m per step): P = 0.9 * 500 * 20 = 9000 W, tau = 150 s, so
    the rise tends to P_i tau / C_i: 127.70 K front, 72.32 K rear (20 tau simulated)."""
    temp = np.full((1, 4), AMBIENT)
    p, brake = params(), brakes()
    for _ in range(int(20 * 150 / PHYSICS_DT)):
        temp = disc_step(
            temp, one(500.0), one(20.0), one(20.0), one(2.0), AMBIENT, NOMINAL, p, brake
        )
    expected = 9000 * SHARES * 150 / NOMINAL[0]
    assert temp[0] - AMBIENT == pytest.approx(expected, rel=1e-6)
    assert temp[0] - AMBIENT == pytest.approx([127.70, 127.70, 72.32, 72.32], abs=0.005)


def test_standstill_cooling_halves_the_rise_in_600_ln_2() -> None:
    """At rest the rise decays as e^(-t / 600 s): exactly so after every step, and halved after
    600 ln 2 = 415.9 s (4159 steps: 49.999 K from 100 K)."""
    temp = np.full((1, 4), AMBIENT + 100)
    p, brake, zero = params(), brakes(), one(0.0)
    for _ in range(4159):
        temp = disc_step(temp, zero, zero, zero, zero, AMBIENT, NOMINAL, p, brake)
    assert temp[0] - AMBIENT == pytest.approx(100 * math.exp(-415.9 / 600), rel=1e-12)
    assert temp[0] - AMBIENT == pytest.approx(50.0, abs=0.002)


def test_rises_scale_with_each_disc_heat_capacity() -> None:
    """Four different capacities, one braking step from ambient: rise_i * C_i / share_i is the
    same for every disc."""
    capacity = np.array([[3600.0, 3800.0, 2700.0, 2900.0]])
    temp = disc_step(
        np.full((1, 4), AMBIENT),
        one(4000.0),
        one(10.0),
        one(9.6),
        one(0.98),
        AMBIENT,
        capacity,
        params(),
        brakes(),
    )
    normalised = (temp[0] - AMBIENT) * capacity[0] / SHARES
    assert normalised == pytest.approx(np.full(4, normalised[0]), rel=1e-12)


def test_sustained_descent_plus_final_stop_stays_in_the_catalogue_range() -> None:
    """Worst sustained case, not a global bound (the range on driven trips is checked by the
    Hypothesis test below). Laden van holding the highest downhill speed (60 km/h) on the steepest
    descent (10 %), lightest front disc (3700 * 0.95 * 0.98 = 3444.7 J/K), hottest ambient (30
    degC, also the thinnest air), tau(16.67 m/s) = 171.4 s, then a full stop from that speed.
    Exact: F_brake = grade - engine drag - rolling - air drag = 1368 N, 20.5 kW to the discs, front
    steady rise 357.4 K. Conservative, rolling resistance and air drag ignored: 1900 N, 28.5 kW,
    496.4 K. Stop from 60 km/h, kinetic energy only, no cooling: 0.9 * 0.35 * 1/2 * 3300 *
    16.67^2 / 3444.7 = 41.9 K. Conservative total 30 + 496.4 + 41.9 = 568.4 degC."""
    model, profile, duty = van_model(), drive_profile(), duty_cycle()
    mass = np.array([model.si("mass_empty") + model.si("payload_max")])
    grade = np.array([-profile.si("downhill_grade_max")])
    speed = profile.si("downhill_target_speed_max")
    ambient = duty.value("ambient_temp_max")
    density = air_density(duty.si("atmospheric_pressure"), ambient)
    capacity = (
        model.si("disc_heat_capacity_front")
        * (1 - model.si("disc_heat_capacity_vehicle_spread"))
        * (1 - model.si("disc_heat_capacity_wheel_spread"))
    )
    share = model.si("front_brake_share") / 2
    to_disc = model.si("brake_energy_to_disc")
    tau = disc_cooling_tau(np.array([speed]), params())[0]
    conservative_force = grade_force(mass, grade) - engine_braking_force(
        mass, model.si("engine_braking_decel")
    )
    exact_force = (
        conservative_force
        - rolling_resistance_force(mass, grade, model.si("rolling_resistance"))
        - drag_force(np.array([speed]), density, model.si("drag_area"))
    )

    def steady_rise(force: np.ndarray) -> float:
        return float(share * to_disc * force[0] * speed * tau / capacity)

    stop_rise = share * to_disc * 0.5 * mass[0] * speed**2 / capacity
    assert capacity == pytest.approx(3444.7, abs=0.05)
    assert tau == pytest.approx(171.43, abs=0.005)
    assert exact_force[0] == pytest.approx(1368, abs=0.5)
    assert steady_rise(exact_force) == pytest.approx(357.4, abs=0.05)
    assert conservative_force[0] == pytest.approx(1900, abs=0.5)
    assert steady_rise(conservative_force) == pytest.approx(496.4, abs=0.05)
    assert stop_rise == pytest.approx(41.9, abs=0.05)
    total = ambient + steady_rise(conservative_force) + stop_rise
    assert total == pytest.approx(568.4, abs=0.05)
    assert total <= catalogue().signal("brake_disc_temp_est_fl").range[1]


def test_held_steps_cool_as_at_standstill() -> None:
    """Seed 11, trip 3: on some held steps a stop cancels a commanded pull-away and the model still
    moves the van (spec section 8). Every held step cools exactly as at standstill, with no heat
    from the holding force over that distance."""
    p = params()
    decay = -math.expm1(-PHYSICS_DT / p.tau_standstill)
    moving_holds = 0
    for number, driven in enumerate(drive_with_steering(11, 3, 300.0, AMBIENT), start=1):
        trace = driven.trace
        temps = disc_temperatures(driven, AMBIENT, capacities(11, number))
        held = held_steps(trace.speed, np.append(trace.speed[1:], 0.0))
        before, after = temps[:-1][held], temps[1:][held]
        assert np.allclose(after, before - (before - AMBIENT) * decay, rtol=0, atol=1e-12)
        distance = np.diff(driven.positions)
        moving_holds += int(np.sum(held & (distance > 0) & (trace.brake_force > 0)))
    assert moving_holds > 0


def test_disc_heat_balances_the_braking_energy_without_cooling() -> None:
    """Cooling switched off (tau 1e12 s): over a whole trip the heat stored in the four discs,
    sum of C_i * rise_i, equals 0.9 * sum of F_brake * distance over the steps that are not held
    (seed 11, trip 3, which has held steps that move)."""
    no_cooling = DiscParameters(tau_standstill=1e12, cooling_speed=params().cooling_speed)
    for number, driven in enumerate(drive_with_steering(11, 3, 300.0, AMBIENT), start=1):
        trace = driven.trace
        capacity = capacities(11, number)
        temps = disc_temperatures(driven, AMBIENT, capacity, no_cooling)
        held = held_steps(trace.speed, np.append(trace.speed[1:], 0.0))
        work = trace.brake_force * np.diff(driven.positions)
        stored = float(np.sum(capacity[0] * (temps[-1] - AMBIENT)))
        assert stored == pytest.approx(0.9 * work[~held].sum(), rel=1e-9)
        assert stored > 0


def test_long_trips_keep_the_axle_rise_ratio_in_its_bounds() -> None:
    """Eight 20-minute trips (seed 7, trip 1): the windowed front/rear ratio of the axle mean rises
    (plausibility.yaml) is evaluated on some windows and stays in [1.69, 1.84]."""
    bounds = plausibility()
    window = round(bounds.si("disc_ratio_window") / PHYSICS_DT)
    low, high = bounds.si("disc_front_rear_ratio_min"), bounds.si("disc_front_rear_ratio_max")
    evaluated = 0
    for number, driven in enumerate(drive_with_steering(7, 1, 1200.0, AMBIENT, vans=8), start=1):
        temps = disc_temperatures(driven, AMBIENT, capacities(7, number))
        ratio = disc_rise_ratio(temps, AMBIENT, window, bounds.si("disc_ratio_min_rear_rise"))
        ratio = ratio[~np.isnan(ratio)]
        assert np.all((low <= ratio) & (ratio <= high))
        evaluated += len(ratio)
    assert evaluated > 0


def test_tyre_target_rises_with_speed_up_to_the_reference_speed() -> None:
    """T_a + 20 K * min(1, v / 20 m/s): +0, +10, +20 K at 0, 10, 20 m/s, still +20 K at 30 m/s."""
    target = tyre_target_temperature(np.array([0.0, 10.0, 20.0, 30.0]), AMBIENT, tyres())
    assert target == pytest.approx(AMBIENT + np.array([0.0, 10.0, 20.0, 20.0]), rel=1e-12)


def test_one_tyre_step_follows_the_exact_first_order_update() -> None:
    """Tyres at ambient + 5 K, speed 9.8 -> 10.2 m/s: mean 10 m/s, target ambient + 10 K, so
    T' = 20 + (15 - 20) e^(-0.1 / 1200) = 15.000416649 degC."""
    temp = tyre_step(np.full((1, 4), AMBIENT + 5), one(9.8), one(10.2), AMBIENT, tyres())[0]
    expected = AMBIENT + 10 - 5 * math.exp(-0.1 / 1200)
    assert temp == pytest.approx(np.full(4, expected), rel=1e-12)
    assert temp == pytest.approx(np.full(4, 15.000416649), abs=1e-9)


def test_tyre_warm_up_at_constant_speed() -> None:
    """At a constant 20 m/s from ambient: rise 20 (1 - e^(-1)) = 12.642 K after one time
    constant (1200 s), 20 K after 20 time constants."""
    p, temp, speed = tyres(), np.full((1, 4), AMBIENT), one(20.0)
    for _ in range(12000):
        temp = tyre_step(temp, speed, speed, AMBIENT, p)
    assert temp[0] - AMBIENT == pytest.approx(np.full(4, 20 * (1 - math.exp(-1))), rel=1e-9)
    assert temp[0, 0] - AMBIENT == pytest.approx(12.642, abs=0.0005)
    for _ in range(19 * 12000):
        temp = tyre_step(temp, speed, speed, AMBIENT, p)
    assert temp[0] - AMBIENT == pytest.approx(np.full(4, 20.0), rel=1e-6)


def test_tyre_pressure_follows_the_ideal_gas_on_absolute_pressure() -> None:
    """Cold 3.8 bar gauge at 20 degC, 1.013 bar atmospheric: 3.8 bar at 20 degC,
    4.813 * 268.15 / 293.15 - 1.013 = 3.390 bar at -5 degC, 4.813 * 313.15 / 293.15 - 1.013
    = 4.128 bar at 40 degC. Each wheel keeps its own cold pressure."""
    p = tyres()
    temps = np.array([[20.0] * 4, [-5.0] * 4, [40.0] * 4])
    pressure = tyre_pressure(temps, np.full(4, 3.8 * BAR), ATMOSPHERIC, p) / BAR
    assert pressure[0] == pytest.approx(np.full(4, 3.8), rel=1e-12)
    assert pressure[1] == pytest.approx(np.full(4, 4.813 * 268.15 / 293.15 - 1.013), rel=1e-12)
    assert pressure[2] == pytest.approx(np.full(4, 4.813 * 313.15 / 293.15 - 1.013), rel=1e-12)
    assert pressure[1:, 0] == pytest.approx([3.3895, 4.1284], abs=5e-5)
    cold = np.array([3.7, 3.75, 3.85, 3.9])
    per_wheel = tyre_pressure(np.full(4, 40.0), cold * BAR, ATMOSPHERIC, p) / BAR
    assert per_wheel == pytest.approx((cold + 1.013) * 313.15 / 293.15 - 1.013, rel=1e-12)


def test_tyre_extremes_stay_in_the_catalogue_ranges() -> None:
    """Coldest day, lowest cold pressure (3.8 - 0.1 bar) at -5 degC: 3.298 bar. Hottest day,
    highest cold pressure (3.9 bar) at the full warm-up rise, 30 + 20 = 50 degC: 4.403 bar.
    Tyre temperatures lie in [-5, 50] degC."""
    model, duty, p = van_model(), duty_cycle(), tyres()
    atmospheric = duty.si("atmospheric_pressure")
    nominal, spread = model.si("tyre_pressure_nominal"), model.si("tyre_pressure_wheel_spread")
    coldest = duty.value("ambient_temp_min")
    hottest = duty.value("ambient_temp_max") + p.warmup_rise
    low = tyre_pressure(np.array([coldest]), np.array([nominal - spread]), atmospheric, p)[0]
    high = tyre_pressure(np.array([hottest]), np.array([nominal + spread]), atmospheric, p)[0]
    assert low / BAR == pytest.approx(3.298, abs=0.0005)
    assert high / BAR == pytest.approx(4.403, abs=0.0005)
    pressure_range = catalogue().signal("tyre_pressure_fl").range
    temp_range = catalogue().signal("tyre_temp_fl").range
    assert pressure_range[0] <= low / BAR < high / BAR <= pressure_range[1]
    assert temp_range[0] <= coldest < hottest <= temp_range[1]


def test_long_trips_warm_the_tyres_with_the_speed_cap_reached() -> None:
    """Eight 20-minute trips (seed 7, trip 1): every van's tyres warm and stay below ambient +
    20 K. Van 2 drives above 20 m/s on most steps: its tyre temperature, replayed step by step from
    the capped target T_a + 20 K * min(1, v / 20 m/s) of the step's mean speed, matches
    tyre_step."""
    decay = math.exp(-PHYSICS_DT / 1200)
    for number, driven in enumerate(drive_with_steering(7, 1, 1200.0, AMBIENT, vans=8), start=1):
        temps = tyre_temperatures(driven, AMBIENT)
        rise = temps[:, 0] - AMBIENT
        assert rise.max() > 1 and rise.max() <= 20
        if number != 2:
            continue
        speed = driven.trace.speed
        mean = (speed + np.append(speed[1:], 0.0)) / 2
        assert np.mean(mean > 20) > 0.5
        replay = [AMBIENT]
        for v in mean:
            target = AMBIENT + 20 * min(1.0, v / 20)
            replay.append(target + (replay[-1] - target) * decay)
        assert np.allclose(temps, np.array(replay)[:, None], rtol=1e-12, atol=0)


@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    duration=st.floats(min_value=60.0, max_value=300.0),
    ambient=st.floats(min_value=-5.0, max_value=30.0),
)
def test_driven_thermal_states_respect_the_invariants(
    seed: int, trip: int, duration: float, ambient: float
) -> None:
    bounds = plausibility()
    rel, absolute = bounds.si("disc_lr_spread_rel_max"), bounds.si("disc_lr_spread_abs_max")
    high = catalogue().signal("brake_disc_temp_est_fl").range[1]
    p_tyre = tyres()
    tyre_low, tyre_high = catalogue().signal("tyre_temp_fl").range
    pressure_low, pressure_high = catalogue().signal("tyre_pressure_fl").range
    largest_step = p_tyre.warmup_rise * -math.expm1(-PHYSICS_DT / p_tyre.tau)
    for number, driven in enumerate(drive_with_steering(seed, trip, duration, ambient), start=1):
        trace = driven.trace
        held = held_steps(trace.speed, np.append(trace.speed[1:], 0.0))
        # Tyres: between ambient and ambient + rise, the four equal; each step lands between the
        # temperature and its target and moves at most rise * (1 - e^(-DT/tau)) (with 1e-9 of
        # relative allowance for rounding); no warming while held.
        tyre = tyre_temperatures(driven, ambient)
        assert np.all(tyre >= ambient) and np.all(tyre <= ambient + p_tyre.warmup_rise)
        assert np.all(tyre == tyre[:, :1])
        assert np.all((tyre_low <= tyre) & (tyre <= tyre_high))
        mean = (trace.speed + np.append(trace.speed[1:], 0.0)) / 2
        target = tyre_target_temperature(mean, ambient, p_tyre)
        start, end = tyre[:-1, 0], tyre[1:, 0]
        assert np.all(np.minimum(start, target) <= end) and np.all(end <= np.maximum(start, target))
        assert np.all(np.abs(end - start) <= largest_step * (1 + 1e-9))
        assert np.all(end[held] <= start[held])
        # Pressure: ideal gas on absolute pressure, (p + p_atm) / T constant per wheel and equal
        # to its value for the cold pressure at 20 degC; catalogue range.
        cold = np.array(drawn(seed, number).tyre_pressure_nominal) * BAR
        pressure = tyre_pressure(tyre, cold, ATMOSPHERIC, p_tyre)
        absolute_per_kelvin = (pressure + ATMOSPHERIC) / (tyre + 273.15)
        expected = (cold + ATMOSPHERIC) / 293.15
        assert np.allclose(absolute_per_kelvin, expected, rtol=1e-12, atol=0)
        assert np.all((pressure_low * BAR <= pressure) & (pressure <= pressure_high * BAR))
        # Discs.
        capacity = capacities(seed, number)
        temps = disc_temperatures(driven, ambient, capacity)
        before, after = temps[:-1], temps[1:]
        # Range: never below ambient (the discs start there and only cool towards it), never
        # above the catalogue range.
        assert np.all(np.isfinite(temps))
        assert np.all(temps >= ambient) and np.all(temps <= high)
        # No friction brake force, or held (static friction): the discs only cool.
        idle = held | (trace.brake_force == 0)
        assert np.all(after[idle] <= before[idle])
        # One tau for the four discs: each rise is proportional to share_i / C_i, so the ratio of
        # the axle mean rises is the vehicle's constant (above 0.01 K of rear rise, where rounding
        # stays far below the tolerance).
        rise = temps - ambient
        front, rear = rise[:, :2].mean(axis=1), rise[:, 2:].mean(axis=1)
        inverse = 1 / capacity[0]
        constant = 0.35 * (inverse[0] + inverse[1]) / (0.15 * (inverse[2] + inverse[3]))
        warm = rear > 0.01
        assert np.allclose(front[warm] / rear[warm], constant, rtol=1e-9, atol=0)
        # Left/right spread on each axle (plausibility.yaml).
        for left, right in ((0, 1), (2, 3)):
            larger = np.maximum(rise[:, left], rise[:, right])
            assert np.all(np.abs(rise[:, left] - rise[:, right]) <= rel * larger + absolute)
