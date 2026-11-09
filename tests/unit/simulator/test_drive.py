"""Speed plan and longitudinal driving (NFR-15): invariants on the true state of driven trips."""

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.config_files import plausibility, van_model
from simulator.constants import PHYSICS_DT
from simulator.model.drive import Trace, drive_trips, step
from simulator.model.forces import air_density
from simulator.model.longitudinal import (
    LongitudinalParameters,
    acceleration_limits,
    split_forces,
    traction_force_limit,
)
from simulator.profile.road import Road, StopKind
from simulator.profile.speed_plan import DriverParameters, SpeedPlanner
from simulator.profile_config import drive_profile, trip_overrun_bound

STOP_TOLERANCE = 0.05  # m: a van stops within 5 cm of a stop's position


def configs():
    model, profile = van_model(), drive_profile()
    return (
        model,
        profile,
        DriverParameters.from_configs(profile, model),
        LongitudinalParameters.from_model(model),
    )


def drive(seed: int, vehicles: list[str], trip: int, duration: float, masses, ambient: float):
    model, profile, driver, longitudinal = configs()
    roads = [Road(seed, vehicle, 0, trip, profile) for vehicle in vehicles]
    density = air_density(101300, ambient)
    traces = drive_trips(
        roads,
        [duration] * len(roads),
        list(masses),
        density,
        longitudinal,
        driver,
        time_limit=duration + 2 * trip_overrun_bound(profile),
    )
    return roads, traces


def curvature(road: Road, position: np.ndarray) -> np.ndarray:
    total = np.zeros_like(position)
    for curve in road.curves:
        total += curve.curvature(position)
    return total


def check_invariants(road: Road, trace: Trace, mass: float) -> None:
    model, profile, _, _ = configs()
    decel_limit = profile.si("approach_decel") + plausibility().si("approach_decel_tolerance")
    # Speed and acceleration stay within the model.
    assert trace.speed.min() >= 0
    assert trace.accel.min() >= -decel_limit - 1e-9
    assert trace.accel.max() <= model.si("accel_max") + 1e-9
    # Traction power never exceeds engine_power_max.
    assert np.max(trace.engine_force * trace.speed) <= model.si("engine_power_max") * (1 + 1e-9)
    # Lateral acceleration v^2 * curvature stays below accel_lat_max.
    lateral = trace.speed**2 * np.abs(curvature(road, trace.position))
    assert lateral.max() <= model.si("accel_lat_max")
    # Every intermediate stop is reached (v = 0 within 5 cm) and held for its duration.
    for stop in road.stops:
        if stop.kind is StopKind.FINAL:
            continue
        held = trace.stopped & (np.abs(trace.position - stop.position) <= STOP_TOLERANCE)
        assert held.sum() * PHYSICS_DT >= stop.duration - PHYSICS_DT - 1e-9, stop
    # Ignition off at the final stop, at rest, within the overrun bound of the planned duration.
    final = road.final_stop
    assert final is not None
    assert trace.speed[-1] * PHYSICS_DT + abs(trace.position[-1] - final.position) <= 0.1
    assert trace.duration - PHYSICS_DT <= trace.end_time
    assert trace.end_time <= trace.duration + trip_overrun_bound(profile)


# Each example drives eight vans for up to five simulated minutes, so this test runs a fifth of
# the active profile's examples (20 in dev, 40 in ci, 400 in thorough).
@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    duration=st.floats(min_value=60.0, max_value=300.0),
    ambient=st.floats(min_value=-5.0, max_value=30.0),
)
def test_driven_trips_respect_the_invariants(
    seed: int, trip: int, duration: float, ambient: float
) -> None:
    """Eight vans side by side, empty to laden, on random roads and durations."""
    vehicles = [f"VAN-{n:03d}" for n in range(1, 9)]
    masses = np.linspace(2300, 3300, len(vehicles))
    roads, traces = drive(seed, vehicles, trip, duration, masses, ambient)
    for road, trace, mass in zip(roads, traces, masses, strict=True):
        check_invariants(road, trace, mass)


def test_long_trips_regression() -> None:
    """Twenty 20-minute trips with fixed inputs: the same invariants over many events."""
    vehicles = [f"VAN-{n:03d}" for n in range(1, 21)]
    masses = np.linspace(2300, 3300, len(vehicles))
    roads, traces = drive(7, vehicles, 1, 1200.0, masses, 10.0)
    assert sum(len(road.stops) for road in roads) > 100
    for road, trace, mass in zip(roads, traces, masses, strict=True):
        check_invariants(road, trace, mass)


def test_driving_is_reproducible_and_independent_of_the_other_vans() -> None:
    _, alone = drive(7, ["VAN-003"], 1, 300.0, [2800], 10.0)
    _, together = drive(7, ["VAN-001", "VAN-002", "VAN-003"], 1, 300.0, [2800] * 3, 10.0)
    assert np.array_equal(alone[0].speed, together[2].speed)
    assert np.array_equal(alone[0].position, together[2].position)


@pytest.mark.parametrize(("speed_value", "power_kw"), [(20.0, 91.9), (25.0, 119.3)])
def test_power_needed_to_pull_away(speed_value: float, power_kw: float) -> None:
    """Laden van (3300 kg), 1.2 m/s2, 20 degC: m a + drag + rolling resistance, times speed."""
    _, _, _, longitudinal = configs()
    speed = np.array([speed_value])
    density = air_density(101300, 20.0)
    forces = split_forces(
        np.array([1.2]), speed, np.zeros(1), np.array([3300.0]), density, longitudinal
    )
    assert float(forces.engine[0] * speed[0]) / 1000 == pytest.approx(power_kw, abs=0.1)


def planner_at_rest_before_stop(gap: float) -> SpeedPlanner:
    """A van at v = 0 on the approach to a stop `gap` metres ahead, no other term binding."""
    _, profile, driver, _ = configs()
    planner = SpeedPlanner.for_trips([Road(7, "VAN-001", 0, 0, profile)], [600.0], driver)
    planner.segment_speed[:] = 10.0
    planner.curve_speed[:] = np.inf
    planner.ramp_position[:], planner.ramp_speed[:] = 0.0, 10.0
    planner.window_position[:] = np.inf
    planner.window_speed[:] = np.inf
    planner.window_position[0, 0], planner.window_speed[0, 0] = 100.0 + gap, 0.0
    planner.next_stop_position[0] = 100.0 + gap
    planner.next_stop_duration[0] = 30.0
    return planner


def test_van_at_rest_short_of_a_stop_moves_up_to_it() -> None:
    """v = 0 with a stop 5 cm ahead (beyond the capture distance): the van must not stay put.

    The exact approach law alone asks -(0 - 0) / (2 gap) = 0 there, which would hold the van
    at rest forever; the state is not shown unreachable, so the command must handle it.
    """
    _, profile, driver, _ = configs()
    planner = planner_at_rest_before_stop(0.05)
    s, v = np.array([100.0]), np.array([0.0])
    _, accel = planner.command(s, v)
    assert accel[0] > 0
    # Driven on, it reaches the stop and is captured there within a few steps.
    for _ in range(50):
        _, accel = planner.command(s, v)
        v_next = np.maximum(v + accel * PHYSICS_DT, 0.0)
        s_next = s + (v + v_next) / 2 * PHYSICS_DT
        planner.advance_time()
        v_next = planner.apply_stops(s_next, v_next, v)
        s, v = s_next, v_next
        if planner.stopped[0]:
            break
    assert planner.stopped[0]
    assert abs(s[0] - 100.05) <= driver.stop_speed_threshold * PHYSICS_DT


def slot_state(planner: SpeedPlanner, i: int) -> dict:
    """Every per-slot array of the planner, for slot i."""
    return {
        name: value[i]
        for name, value in vars(planner).items()
        if isinstance(value, np.ndarray) and value.shape[:1] == (planner.n,)
    }


def test_a_reused_slot_starts_like_a_fresh_planner() -> None:
    """start_trip resets every per-slot state left by the previous trip of that slot."""
    _, profile, driver, _ = configs()
    planner = SpeedPlanner.for_trips([Road(7, "VAN-001", 0, 0, profile)], [600.0], driver)
    # Leave state behind, as a trip that ended at a stop in a curve would.
    planner.time[0], planner.stopped[0], planner.stop_until[0] = 812.3, True, 830.0
    planner.final_inserted[0], planner.curve_speed[0], planner.curve_end[0] = True, 4.5, 950.0
    planner.ramp_position[0], planner.ramp_speed[0] = 900.0, 3.0
    planner.ended[0], planner.end_time[0] = True, 812.3
    road = Road(7, "VAN-001", 0, 1, profile)
    planner.start_trip(0, road, 900.0)
    fresh = SpeedPlanner.for_trips([Road(7, "VAN-001", 0, 1, profile)], [900.0], driver)
    reused, expected = slot_state(planner, 0), slot_state(fresh, 0)
    assert reused.keys() == expected.keys()
    assert {"time", "stopped", "window_position"} <= reused.keys()  # enumeration is not empty
    for name, value in expected.items():
        assert np.array_equal(reused[name], value, equal_nan=True), name
    # The slot's Python-side trip state, field by field (the road compared by its content).
    reused_trip, fresh_trip = planner.trips[0], fresh.trips[0]
    assert reused_trip is not None and fresh_trip is not None
    for name, value in vars(fresh_trip).items():
        if name == "road":
            assert reused_trip.road.plans == value.plans
            assert reused_trip.road.final_stop == value.final_stop
        else:
            assert getattr(reused_trip, name) == value, name


def drive_staggered(starts: list[int], trips: list[tuple[str, int]], duration: float = 300.0):
    _, profile, driver, longitudinal = configs()
    roads = [Road(7, vehicle, 0, trip, profile) for vehicle, trip in trips]
    return drive_trips(
        roads,
        [duration] * len(roads),
        [2800.0] * len(roads),
        air_density(101300, 10.0),
        longitudinal,
        driver,
        time_limit=duration + 2 * trip_overrun_bound(profile),
        start_steps=starts,
    )


def same_trace(a: Trace, b: Trace) -> bool:
    return all(
        np.array_equal(getattr(a, f.name), getattr(b, f.name))
        for f in dataclasses.fields(Trace)
        if f.name != "start_step"
    )


@pytest.mark.parametrize("k", [1, 137, 1500])
def test_a_trip_started_later_is_independent_and_shifted(k: int) -> None:
    """Slot 1 starts at fleet step k while slot 0 drives: slot 0 is unchanged (bit for bit) and
    slot 1 drives exactly as a fresh planner from step 0, shifted by k steps."""
    trips = [("VAN-001", 1), ("VAN-002", 2)]
    together = drive_staggered([0, k], trips)
    alone_0 = drive_staggered([0], trips[:1])[0]
    alone_1 = drive_staggered([0], trips[1:])[0]
    assert together[1].start_step == k and alone_1.start_step == 0
    assert same_trace(together[0], alone_0)
    assert same_trace(together[1], alone_1)


def test_empty_and_ended_slots_are_held_at_rest() -> None:
    """Slot 0 drives and ends, slot 1 starts after that, slot 2 never starts: a slot without a
    trip in progress has v = 0, a = 0, zero forces and reference speed, and nothing is NaN."""
    _, profile, driver, longitudinal = configs()
    planner = SpeedPlanner(3, driver)
    mass, density = np.full(3, 2800.0), air_density(101300, 10.0)
    s, v = np.zeros(3), np.zeros(3)
    planner.start_trip(0, Road(7, "VAN-001", 0, 0, profile), 60.0)
    seen_ended = False
    for k in range(4000):
        if k == 2500:
            planner.start_trip(1, Road(7, "VAN-002", 0, 0, profile), 60.0)
            s[1], v[1] = 0.0, 0.0
        result = step(planner, s, v, mass, density, longitudinal)
        for name in ("speed", "accel", "speed_ref", "grade", "engine_force", "brake_force"):
            values = getattr(result, name)
            assert np.all(np.isfinite(values)), name
            assert np.all(values[~result.active] == 0), name
        assert np.all(result.next_speed[~result.active] == 0)
        assert np.array_equal(result.next_position[~result.active], s[~result.active])
        seen_ended |= bool(k < 2500 and not result.active[0])
        s, v = result.next_position, result.next_speed
    assert seen_ended, "slot 0 must end before slot 1 starts"
    assert planner.ended.all()
    assert not result.active[2]


def test_a_slot_with_a_trip_in_progress_cannot_start_another() -> None:
    _, profile, driver, _ = configs()
    planner = SpeedPlanner.for_trips([Road(7, "VAN-001", 0, 0, profile)], [600.0], driver)
    with pytest.raises(RuntimeError, match="already has a trip in progress"):
        planner.start_trip(0, Road(7, "VAN-001", 0, 1, profile), 600.0)


def test_power_limit_caps_acceleration_at_speed() -> None:
    """At 25 m/s the 100 kW limit caps a laden van below the 1.2 m/s2 pull-away rate."""
    model, _, _, longitudinal = configs()
    mass, speed = np.array([3300.0]), np.array([25.0])
    density = air_density(101300, 20.0)
    _, highest = acceleration_limits(speed, np.zeros(1), mass, density, longitudinal)
    assert highest[0] < 1.2
    capped = split_forces(highest, speed, np.zeros(1), mass, density, longitudinal)
    assert float(capped.engine[0] * speed[0]) == pytest.approx(model.si("engine_power_max"))


def test_traction_force_limit_has_a_speed_floor() -> None:
    _, _, _, longitudinal = configs()
    at_rest = traction_force_limit(np.array([0.0, 1.0, 3.0, 10.0]), longitudinal)
    assert at_rest[0] == at_rest[1] == at_rest[2] == pytest.approx(100000 / 3)
    assert at_rest[3] == pytest.approx(10000)


@pytest.mark.parametrize("accel", [-3.0, -1.0, -0.2, 0.0, 0.5])
def test_force_split_balances(accel: float) -> None:
    """engine - brake = required; brakes only beyond full engine braking."""
    _, _, _, longitudinal = configs()
    mass, speed, grade = np.array([3000.0]), np.array([15.0]), np.array([-0.05])
    forces = split_forces(np.array([accel]), speed, grade, mass, 1.2, longitudinal)
    assert float(forces.engine[0] - forces.brake[0]) == pytest.approx(float(forces.required[0]))
    engine_drag = mass[0] * longitudinal.engine_braking_decel
    assert forces.engine[0] >= -engine_drag - 1e-9
    assert (forces.brake[0] > 0) == (forces.required[0] < -engine_drag)


def test_descent_needs_brakes_to_hold_speed() -> None:
    _, _, _, longitudinal = configs()
    forces = split_forces(
        np.zeros(1), np.array([15.3]), np.array([-0.10]), np.array([3300.0]), 1.204, longitudinal
    )
    assert float(forces.brake[0] * 15.3) / 1000 == pytest.approx(21.3, abs=0.2)
