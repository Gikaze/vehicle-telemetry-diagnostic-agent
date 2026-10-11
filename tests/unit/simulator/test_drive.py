"""Speed plan and longitudinal driving (NFR-15): invariants on the true state of driven trips."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.config_files import plausibility, van_model
from simulator.constants import PHYSICS_DT
from simulator.model.drive import Trace, drive_trips
from simulator.model.forces import air_density
from simulator.model.longitudinal import (
    LongitudinalParameters,
    acceleration_limits,
    split_forces,
    traction_force_limit,
)
from simulator.profile.road import Road, StopKind
from simulator.profile.speed_plan import DriverParameters
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
