"""Steering, yaw rate and lateral acceleration (spec section 4, NFR-15)."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.catalogue import catalogue
from simulator.config_files import van_model
from simulator.constants import PHYSICS_DT
from simulator.model.drive import drive_trips
from simulator.model.forces import air_density
from simulator.model.longitudinal import LongitudinalParameters
from simulator.model.steering import (
    CurveTracker,
    SteeringParameters,
    steering_state,
    steering_wheel_angle,
)
from simulator.profile.road import Road
from simulator.profile.speed_plan import DriverParameters
from simulator.profile_config import drive_profile, trip_overrun_bound


def road_curvature(road: Road, s: float) -> float:
    return float(sum(curve.curvature(s) for curve in road.curves))


@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    walk=st.integers(min_value=0, max_value=2**32 - 1),
    end_after=st.floats(min_value=500.0, max_value=4000.0),
)
def test_tracker_matches_the_road_curvature(
    seed: int, trip: int, walk: int, end_after: float
) -> None:
    """Walking a road in steps of up to 2.5 m (90 km/h for 0.1 s), the tracker's curvature equals
    the road's, bit for bit, also after a final stop removes the curves beyond it."""
    road = Road(seed, "VAN-001", 0, trip, drive_profile())
    tracker = CurveTracker(1)
    tracker.start_trip(0, road)
    rng = np.random.default_rng(walk)
    s, ended = 0.0, False
    while True:
        tracker.update(np.array([s]), np.array([True]))
        assert tracker.curvature(np.array([s]))[0] == road_curvature(road, s), s
        if not ended and s >= end_after:
            road.end_at(s + float(rng.uniform(0.0, 300.0)))
            ended = True
        if ended and road.final_stop is not None and s >= road.final_stop.position:
            break
        s += float(rng.uniform(0.0, 2.5))


def test_a_curve_removed_by_the_final_stop_is_never_entered() -> None:
    """The final stop lands exactly at the next curve's start, which end_at then removes; a van
    stopping a few millimetres past the stop (capture tolerance) must see no curvature."""
    road = Road(3, "VAN-001", 0, 0, drive_profile())
    tracker = CurveTracker(1)
    tracker.start_trip(0, road)
    road.ensure(20000.0)
    curve = road.curves[0]
    before = curve.start - 10.0
    tracker.update(np.array([before]), np.array([True]))
    assert tracker.start[0] == curve.start  # the tracker holds the curve ahead
    road.end_at(curve.start)
    assert all(c.start != curve.start for c in road.curves)
    past = np.array([curve.start + 0.005])
    tracker.update(past, np.array([True]))
    assert tracker.curvature(past)[0] == 0.0 == road_curvature(road, float(past[0]))


def test_steering_at_the_tightest_curve() -> None:
    """Radius 10 m at its curve speed sqrt(2 * 10): delta = atan(3.6 / 10) = 19.80 deg."""
    p = SteeringParameters.from_model(van_model())
    kappa, v = np.array([0.1]), np.array([math.sqrt(20.0)])
    state = steering_state(kappa, kappa, v, p)
    assert math.degrees(state.road_wheel_angle[0]) == pytest.approx(19.799, abs=1e-3)
    assert math.degrees(state.steering_wheel_angle[0]) == pytest.approx(336.6, abs=0.1)
    assert state.yaw_rate[0] == pytest.approx(v[0] * math.tan(state.road_wheel_angle[0]) / 3.6)
    assert state.accel_lat[0] == pytest.approx(2.0)


def test_no_yaw_rate_or_lateral_acceleration_at_standstill() -> None:
    p = SteeringParameters.from_model(van_model())
    kappa = np.array([0.0, 0.05, -0.1])
    state = steering_state(kappa, kappa, np.zeros(3), p)
    assert np.all(state.yaw_rate == 0)
    assert np.all(state.accel_lat == 0)


def test_no_steering_rate_in_a_step_without_motion() -> None:
    """Identity of the law: same position (same curvature) at both ends of the step, rate 0."""
    p = SteeringParameters.from_model(van_model())
    kappa = np.array([0.0, 0.05, -0.1, 0.02])
    state = steering_state(kappa, kappa, np.array([0.0, 0.0, 0.0, 3.0]), p)
    assert np.all(state.steering_wheel_rate == 0)


def hand_torques(speed: float, angle_deg: float, rate_deg_s: float, accel_lat: float):
    """Driver and assist torques from van_model.yaml, in its stated units (deg, deg/s)."""
    model = van_model()
    fade = math.exp(-speed / model.value("eps_assist_fade_speed"))
    driver = (
        model.value("driver_torque_gain_rate") * rate_deg_s
        + model.value("driver_torque_gain_lat") * accel_lat
        + model.value("driver_torque_gain_park") * angle_deg * fade
    )
    high = model.value("eps_assist_ratio_high_speed")
    ratio = high + (model.value("eps_assist_ratio_standstill") - high) * fade
    return driver, ratio * driver


@pytest.mark.parametrize("sign", [1, -1])
def test_torques_at_standstill_with_the_wheel_turned(sign: int) -> None:
    """v = 0, wheel at 17 * atan(3.6 * 0.08) deg = 273.1 deg: driver = k_park * angle = 2.73 Nm,
    assist = 6 * driver; both with the sign of the angle."""
    p = SteeringParameters.from_model(van_model())
    kappa = np.array([sign * 0.08])
    state = steering_state(kappa, kappa, np.zeros(1), p)
    angle_deg = 17 * math.degrees(math.atan(3.6 * 0.08)) * sign
    driver, assist = hand_torques(0.0, angle_deg, 0.0, 0.0)
    assert driver == pytest.approx(sign * 2.731, abs=1e-3)
    assert state.driver_torque[0] == pytest.approx(driver, rel=1e-12)
    assert state.assist_torque[0] == pytest.approx(assist, rel=1e-12)
    assert assist == pytest.approx(6 * driver)
    assert np.sign(state.driver_torque[0]) == np.sign(state.assist_torque[0]) == sign


@pytest.mark.parametrize("sign", [1, -1])
def test_torques_while_driving(sign: int) -> None:
    """On an arc of radius 100 m at 10 m/s (a_lat = 1 m/s2): driver = 0.8 * 1 + 0.01 * 35.05
    * exp(-1) = 0.929 Nm, assist = (2 + 4 exp(-1)) * driver. Entering a curve at 5 m/s (angle 0,
    rate 35.06 deg/s over the step): driver = 0.004 * 35.06 Nm, assist = (2 + 4 exp(-0.5))."""
    p = SteeringParameters.from_model(van_model())
    arc = steering_state(np.array([sign * 0.01]), np.array([sign * 0.01]), np.array([10.0]), p)
    angle_deg = 17 * math.degrees(math.atan(3.6 * 0.01)) * sign
    driver, assist = hand_torques(10.0, angle_deg, 0.0, sign * 1.0)
    assert driver == pytest.approx(sign * 0.929, abs=1e-3)
    assert arc.driver_torque[0] == pytest.approx(driver, rel=1e-12)
    assert arc.assist_torque[0] == pytest.approx(assist, rel=1e-12)
    entry = steering_state(np.zeros(1), np.array([sign * 0.001]), np.array([5.0]), p)
    rate_deg_s = 17 * math.degrees(math.atan(3.6 * 0.001)) / 0.1 * sign
    driver, assist = hand_torques(5.0, 0.0, rate_deg_s, 0.0)
    assert entry.driver_torque[0] == pytest.approx(driver, rel=1e-12)
    assert entry.assist_torque[0] == pytest.approx(assist, rel=1e-12)
    assert np.sign(entry.assist_torque[0]) == sign


def drive_with_steering(seed: int, trip: int, duration: float, ambient: float):
    model, profile = van_model(), drive_profile()
    vehicles = [f"VAN-{n:03d}" for n in range(1, 5)]
    roads = [Road(seed, vehicle, 0, trip, profile) for vehicle in vehicles]
    traces = drive_trips(
        roads,
        [duration] * len(roads),
        list(np.linspace(2300, 3300, len(roads))),
        air_density(101300, ambient),
        LongitudinalParameters.from_model(model),
        DriverParameters.from_configs(profile, model),
        time_limit=duration + 2 * trip_overrun_bound(profile),
    )
    p = SteeringParameters.from_model(model)
    results = []
    for road, trace in zip(roads, traces, strict=True):
        tracker = CurveTracker(1)
        tracker.start_trip(0, road)
        # Positions at the start of each step, then the position after the last step.
        next_speed = trace.speed + trace.accel * PHYSICS_DT
        end = trace.position[-1] + (trace.speed[-1] + next_speed[-1]) / 2 * PHYSICS_DT
        positions = np.append(trace.position, end)
        kappa = np.empty_like(positions)
        for k, s in enumerate(positions):
            tracker.update(np.array([s]), np.array([True]))
            kappa[k] = tracker.curvature(np.array([s]))[0]
        state = steering_state(kappa[:-1], kappa[1:], trace.speed, p)
        results.append((trace, positions, kappa, state))
    return results


@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    duration=st.floats(min_value=60.0, max_value=300.0),
    ambient=st.floats(min_value=-5.0, max_value=30.0),
)
def test_driven_steering_respects_the_invariants(
    seed: int, trip: int, duration: float, ambient: float
) -> None:
    model = van_model()
    lock, rate_max = model.si("steering_wheel_lock"), model.si("steering_wheel_rate_max")
    driver_range = catalogue().signal("driver_torque").range
    assist_range = catalogue().signal("eps_assist_torque").range
    p = SteeringParameters.from_model(model)
    for trace, positions, kappa, state in drive_with_steering(seed, trip, duration, ambient):
        # Single-track couplings, exact on the true state.
        assert np.array_equal(state.yaw_rate, trace.speed * state.curvature)
        assert np.array_equal(state.accel_lat, trace.speed * state.yaw_rate)
        assert np.max(np.abs(state.accel_lat)) <= model.si("accel_lat_max")
        # Mechanical and driver limits.
        assert np.max(np.abs(state.steering_wheel_angle)) <= lock
        assert np.max(np.abs(state.steering_wheel_rate)) <= rate_max
        # The rate over each step adds up to the angle change over the trip.
        angles = steering_wheel_angle(kappa, p)
        total = np.sum(state.steering_wheel_rate) * PHYSICS_DT
        assert total == pytest.approx(angles[-1] - angles[0], abs=1e-9)
        # Stationary: no yaw or lateral acceleration at v = 0; no steering motion in a step
        # where the van does not move.
        still = trace.speed == 0
        assert np.all(state.yaw_rate[still] == 0)
        assert np.all(state.accel_lat[still] == 0)
        assert np.all(state.steering_wheel_rate[np.diff(positions) == 0] == 0)
        # Torques stay within the sensor ranges of the catalogue.
        assert driver_range[0] <= state.driver_torque.min() <= state.driver_torque.max()
        assert state.driver_torque.max() <= driver_range[1]
        assert assist_range[0] <= state.assist_torque.min()
        assert state.assist_torque.max() <= assist_range[1]
