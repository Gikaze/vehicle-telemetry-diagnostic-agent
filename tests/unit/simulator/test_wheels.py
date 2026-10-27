"""Wheel speeds, longitudinal slip and odometer (spec section 4, NFR-15)."""

import math

import numpy as np
import pytest
from driven import drive_with_steering, slip_bound
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.config_files import van_model
from simulator.constants import GRAVITY
from simulator.model.forces import drag_force, grade_force, rolling_resistance_force
from simulator.model.longitudinal import LongitudinalParameters, split_forces
from simulator.model.wheels import (
    WheelParameters,
    axle_forces,
    axle_slip,
    ground_speeds,
    odometer_increment,
    wheel_speed_readings,
)


def params() -> WheelParameters:
    return WheelParameters.from_model(van_model())


@pytest.mark.parametrize("direction", [1, -1])
def test_ground_speeds_at_the_tightest_curve(direction: int) -> None:
    """Radius 10 m at 4.47 m/s, rear axle centre reference: inner rear 4.092, outer rear 4.852,
    inner front 4.397, outer front 5.112 m/s (front wheels on sqrt((R -/+ b/2)^2 + L^2))."""
    v = math.sqrt(20.0)
    speeds = ground_speeds(np.array([v]), np.array([direction * v / 10]), params())[0]
    inner_front, outer_front, inner_rear, outer_rear = 4.397, 5.112, 4.092, 4.852
    if direction == 1:  # left turn: left wheels inner
        expected = [inner_front, outer_front, inner_rear, outer_rear]
    else:
        expected = [outer_front, inner_front, outer_rear, inner_rear]
    assert speeds == pytest.approx(expected, abs=1e-3)


def test_rear_wheels_average_the_reference_speed_in_a_curve() -> None:
    v, yaw = np.array([4.0, 12.0, 20.0]), np.array([0.4, -0.15, 0.05])
    speeds = ground_speeds(v, yaw, params())
    assert np.allclose((speeds[:, 2] + speeds[:, 3]) / 2, v, rtol=0, atol=1e-12)
    straight = ground_speeds(v, np.zeros(3), params())
    assert np.all(straight == v[:, None])


def test_wheels_read_zero_at_standstill_with_the_brakes_on() -> None:
    p = params()
    mass = np.array([2300.0, 3300.0])
    slip_front, slip_rear = axle_slip(np.array([0.0, 800.0]), np.array([5000.0, 0.0]), mass, p)
    readings = wheel_speed_readings(
        np.zeros(2), np.zeros(2), slip_front, slip_rear, np.array([0.35, 0.3535]), p
    )
    assert np.all(readings == 0)


def test_braking_slip_from_the_longitudinal_model() -> None:
    """Laden van (3300 kg; rear axle 0.45 * 2300 + 0.88 * 1000 = 1915 kg, front 1385 kg) at the
    service braking limit, a = -4 m/s2, on a flat road at 10 m/s, air density 1.2 kg/m3.
    split_forces: rolling resistance 3300 * g * 0.01 = 323.62 N and drag 0.5 * 1.2 * 1.3 * 10^2
    = 78 N help, the engine drags with its full 3300 * 0.4 = 1320 N, and the brakes supply the
    rest: 13200 - 323.62 - 78 - 1320 = 11478.38 N. Front slip -0.70 * 11478.38 / (19 * 1385 g)
    = -3.1135 %, rear -(0.30 * 11478.38 + 1320) / (19 * 1915 g) = -1.3350 %."""
    p = params()
    longitudinal = LongitudinalParameters.from_model(van_model())
    mass, speed, accel = np.array([3300.0]), np.array([10.0]), np.array([-4.0])
    forces = split_forces(accel, speed, np.zeros(1), mass, 1.2, longitudinal)
    rolling = 3300 * GRAVITY * 0.01
    drag = 0.5 * 1.2 * 1.3 * 10.0**2
    # Force balance: engine - brake = m * a + rolling resistance + drag on a flat road.
    assert forces.engine[0] - forces.brake[0] == pytest.approx(3300 * -4.0 + rolling + drag)
    assert forces.engine[0] == pytest.approx(-1320.0)
    brake = 13200 - rolling - drag - 1320
    assert forces.brake[0] == pytest.approx(brake, rel=1e-12)
    assert brake == pytest.approx(11478.38, abs=0.01)
    front, rear = axle_slip(forces.engine, forces.brake, mass, p)
    assert front[0] == pytest.approx(-0.70 * brake / (19 * 1385 * GRAVITY), rel=1e-12)
    assert rear[0] == pytest.approx(-(0.30 * brake + 1320) / (19 * 1915 * GRAVITY), rel=1e-12)
    assert front[0] == pytest.approx(-0.031135, abs=1e-6)
    assert rear[0] == pytest.approx(-0.013350, abs=1e-6)


def test_traction_slip_on_the_rear_axle() -> None:
    """Empty van (rear axle 0.45 * 2300 = 1035 kg) pulling 5290 N: rear 5290 / (19 * 1035 g)
    = +2.74 %, front 0."""
    front, rear = axle_slip(np.array([5290.0]), np.array([0.0]), np.array([2300.0]), params())
    assert front[0] == 0
    assert rear[0] == pytest.approx(5290 / (19 * 1035 * GRAVITY), rel=1e-12)
    assert rear[0] == pytest.approx(0.0274, abs=1e-4)


def test_slip_bound_from_the_configuration() -> None:
    """The bound is the laden van braking at 4 m/s2 on a 10 % descent. The brakes supply
    m * (4 + g sin(atan 0.1) - 0.01 g cos(atan 0.1) - 0.4) = 4.478 m = 14777 N beyond rolling
    resistance and full engine drag; front slip 0.70 * 14777 / (19 * 1385 g) = 4.01 %."""
    assert slip_bound(np.linspace(2300, 3300, 11)) == pytest.approx(0.0401, abs=1e-4)


def test_readings_carry_the_tyre_radius_bias() -> None:
    """The ECU uses the nominal radius: a tyre 1 % larger reads 1/1.01 of the rolling speed."""
    p = params()
    v, yaw, zero = np.array([15.0]), np.array([0.1]), np.zeros(1)
    nominal = wheel_speed_readings(v, yaw, zero, zero, np.array([p.radius_nominal]), p)
    larger = wheel_speed_readings(v, yaw, zero, zero, np.array([1.01 * p.radius_nominal]), p)
    assert np.allclose(larger, nominal / 1.01, rtol=1e-12)
    assert np.array_equal(nominal, ground_speeds(v, yaw, p))


def test_odometer_counts_the_rear_wheels_distance() -> None:
    """In a curve the rear wheels' mean speed is v (not the front wheels', which run wider), so
    the odometer reads the distance travelled, scaled by the radius bias and the rear slip."""
    p = params()
    v, yaw = np.array([4.5]), np.array([0.45])
    radius, zero = np.array([1.01 * p.radius_nominal]), np.zeros(1)
    readings = wheel_speed_readings(v, yaw, zero, zero, radius, p)[0]
    assert (readings[2] + readings[3]) / 2 == pytest.approx(v[0] / 1.01, rel=1e-12)
    assert (readings[0] + readings[1]) / 2 > v[0] / 1.01 * 1.05
    increment = odometer_increment(np.array([0.45]), zero, radius, p)
    assert increment[0] == pytest.approx(0.45 / 1.01, rel=1e-12)
    slipping = odometer_increment(np.array([0.45]), np.array([0.02]), radius, p)
    assert slipping[0] == pytest.approx(0.45 * 1.02 / 1.01, rel=1e-12)


@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    duration=st.floats(min_value=60.0, max_value=300.0),
    ambient=st.floats(min_value=-5.0, max_value=30.0),
    radius_factor=st.floats(min_value=0.99, max_value=1.01),
)
def test_driven_wheel_speeds_respect_the_invariants(
    seed: int, trip: int, duration: float, ambient: float, radius_factor: float
) -> None:
    p, longitudinal = params(), LongitudinalParameters.from_model(van_model())
    drives = drive_with_steering(seed, trip, duration, ambient)
    bound = slip_bound(np.array([d.mass for d in drives]))
    for driven in drives:
        trace, yaw = driven.trace, driven.steering.yaw_rate
        n = len(trace.speed)
        mass, radius = np.full(n, driven.mass), np.full(n, radius_factor * p.radius_nominal)
        slip_front, slip_rear = axle_slip(trace.engine_force, trace.brake_force, mass, p)
        # Force balance: the axle forces drive m * a against gravity, rolling resistance and drag
        # (resistances act only while moving, as in the longitudinal model), on every step that
        # is not held at a stop and does not end at rest (where the speed is set to 0).
        front_force, rear_force = axle_forces(trace.engine_force, trace.brake_force, p)
        moving = trace.speed > 0
        resistance = np.where(
            moving,
            rolling_resistance_force(mass, trace.grade, longitudinal.rolling_resistance)
            + drag_force(trace.speed, driven.density, longitudinal.drag_area),
            0.0,
        )
        needed = mass * trace.accel - grade_force(mass, trace.grade) + resistance
        # Speed after each step, read from the trace (0 after the last step: ignition off), not
        # rebuilt as v + a * DT, which leaves a 1e-17 residue where the speed was set to 0.
        next_speed = np.append(trace.speed[1:], 0.0)
        free = ~trace.stopped & (next_speed > 0)
        assert np.allclose((front_force + rear_force)[free], needed[free], rtol=1e-9, atol=1e-6)
        # At rest the brakes hold the van: brake force |F_grade|, no engine force. The axle forces
        # take the brake force as acting backward, which balances gravity at rest on a descent or
        # the flat; on a climb the brakes push forward, so the sum is off by exactly 2 |F_grade|
        # (no effect on the readings or the odometer, both exactly 0 at rest).
        held = (trace.speed == 0) & (next_speed == 0)
        gravity = grade_force(mass, trace.grade)
        total = front_force + rear_force
        assert np.allclose(trace.brake_force[held], np.abs(gravity[held]), rtol=1e-12, atol=0)
        assert np.all(trace.engine_force[held] == 0)
        level_or_down = held & (trace.grade <= 0)
        assert np.allclose(total[level_or_down] + gravity[level_or_down], 0, rtol=0, atol=1e-6)
        up = held & (trace.grade > 0)
        gap = (total + gravity)[up]
        assert np.allclose(gap, -2 * np.abs(gravity[up]), rtol=1e-12, atol=1e-9)
        readings = wheel_speed_readings(trace.speed, yaw, slip_front, slip_rear, radius, p)
        # Readings are 0 exactly at standstill and positive otherwise.
        still = trace.speed == 0
        assert np.all(readings[still] == 0)
        assert np.all(readings[~still] > 0)
        # Slip stays within the bound the configuration allows.
        assert np.max(np.abs(slip_front)) <= bound and np.max(np.abs(slip_rear)) <= bound
        # Outer wheels are faster: right side faster in a left turn (yaw > 0).
        turning = np.abs(yaw) > 1e-6
        assert np.all(np.sign(readings[turning, 3] - readings[turning, 2]) == np.sign(yaw[turning]))
        # The rear wheels' mean is the reference speed with the rear slip and radius bias.
        scale = (1 + slip_rear) * p.radius_nominal / radius
        rear_mean = (readings[:, 2] + readings[:, 3]) / 2
        assert np.allclose(rear_mean, trace.speed * scale, rtol=1e-12, atol=0)
        # The odometer never decreases and reads the distance within the slip and radius bias.
        distance = np.diff(driven.positions)
        increments = odometer_increment(distance, slip_rear, radius, p)
        assert np.all(increments >= 0)
        travelled, read = distance.sum(), increments.sum()
        bias = p.radius_nominal / radius[0]
        assert abs(read - travelled * bias) <= np.max(np.abs(slip_rear)) * travelled * bias + 1e-9
