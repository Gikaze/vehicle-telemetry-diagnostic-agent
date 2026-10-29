"""Sensor layer: units, encoding, noise statistics and bounds, noise streams (spec section 2,
NFR-15)."""

import itertools
import math

import numpy as np
import pytest
from driven import DrivenTrip, drive_with_steering
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.catalogue import catalogue, on_grid
from simulator.config_files import NOISE_FREE, sensors, van_model
from simulator.model.wheels import WheelParameters, axle_slip, wheel_speed_readings
from simulator.sensors import (
    CHANNELS,
    GROUP,
    SensorNoise,
    SensorParameters,
    encode,
    error_bound,
    from_catalogue_unit,
    measure,
    measure_signal,
    to_catalogue_unit,
)

KMH = 1 / 3.6
SAMPLES = 200_000


def params(**changes) -> SensorParameters:
    p = SensorParameters.from_configs(sensors(), catalogue())
    return SensorParameters(**{**p.__dict__, **changes})


def clipped_variance(a: float) -> float:
    """E[clip(Z, -a, a)^2] for a standard normal Z: 1 - 2 Phi(-a) - 2 a phi(a) + 2 a^2 Phi(-a)."""
    tail = 0.5 * math.erfc(a / math.sqrt(2))
    density = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    return 1 - 2 * tail - 2 * a * density + 2 * a * a * tail


@pytest.mark.parametrize(
    ("name", "si", "expected"),
    [
        ("wheel_speed_fl", 10.0, 36.0),
        ("yaw_rate", 0.1, 5.729578),
        ("master_cyl_pressure", 2e5, 2.0),
        ("accel_pedal_position", 0.5, 50.0),
        ("odometer", 12345.0, 12.345),
        ("coolant_temp", 90.0, 90.0),
        ("engine_speed", 2075.0, 2075.0),
    ],
)
def test_conversion_to_catalogue_units(name: str, si: float, expected: float) -> None:
    signal = catalogue().signal(name)
    value = to_catalogue_unit(np.array([si]), signal)
    assert value[0] == pytest.approx(expected, rel=1e-6)
    assert from_catalogue_unit(value, signal)[0] == pytest.approx(si, rel=1e-12)


def test_encode_quantises_and_saturates() -> None:
    wheel, pressure = (
        catalogue().signal("wheel_speed_fl"),
        catalogue().signal("master_cyl_pressure"),
    )
    assert list(encode(np.array([12.344, 12.346, -0.2, 260.0]), wheel)) == [
        12.34,
        12.35,
        0.0,
        250.0,
    ]
    assert list(encode(np.array([0.04, 0.06]), pressure)) == [0.0, 0.1]


def true_values(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """SI true values of every noisy and noise-free signal, uniform inside its range."""
    p, values = params(), {}
    for name in (*CHANNELS, *NOISE_FREE):
        signal = p.signals[name]
        low, high = signal.range
        values[name] = from_catalogue_unit(rng.uniform(low, high, n), signal)
    return values


def test_noise_free_signals_are_their_encoded_true_values() -> None:
    p, rng = params(), np.random.default_rng(1)
    true = true_values(rng, 1000)
    measured = measure(true, rng.standard_normal((1000, len(CHANNELS))), p)
    for name in NOISE_FREE:
        signal = p.signals[name]
        assert np.array_equal(measured[name], encode(to_catalogue_unit(true[name], signal), signal))


def test_measure_needs_every_channel_and_rejects_unknown_signals() -> None:
    p, rng = params(), np.random.default_rng(2)
    true = true_values(rng, 3)
    z = np.zeros((3, len(CHANNELS)))
    with pytest.raises(ValueError, match="missing \\['accel_lat'\\]"):
        measure({k: v for k, v in true.items() if k != "accel_lat"}, z, p)
    with pytest.raises(ValueError, match="not measurable \\['abs_active'\\]"):
        measure({**true, "abs_active": np.zeros(3)}, z, p)


def test_measured_keys_come_in_a_fixed_order() -> None:
    """CHANNELS then NOISE_FREE, whatever the order of the true values (FR-02: same bytes)."""
    p, rng = params(), np.random.default_rng(3)
    true = true_values(rng, 3)
    z = np.zeros((3, len(CHANNELS)))
    for keys in (list(true), list(reversed(true))):
        measured = measure({k: true[k] for k in keys}, z, p)
        assert list(measured) == [*CHANNELS, *NOISE_FREE]


def dithered(name: str, rng: np.random.Generator, p: SensorParameters) -> np.ndarray:
    """Catalogue-unit true values uniform over one resolution cell, on a grid point at the middle
    of the range, so the quantisation error is uniform and independent of the noise; at least
    error_bound away from the range bounds, so saturation never applies."""
    signal = p.signals[name]
    low, high = signal.range
    centre = float(encode(np.array([(low + high) / 2]), signal)[0])
    values = centre + rng.uniform(0.0, signal.resolution, SAMPLES)
    bound = error_bound(name, p)
    assert values.min() - bound >= low and values.max() + bound <= high
    return values


@pytest.mark.parametrize("name", CHANNELS)
def test_measured_noise_has_the_declared_sigma(name: str) -> None:
    """Large sample: std(measured - true) = sqrt(sigma^2 c_3 + resolution^2 / 12) within 1 %,
    with sigma as sensors.yaml states it, c_3 = E[clip(Z, -3, 3)^2] = 0.99501 (the clip) and the
    uniform quantisation error; the mean error is 0 within 4 standard errors."""
    p, rng = params(), np.random.default_rng(3)
    signal = p.signals[name]
    true = dithered(name, rng, p)
    z = rng.standard_normal(SAMPLES)
    error = measure_signal(name, from_catalogue_unit(true, signal), z, p) - true
    declared = sensors().value(GROUP[name])
    clip = sensors().value("noise_clip_sigma")
    sigma = math.sqrt(declared**2 * clipped_variance(clip) + signal.resolution**2 / 12)
    assert clipped_variance(3.0) == pytest.approx(0.99501, abs=5e-6)
    assert error.std() == pytest.approx(sigma, rel=0.01)
    assert abs(error.mean()) <= 4 * sigma / math.sqrt(SAMPLES)


@pytest.mark.parametrize(
    ("name", "bound"),
    [
        ("wheel_speed_fl", 0.155),  # 3 * 0.05 km/h + 0.01 / 2
        ("yaw_rate", 0.305),  # 3 * 0.1 deg/s + 0.01 / 2
        ("steering_wheel_angle", 0.35),  # 3 * 0.1 deg + 0.1 / 2
        ("accel_lat", 0.065),  # 3 * 0.02 m/s2 + 0.01 / 2
        ("tyre_temp_rr", 2.0),  # 3 * 0.5 K + 1 / 2
        ("master_cyl_pressure", 0.65),  # 3 * 0.2 bar + 0.1 / 2
        ("engine_speed", 0.5),  # noise-free: 1 rpm / 2
        ("vehicle_ref_speed", 0.005),  # noise-free: 0.01 km/h / 2
    ],
)
def test_error_bounds_by_hand(name: str, bound: float) -> None:
    assert error_bound(name, params()) == pytest.approx(bound, rel=1e-12)


@pytest.mark.parametrize("name", CHANNELS)
def test_measurement_errors_stay_within_the_bound(name: str) -> None:
    """|measured - true| <= error_bound on every sample, and the clip is reached: at least 0.99 of
    the bound (about 540 of the 200 000 draws lie beyond 3 sigma)."""
    p, rng = params(), np.random.default_rng(4)
    signal = p.signals[name]
    true = dithered(name, rng, p)
    z = rng.standard_normal(SAMPLES)
    error = np.abs(measure_signal(name, from_catalogue_unit(true, signal), z, p) - true)
    bound = error_bound(name, p)
    assert error.max() <= bound + 1e-9 * max(1.0, abs(true).max())
    assert error.max() >= 0.99 * bound


def test_a_wheel_at_rest_reads_zero() -> None:
    """No pulses at standstill: a wheel speed of exactly 0 reads 0 whatever the draw; any other
    noisy signal at 0 gets its noise (the yaw rate at standstill)."""
    p = params()
    z = np.array([-3.5, -1.0, 0.0, 1.0, 3.5])
    for name in ("wheel_speed_fl", "wheel_speed_rr"):
        assert np.array_equal(measure_signal(name, np.zeros(5), z, p), np.zeros(5))
    yaw = measure_signal("yaw_rate", np.zeros(5), z, p)
    assert list(yaw) == [-0.3, -0.1, 0.0, 0.1, 0.3]


@pytest.mark.parametrize("name", ["master_cyl_pressure", "accel_pedal_position"])
def test_readings_at_zero_saturate_at_the_range_bottom(name: str) -> None:
    """A released brake (0 bar) or accelerator (0 %) reads between 0 and error_bound, never below
    0: noise below half a resolution step reads 0 (negative noise saturates), so a fraction
    Phi(resolution / 2 / sigma) = Phi(0.25) = 0.599 of the readings is exactly 0."""
    p, rng = params(), np.random.default_rng(5)
    measured = measure_signal(name, np.zeros(SAMPLES), rng.standard_normal(SAMPLES), p)
    assert measured.min() == 0.0
    assert measured.max() <= error_bound(name, p)
    ratio = p.signals[name].resolution / 2 / p.sigma[name]
    expected = 0.5 * math.erfc(-ratio / math.sqrt(2))
    assert expected == pytest.approx(0.599, abs=5e-4)
    assert np.mean(measured == 0) == pytest.approx(expected, abs=0.005)


def test_measured_values_lie_on_the_grid_and_in_the_range() -> None:
    p, rng = params(), np.random.default_rng(6)
    true = true_values(rng, 2000)
    measured = measure(true, 3 * rng.standard_normal((2000, len(CHANNELS))), p)
    for name, values in measured.items():
        signal = p.signals[name]
        low, high = signal.range
        assert np.all((values >= low) & (values <= high)), name
        assert all(on_grid(float(v), signal.resolution, signal.offset) for v in values), name


def draws(n_steps: int, block_steps: int, vehicle: str = "VAN-001", trip: int = 0) -> np.ndarray:
    noise = SensorNoise(1, params(block_steps=block_steps))
    noise.start_trip(0, 42, vehicle, 0, trip)
    active = np.array([True])
    return np.array([noise.draw(active)[0] for _ in range(n_steps)])


def test_noise_is_the_trip_stream_in_order_whatever_the_block_size() -> None:
    """Every active step takes the next len(CHANNELS) draws of the trip's SENSOR_NOISE stream: the
    block size (here 1, 7, 600 and the configured one) changes no value."""
    reference = draws(1500, 600)
    for block in (1, 7, params().block_steps):
        assert np.array_equal(draws(1500, block), reference), block
    assert params().block_steps == 600


def test_noise_streams_are_per_vehicle_and_trip() -> None:
    a = draws(10, 600)
    assert np.array_equal(draws(10, 600), a)
    assert not np.array_equal(draws(10, 600, vehicle="VAN-002"), a)
    assert not np.array_equal(draws(10, 600, trip=1), a)


def test_a_slot_draws_only_while_active_and_independently_of_the_others() -> None:
    """Three slots, VAN-002 inactive on every other step and VAN-003 starting late: each slot
    gets its own trip stream in order, inactive steps return 0 and do not advance it."""
    noise = SensorNoise(3, params(block_steps=7))
    noise.start_trip(0, 42, "VAN-001", 0, 0)
    noise.start_trip(1, 42, "VAN-002", 0, 0)
    rows: list[list[np.ndarray]] = [[], [], []]
    for k in range(40):
        if k == 25:
            noise.start_trip(2, 42, "VAN-003", 0, 0)
        active = np.array([True, k % 2 == 0, k >= 25])
        z = noise.draw(active)
        for i in range(3):
            if active[i]:
                rows[i].append(z[i])
            else:
                assert np.all(z[i] == 0)
    for i, vehicle in enumerate(("VAN-001", "VAN-002", "VAN-003")):
        assert np.array_equal(np.array(rows[i]), draws(len(rows[i]), 600, vehicle)), vehicle


def test_a_slot_without_a_trip_cannot_draw() -> None:
    with pytest.raises(RuntimeError, match="slot 0 has no trip"):
        SensorNoise(1, params()).draw(np.array([True]))


COLUMN = {name: j for j, name in enumerate(CHANNELS)}


def lateral_residual(
    driven: DrivenTrip, radius_factor: float, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(|a_m - v_m psi_m|, tolerance) on every step of a driven trip measured with draws z.

    v_m is the mean of the two measured rear wheel speeds, whose true value is
    v (1 + slip_rear) r_nom / r (the yaw offsets cancel). With a = v psi, the propagated error is

        |a_m - v_m psi_m| <= eps_a + |v| eps_psi + |psi| eps_v + eps_v eps_psi

    eps_a and eps_psi the error bounds of accel_lat and yaw_rate (rad/s), eps_v the wheel speed
    error bound plus |v| |(1 + slip_rear) r_nom / r - 1| (rear slip and tyre radius bias)."""
    p, wheels = params(), WheelParameters.from_model(van_model())
    trace, steering = driven.trace, driven.steering
    n = len(trace.speed)
    mass, radius = np.full(n, driven.mass), np.full(n, radius_factor * wheels.radius_nominal)
    slip_front, slip_rear = axle_slip(trace.engine_force, trace.brake_force, mass, wheels)
    readings = wheel_speed_readings(
        trace.speed, steering.yaw_rate, slip_front, slip_rear, radius, wheels
    )

    def measured(name: str, true: np.ndarray) -> np.ndarray:
        value = measure_signal(name, true, z[:, COLUMN[name]], p)
        return from_catalogue_unit(value, p.signals[name])

    a_m = measured("accel_lat", steering.accel_lat)
    psi_m = measured("yaw_rate", steering.yaw_rate)
    v_m = (
        measured("wheel_speed_rl", readings[:, 2]) + measured("wheel_speed_rr", readings[:, 3])
    ) / 2
    v, psi = trace.speed, np.abs(steering.yaw_rate)
    eps_a = error_bound("accel_lat", p)
    eps_psi = math.radians(error_bound("yaw_rate", p))
    bias = np.abs((1 + slip_rear) * wheels.radius_nominal / radius - 1)
    eps_v = error_bound("wheel_speed_rl", p) * KMH + v * bias
    tolerance = eps_a + v * eps_psi + psi * eps_v + eps_v * eps_psi
    return np.abs(a_m - v_m * psi_m), tolerance


@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    duration=st.floats(min_value=60.0, max_value=300.0),
    ambient=st.floats(min_value=-5.0, max_value=30.0),
    radius_factor=st.floats(min_value=0.99, max_value=1.01),
)
def test_lateral_acceleration_matches_speed_times_yaw_rate(
    seed: int, trip: int, duration: float, ambient: float, radius_factor: float
) -> None:
    """True states: accel_lat == v * yaw_rate exactly. Measured with the trip's real noise (each
    van's SENSOR_NOISE stream): within the propagated tolerance of lateral_residual."""
    for number, driven in enumerate(drive_with_steering(seed, trip, duration, ambient), start=1):
        trace, steering = driven.trace, driven.steering
        assert np.array_equal(steering.accel_lat, trace.speed * steering.yaw_rate)
        noise = SensorNoise(1, params())
        noise.start_trip(0, seed, f"VAN-{number:03d}", 0, trip)
        z = np.array([noise.draw(np.array([True]))[0] for _ in trace.speed])
        residual, tolerance = lateral_residual(driven, radius_factor, z)
        assert np.all(residual <= tolerance + 1e-9)


@pytest.mark.parametrize("radius_factor", [0.99, 1.01])
def test_lateral_tolerance_is_reached_with_the_errors_at_their_bounds(
    radius_factor: float,
) -> None:
    """Seed 7, trip 1, both ends of the tyre radius spread: with the draws of accel_lat, yaw_rate
    and the two rear wheels at +/- noise_clip_sigma (every sign pattern, rear wheels together),
    the residual stays within the tolerance and reaches at least 0.97 of it: no term of the
    tolerance is spare."""
    p = params()
    drives = drive_with_steering(7, 1, 300.0, 10.0)
    highest = 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        for driven in drives:
            z = np.zeros((len(driven.trace.speed), len(CHANNELS)))
            z[:, COLUMN["accel_lat"]] = signs[0] * p.clip
            z[:, COLUMN["yaw_rate"]] = signs[1] * p.clip
            z[:, [COLUMN["wheel_speed_rl"], COLUMN["wheel_speed_rr"]]] = signs[2] * p.clip
            residual, tolerance = lateral_residual(driven, radius_factor, z)
            assert np.all(residual <= tolerance + 1e-9)
            moving = driven.trace.speed > 0
            highest = max(highest, float((residual[moving] / tolerance[moving]).max()))
    assert highest >= 0.97
