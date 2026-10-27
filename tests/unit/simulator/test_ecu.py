"""Brake ECU: corrections, reference speed, ABS/TCS/ESC flags and their margins (spec section 4,
NFR-15)."""

import dataclasses
import itertools
import math

import numpy as np
import pytest
from driven import drive_with_steering, slip_bound
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.catalogue import catalogue, quantise
from simulator.config_files import sensors, van_model
from simulator.constants import PHYSICS_DT
from simulator.model.brakes import (
    BrakeParameters,
    brake_pedal_pressed,
    held_steps,
    hold_force,
    master_cylinder_pressure,
)
from simulator.model.ecu import (
    EcuParameters,
    abs_active,
    braking_slip,
    corrected_wheel_speeds,
    esc_active,
    reference_speed,
    tcs_active,
    traction_slip,
    yaw_rate_error,
)
from simulator.model.forces import air_density
from simulator.model.longitudinal import LongitudinalParameters, acceleration_limits, split_forces
from simulator.model.wheels import (
    WheelParameters,
    axle_slip,
    ground_speeds,
    wheel_speed_readings,
)
from simulator.profile.road import Curve
from simulator.profile_config import (
    CURVES,
    SEGMENT_TYPES,
    curve_ramp_time,
    curve_speed,
    drive_profile,
    duty_cycle,
)

KMH = 1 / 3.6
FLAGS = ("ABS", "TCS", "ESC")
# Every combination of error signs on the four wheel speeds, the yaw rate and the steering angle.
SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
# Conservative case of the ecu_slip_speed_min rationale (van_model.yaml): tyre_slip_stiffness floor
# and the range of payload_rear_axle_share. The worst ABS case falls steadily from share 0.75 to
# 1.0 and the worst TCS and ESC cases are the empty van, where the share plays no part: the two
# ends and the nominal value cover the range.
STIFFNESS_FLOOR = 15
REAR_SHARES = (0.75, 0.88, 1.0)
# Front wheel error left by the single road-wheel angle in the 10 m curve at its curve speed
# (spec section 8): inner +1.01 %, outer -0.94 % of the rear axle speed.
ARC_INNER, ARC_OUTER = 1.0101, 0.9906


def params() -> EcuParameters:
    return EcuParameters.from_model(van_model())


def noise_bounds() -> tuple[float, float, float]:
    """3 sigma (the clip) of the wheel speed, m/s, yaw rate, rad/s, and steering angle, rad."""
    clip, noise = sensors().value("noise_clip_sigma"), sensors()
    return (
        clip * noise.si("wheel_speed"),
        clip * noise.si("yaw_rate"),
        clip * noise.si("steering_wheel_angle"),
    )


def error_bounds() -> tuple[float, float, float]:
    """Largest measurement error, 3 sigma plus half the resolution, in the units above."""
    wheel, yaw, angle = noise_bounds()
    signal = catalogue().signal
    return (
        wheel + signal("wheel_speed_fl").resolution / 2 * KMH,
        yaw + math.radians(signal("yaw_rate").resolution / 2),
        angle + math.radians(signal("steering_wheel_angle").resolution / 2),
    )


def measure(
    readings: np.ndarray, yaw: np.ndarray, angle: np.ndarray, signs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sensor path: the true value plus sign * 3 sigma, quantised to the catalogue resolution in
    catalogue units (km/h, deg/s, deg), back in SI units. A wheel at rest reads exactly 0."""
    wheel_noise, yaw_noise, angle_noise = noise_bounds()
    signal = catalogue().signal
    kmh = quantise((readings + signs[..., :4] * wheel_noise) / KMH, signal("wheel_speed_fl"))
    wheels = np.where(readings == 0, 0.0, kmh * KMH)
    deg_s = quantise(np.degrees(yaw + signs[..., 4] * yaw_noise), signal("yaw_rate"))
    deg = quantise(np.degrees(angle + signs[..., 5] * angle_noise), signal("steering_wheel_angle"))
    return wheels, np.radians(deg_s), np.radians(deg)


def flags(
    wheels: np.ndarray, yaw: np.ndarray, angle: np.ndarray, braking: np.ndarray, p: EcuParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrected = corrected_wheel_speeds(wheels, yaw, angle, p)
    reference = reference_speed(corrected, braking)
    return (
        abs_active(corrected, reference, braking, p),
        tcs_active(corrected, reference, braking, p),
        esc_active(yaw, angle, reference, p),
    )


@pytest.mark.parametrize("direction", [1, -1])
def test_corrections_in_the_tightest_curve(direction: int) -> None:
    """10 m curve at its curve speed sqrt(2 * 10) = 4.472 m/s, no slip, nominal tyres. Front
    projection then yaw term: inner front sqrt(9.15^2 + 3.6^2) / 10 * cos(atan 0.36) + 0.085
    = 1.0101 v, outer sqrt(10.85^2 + 3.6^2) / 10 * cos(atan 0.36) - 0.085 = 0.9906 v; the rear
    wheels v exactly."""
    p, wheels = params(), WheelParameters.from_model(van_model())
    v = math.sqrt(2.0 * 10)
    yaw = np.array([direction * v / 10])
    angle = np.array([direction * p.steering_ratio * math.atan(p.wheelbase / 10)])
    readings = ground_speeds(np.array([v]), yaw, wheels)
    corrected = corrected_wheel_speeds(readings, yaw, angle, p)[0] / v
    inner, outer = (0, 1) if direction == 1 else (1, 0)
    assert corrected[inner] == pytest.approx(ARC_INNER, abs=5e-5)
    assert corrected[outer] == pytest.approx(ARC_OUTER, abs=5e-5)
    assert corrected[2:] == pytest.approx([1.0, 1.0], rel=1e-12)


def test_corrections_leave_a_straight_road_and_a_standstill_unchanged() -> None:
    p = params()
    readings = np.array([[10.0, 10.1, 9.9, 10.2], [0.0, 0.0, 0.0, 0.0]])
    yaw = np.array([0.0, math.radians(0.3)])  # yaw rate noise at standstill
    corrected = corrected_wheel_speeds(readings, yaw, np.zeros(2), p)
    assert np.array_equal(corrected, readings)


def test_reference_speed() -> None:
    corrected = np.array([[10.0, 9.0, 11.0, 10.5]])
    assert reference_speed(corrected, np.array([True]))[0] == 11.0
    assert reference_speed(corrected, np.array([False]))[0] == 9.5


def test_slip_ratio_uses_the_reference_speed_guard() -> None:
    """Below ecu_reference_speed_min (5 km/h) the ratio is taken over 5 km/h: 0.4 m/s below a
    1 m/s reference is 0.4 / 1.3889 = 0.288, not 0.4."""
    p = params()
    corrected = np.array([[0.6, 1.0, 1.0, 1.0]])
    difference, ratio = braking_slip(corrected, np.array([1.0]), p)
    assert difference[0, 0] == pytest.approx(0.4, rel=1e-12)
    assert ratio[0, 0] == pytest.approx(0.4 / (5 * KMH), rel=1e-12)
    corrected = np.array([[10.0, 10.0, 11.2, 10.0]])
    difference, ratio = traction_slip(corrected, np.array([10.0]), p)
    assert difference[0] == pytest.approx([1.2, 0.0], abs=1e-12)
    assert ratio[0] == pytest.approx([0.12, 0.0], abs=1e-12)


@pytest.mark.parametrize(
    ("reference", "wheel", "expected"),
    [
        # 0.4 m/s = 1.44 km/h below 1 m/s: ratio 0.288 above 0.15, difference below 2 km/h.
        (1.0, 0.6, False),
        # 0.8 m/s = 2.88 km/h below 25 m/s: difference above 2 km/h, ratio 3.2 % below 15 %.
        (25.0, 24.2, False),
        # 2 m/s = 7.2 km/h below 10 m/s: both above, ratio 0.2.
        (10.0, 8.0, True),
    ],
)
def test_abs_needs_both_the_ratio_and_the_difference(
    reference: float, wheel: float, expected: bool
) -> None:
    p = params()
    corrected = np.array([[wheel, reference, reference, reference]])
    braking = abs_active(corrected, np.array([reference]), np.array([True]), p)
    released = abs_active(corrected, np.array([reference]), np.array([False]), p)
    assert braking[0] == expected and not released[0]


@pytest.mark.parametrize(
    ("front", "rear", "expected"),
    [
        # 1.2 m/s = 4.32 km/h and 12 % above the front mean: both above.
        (10.0, 11.2, True),
        # 0.5 m/s = 1.8 km/h and 5 %: neither.
        (10.0, 10.5, False),
        # 1.6 m/s = 5.76 km/h above 24 m/s, but 6.7 %: the ratio alone prevents the flag.
        (24.0, 25.6, False),
    ],
)
def test_tcs_on_the_rear_wheels_with_the_pedal_released(
    front: float, rear: float, expected: bool
) -> None:
    p = params()
    corrected = np.array([[front, front, rear, front]])
    reference = np.array([front])
    released = tcs_active(corrected, reference, np.array([False]), p)
    braking = tcs_active(corrected, reference, np.array([True]), p)
    assert released[0] == expected and not braking[0]
    # A front wheel above the reference is not a driven wheel: no TCS.
    corrected = np.array([[12.0, 8.0, 10.0, 10.0]])
    assert not tcs_active(corrected, np.array([10.0]), np.array([False]), p)[0]


def test_esc_compares_the_yaw_rate_with_the_single_track_reference() -> None:
    """10 m/s, road-wheel angle atan(3.6 / 50): reference 10 * (3.6 / 50) / 3.6 = 0.2 rad/s."""
    p = params()
    angle = np.full(3, p.steering_ratio * math.atan(p.wheelbase / 50))
    reference = np.full(3, 10.0)
    yaw = 0.2 + np.radians([4.9, 5.1, -5.1])
    error = yaw_rate_error(yaw, angle, reference, p)
    assert error == pytest.approx(np.radians([4.9, 5.1, -5.1]), abs=1e-12)
    assert list(esc_active(yaw, angle, reference, p)) == [False, True, True]
    # Inactive below ecu_reference_speed_min: 4.68 km/h with a 20 deg/s error.
    slow = esc_active(np.radians([20.0]), np.zeros(1), np.array([1.3]), p)
    assert not slow[0]


def lowest_margins(
    stiffness: float, shares: tuple[float, ...]
) -> dict[tuple[str, str], tuple[float, str]]:
    """Worst case of normal driving per flag (van_model.yaml, ecu_slip_speed_min).

    Every segment kind with its grades (end points and middle), its speeds every 0.1 km/h up to
    its highest target speed in the tightest curve allowed at each speed (its smallest radius,
    wider where the curve speed law requires), both turn directions, empty and laden, coldest
    and hottest ambient, tyre radius at both ends of its spread. Forces: service braking
    (pedal pressed), the largest traction and engine drag alone (pedal released). Measured
    signals carry every sign combination of the largest error (3 sigma plus half the
    resolution). Asserts that no flag is set; returns, per flag and segment kind, the lowest
    margin max(ecu_slip_speed_min / difference, threshold / ratio) (ABS, TCS) or
    threshold / |yaw error| (ESC) and where it occurs.
    """
    model, profile, duty = van_model(), drive_profile(), duty_cycle()
    p, longitudinal = params(), LongitudinalParameters.from_model(model)
    e_wheel, e_yaw, e_angle = error_bounds()
    errors = SIGNS * np.array([e_wheel] * 4 + [e_yaw, e_angle])
    spread, nominal = model.si("tyre_radius_spread"), model.si("tyre_rolling_radius")
    lateral = profile.si("curve_lateral_accel")
    worst = {(name, kind): (math.inf, "") for name in FLAGS for kind in SEGMENT_TYPES}
    for share, mass, ambient, radius, kind in itertools.product(
        shares,
        (model.si("mass_empty"), model.si("mass_empty") + model.si("payload_max")),
        (duty.value("ambient_temp_min"), duty.value("ambient_temp_max")),
        (nominal * (1 - spread), nominal * (1 + spread)),
        SEGMENT_TYPES,
    ):
        wheels = dataclasses.replace(
            WheelParameters.from_model(model), slip_stiffness=stiffness, payload_rear_share=share
        )
        density = air_density(duty.si("atmospheric_pressure"), ambient)
        if kind == "downhill":
            low, high = -profile.si("downhill_grade_max"), -profile.si("downhill_grade_min")
        else:
            low, high = -profile.si(f"{kind}_grade_max"), profile.si(f"{kind}_grade_max")
        top = profile.value(f"{kind}_target_speed_max")
        grid = np.meshgrid(
            np.linspace(low, high, 3), np.arange(1, round(top * 10) + 1) / 10 * KMH, (1.0, -1.0)
        )
        grade, speed, turn = (axis.ravel() for axis in grid)
        masses = np.full_like(speed, mass)
        curvature = np.minimum(1 / profile.si(CURVES[kind][0]), lateral / speed**2)
        yaw = turn * speed * curvature
        angle = turn * p.steering_ratio * np.arctan(p.wheelbase * curvature)
        _, highest = acceleration_limits(speed, grade, masses, density, longitudinal)
        service = split_forces(
            np.full_like(speed, -longitudinal.decel_service_max),
            speed,
            grade,
            masses,
            density,
            longitudinal,
        )
        traction = split_forces(highest, speed, grade, masses, density, longitudinal)
        drag = -masses * longitudinal.engine_braking_decel
        for case, engine, brake, braking in (
            ("service braking", service.engine, service.brake, True),
            ("largest traction", traction.engine, traction.brake, False),
            ("engine drag", drag, np.zeros_like(speed), False),
        ):
            slip_front, slip_rear = axle_slip(engine, brake, masses, wheels)
            readings = wheel_speed_readings(
                speed, yaw, slip_front, slip_rear, np.full_like(speed, radius), wheels
            )
            measured = readings[:, None, :] + errors[None, :, :4]
            yaw_m = yaw[:, None] + errors[None, :, 4]
            angle_m = angle[:, None] + errors[None, :, 5]
            pressed = np.full(yaw_m.shape, braking)
            corrected = corrected_wheel_speeds(measured, yaw_m, angle_m, p)
            reference = reference_speed(corrected, pressed)
            assert not np.any(abs_active(corrected, reference, pressed, p)), (kind, case)
            assert not np.any(tcs_active(corrected, reference, pressed, p)), (kind, case)
            assert not np.any(esc_active(yaw_m, angle_m, reference, p)), (kind, case)
            with np.errstate(divide="ignore", invalid="ignore"):
                margins = {}
                if braking:
                    difference, ratio = braking_slip(corrected, reference, p)
                    threshold = p.abs_threshold
                    margins["ABS"] = (difference, ratio, threshold)
                else:
                    difference, ratio = traction_slip(corrected, reference, p)
                    margins["TCS"] = (difference, ratio, p.tcs_threshold)
                for name, (difference, ratio, threshold) in margins.items():
                    per_wheel = np.where(
                        difference > 0,
                        np.maximum(p.slip_speed_min / difference, threshold / ratio),
                        np.inf,
                    )
                    lowest = per_wheel.min(axis=-1)
                    k = np.unravel_index(np.argmin(lowest), lowest.shape)
                    if lowest[k] < worst[name, kind][0]:
                        worst[name, kind] = (
                            float(lowest[k]),
                            f"{kind} {case}, grade {grade[k[0]]:+.1%}, "
                            f"{speed[k[0]] / KMH:.1f} km/h, mass {mass:.0f} kg, share {share}, "
                            f"{ambient} degC, radius {radius:.4f} m, curve "
                            f"{1 / curvature[k[0]]:.1f} m, slips {slip_front[k[0]]:.2%} / "
                            f"{slip_rear[k[0]]:.2%}, worst wheel "
                            f"{difference[k].max() / KMH:.2f} km/h, {ratio[k].max():.1%}",
                        )
                error = np.abs(yaw_rate_error(yaw_m, angle_m, reference, p))
                esc = np.where(reference >= p.reference_speed_min, p.esc_threshold / error, np.inf)
                k = np.unravel_index(np.argmin(esc), esc.shape)
                if esc[k] < worst["ESC", kind][0]:
                    worst["ESC", kind] = (
                        float(esc[k]),
                        f"{kind} {case}, grade {grade[k[0]]:+.1%}, "
                        f"{speed[k[0]] / KMH:.1f} km/h, mass {mass:.0f} kg, {ambient} degC, "
                        f"radius {radius:.4f} m, curve {1 / curvature[k[0]]:.1f} m, yaw error "
                        f"{math.degrees(error[k]):.2f} deg/s",
                    )
    return worst


def lowest(worst: dict[tuple[str, str], tuple[float, str]], name: str) -> tuple[float, str]:
    """Lowest margin of one flag over the segment kinds."""
    return min(margin for (flag, _), margin in worst.items() if flag == name)


def test_ecu_margins_from_the_configuration() -> None:
    """No flag in the worst case of normal driving and every margin at least 1; the lowest
    margins round to the figures the rationales of van_model.yaml state, with the conservative
    stiffness and payload share: ABS 2.13, TCS 1.86, ESC 3.93, and ABS 2.38 on descents."""
    worst = lowest_margins(STIFFNESS_FLOOR, REAR_SHARES)
    for (name, kind), (margin, where) in worst.items():
        assert margin >= 1, f"{name} margin {margin:.2f} on {kind} segments at {where}"
    assert lowest(worst, "ABS")[0] == pytest.approx(2.13, abs=0.005), lowest(worst, "ABS")
    assert lowest(worst, "TCS")[0] == pytest.approx(1.86, abs=0.005), lowest(worst, "TCS")
    assert lowest(worst, "ESC")[0] == pytest.approx(3.93, abs=0.005), lowest(worst, "ESC")
    downhill = worst["ABS", "downhill"]
    assert downhill[0] == pytest.approx(2.38, abs=0.005), downhill


def test_ecu_margins_with_the_nominal_tyre_and_payload() -> None:
    """Nominal stiffness (19) and payload share (0.88): ABS 2.54, TCS 2.18, ESC 4.43."""
    model = van_model()
    worst = lowest_margins(
        model.value("tyre_slip_stiffness"), (model.value("payload_rear_axle_share"),)
    )
    assert lowest(worst, "ABS")[0] == pytest.approx(2.54, abs=0.005), lowest(worst, "ABS")
    assert lowest(worst, "TCS")[0] == pytest.approx(2.18, abs=0.005), lowest(worst, "TCS")
    assert lowest(worst, "ESC")[0] == pytest.approx(4.43, abs=0.005), lowest(worst, "ESC")


def tightest_curve(direction: int, drawn_ramp_time: float) -> Curve:
    """The urban junction turn at the smallest radius, driven at its curve speed."""
    profile = drive_profile()
    radius, angle = profile.si("urban_turn_radius_min"), profile.si("urban_turn_angle")
    speed = curve_speed(
        radius, profile.si("urban_target_speed_max"), profile.si("curve_lateral_accel")
    )
    ramp = speed * curve_ramp_time(radius, angle, speed, drawn_ramp_time)
    return Curve(
        start=0.0,
        radius=radius,
        angle=angle,
        direction=direction,
        speed=speed,
        ramp_length=ramp,
        arc_length=max(radius * angle - ramp, 0.0),
    )


@dataclasses.dataclass(frozen=True)
class CurveDrive:
    """The whole tightest curve at its curve speed, one entry per position (SI units)."""

    curve: Curve
    speed: np.ndarray
    yaw: np.ndarray
    angle: np.ndarray  # steering wheel angle
    on_arc: np.ndarray  # positions at the peak curvature 1 / R


def drive_tightest_curve(direction: int, ramp_time: str) -> CurveDrive:
    """Positions every step (v * DT) through the 10 m junction turn (90 deg) at its curve speed,
    16.1 km/h, plus the point where the ramps end. Ramps of 2 s (curve_transition_time_min, with
    an arc of 6.76 m) or of 3.51 s (curve_transition_time_max, 4 s capped by R * angle / v: two
    ramps meeting at 1 / R, no arc)."""
    p = params()
    curve = tightest_curve(direction, drive_profile().si(ramp_time))
    v = curve.speed
    positions = np.union1d(
        np.arange(0.0, curve.end + v * PHYSICS_DT, v * PHYSICS_DT),
        [curve.start + curve.ramp_length],
    )
    curvature = curve.curvature(positions)
    speed = np.full_like(positions, v)
    on_arc = np.abs(curvature) == 1 / curve.radius
    assert on_arc.sum() >= (10 if curve.arc_length > 0 else 1)
    return CurveDrive(
        curve=curve,
        speed=speed,
        yaw=speed * curvature,
        angle=p.steering_ratio * np.arctan(p.wheelbase * curvature),
        on_arc=on_arc,
    )


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("ramp_time", ["curve_transition_time_min", "curve_transition_time_max"])
def test_corrections_through_the_tightest_curve(direction: int, ramp_time: str) -> None:
    """Kinematics alone (no slip, no noise), through the whole 10 m curve: the front wheels stay
    within the arc's band (inner +1.01 %, outer -0.94 %) and reach it at the peak curvature;
    the rear wheels read v exactly."""
    p, drive = params(), drive_tightest_curve(direction, ramp_time)
    v = drive.curve.speed
    readings = ground_speeds(drive.speed, drive.yaw, WheelParameters.from_model(van_model()))
    corrected = corrected_wheel_speeds(readings, drive.yaw, drive.angle, p)
    front = corrected[:, :2] / v
    assert np.all((front >= ARC_OUTER - 5e-5) & (front <= ARC_INNER + 5e-5))
    inner, outer = (0, 1) if direction == 1 else (1, 0)
    assert front[drive.on_arc, inner] == pytest.approx(ARC_INNER, abs=5e-5)
    assert front[drive.on_arc, outer] == pytest.approx(ARC_OUTER, abs=5e-5)
    assert corrected[:, 2:] / v == pytest.approx(np.ones((len(drive.speed), 2)), rel=1e-12)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("ramp_time", ["curve_transition_time_min", "curve_transition_time_max"])
@pytest.mark.parametrize("braked", [False, True])
def test_no_flag_through_the_tightest_curve(direction: int, ramp_time: str, braked: bool) -> None:
    """The whole 10 m curve at its curve speed, step by step: steady (pedal released), or with
    the forces of a deceleration at approach_decel (pedal pressed, so ABS is checked; the speed
    is held at the curve speed, the highest with that braking force). Empty and laden, urban
    grades, conservative and nominal tyre and payload share, both ends of the tyre radius
    spread, every error sign pattern on the measured signals. No flag on any step."""
    model, profile, duty = van_model(), drive_profile(), duty_cycle()
    p, longitudinal = params(), LongitudinalParameters.from_model(model)
    drive = drive_tightest_curve(direction, ramp_time)
    speed, yaw, angle = drive.speed, drive.yaw, drive.angle
    density = air_density(duty.si("atmospheric_pressure"), duty.value("ambient_temp_max"))
    accel = -profile.si("approach_decel") if braked else 0.0
    grade_max = profile.si("urban_grade_max")
    nominal, spread = model.si("tyre_rolling_radius"), model.si("tyre_radius_spread")
    for stiffness, share, mass, grade, radius in itertools.product(
        (STIFFNESS_FLOOR, model.value("tyre_slip_stiffness")),
        REAR_SHARES,
        (model.si("mass_empty"), model.si("mass_empty") + model.si("payload_max")),
        (-grade_max, 0.0, grade_max),
        (nominal * (1 - spread), nominal * (1 + spread)),
    ):
        wheels = dataclasses.replace(
            WheelParameters.from_model(model), slip_stiffness=stiffness, payload_rear_share=share
        )
        masses, grades = np.full_like(speed, mass), np.full_like(speed, grade)
        forces = split_forces(
            np.full_like(speed, accel), speed, grades, masses, density, longitudinal
        )
        assert np.all(forces.brake > 0) if braked else np.all(forces.brake == 0)
        slip_front, slip_rear = axle_slip(forces.engine, forces.brake, masses, wheels)
        radii = np.full_like(speed, radius)
        readings = wheel_speed_readings(speed, yaw, slip_front, slip_rear, radii, wheels)
        wheels_m, yaw_m, angle_m = measure(
            readings[:, None, :], yaw[:, None], angle[:, None], SIGNS[None, :, :]
        )
        pressed = np.full(yaw_m.shape, braked)
        abs_flag, tcs_flag, esc_flag = flags(wheels_m, yaw_m, angle_m, pressed, p)
        where = (stiffness, share, mass, grade, radius)
        assert not np.any(abs_flag) and not np.any(tcs_flag) and not np.any(esc_flag), where


@settings(max_examples=max(10, settings.default.max_examples // 5))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    trip=st.integers(min_value=0, max_value=5),
    duration=st.floats(min_value=60.0, max_value=300.0),
    ambient=st.floats(min_value=-5.0, max_value=30.0),
    radius_factor=st.floats(min_value=0.99, max_value=1.01),
    signs=st.tuples(*[st.sampled_from((-1.0, 1.0))] * 6),
)
def test_driven_ecu_respects_the_invariants(
    seed: int,
    trip: int,
    duration: float,
    ambient: float,
    radius_factor: float,
    signs: tuple[float, ...],
) -> None:
    """On driven trips, with the measured signals off by the drawn error sign pattern: no flag
    on any step; the reference speed is 0 at rest and otherwise within the slip bound, the
    single-angle error and the measurement error of the rear axle speed read with the tyre
    radius bias."""
    model, profile = van_model(), drive_profile()
    p, wheels = params(), WheelParameters.from_model(model)
    brakes = BrakeParameters.from_configs(model, profile)
    drives = drive_with_steering(seed, trip, duration, ambient)
    bound = slip_bound(np.array([d.mass for d in drives]))
    e_wheel, e_yaw, e_angle = error_bounds()
    for driven in drives:
        trace, steering = driven.trace, driven.steering
        n = len(trace.speed)
        mass = np.full(n, driven.mass)
        radius = np.full(n, radius_factor * wheels.radius_nominal)
        slip_front, slip_rear = axle_slip(trace.engine_force, trace.brake_force, mass, wheels)
        readings = wheel_speed_readings(
            trace.speed, steering.yaw_rate, slip_front, slip_rear, radius, wheels
        )
        next_speed = np.append(trace.speed[1:], 0.0)  # 0 after the last step: ignition off
        held = held_steps(trace.speed, next_speed)
        holding = hold_force(mass, trace.grade)
        pressure = master_cylinder_pressure(
            trace.brake_force, held, trace.at_delivery, holding, brakes
        )
        pressed = brake_pedal_pressed(pressure, brakes)
        wheels_m, yaw_m, angle_m = measure(
            readings, steering.yaw_rate, steering.steering_wheel_angle, np.array(signs)
        )
        abs_flag, tcs_flag, esc_flag = flags(wheels_m, yaw_m, angle_m, pressed, p)
        assert not np.any(abs_flag) and not np.any(tcs_flag) and not np.any(esc_flag)
        corrected = corrected_wheel_speeds(wheels_m, yaw_m, angle_m, p)
        reference = reference_speed(corrected, pressed)
        still = trace.speed == 0
        assert np.all(reference[still] == 0)
        # Rear axle speed read with the radius bias; the reference stays within the slip bound
        # and the single-angle error of it, plus the measurement error through the corrections.
        read = trace.speed * wheels.radius_nominal / radius
        kinematic = read * ((1 + bound) * ARC_INNER - 1)
        measurement = (
            e_wheel + e_yaw * p.track / 2 + readings.max(axis=1) * e_angle / p.steering_ratio
        )
        assert np.all(np.abs(reference - read) <= kinematic + measurement + 1e-9)
