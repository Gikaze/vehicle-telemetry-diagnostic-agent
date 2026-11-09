"""Duty cycle and drive profile: single-file and cross-file checks, each seen failing once."""

import math
from pathlib import Path

import pytest
import yaml

from simulator.catalogue import catalogue
from simulator.config_files import van_model
from simulator.model.forces import air_density, brake_force_to_hold_speed
from simulator.params import UNIT_FACTORS, ConfigError, config_path
from simulator.profile_config import (
    check_profile_against_model,
    curve_ramp_time,
    downhill_worst_brake_force,
    drive_profile,
    duty_cycle,
    load_drive_profile,
    load_duty_cycle,
    longest_trip,
    shortest_trip,
)


def corrupted(tmp_path: Path, name: str, group: str, parameter: str, value) -> Path:
    """Copy of a repository config file with one parameter value changed."""
    document = yaml.safe_load(config_path(name).read_text())
    document[group][parameter]["value"] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document))
    return path


def test_repository_files_pass_every_check() -> None:
    profile = drive_profile()
    check_profile_against_model(profile, van_model(), duty_cycle())


def test_every_unit_used_has_a_known_si_factor() -> None:
    for config in (van_model(), duty_cycle(), drive_profile()):
        for parameter in config.parameters.values():
            assert parameter.unit in UNIT_FACTORS


def test_unknown_unit_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(config_path("duty_cycle.yaml").read_text())
    document["shift"]["short_break_min"]["unit"] = "fortnight"
    path = tmp_path / "duty_cycle.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigError, match="unknown unit 'fortnight'"):
        load_duty_cycle(path)


def test_si_conversion() -> None:
    profile = drive_profile()
    assert profile.si("urban_target_speed_max") == pytest.approx(50 / 3.6)
    assert profile.si("downhill_grade_min") == pytest.approx(0.07)
    assert profile.si("urban_turn_angle") == pytest.approx(math.pi / 2)
    with pytest.raises(ConfigError, match="no plain SI factor"):
        duty_cycle().si("ambient_temp_min")


# Duty cycle -----------------------------------------------------------------------------------


def test_day_plan_worst_cases() -> None:
    """Hand check: 385 min over 6 trips at weight 0.7 / 7.2; 670 min over 4 at 1.3 / 3.4."""
    duty = duty_cycle()
    assert shortest_trip(duty) / 60 == pytest.approx(385 * 0.7 / 7.2)
    assert longest_trip(duty) / 60 == pytest.approx(670 * 1.3 / 3.4)


@pytest.mark.parametrize(
    ("group", "parameter", "value", "message"),
    [
        ("shift", "trips_max", 9, "below trip_duration_min"),
        ("shift", "shift_end_latest", 21.0, "above trip_duration_max"),
        # Start 9:00, end 16:00, 6 trips, 125 min of breaks: 295 min * 0.7 / 7.2 = 28.7 min.
        ("shift", "shift_start_latest", 9.0, "below trip_duration_min"),
        ("shift", "shift_end_earliest", 7.0, "earliest shift end must follow"),
        ("shift", "trips_min", 7, "trips_min is above trips_max"),
        ("shift", "trip_weight_spread", 1.0, "trip_weight_spread must be in"),
        ("ambient", "ambient_temp_min", 40, "ambient_temp_min is above ambient_temp_max"),
        ("ambient", "ambient_temp_max", 70, "within the catalogue range"),
    ],
)
def test_bad_duty_cycles_are_rejected(
    tmp_path: Path, group: str, parameter: str, value, message: str
) -> None:
    path = corrupted(tmp_path, "duty_cycle.yaml", group, parameter, value)
    with pytest.raises(ConfigError, match=message):
        load_duty_cycle(path, catalogue())


# Drive profile, single file -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "parameter", "value", "message"),
    [
        ("segments", "weight_urban", 0.5, "sum to 1"),
        ("segments", "weight_downhill", -0.1, "non-negative and sum to 1"),
        ("urban", "delivery_stop_probability", 1.2, "delivery_stop_probability must be in"),
        ("rural", "rural_target_speed_min", 95, "rural_target_speed_min is above"),
        ("downhill", "downhill_grade_min", 12, "downhill_grade_min is above"),
    ],
)
def test_bad_drive_profiles_are_rejected(
    tmp_path: Path, group: str, parameter: str, value, message: str
) -> None:
    path = corrupted(tmp_path, "drive_profile.yaml", group, parameter, value)
    with pytest.raises(ConfigError, match=message):
        load_drive_profile(path)


# Drive profile against the vehicle model ------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "parameter", "value", "message"),
    [
        ("driver", "curve_lateral_accel", 3.0, "curve_lateral_accel must be below accel_lat_max"),
        ("driver", "approach_decel", 4.5, "approach_decel exceeds decel_service_max"),
        ("driver", "pull_away_accel", 2.5, "pull_away_accel exceeds accel_max"),
        ("driver", "stop_speed_threshold", 0.2, "decelerates faster than approach_decel"),
        ("driver", "standstill_brake_pressure", 2, "must exceed brake_switch_pressure"),
        # Laden van on 10 %: 3300 * g * sin(atan 0.1) = 3220 N, i.e. 11.5 bar > 10 bar.
        ("urban", "urban_grade_max", 10, "does not hold the laden van .* 11.5 bar"),
        # 3.6 m * 17 / 6 m = 10.2 rad = 584 deg > 540 deg lock.
        ("urban", "urban_turn_radius_min", 6, "beyond the 540 deg lock"),
        # 351 deg of steering wheel in a 0.9 s ramp = 390 deg/s > 360 deg/s.
        ("driver", "curve_transition_time_min", 0.9, "above steering_wheel_rate_max"),
        # 6 %, 60 km/h, empty van, -5 degC: brake force -32 N.
        ("downhill", "downhill_grade_min", 6, "needs no braking"),
    ],
)
def test_profile_inconsistent_with_the_model_is_rejected(
    tmp_path: Path, group: str, parameter: str, value, message: str
) -> None:
    path = corrupted(tmp_path, "drive_profile.yaml", group, parameter, value)
    profile = load_drive_profile(path)
    with pytest.raises(ConfigError, match=message):
        check_profile_against_model(profile, van_model(), duty_cycle())


def test_steering_rate_uses_the_shortened_ramp(tmp_path: Path) -> None:
    """A small rural bend angle shortens the ramp and raises the steering rate needed."""
    # 150 m, 0.4 deg, 17.3 m/s: ramp shortened to 150 * 0.00698 / 17.3 = 0.06 s, so the
    # 23.4 deg of steering wheel need 387 deg/s. At 2 deg the ramp is 0.30 s: 77 deg/s, accepted.
    path = corrupted(tmp_path, "drive_profile.yaml", "rural", "rural_bend_angle_min", 0.4)
    profile = load_drive_profile(path)
    with pytest.raises(ConfigError, match="rural curves need .* ramp 0.06 s"):
        check_profile_against_model(profile, van_model(), duty_cycle())


# Curve geometry and forces --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("radius", "angle_deg", "speed", "drawn"),
    [(10, 90, 4.5, 4.0), (150, 20, 17.3, 4.0), (300, 120, 14.0, 2.0), (600, 20, 25.0, 4.0)],
)
def test_ramps_never_turn_more_than_the_curve_angle(
    radius: float, angle_deg: float, speed: float, drawn: float
) -> None:
    """Entry plus exit ramps turn L / R with L = v t; the arc covers the rest of the angle."""
    angle = math.radians(angle_deg)
    ramp = curve_ramp_time(radius, angle, speed, drawn)
    ramps_turn = speed * ramp / radius
    arc_turn = (radius * angle - speed * ramp) / radius
    assert ramps_turn <= angle + 1e-12
    assert arc_turn >= -1e-12
    assert ramps_turn + arc_turn == pytest.approx(angle)


def test_review_cases_of_ramp_overturn() -> None:
    """10 m junction at 4.5 m/s, 4 s ramps: 102 deg unshortened, so the ramp is shortened."""
    unshortened = math.degrees(4.5 * 4.0 / 10)
    assert unshortened == pytest.approx(103.1, abs=0.1)
    assert curve_ramp_time(10, math.radians(90), 4.5, 4.0) == pytest.approx(10 * math.pi / 2 / 4.5)


def test_air_density_at_the_range_ends() -> None:
    assert air_density(101300, -5) == pytest.approx(1.316, abs=1e-3)
    assert air_density(101300, 30) == pytest.approx(1.164, abs=1e-3)
    assert air_density(101300, 20) == pytest.approx(1.204, abs=1e-3)


def test_downhill_figures_quoted_in_the_rationales() -> None:
    model = van_model()
    common = {
        "drag_area": model.si("drag_area"),
        "rolling_resistance": model.si("rolling_resistance"),
        "engine_braking_decel": model.si("engine_braking_decel"),
    }
    cold = air_density(101300, -5)
    at_6 = brake_force_to_hold_speed(2300, 60 / 3.6, -0.06, density=cold, **common)
    assert float(at_6) == pytest.approx(-32, abs=1)
    assert downhill_worst_brake_force(drive_profile(), model, duty_cycle()) == pytest.approx(
        192, abs=1
    )
    force = brake_force_to_hold_speed(
        3300, 55 / 3.6, -0.10, density=air_density(101300, 20), **common
    )
    assert float(force) * 55 / 3.6 / 1000 == pytest.approx(21.3, abs=0.1)
