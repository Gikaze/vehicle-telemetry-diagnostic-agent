"""Layouts and checks of the duty cycle and drive profile files, including cross-file checks.

Cross-file checks tie the drive profile to the vehicle model (lateral acceleration, steering lock
and rate, braking and traction limits, downhill braking) and the ambient range to the catalogue.
"""

import math
from functools import cache
from pathlib import Path

from simulator.catalogue import Catalogue, catalogue
from simulator.config_files import van_model
from simulator.constants import PHYSICS_DT
from simulator.model.forces import air_density, brake_force_to_hold_speed
from simulator.params import Config, ConfigError, Layout, config_path, load_config

SEGMENT_TYPES = ("urban", "rural", "downhill")

DUTY_CYCLE_LAYOUT = Layout(
    structural={},
    parameters=frozenset(
        {
            "ambient_temp_min",
            "ambient_temp_max",
            "atmospheric_pressure",
            "shift_start_earliest",
            "shift_start_latest",
            "shift_end_earliest",
            "shift_end_latest",
            "trips_min",
            "trips_max",
            "short_break_min",
            "short_break_max",
            "lunch_break_min",
            "lunch_break_max",
            "trip_weight_spread",
            "trip_duration_min",
            "trip_duration_max",
        }
    ),
)

# Curves of each segment type: (radius min, radius max, angle min, angle max, curves per km).
# Urban turns have a single angle.
CURVES = {
    "urban": (
        "urban_turn_radius_min",
        "urban_turn_radius_max",
        "urban_turn_angle",
        "urban_turn_angle",
        "urban_turns_per_km",
    ),
    "rural": (
        "rural_bend_radius_min",
        "rural_bend_radius_max",
        "rural_bend_angle_min",
        "rural_bend_angle_max",
        "rural_bends_per_km",
    ),
    "downhill": (
        "downhill_bend_radius_min",
        "downhill_bend_radius_max",
        "downhill_bend_angle_min",
        "downhill_bend_angle_max",
        "downhill_bends_per_km",
    ),
}

DRIVE_PROFILE_LAYOUT = Layout(
    structural={},
    parameters=frozenset(
        {
            *(f"weight_{kind}" for kind in SEGMENT_TYPES),
            *(f"{kind}_target_speed_{bound}" for kind in SEGMENT_TYPES for bound in ("min", "max")),
            *(f"{kind}_duration_{bound}" for kind in SEGMENT_TYPES for bound in ("min", "max")),
            *(name for names in CURVES.values() for name in names),
            "urban_grade_max",
            "rural_grade_max",
            "downhill_grade_min",
            "downhill_grade_max",
            "urban_traffic_stops_per_km",
            "traffic_stop_min",
            "traffic_stop_max",
            "delivery_stop_probability",
            "delivery_stop_min",
            "delivery_stop_max",
            "curve_lateral_accel",
            "approach_decel",
            "pull_away_accel",
            "curve_transition_time_min",
            "curve_transition_time_max",
            "stop_speed_threshold",
            "standstill_brake_pressure",
        }
    ),
)

WEIGHT_SUM_TOLERANCE = 1e-9


@cache
def duty_cycle() -> Config:
    """The repository's duty cycle, loaded and checked on first use (never at import)."""
    return load_duty_cycle()


@cache
def drive_profile() -> Config:
    """The repository's drive profile, loaded and checked against the vehicle model on first use."""
    profile = load_drive_profile()
    check_profile_against_model(profile, van_model(), duty_cycle())
    return profile


def load_duty_cycle(path: Path | None = None, catalogue_: Catalogue | None = None) -> Config:
    config = load_config(path or config_path("duty_cycle.yaml"), DUTY_CYCLE_LAYOUT)
    _check_ordered_pairs(config, "duty_cycle.yaml")
    _check_day_plan(config)
    _check_ambient_in_catalogue(config, catalogue_ or catalogue())
    return config


def load_drive_profile(path: Path | None = None) -> Config:
    config = load_config(path or config_path("drive_profile.yaml"), DRIVE_PROFILE_LAYOUT)
    _check_ordered_pairs(config, "drive_profile.yaml")
    weights = [config.value(f"weight_{kind}") for kind in SEGMENT_TYPES]
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1) > WEIGHT_SUM_TOLERANCE:
        raise ConfigError(
            f"drive_profile.yaml: segment weights must be non-negative and sum to 1, got {weights}"
        )
    if not 0 <= config.value("delivery_stop_probability") <= 1:
        raise ConfigError("drive_profile.yaml: delivery_stop_probability must be in [0, 1]")
    return config


def _check_ordered_pairs(config: Config, file: str) -> None:
    """Every x_min <= x_max and x_earliest <= x_latest pair is ordered."""
    for low_suffix, high_suffix in (("_min", "_max"), ("_earliest", "_latest")):
        for name in config.parameters:
            if not name.endswith(low_suffix):
                continue
            partner = name[: -len(low_suffix)] + high_suffix
            if partner in config.parameters and config.value(name) > config.value(partner):
                raise ConfigError(f"{file}: {name} is above {partner}")


def _check_day_plan(config: Config) -> None:
    """Trip durations stay within their bounds in both worst cases of the day plan draws."""
    if config.value("trips_min") < 1:
        raise ConfigError("duty_cycle.yaml: at least one trip per day")
    if config.si("shift_end_earliest") <= config.si("shift_start_latest"):
        raise ConfigError("duty_cycle.yaml: the earliest shift end must follow the latest start")
    spread = config.value("trip_weight_spread")
    if not 0 <= spread < 1:
        raise ConfigError("duty_cycle.yaml: trip_weight_spread must be in [0, 1)")
    shortest = shortest_trip(config)
    longest = longest_trip(config)
    if shortest < config.si("trip_duration_min"):
        raise ConfigError(
            f"duty_cycle.yaml: the day plan can give a {shortest / 60:.0f} min trip, below "
            f"trip_duration_min"
        )
    if longest > config.si("trip_duration_max"):
        raise ConfigError(
            f"duty_cycle.yaml: the day plan can give a {longest / 60:.0f} min trip, above "
            f"trip_duration_max"
        )


def _breaks(trips: int, short: float, lunch: float) -> float:
    """Total break time between `trips` trips: one lunch break, short breaks elsewhere."""
    return lunch + max(trips - 2, 0) * short if trips > 1 else 0.0


def shortest_trip(config: Config) -> float:
    """Shortest possible trip, s: latest start, earliest end, most trips, longest breaks."""
    trips = config.value("trips_max")
    available = config.si("shift_end_earliest") - config.si("shift_start_latest")
    driving = available - _breaks(trips, config.si("short_break_max"), config.si("lunch_break_max"))
    spread = config.value("trip_weight_spread")
    return driving * (1 - spread) / ((1 - spread) + (trips - 1) * (1 + spread))


def longest_trip(config: Config) -> float:
    """Longest possible trip, s: earliest start, latest end, fewest trips, shortest breaks."""
    trips = config.value("trips_min")
    available = config.si("shift_end_latest") - config.si("shift_start_earliest")
    driving = available - _breaks(trips, config.si("short_break_min"), config.si("lunch_break_min"))
    spread = config.value("trip_weight_spread")
    return driving * (1 + spread) / ((1 + spread) + (trips - 1) * (1 - spread))


def _check_ambient_in_catalogue(config: Config, catalogue_: Catalogue) -> None:
    signal = catalogue_.signal("ambient_temp")
    assert signal.range is not None
    low, high = signal.range
    if not low <= config.value("ambient_temp_min") <= config.value("ambient_temp_max") <= high:
        raise ConfigError(
            f"duty_cycle.yaml: the ambient range must lie within the catalogue range "
            f"[{low}, {high}]"
        )


def check_profile_against_model(profile: Config, model: Config, duty: Config) -> None:
    """Cross-file checks between drive_profile.yaml, van_model.yaml and duty_cycle.yaml."""
    if profile.si("curve_lateral_accel") >= model.si("accel_lat_max"):
        raise ConfigError("drive_profile.yaml: curve_lateral_accel must be below accel_lat_max")
    if profile.si("approach_decel") > model.si("decel_service_max"):
        raise ConfigError("drive_profile.yaml: approach_decel exceeds decel_service_max")
    if profile.si("pull_away_accel") > model.si("accel_max"):
        raise ConfigError("drive_profile.yaml: pull_away_accel exceeds accel_max")
    if profile.si("stop_speed_threshold") / PHYSICS_DT > profile.si("approach_decel"):
        raise ConfigError(
            "drive_profile.yaml: setting stop_speed_threshold to 0 in one step decelerates "
            "faster than approach_decel"
        )
    if profile.si("standstill_brake_pressure") <= model.si("brake_switch_pressure"):
        raise ConfigError(
            "drive_profile.yaml: standstill_brake_pressure must exceed brake_switch_pressure"
        )
    for kind in SEGMENT_TYPES:
        _check_curves(kind, profile, model)
    _check_downhill_braking(profile, model, duty)
    _check_day_end_margin(profile, duty)


def longest_curve_time(profile: Config) -> float:
    """Longest time to drive through one curve, s: widest radius, largest angle, lowest speed."""
    times = []
    for kind, (_, radius_max_name, _, angle_max_name, _) in CURVES.items():
        radius = profile.si(radius_max_name)
        speed = curve_speed(
            radius, profile.si(f"{kind}_target_speed_min"), profile.si("curve_lateral_accel")
        )
        times.append(radius * profile.si(angle_max_name) / speed)
    return max(times)


def trip_overrun_bound(profile: Config) -> float:
    """Upper bound of the time a trip can run past its planned duration, s.

    The final stop is inserted when the remaining time equals the time to stop at approach_decel;
    if that stop falls in or just before a curve it moves to the curve end, which adds at most the
    whole curve plus the stop from the curve speed (at most the highest segment speed), plus one
    physics step for the step at which the insertion is detected.
    """
    top_speed = max(profile.si(f"{kind}_target_speed_max") for kind in SEGMENT_TYPES)
    return longest_curve_time(profile) + top_speed / profile.si("approach_decel") + PHYSICS_DT


def day_end_margin(profile: Config, duty: Config) -> float:
    """Margin kept before shift_end_latest so that the last trip ends by it in the worst case, s."""
    return duty.value("trips_max") * trip_overrun_bound(profile)


def effective_shift_end_latest(profile: Config, duty: Config) -> float:
    """Latest planned end of the day, s after midnight, so the actual end is at most 18:30."""
    return duty.si("shift_end_latest") - day_end_margin(profile, duty)


def _check_day_end_margin(profile: Config, duty: Config) -> None:
    if effective_shift_end_latest(profile, duty) < duty.si("shift_end_earliest"):
        raise ConfigError(
            f"duty_cycle.yaml: with {duty.value('trips_max')} trips each overrunning by up to "
            f"{trip_overrun_bound(profile):.0f} s, no planned end in the shift end window keeps "
            "the day within shift_end_latest"
        )


def curve_ramp_time(radius: float, angle: float, speed: float, drawn_ramp_time: float) -> float:
    """Ramp time shortened so that entry plus exit ramps turn no more than the curve angle.

    A linear curvature ramp of length L to 1/R turns the heading by L / (2 R); entry plus exit turn
    L / R. With L = v * t, the ramps fit the curve when v * t <= R * angle; the constant-curvature
    arc then covers the remaining R * angle - L.
    """
    return min(drawn_ramp_time, radius * angle / speed)


def curve_speed(radius: float, segment_speed: float, lateral_accel: float) -> float:
    return min(segment_speed, math.sqrt(lateral_accel * radius))


def _check_curves(kind: str, profile: Config, model: Config) -> None:
    radius_min_name, _, angle_min_name, _, _ = CURVES[kind]
    radius = profile.si(radius_min_name)
    lock = model.si("steering_wheel_lock")
    wheelbase, ratio = model.si("wheelbase"), model.si("steering_ratio")
    needed = wheelbase * ratio / radius  # steering-wheel angle, rad (small-angle single track)
    if needed > lock:
        raise ConfigError(
            f"drive_profile.yaml: {radius_min_name} {radius:g} m needs "
            f"{math.degrees(needed):.0f} deg of steering wheel, beyond the "
            f"{math.degrees(lock):.0f} deg lock"
        )
    # Worst steering rate: tightest radius, smallest angle, fastest curve speed, shortest ramp.
    speed = curve_speed(
        radius, profile.si(f"{kind}_target_speed_max"), profile.si("curve_lateral_accel")
    )
    ramp = curve_ramp_time(
        radius, profile.si(angle_min_name), speed, profile.si("curve_transition_time_min")
    )
    rate = needed / ramp
    if rate > model.si("steering_wheel_rate_max"):
        raise ConfigError(
            f"drive_profile.yaml: {kind} curves need {math.degrees(rate):.0f} deg/s of steering "
            f"rate (radius {radius:g} m, ramp {ramp:.2f} s), above steering_wheel_rate_max"
        )


def downhill_worst_brake_force(profile: Config, model: Config, duty: Config) -> float:
    """Brake force on the gentlest, fastest descent with an empty van in the densest air, N.

    Drag grows with speed and its share grows as mass falls, so the highest speed, the empty van
    and the coldest ambient leave the least work to the brakes.
    """
    density = air_density(duty.si("atmospheric_pressure"), duty.value("ambient_temp_min"))
    force = brake_force_to_hold_speed(
        model.si("mass_empty"),
        profile.si("downhill_target_speed_max"),
        -profile.si("downhill_grade_min"),
        density=density,
        drag_area=model.si("drag_area"),
        rolling_resistance=model.si("rolling_resistance"),
        engine_braking_decel=model.si("engine_braking_decel"),
    )
    return float(force)


def _check_downhill_braking(profile: Config, model: Config, duty: Config) -> None:
    force = downhill_worst_brake_force(profile, model, duty)
    if force <= 0:
        raise ConfigError(
            f"drive_profile.yaml: a descent of {profile.value('downhill_grade_min')} % at "
            f"{profile.value('downhill_target_speed_max')} km/h needs no braking for an empty van "
            f"in cold air (brake force {force:.0f} N): downhill segments would not heat the discs"
        )
