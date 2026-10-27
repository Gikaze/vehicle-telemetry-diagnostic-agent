"""Layouts and semantic checks of the vehicle model, sensor and plausibility configuration files.

The parameter sets below are the complete list of constants the vehicle model, the brake ECU logic,
the sensor layer and the invariant checks may use; a constant used in code must appear here and in
the matching YAML file with its rationale.
"""

from functools import cache
from itertools import pairwise
from pathlib import Path

from simulator.params import Config, ConfigError, Layout, config_path, load_config

DRIVE_AXLES = ("front", "rear")

VAN_MODEL_LAYOUT = Layout(
    structural={("model", "id"): str, ("model", "name"): str, ("drive_axle",): str},
    parameters=frozenset(
        {
            # mass
            "mass_empty",
            "payload_max",
            "unladen_rear_axle_share",
            "payload_rear_axle_share",
            # geometry
            "wheelbase",
            "track_width",
            "tyre_rolling_radius",
            # steering
            "steering_ratio",
            "steering_wheel_lock",
            "steering_wheel_rate_max",
            "driver_torque_gain_rate",
            "driver_torque_gain_lat",
            "driver_torque_gain_park",
            "eps_assist_ratio_standstill",
            "eps_assist_ratio_high_speed",
            "eps_assist_fade_speed",
            # longitudinal and lateral
            "accel_max",
            "decel_service_max",
            "speed_tracking_gain",
            "drag_area",
            "rolling_resistance",
            "engine_braking_decel",
            "accel_lat_max",
            # powertrain
            "engine_power_max",
            "pedal_mapping_min_speed",
            "gear_overall_ratios",
            "upshift_speed",
            "downshift_speed",
            "shift_margin",
            "idle_speed",
            "engine_speed_max",
            # brakes
            "front_brake_share",
            "brake_pressure_gain",
            "brake_reference_mass",
            "brake_switch_pressure",
            "brake_energy_to_disc",
            "disc_heat_capacity_front",
            "disc_heat_capacity_rear",
            "disc_cooling_tau_standstill",
            "disc_cooling_tau_reference",
            "disc_cooling_reference_speed",
            # brake ECU logic
            "tyre_slip_stiffness",
            "abs_slip_threshold",
            "tcs_slip_threshold",
            "esc_yaw_error_threshold",
            "ecu_reference_speed_min",
            "ecu_slip_speed_min",
            # tyres
            "tyre_pressure_nominal",
            "tyre_pressure_reference_temp",
            "tyre_warmup_rise",
            "tyre_warmup_reference_speed",
            "tyre_thermal_tau",
            # engine
            "coolant_thermostat_temp",
            "coolant_load_rise",
            "coolant_warmup_tau",
            "coolant_cooldown_tau",
            # per-vehicle and per-wheel variation
            "tyre_radius_spread",
            "disc_heat_capacity_vehicle_spread",
            "disc_heat_capacity_wheel_spread",
            "tyre_pressure_wheel_spread",
        }
    ),
)

# Numeric catalogue signals that get measurement noise (sigma in sensors.yaml). The others are
# noise-free, for the reason given in NOISE_FREE.
NOISY_SIGNALS = {
    "wheel_speed": ("wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr"),
    "accel_long": ("accel_long",),
    "accel_lat": ("accel_lat",),
    "yaw_rate": ("yaw_rate",),
    "steering_wheel_angle": ("steering_wheel_angle",),
    "steering_wheel_rate": ("steering_wheel_rate",),
    "driver_torque": ("driver_torque",),
    "eps_assist_torque": ("eps_assist_torque",),
    "master_cyl_pressure": ("master_cyl_pressure",),
    "tyre_pressure": (
        "tyre_pressure_fl",
        "tyre_pressure_fr",
        "tyre_pressure_rl",
        "tyre_pressure_rr",
    ),
    "tyre_temp": ("tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr"),
    "coolant_temp": ("coolant_temp",),
    "accel_pedal_position": ("accel_pedal_position",),
}
NOISE_FREE = {
    "vehicle_ref_speed": "estimated by the brake ECU from the (noisy) wheel speeds",
    "brake_disc_temp_est_fl": "estimate computed by the brake ECU from braking energy",
    "brake_disc_temp_est_fr": "estimate computed by the brake ECU from braking energy",
    "brake_disc_temp_est_rl": "estimate computed by the brake ECU from braking energy",
    "brake_disc_temp_est_rr": "estimate computed by the brake ECU from braking energy",
    "engine_speed": "crankshaft position sensor; quantisation dominates",
    "odometer": "integrated distance kept by the instrument cluster",
    "ambient_temp": "fixed for the simulated day (spec section 3)",
    "eps_temperature": "nominal in v1.0 (spec section 3)",
}

SENSORS_LAYOUT = Layout(
    structural={},
    parameters=frozenset({"noise_clip_sigma", *NOISY_SIGNALS}),
)

PLAUSIBILITY_LAYOUT = Layout(
    structural={},
    parameters=frozenset(
        {
            "disc_lr_spread_rel_max",
            "disc_lr_spread_abs_max",
            "disc_front_rear_ratio_min",
            "disc_front_rear_ratio_max",
            "disc_ratio_window",
            "disc_ratio_min_rear_rise",
            "coolant_rate_max",
            "coolant_band_max",
            "approach_decel_tolerance",
        }
    ),
)


@cache
def van_model() -> Config:
    """The repository's vehicle model, loaded and validated on first use (never at import)."""
    return load_van_model()


@cache
def sensors() -> Config:
    return load_sensors()


@cache
def plausibility() -> Config:
    return load_plausibility()


def load_van_model(path: Path | None = None) -> Config:
    config = load_config(path or config_path("van_model.yaml"), VAN_MODEL_LAYOUT)
    _check_drive_axle(config)
    _check_gear_schedule(config)
    _check_brakes(config)
    _check_disc_cooling(config)
    _check_tyres(config)
    _check_coolant(config)
    _check_ecu(config)
    return config


def load_sensors(path: Path | None = None) -> Config:
    return load_config(path or config_path("sensors.yaml"), SENSORS_LAYOUT)


def load_plausibility(path: Path | None = None) -> Config:
    config = load_config(path or config_path("plausibility.yaml"), PLAUSIBILITY_LAYOUT)
    if not config.value("disc_front_rear_ratio_min") < config.value("disc_front_rear_ratio_max"):
        raise ConfigError("plausibility.yaml: disc_front_rear_ratio_min must be below the maximum")
    return config


def _check_drive_axle(config: Config) -> None:
    axle = config.structure[("drive_axle",)]
    if axle not in DRIVE_AXLES:
        raise ConfigError(f"van_model.yaml: drive_axle must be one of {DRIVE_AXLES}, got {axle!r}")


def _check_brakes(config: Config) -> None:
    """Pressure = gain * F_brake / reference mass is non-negative for a non-negative force."""
    for name in ("brake_pressure_gain", "brake_reference_mass"):
        if not config.value(name) > 0:
            raise ConfigError(f"van_model.yaml: {name} must be positive")


def _check_disc_cooling(config: Config) -> None:
    """Heat capacities positive; 1/tau = (1/tau_standstill) * (1 + v / v_c) needs
    0 < tau_reference < tau_standstill and a positive reference speed, so that v_c > 0."""
    for name in ("disc_heat_capacity_front", "disc_heat_capacity_rear"):
        if not config.value(name) > 0:
            raise ConfigError(f"van_model.yaml: {name} must be positive")
    if not config.value("disc_cooling_reference_speed") > 0:
        raise ConfigError("van_model.yaml: disc_cooling_reference_speed must be positive")
    reference = config.value("disc_cooling_tau_reference")
    if not 0 < reference < config.value("disc_cooling_tau_standstill"):
        raise ConfigError(
            "van_model.yaml: expected 0 < disc_cooling_tau_reference "
            "< disc_cooling_tau_standstill (discs cool faster when moving)"
        )


def _check_tyres(config: Config) -> None:
    """Target T_a + rise * min(1, v / reference speed) and the first-order step need a positive
    time constant and reference speed and a non-negative rise; every drawn cold pressure must be
    positive."""
    for name in ("tyre_thermal_tau", "tyre_warmup_reference_speed"):
        if not config.value(name) > 0:
            raise ConfigError(f"van_model.yaml: {name} must be positive")
    if not config.value("tyre_warmup_rise") >= 0:
        raise ConfigError("van_model.yaml: tyre_warmup_rise must not be negative")
    if not config.value("tyre_pressure_nominal") - config.value("tyre_pressure_wheel_spread") > 0:
        raise ConfigError(
            "van_model.yaml: tyre_pressure_nominal minus tyre_pressure_wheel_spread must be "
            "positive (every drawn cold pressure)"
        )


def _check_coolant(config: Config) -> None:
    """Target thermostat + load_rise * pedal / 100 and the first-order step need a positive time
    constant and a non-negative load rise."""
    if not config.value("coolant_warmup_tau") > 0:
        raise ConfigError("van_model.yaml: coolant_warmup_tau must be positive")
    if not config.value("coolant_load_rise") >= 0:
        raise ConfigError("van_model.yaml: coolant_load_rise must not be negative")


def _check_ecu(config: Config) -> None:
    """Slip ratios in (0, 1); yaw error threshold and the two speed guards positive (the slip
    ratio divides by max(reference speed, ecu_reference_speed_min))."""
    for name in ("abs_slip_threshold", "tcs_slip_threshold"):
        if not 0 < config.value(name) < 1:
            raise ConfigError(f"van_model.yaml: {name} must lie between 0 and 1")
    for name in ("esc_yaw_error_threshold", "ecu_reference_speed_min", "ecu_slip_speed_min"):
        if not config.value(name) > 0:
            raise ConfigError(f"van_model.yaml: {name} must be positive")


def _check_gear_schedule(config: Config) -> None:
    """Ratios strictly decreasing; an upshift never lands below the downshift speed (no hunting)."""
    ratios = config.value("gear_overall_ratios")
    if not isinstance(ratios, list) or len(ratios) < 2:
        raise ConfigError("van_model.yaml: gear_overall_ratios must list at least two gears")
    if any(ratio <= 0 for ratio in ratios) or any(b >= a for a, b in pairwise(ratios)):
        raise ConfigError("van_model.yaml: gear_overall_ratios must be positive and decreasing")
    upshift = config.value("upshift_speed")
    downshift = config.value("downshift_speed")
    idle = config.value("idle_speed")
    smallest_step = min(b / a for a, b in pairwise(ratios))
    after_upshift = upshift * smallest_step
    if downshift > after_upshift - config.value("shift_margin"):
        raise ConfigError(
            f"van_model.yaml: downshift_speed {downshift} rpm is not at least shift_margin below "
            f"the engine speed after the largest upshift step ({after_upshift:.0f} rpm): "
            "the gear schedule would hunt"
        )
    if not idle < downshift < upshift < config.value("engine_speed_max"):
        raise ConfigError(
            "van_model.yaml: expected idle_speed < downshift_speed < upshift_speed "
            "< engine_speed_max"
        )
