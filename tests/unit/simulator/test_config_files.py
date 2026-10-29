"""Semantic checks of the vehicle model, sensor and plausibility configuration files."""

from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from simulator.catalogue import catalogue
from simulator.config_files import (
    NOISE_FREE,
    NOISY_SIGNALS,
    load_plausibility,
    load_sensors,
    load_van_model,
    sensors,
)
from simulator.params import ConfigError, config_path


def van_model_copy(tmp_path: Path, edit) -> Path:
    document = yaml.safe_load(config_path("van_model.yaml").read_text())
    edit(document)
    path = tmp_path / "van_model.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def set_value(group: str, name: str, value):
    def edit(document):
        document[group][name]["value"] = value

    return edit


def set_shift_speeds(upshift: int, downshift: int):
    def edit(document):
        document["powertrain"]["upshift_speed"]["value"] = upshift
        document["powertrain"]["downshift_speed"]["value"] = downshift

    return edit


def test_gear_schedule_does_not_hunt() -> None:
    """After the largest upshift step the engine stays above the downshift speed plus margin."""
    model = load_van_model()
    ratios = model.value("gear_overall_ratios")
    after_upshift = model.value("upshift_speed") * min(b / a for a, b in pairwise(ratios))
    assert model.value("downshift_speed") <= after_upshift - model.value("shift_margin")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        # Upshift at 2200 rpm in first gives 2200 * 10.14 / 17.55 = 1271 rpm in second, below a
        # 1300 rpm downshift: immediate downshift (the hunting case of the review).
        (set_shift_speeds(2200, 1300), "would hunt"),
        # 2000 * 0.578 = 1156 rpm, minus the 100 rpm margin, is below the 1200 rpm downshift.
        (set_value("powertrain", "upshift_speed", 2000), "would hunt"),
        # 1444 rpm after the upshift leaves only 94 rpm above a 1350 rpm downshift: inside the
        # margin.
        (set_shift_speeds(2500, 1350), "would hunt"),
        (set_value("powertrain", "gear_overall_ratios", [17.55, 18.0, 6.63]), "decreasing"),
        (set_value("powertrain", "gear_overall_ratios", [17.55]), "at least two gears"),
        (set_value("powertrain", "downshift_speed", 700), "idle_speed < downshift_speed"),
        (set_value("powertrain", "upshift_speed", 4600), "upshift_speed"),
    ],
)
def test_bad_gear_schedules_are_rejected(tmp_path: Path, edit, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_van_model(van_model_copy(tmp_path, edit))


def test_disc_cooling_law_parameters_load() -> None:
    model = load_van_model()
    assert model.si("disc_cooling_tau_standstill") == 600
    assert model.si("disc_cooling_tau_reference") == 150
    assert model.si("disc_cooling_reference_speed") == 20


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        # Equal or longer than at standstill: discs would not cool faster when moving (v_c
        # infinite or negative).
        (set_value("brakes", "disc_cooling_tau_reference", 600), "cool faster when moving"),
        (set_value("brakes", "disc_cooling_tau_reference", 700), "cool faster when moving"),
        (set_value("brakes", "disc_cooling_tau_reference", 0), "cool faster when moving"),
        (set_value("brakes", "disc_cooling_tau_reference", -150), "cool faster when moving"),
        (set_value("brakes", "disc_cooling_reference_speed", 0), "reference_speed must be"),
        (set_value("brakes", "disc_heat_capacity_front", 0), "capacity_front must be positive"),
        (set_value("brakes", "disc_heat_capacity_rear", -1), "capacity_rear must be positive"),
    ],
)
def test_bad_disc_thermal_constants_are_rejected(tmp_path: Path, edit, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_van_model(van_model_copy(tmp_path, edit))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (set_value("tyres", "tyre_thermal_tau", 0), "tyre_thermal_tau must be positive"),
        (
            set_value("tyres", "tyre_warmup_reference_speed", -1),
            "tyre_warmup_reference_speed must be positive",
        ),
        (set_value("tyres", "tyre_warmup_rise", -5), "tyre_warmup_rise must not be negative"),
        # 0.1 bar of spread around 0.1 bar: a drawn cold pressure could be 0.
        (set_value("tyres", "tyre_pressure_nominal", 0.1), "every drawn cold pressure"),
        (set_value("variation", "tyre_pressure_wheel_spread", 4), "every drawn cold pressure"),
    ],
)
def test_bad_tyre_constants_are_rejected(tmp_path: Path, edit, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_van_model(van_model_copy(tmp_path, edit))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (set_value("engine", "coolant_warmup_tau", 0), "coolant_warmup_tau must be positive"),
        (set_value("engine", "coolant_load_rise", -1), "coolant_load_rise must not be negative"),
    ],
)
def test_bad_coolant_constants_are_rejected(tmp_path: Path, edit, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_van_model(van_model_copy(tmp_path, edit))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (set_value("ecu_logic", "abs_slip_threshold", 0), "abs_slip_threshold must lie between"),
        (set_value("ecu_logic", "tcs_slip_threshold", 1), "tcs_slip_threshold must lie between"),
        (
            set_value("ecu_logic", "esc_yaw_error_threshold", 0),
            "esc_yaw_error_threshold must be positive",
        ),
        (
            set_value("ecu_logic", "ecu_reference_speed_min", -5),
            "ecu_reference_speed_min must be positive",
        ),
        (set_value("ecu_logic", "ecu_slip_speed_min", 0), "ecu_slip_speed_min must be positive"),
    ],
)
def test_bad_ecu_constants_are_rejected(tmp_path: Path, edit, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_van_model(van_model_copy(tmp_path, edit))


@pytest.mark.parametrize("name", ["brake_pressure_gain", "brake_reference_mass"])
@pytest.mark.parametrize("value", [0, -10])
def test_brake_pressure_law_needs_positive_constants(tmp_path: Path, name: str, value) -> None:
    """p = gain * F_brake / reference mass must be non-negative for every F_brake >= 0."""
    with pytest.raises(ConfigError, match=f"{name} must be positive"):
        load_van_model(van_model_copy(tmp_path, set_value("brakes", name, value)))


def test_drive_axle_must_be_front_or_rear(tmp_path: Path) -> None:
    def edit(document):
        document["drive_axle"] = "middle"

    with pytest.raises(ConfigError, match="drive_axle"):
        load_van_model(van_model_copy(tmp_path, edit))


def test_drive_axle_is_rear() -> None:
    assert load_van_model().structure[("drive_axle",)] == "rear"


def test_front_rear_ratio_bounds_are_ordered(tmp_path: Path) -> None:
    document = yaml.safe_load(config_path("plausibility.yaml").read_text())
    document["disc_temperature"]["disc_front_rear_ratio_min"]["value"] = 2.0
    path = tmp_path / "plausibility.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigError, match="ratio_min must be below"):
        load_plausibility(path)


def test_every_numeric_signal_is_noisy_or_noise_free_with_a_reason() -> None:
    """sensors.yaml lists a sigma for each noisy signal; every other one has a stated reason."""
    numeric = {signal.name for signal in catalogue().signals if signal.is_number}
    noisy = [name for names in NOISY_SIGNALS.values() for name in names]
    assert len(noisy) == len(set(noisy)), "a signal is listed under two sigmas"
    assert not set(noisy) & set(NOISE_FREE), "a signal is both noisy and noise-free"
    assert numeric - set(noisy) == set(NOISE_FREE)
    assert set(noisy) <= numeric
    assert all(isinstance(reason, str) and reason.strip() for reason in NOISE_FREE.values())
    assert set(sensors().parameters) == {
        "noise_clip_sigma",
        "noise_block_duration",
        *NOISY_SIGNALS,
    }


def sensors_copy(tmp_path: Path, group: str, name: str, value) -> Path:
    document = yaml.safe_load(config_path("sensors.yaml").read_text())
    document[group][name]["value"] = value
    path = tmp_path / "sensors.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.mark.parametrize(
    ("group", "name", "value", "message"),
    [
        ("noise", "noise_clip_sigma", 0, "noise_clip_sigma must be positive"),
        ("noise_sigma", "yaw_rate", -0.1, "yaw_rate must be positive"),
        # 6.05 s is 60.5 steps: a block must be a whole number of steps.
        ("noise", "noise_block_duration", 6.05, "whole number of physics steps"),
        ("noise", "noise_block_duration", 0, "whole number of physics steps"),
    ],
)
def test_sensor_checks_reject(
    tmp_path: Path, group: str, name: str, value: float, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_sensors(sensors_copy(tmp_path, group, name, value))


def test_noise_sigmas_are_in_the_catalogue_units() -> None:
    """A sigma is added in the unit of its signals (K for a temperature in degC)."""
    signal = catalogue().signal
    for group, names in NOISY_SIGNALS.items():
        unit = sensors().parameters[group].unit
        for name in names:
            expected = "K" if signal(name).unit == "degC" else signal(name).unit
            assert unit == expected, (group, name)
