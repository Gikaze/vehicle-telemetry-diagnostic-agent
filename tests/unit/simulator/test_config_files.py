"""Semantic checks of the vehicle model and plausibility configuration files."""

from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from simulator.config_files import load_plausibility, load_van_model
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
