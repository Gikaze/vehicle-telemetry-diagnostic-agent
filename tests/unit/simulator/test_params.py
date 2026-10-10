"""Spec section 8: every simulator constant lives in config/ with value, unit and rationale."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from simulator.catalogue import load_catalogue
from simulator.config_files import (
    NOISE_FREE,
    NOISY_SIGNALS,
    load_plausibility,
    load_sensors,
    load_van_model,
)
from simulator.params import ConfigError, Layout, load_config

LAYOUT = Layout(
    structural={("model", "id"): str, ("batches",): list},
    parameters=frozenset({"wheelbase", "ratios"}),
)


def parameter(value: Any = 3.6, unit: Any = "m", rationale: Any = "Mid-length van.") -> dict:
    return {"value": value, "unit": unit, "rationale": rationale}


def document(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "model": {"id": "nordwind-lv1"},
        "batches": [],
        "geometry": {"wheelbase": parameter(), "ratios": parameter(value=[4.5, 2.6], unit="-")},
    }
    doc.update(overrides)
    return doc


def load(tmp_path: Path, doc: dict[str, Any]):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(doc))
    return load_config(path, LAYOUT)


def test_valid_document_loads(tmp_path: Path) -> None:
    config = load(tmp_path, document())
    assert config.value("wheelbase") == 3.6
    assert config.value("ratios") == [4.5, 2.6]
    assert config.structure[("model", "id")] == "nordwind-lv1"


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"wheelbase": 3.6, "ratios": parameter()}, "expected a parameter"),
        ({"wheelbase": {"unit": "m"}, "ratios": parameter()}, "needs exactly value, unit"),
        ({"wheelbase": parameter(rationale=""), "ratios": parameter()}, "rationale must be"),
        ({"wheelbase": parameter(rationale=None), "ratios": parameter()}, "rationale must be"),
        ({"wheelbase": parameter(unit=""), "ratios": parameter()}, "unit must be"),
        ({"wheelbase": parameter(value="long"), "ratios": parameter()}, "must be a number"),
        ({"wheelbase": parameter(value=True), "ratios": parameter()}, "must be a number"),
        ({"wheelbase": parameter(value=[]), "ratios": parameter()}, "must be a number"),
        (
            {"wheelbase": {**parameter(), "source": "x"}, "ratios": parameter()},
            "needs exactly value, unit",
        ),
        ({"wheelbase": parameter()}, "missing \\['ratios'\\]"),
        (
            {"wheelbase": parameter(), "ratios": parameter(), "track": parameter()},
            "unexpected \\['track'\\]",
        ),
        ({"wheelbase": parameter(), "ratios": parameter(), "empty": {}}, "expected a parameter"),
    ],
)
def test_parameter_rules(tmp_path: Path, geometry: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load(tmp_path, document(geometry=geometry))


def test_parameter_defined_twice_is_rejected(tmp_path: Path) -> None:
    doc = document(other={"wheelbase": parameter()})
    with pytest.raises(ConfigError, match="defined twice"):
        load(tmp_path, doc)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": {"id": 7}}, "expected str"),
        ({"model": {"id": True}}, "expected str"),
        ({"batches": "2025-T1"}, "expected list"),
        ({"colour": "red"}, "expected a parameter"),
        ({"model": {"id": "x", "name": "y"}}, "expected a parameter"),
    ],
)
def test_structural_rules(tmp_path: Path, overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load(tmp_path, document(**overrides))


def test_missing_structural_field_is_rejected(tmp_path: Path) -> None:
    doc = document()
    del doc["batches"]
    with pytest.raises(ConfigError, match="missing structural fields"):
        load(tmp_path, doc)


def test_repository_config_files_load() -> None:
    assert load_van_model().value("steering_wheel_lock") == 540
    assert load_sensors().value("noise_clip_sigma") == 3
    assert load_plausibility().value("disc_ratio_min_rear_rise") == 20


def test_every_numeric_signal_has_a_noise_decision() -> None:
    """Each numeric catalogue signal is noisy (sigma in sensors.yaml) or noise-free, with reason."""
    noisy = {name for names in NOISY_SIGNALS.values() for name in names}
    numeric = {signal.name for signal in load_catalogue().signals if signal.is_number}
    assert not noisy & set(NOISE_FREE)
    assert noisy | set(NOISE_FREE) == numeric
