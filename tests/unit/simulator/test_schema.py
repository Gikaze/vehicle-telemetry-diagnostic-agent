"""FR-16: the telemetry schema is generated from the catalogue; the committed copy has no drift."""

import copy
import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from simulator.catalogue import load_catalogue
from simulator.paths import schema_path
from simulator.schema import build_schema, main, render

CATALOGUE = load_catalogue()
SCHEMA = build_schema(CATALOGUE)
VALIDATOR = Draft202012Validator(SCHEMA)


def periodic_message() -> dict[str, Any]:
    """A valid periodic message: every numeric statistic at its range minimum."""
    signals: dict[str, Any] = {}
    for signal in CATALOGUE.signals:
        if signal.is_number:
            assert signal.range is not None
            low = signal.range[0]
            signals[signal.name] = {"mean": low, "min": low, "max": low, "last": low}
        else:
            signals[signal.name] = {"last": False, "count": 0}
    return {
        "message_type": "periodic",
        "schema_version": CATALOGUE.schema_version,
        "vehicle_id": "VAN-042",
        "seq": 0,
        "window_start": "2026-01-05T07:30:00.000Z",
        "window_s": 10,
        "signals": signals,
    }


def test_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(SCHEMA)


def test_committed_schema_is_valid_draft_2020_12() -> None:
    """The file consumers load is itself a valid schema, independently of the drift check."""
    Draft202012Validator.check_schema(json.loads(schema_path().read_text()))


def test_committed_schema_matches_the_catalogue() -> None:
    """Drift check: regenerate with `make schema` after changing catalogue.yaml."""
    assert schema_path().read_text() == render(SCHEMA)


def test_check_mode_reports_drift(tmp_path) -> None:
    stale = tmp_path / "telemetry.schema.json"
    stale.write_text("{}\n")
    assert main(["--check", "--out", str(stale)]) == 1
    assert main(["--out", str(stale)]) == 0
    assert main(["--check", "--out", str(stale)]) == 0


def test_valid_message_passes() -> None:
    VALIDATOR.validate(periodic_message())


def test_invalid_value_passes_after_a_json_round_trip() -> None:
    """Every numeric signal may carry its catalogue invalid value, exactly, through JSON."""
    message = periodic_message()
    for signal in CATALOGUE.signals:
        if signal.is_number:
            message["signals"][signal.name]["last"] = signal.invalid
    VALIDATOR.validate(json.loads(json.dumps(message)))


@pytest.mark.parametrize("name", ["wheel_speed_fl", "accel_long", "coolant_temp"])
def test_out_of_range_value_fails(name: str) -> None:
    signal = CATALOGUE.signal(name)
    assert signal.range is not None and signal.resolution is not None
    message = periodic_message()
    message["signals"][name]["max"] = signal.range[1] + signal.resolution
    assert not VALIDATOR.is_valid(message)


def test_missing_signal_fails() -> None:
    message = periodic_message()
    del message["signals"]["yaw_rate"]
    assert not VALIDATOR.is_valid(message)


def test_unknown_signal_fails() -> None:
    message = periodic_message()
    message["signals"]["gearbox_temp"] = {"mean": 1, "min": 1, "max": 1, "last": 1}
    assert not VALIDATOR.is_valid(message)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_s", 0),
        ("window_s", 10.5),
        ("seq", -1),
        ("vehicle_id", "van-042"),
        ("window_start", "2026-01-05 07:30:00"),
        ("schema_version", 2),
    ],
)
def test_envelope_violations_fail(field: str, value: Any) -> None:
    message = copy.deepcopy(periodic_message())
    message[field] = value
    assert not VALIDATOR.is_valid(message)


def test_boolean_count_must_be_a_non_negative_integer() -> None:
    message = periodic_message()
    message["signals"]["abs_active"]["count"] = -1
    assert not VALIDATOR.is_valid(message)


def test_resolution_is_annotated_not_validated() -> None:
    """The encoding grid is carried as annotations; no float multipleOf in the schema."""
    value = SCHEMA["$defs"]["value.wheel_speed_fl"]
    assert (value["x-unit"], value["x-resolution"], value["x-offset"]) == ("km/h", 0.01, 0)
    assert "multipleOf" not in json.dumps(SCHEMA)
