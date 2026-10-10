"""FR-16: the signal catalogue is complete, consistent and encodes every value it declares."""

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from simulator.catalogue import (
    ENCODING_FIELDS,
    REQUIRED_FIELDS,
    CatalogueError,
    Signal,
    catalogue,
    load_catalogue,
    on_grid,
    quantise,
)

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs" / "simulator-spec.md"

# Rows of the spec section 3 table that are deliberately not catalogue signals, keyed by the
# first name in the row. The whole row is excluded, including the names it lists after it.
EXCLUDED_FROM_SIGNALS = {
    # Diagnostic trouble codes (code + status bits test_failed, pending, confirmed,
    # warning_indicator) are not signals: they enter catalogue.yaml as a separate `diagnostics`
    # section, and the schema as `dtcs[]`, in the fault-scenario PR (M1 PR 4,
    # feat/m1-fault-scenarios).
    "dtcs[]": "fault-scenario PR (M1 PR 4)",
}


def _signals(numeric: bool) -> list[Signal]:
    """Signals for per-signal parametrisation, read at collection time.

    A catalogue that fails to load yields no cases here instead of breaking collection;
    test_repository_catalogue_loads then fails with the loader's message.
    """
    try:
        signals = catalogue().signals
    except (CatalogueError, OSError, KeyError, TypeError, ValueError):
        return []
    return [signal for signal in signals if signal.is_number == numeric]


NUMERIC = _signals(numeric=True)
BOOLEAN = _signals(numeric=False)


def test_repository_catalogue_loads() -> None:
    load_catalogue()


def spec_section_3_rows() -> list[list[str]]:
    """Names listed per row of the signal-catalogue table of docs/simulator-spec.md, expanded."""
    section = SPEC.read_text().split("## 3.", 1)[1].split("\n## ", 1)[0]
    rows: list[list[str]] = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 3 or not cells[2].startswith("`"):
            continue
        rows.append(
            [name for token in re.findall(r"`([^`]+)`", cells[2]) for name in _expand(token)]
        )
    return rows


def _expand(token: str) -> list[str]:
    match = re.fullmatch(r"(.*)\{([^}]+)\}(.*)", token)
    if not match:
        return [token]
    prefix, options, suffix = match.groups()
    return [f"{prefix}{option}{suffix}" for option in options.split(",")]


def test_every_spec_signal_is_in_the_catalogue() -> None:
    """Spec section 3 and the catalogue list the same signals, apart from the named exclusions."""
    rows = spec_section_3_rows()
    excluded_rows = [row for row in rows if row[0] in EXCLUDED_FROM_SIGNALS]
    assert {row[0] for row in excluded_rows} == set(EXCLUDED_FROM_SIGNALS), (
        "an exclusion no longer appears in the spec"
    )
    expected = {name for row in rows if row[0] not in EXCLUDED_FROM_SIGNALS for name in row}
    assert set(catalogue().names) == expected


def _number_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": "x",
        "group": "g",
        "ecu": "ecm",
        "type": "number",
        "unit": "rpm",
        "range": [0, 100],
        "resolution": 1,
        "raw_bits": 8,
        "offset": 0,
        "invalid": 255,
        "dynamic": True,
    }
    entry.update(overrides)
    return entry


def _boolean_entry(**overrides: object) -> dict[str, object]:
    entry = _number_entry(type="boolean", unit="-")
    entry.update({field: None for field in ENCODING_FIELDS})
    entry.update(overrides)
    return entry


def _without(entry: dict[str, object], field: str) -> dict[str, object]:
    return {key: value for key, value in entry.items() if key != field}


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        *[([_without(_number_entry(), field)], "missing fields") for field in REQUIRED_FIELDS],
        ([_number_entry(colour="red")], "unknown fields"),
        ([_number_entry(description=5)], "description must be text"),
        ([_number_entry(ecu="gearbox")], "unknown ECU"),
        ([_number_entry(), _number_entry()], "duplicate signal names"),
        ([_number_entry(type="string")], "unknown type"),
        ([_number_entry(resolution=None)], "numbers need"),
        ([_number_entry(range=[100, 0])], "range minimum must be below maximum"),
        ([_number_entry(resolution=0)], "resolution must be positive"),
        *[
            ([_boolean_entry(**{field: 1})], "booleans have no encoding")
            for field in ENCODING_FIELDS
        ],
    ],
)
def test_loader_rejects_malformed_entries(
    tmp_path: Path, entries: list[dict[str, object]], message: str
) -> None:
    """Structural rules: every field explicit, known ECU, unique names, consistent nulls."""
    path = tmp_path / "catalogue.yaml"
    document = {
        "schema_version": 1,
        "ecus": [{"id": "ecm", "description": "e"}],
        "signals": entries,
    }
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(CatalogueError, match=message):
        load_catalogue(path)


def test_loader_accepts_the_minimal_entries(tmp_path: Path) -> None:
    path = tmp_path / "catalogue.yaml"
    document = {
        "schema_version": 1,
        "ecus": [{"id": "ecm", "description": "e"}],
        "signals": [_number_entry(), _boolean_entry(name="y")],
    }
    path.write_text(yaml.safe_dump(document))
    assert load_catalogue(path).names == ("x", "y")


def test_signal_names_are_unique() -> None:
    assert len(set(catalogue().names)) == len(catalogue().names)


@pytest.mark.parametrize("signal", BOOLEAN, ids=lambda s: s.name)
def test_booleans_have_no_encoding(signal: Signal) -> None:
    """Booleans have no invalid value in v1.0 (decision), nor range or resolution."""
    assert signal.unit == "-"
    assert (signal.range, signal.resolution, signal.raw_bits, signal.offset, signal.invalid) == (
        None,
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize("signal", NUMERIC, ids=lambda s: s.name)
def test_offset_is_on_the_resolution_grid(signal: Signal) -> None:
    """The offset is an integer number of resolution steps, so raw 0 is a grid point."""
    assert signal.resolution is not None and signal.offset is not None
    assert on_grid(signal.offset, signal.resolution), f"{signal.name}: offset off the grid"


@pytest.mark.parametrize("signal", NUMERIC, ids=lambda s: s.name)
def test_invalid_is_the_maximum_raw_value(signal: Signal) -> None:
    """invalid = offset + (2**raw_bits - 1) * resolution (catalogue header rule)."""
    assert signal.resolution is not None and signal.offset is not None
    assert signal.raw_bits is not None and signal.invalid is not None
    expected = signal.offset + (2**signal.raw_bits - 1) * signal.resolution
    assert signal.invalid == pytest.approx(expected, abs=signal.resolution * 1e-6), (
        f"{signal.name}: invalid {signal.invalid}, rule gives {expected}"
    )


@pytest.mark.parametrize("signal", NUMERIC, ids=lambda s: s.name)
def test_range_is_encodable_and_below_invalid(signal: Signal) -> None:
    """The measurement range lies on the encodable grid and strictly below the invalid value."""
    assert signal.range is not None and signal.resolution is not None
    assert signal.offset is not None and signal.invalid is not None
    low, high = signal.range
    assert low >= signal.offset, f"{signal.name}: range minimum below the offset"
    assert high < signal.invalid, f"{signal.name}: range maximum not below the invalid value"
    assert on_grid(low, signal.resolution, signal.offset), f"{signal.name}: minimum off the grid"
    assert on_grid(high, signal.resolution, signal.offset), f"{signal.name}: maximum off the grid"


@pytest.mark.parametrize("signal", NUMERIC, ids=lambda s: s.name)
def test_invalid_value_round_trips_through_quantisation(signal: Signal) -> None:
    """Quantising the invalid value returns the catalogue float itself, bit for bit."""
    assert signal.invalid is not None
    assert quantise(np.array([signal.invalid]), signal)[0] == signal.invalid


def test_quantise_rounds_to_the_decimal_literal() -> None:
    """offset + raw * resolution can miss the decimal literal; quantise returns the literal."""
    signal = catalogue().signal("accel_pedal_position")
    assert 1023 * 0.1 == 102.30000000000001  # the float hazard this guards against
    assert quantise(np.array([102.3, 0.07, 12.34]), signal).tolist() == [102.3, 0.1, 12.3]


def test_quantise_applies_a_negative_offset() -> None:
    signal = catalogue().signal("accel_long")
    assert quantise(np.array([-1.234, 0.0, 2.005]), signal).tolist() == [-1.23, 0.0, 2.0]


@pytest.mark.parametrize(
    ("value", "resolution", "offset", "expected"),
    [
        (0.07, 0.01, 0.0, True),  # 0.07 / 0.01 = 7.000000000000001
        (0.075, 0.01, 0.0, False),
        (-327.68, 0.01, -327.68, True),
        (983.0, 1, -40, True),
        (87.5, 0.5, -40, True),
        (87.25, 0.5, -40, False),
    ],
)
def test_on_grid_tolerates_float_error_only(
    value: float, resolution: float, offset: float, expected: bool
) -> None:
    assert on_grid(value, resolution, offset) is expected
