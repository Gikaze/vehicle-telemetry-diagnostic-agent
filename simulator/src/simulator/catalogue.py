"""Signal catalogue (FR-16): loading, structural validation and the encoding grid of each signal."""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from simulator.paths import catalogue_path

SignalType = Literal["number", "boolean"]

REQUIRED_FIELDS = (
    "name",
    "group",
    "ecu",
    "type",
    "unit",
    "range",
    "resolution",
    "raw_bits",
    "offset",
    "invalid",
    "dynamic",
)
OPTIONAL_FIELDS = ("description",)
ENCODING_FIELDS = ("range", "resolution", "raw_bits", "offset", "invalid")

# Raw-unit tolerance when checking that a float lies on the resolution grid: float division
# leaves errors of order 1e-15 relative (e.g. 0.07 / 0.01 = 7.000000000000001).
GRID_TOLERANCE = 1e-6


class CatalogueError(ValueError):
    """The catalogue file does not have the expected structure."""


@dataclass(frozen=True)
class Ecu:
    id: str
    description: str


@dataclass(frozen=True)
class Signal:
    name: str
    group: str
    ecu: str
    type: SignalType
    unit: str
    range: tuple[float, float] | None
    resolution: float | None
    raw_bits: int | None
    offset: float | None
    invalid: float | None
    dynamic: bool
    description: str | None = None

    @property
    def is_number(self) -> bool:
        return self.type == "number"

    @property
    def decimals(self) -> int:
        """Number of decimal places of the resolution (0.01 -> 2, 0.5 -> 1, 1 -> 0)."""
        if self.resolution is None:
            raise ValueError(f"{self.name} has no resolution")
        exponent = Decimal(str(self.resolution)).normalize().as_tuple().exponent
        assert isinstance(exponent, int)
        return max(0, -exponent)


@dataclass(frozen=True)
class Catalogue:
    schema_version: int
    ecus: tuple[Ecu, ...]
    signals: tuple[Signal, ...]

    def signal(self, name: str) -> Signal:
        for signal in self.signals:
            if signal.name == name:
                return signal
        raise KeyError(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(signal.name for signal in self.signals)


def load_catalogue(path: Path | None = None) -> Catalogue:
    """Load and structurally validate the catalogue (default: `simulator/catalogue.yaml`)."""
    document = yaml.safe_load((path or catalogue_path()).read_text())
    if not isinstance(document, dict):
        raise CatalogueError("the catalogue must be a mapping")
    ecus = tuple(
        Ecu(id=entry["id"], description=entry["description"]) for entry in document["ecus"]
    )
    ecu_ids = {ecu.id for ecu in ecus}
    signals = tuple(_parse_signal(entry, ecu_ids) for entry in document["signals"])
    names = [signal.name for signal in signals]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CatalogueError(f"duplicate signal names: {duplicates}")
    return Catalogue(schema_version=document["schema_version"], ecus=ecus, signals=signals)


def _parse_signal(entry: dict[str, Any], ecu_ids: set[str]) -> Signal:
    name = entry.get("name", "<unnamed>")
    missing = [field for field in REQUIRED_FIELDS if field not in entry]
    if missing:
        raise CatalogueError(f"{name}: missing fields {missing}")
    unknown = sorted(set(entry) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise CatalogueError(f"{name}: unknown fields {unknown}")
    if "description" in entry and not isinstance(entry["description"], str):
        raise CatalogueError(f"{name}: description must be text")
    if entry["ecu"] not in ecu_ids:
        raise CatalogueError(f"{name}: unknown ECU {entry['ecu']!r}")
    if entry["type"] == "boolean":
        set_fields = [field for field in ENCODING_FIELDS if entry[field] is not None]
        if set_fields:
            raise CatalogueError(f"{name}: booleans have no encoding, but {set_fields} are set")
        signal_range = None
    elif entry["type"] == "number":
        null_fields = [field for field in ENCODING_FIELDS if entry[field] is None]
        if null_fields:
            raise CatalogueError(f"{name}: numbers need {null_fields}")
        low, high = entry["range"]
        if not low < high:
            raise CatalogueError(f"{name}: range minimum must be below maximum")
        if entry["resolution"] <= 0:
            raise CatalogueError(f"{name}: resolution must be positive")
        signal_range = (low, high)
    else:
        raise CatalogueError(f"{name}: unknown type {entry['type']!r}")
    return Signal(
        name=name,
        group=entry["group"],
        ecu=entry["ecu"],
        type=entry["type"],
        unit=entry["unit"],
        range=signal_range,
        resolution=entry["resolution"],
        raw_bits=entry["raw_bits"],
        offset=entry["offset"],
        invalid=entry["invalid"],
        dynamic=entry["dynamic"],
        description=entry.get("description"),
    )


def quantise(values: np.ndarray, signal: Signal) -> np.ndarray:
    """Round physical values onto the signal's encoding grid.

    value = offset + round(raw) * resolution, then rounded to the decimals of the resolution so
    that the float matches the decimal literal a consumer parses (no 655.3500000000001).
    Saturation to the measurement range and invalid values are handled by the sensor layer;
    the invalid value is always copied from the catalogue, never recomputed here.
    """
    assert signal.resolution is not None and signal.offset is not None
    raw = np.round((np.asarray(values, dtype=float) - signal.offset) / signal.resolution)
    return np.round(signal.offset + raw * signal.resolution, signal.decimals)


def on_grid(value: float, resolution: float, offset: float = 0.0) -> bool:
    """True if `value` is `offset` plus a whole number of `resolution` steps, within tolerance."""
    steps = (value - offset) / resolution
    return abs(steps - round(steps)) <= GRID_TOLERANCE
