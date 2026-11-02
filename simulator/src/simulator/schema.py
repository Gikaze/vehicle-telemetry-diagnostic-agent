"""Generate the telemetry JSON Schema (Draft 2020-12) from the signal catalogue (FR-16).

Run `python -m simulator.schema` (or `make schema`) to rewrite
simulator/schema/telemetry.schema.json, and `python -m simulator.schema --check` to fail when the
committed file is out of date.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from simulator.catalogue import Catalogue, Signal, load_catalogue
from simulator.paths import schema_path

DRAFT = "https://json-schema.org/draft/2020-12/schema"
VEHICLE_ID_PATTERN = r"^[A-Z][A-Z0-9]*-[0-9]{3,}$"
# ISO 8601 UTC with milliseconds, e.g. 2026-01-05T07:30:00.000Z (simulated time).
TIMESTAMP_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$"
WINDOW_S_MAX = 10

BOOLEAN_STATS = {
    "type": "object",
    "properties": {
        "last": {"type": "boolean", "description": "Value of the last sample in the window."},
        "count": {
            "type": "integer",
            "minimum": 0,
            "description": (
                "Number of rising edges (false -> true) in the window, as for ABS activations "
                "(docs/simulator-spec.md, section 5)."
            ),
        },
    },
    "required": ["last", "count"],
    "additionalProperties": False,
}


def value_definition(signal: Signal) -> dict[str, Any]:
    """A value within the measurement range, or exactly the catalogue's invalid value.

    Resolution is not validated by the schema (float `multipleOf` is unreliable); it is carried as
    the `x-resolution` / `x-offset` annotations and checked in code with a tolerance.
    """
    assert signal.range is not None
    low, high = signal.range
    definition: dict[str, Any] = {
        "anyOf": [
            {"type": "number", "minimum": low, "maximum": high},
            {"const": signal.invalid},
        ],
        "x-unit": signal.unit,
        "x-resolution": signal.resolution,
        "x-offset": signal.offset,
    }
    if signal.description:
        definition["description"] = signal.description
    return definition


def numeric_stats_definition(signal: Signal) -> dict[str, Any]:
    ref = {"$ref": f"#/$defs/value.{signal.name}"}
    return {
        "type": "object",
        "properties": {"mean": ref, "min": ref, "max": ref, "last": ref},
        "required": ["mean", "min", "max", "last"],
        "additionalProperties": False,
    }


def build_schema(catalogue: Catalogue) -> dict[str, Any]:
    defs: dict[str, Any] = {}
    signal_properties: dict[str, Any] = {}
    for signal in catalogue.signals:
        if signal.is_number:
            defs[f"value.{signal.name}"] = value_definition(signal)
            defs[f"stats.{signal.name}"] = numeric_stats_definition(signal)
            signal_properties[signal.name] = {"$ref": f"#/$defs/stats.{signal.name}"}
        else:
            signal_properties[signal.name] = {"$ref": "#/$defs/stats.boolean"}
    defs["stats.boolean"] = BOOLEAN_STATS
    defs["periodic_message"] = {
        "type": "object",
        "description": "Summary of one vehicle over one aggregation window (spec section 5).",
        "properties": {
            "message_type": {"const": "periodic"},
            "schema_version": {"const": catalogue.schema_version},
            "vehicle_id": {"type": "string", "pattern": VEHICLE_ID_PATTERN},
            "seq": {
                "type": "integer",
                "minimum": 0,
                "description": "Per-vehicle monotonic sequence counter.",
            },
            "window_start": {
                "type": "string",
                "pattern": TIMESTAMP_PATTERN,
                "description": "Start of the window in simulated time, ISO 8601 UTC.",
            },
            "window_s": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": WINDOW_S_MAX,
                "description": "Window length in seconds; below 10 for the last window of a trip.",
            },
            "signals": {
                "type": "object",
                "properties": signal_properties,
                "required": list(catalogue.names),
                "additionalProperties": False,
            },
        },
        "required": [
            "message_type",
            "schema_version",
            "vehicle_id",
            "seq",
            "window_start",
            "window_s",
            "signals",
        ],
        "additionalProperties": False,
    }
    return {
        "$schema": DRAFT,
        "title": "Nordwind van telemetry message",
        "description": (
            "Generated from simulator/catalogue.yaml by `make schema`; do not edit by hand."
        ),
        "oneOf": [{"$ref": "#/$defs/periodic_message"}],
        "$defs": defs,
    }


def render(schema: dict[str, Any]) -> str:
    """Stable text form of the schema, as committed."""
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=None, help="output file (default: committed path)"
    )
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args(argv)
    target: Path = args.out or schema_path()
    text = render(build_schema(load_catalogue()))
    if args.check:
        if not target.exists() or target.read_text() != text:
            print(f"{target} is out of date: run `make schema`", file=sys.stderr)
            return 1
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
