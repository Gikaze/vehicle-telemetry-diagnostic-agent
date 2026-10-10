"""Configuration files: declared structure, and parameters with a value, a unit and a rationale.

Each file has a `Layout`: the structural paths it may contain (with their types) and the exact set
of parameter names the code reads. Every other node must be a parameter or a group of parameters:
- a parameter is a mapping with exactly the keys `value`, `unit` and `rationale`, where the unit
  and the rationale are non-empty text and the value is a number or a list of numbers;
- a group is a mapping whose children are all parameters or groups.
Anything else is rejected, so a bare `wheelbase: 3.6` or a partial `wheelbase: {unit: m}` never
passes as structure, and a parameter without a rationale never loads.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from simulator.paths import simulator_home

PARAMETER_KEYS = frozenset({"value", "unit", "rationale"})

Path_ = tuple[str, ...]


class ConfigError(ValueError):
    """A configuration file breaks its layout or the parameter rules."""


@dataclass(frozen=True)
class Parameter:
    value: Any
    unit: str
    rationale: str


@dataclass(frozen=True)
class Layout:
    """What one configuration file may contain."""

    structural: Mapping[Path_, type | tuple[type, ...]]
    parameters: frozenset[str]


@dataclass(frozen=True)
class Config:
    structure: dict[Path_, Any]
    parameters: dict[str, Parameter]

    def value(self, name: str) -> Any:
        return self.parameters[name].value


def config_path(name: str) -> Path:
    return simulator_home() / "config" / name


def load_config(path: Path, layout: Layout) -> Config:
    """Load a configuration file and check it against its layout."""
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ConfigError(f"{path.name}: the file must be a mapping")
    structure: dict[Path_, Any] = {}
    found: dict[str, Parameter] = {}
    _walk(document, (), layout, structure, found, path.name)
    missing_structure = sorted(".".join(p) for p in set(layout.structural) - set(structure))
    if missing_structure:
        raise ConfigError(f"{path.name}: missing structural fields {missing_structure}")
    missing = sorted(layout.parameters - set(found))
    unexpected = sorted(set(found) - layout.parameters)
    if missing or unexpected:
        raise ConfigError(
            f"{path.name}: parameters differ from what the code uses "
            f"(missing {missing}, unexpected {unexpected})"
        )
    return Config(structure=structure, parameters=found)


def _walk(
    node: Any,
    path: Path_,
    layout: Layout,
    structure: dict[Path_, Any],
    found: dict[str, Parameter],
    file: str,
) -> None:
    where = f"{file}:{'.'.join(path) or '<root>'}"
    if path in layout.structural:
        expected = layout.structural[path]
        if isinstance(node, bool) or not isinstance(node, expected):
            raise ConfigError(f"{where}: expected {_type_names(expected)}, got {node!r}")
        structure[path] = node
        return
    if isinstance(node, Mapping) and set(node) & PARAMETER_KEYS:
        name = path[-1]
        if name in found:
            raise ConfigError(f"{where}: parameter {name} is defined twice")
        found[name] = _parameter(node, where)
        return
    if isinstance(node, Mapping) and node:
        for key, child in node.items():
            _walk(child, (*path, str(key)), layout, structure, found, file)
        return
    raise ConfigError(f"{where}: expected a parameter {{value, unit, rationale}}, got {node!r}")


def _parameter(node: Mapping[str, Any], where: str) -> Parameter:
    keys = set(node)
    if keys != PARAMETER_KEYS:
        raise ConfigError(
            f"{where}: a parameter needs exactly value, unit and rationale "
            f"(missing {sorted(PARAMETER_KEYS - keys)}, unexpected {sorted(keys - PARAMETER_KEYS)})"
        )
    for field in ("unit", "rationale"):
        if not isinstance(node[field], str) or not node[field].strip():
            raise ConfigError(f"{where}: the {field} must be non-empty text")
    value = node["value"]
    numbers = value if isinstance(value, list) else [value]
    if not numbers or not all(_is_number(item) for item in numbers):
        raise ConfigError(f"{where}: the value must be a number or a list of numbers")
    return Parameter(value=value, unit=node["unit"], rationale=node["rationale"])


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _type_names(expected: type | tuple[type, ...]) -> str:
    types = expected if isinstance(expected, tuple) else (expected,)
    return " or ".join(t.__name__ for t in types)
