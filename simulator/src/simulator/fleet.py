"""Fleet registry and per-vehicle parameters (spec section 4: no two vans are identical)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulator.config_files import van_model
from simulator.params import Config, ConfigError, Layout, config_path, load_config
from simulator.rng import Stream, vehicle_generator

WHEELS = ("fl", "fr", "rl", "rr")

FLEET_LAYOUT = Layout(
    structural={("id_prefix",): str, ("id_digits",): int, ("model",): str, ("batches",): list},
    parameters=frozenset({"size"}),
)


@dataclass(frozen=True)
class Batch:
    id: str
    first: int
    last: int


@dataclass(frozen=True)
class Vehicle:
    id: str
    number: int
    model: str
    batch: str


@dataclass(frozen=True)
class VehicleParameters:
    """Drawn from the vehicle's own PARAMETERS stream; per-wheel tuples follow WHEELS order."""

    vehicle_id: str
    payload: float  # kg
    tyre_rolling_radius: float  # m, same for the four wheels
    disc_heat_capacity: tuple[float, float, float, float]  # J/K
    tyre_pressure_nominal: tuple[float, float, float, float]  # bar, gauge, at the reference temp


def load_fleet(size: int | None = None, path: Path | None = None) -> tuple[Vehicle, ...]:
    """Vehicles numbered 1..size in order; `size` overrides the configured fleet size."""
    config = load_config(path or config_path("fleet.yaml"), FLEET_LAYOUT)
    batches = _batches(config)
    count = config.value("size") if size is None else size
    _check_size(count, config, batches)
    prefix, digits = config.structure[("id_prefix",)], config.structure[("id_digits",)]
    model = config.structure[("model",)]
    return tuple(
        Vehicle(
            id=f"{prefix}-{number:0{digits}d}",
            number=number,
            model=model,
            batch=_batch_of(number, batches),
        )
        for number in range(1, count + 1)
    )


def _batches(config: Config) -> tuple[Batch, ...]:
    entries = config.structure[("batches",)]
    batches = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "first", "last"}:
            raise ConfigError(f"fleet.yaml: a batch needs exactly id, first, last: {entry!r}")
        first, last = entry["first"], entry["last"]
        if not (isinstance(first, int) and isinstance(last, int) and 1 <= first <= last):
            raise ConfigError(f"fleet.yaml: batch {entry['id']} has an invalid range")
        batches.append(Batch(id=str(entry["id"]), first=first, last=last))
    if not batches:
        raise ConfigError("fleet.yaml: at least one production batch is needed")
    expected_first = 1
    for batch in batches:
        if batch.first < expected_first:
            raise ConfigError(f"fleet.yaml: batch {batch.id} overlaps the previous batch")
        if batch.first > expected_first:
            raise ConfigError(
                f"fleet.yaml: gap before batch {batch.id} (vehicles {expected_first}"
                f"..{batch.first - 1} are in no batch)"
            )
        expected_first = batch.last + 1
    return tuple(batches)


def _check_size(count: int, config: Config, batches: tuple[Batch, ...]) -> None:
    if count < 1:
        raise ConfigError("fleet.yaml: the fleet needs at least one vehicle")
    covered = batches[-1].last
    if count > covered:
        raise ConfigError(
            f"fleet.yaml: fleet size {count} exceeds the batch ranges (vehicles 1..{covered})"
        )
    digits = config.structure[("id_digits",)]
    if count > 10**digits - 1:
        raise ConfigError(f"fleet.yaml: fleet size {count} does not fit {digits}-digit vehicle IDs")


def _batch_of(number: int, batches: tuple[Batch, ...]) -> str:
    for batch in batches:
        if batch.first <= number <= batch.last:
            return batch.id
    raise ConfigError(f"vehicle number {number} is in no production batch")


def draw_vehicle_parameters(root_seed: int, vehicle: Vehicle, model: Config) -> VehicleParameters:
    """Variation around the model values, from this vehicle's PARAMETERS stream only.

    The draw order is part of the reproducibility contract: payload, tyre radius, vehicle disc
    factor, then per wheel (WHEELS order) the disc factor and the cold pressure offset.
    """
    rng = vehicle_generator(root_seed, vehicle.id, Stream.PARAMETERS)

    def spread(name: str, size: int | None = None) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size) * model.value(name)

    payload = float(rng.uniform(0.0, model.value("payload_max")))
    radius = model.value("tyre_rolling_radius") * (1.0 + float(spread("tyre_radius_spread")))
    vehicle_disc = 1.0 + float(spread("disc_heat_capacity_vehicle_spread"))
    wheel_disc = 1.0 + spread("disc_heat_capacity_wheel_spread", len(WHEELS))
    pressure_offset = spread("tyre_pressure_wheel_spread", len(WHEELS))
    nominal_capacity = (
        model.value("disc_heat_capacity_front"),
        model.value("disc_heat_capacity_front"),
        model.value("disc_heat_capacity_rear"),
        model.value("disc_heat_capacity_rear"),
    )
    capacity = tuple(
        float(c * vehicle_disc * w) for c, w in zip(nominal_capacity, wheel_disc, strict=True)
    )
    pressure = tuple(float(model.value("tyre_pressure_nominal") + o) for o in pressure_offset)
    return VehicleParameters(
        vehicle_id=vehicle.id,
        payload=payload,
        tyre_rolling_radius=radius,
        disc_heat_capacity=capacity,  # type: ignore[arg-type]
        tyre_pressure_nominal=pressure,  # type: ignore[arg-type]
    )


def draw_fleet_parameters(
    root_seed: int, fleet: tuple[Vehicle, ...], model: Config | None = None
) -> tuple[VehicleParameters, ...]:
    model = model or van_model()
    return tuple(draw_vehicle_parameters(root_seed, vehicle, model) for vehicle in fleet)
