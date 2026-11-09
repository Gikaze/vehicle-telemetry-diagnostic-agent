"""Fleet registry: IDs, production batches and per-vehicle variation (spec section 4)."""

from pathlib import Path

import pytest
import yaml

from simulator.config_files import van_model
from simulator.fleet import (
    WHEELS,
    draw_fleet_parameters,
    fleet_config,
    load_fleet,
    load_fleet_config,
)
from simulator.params import ConfigError, config_path


def fleet_copy(tmp_path: Path, edit) -> Path:
    document = yaml.safe_load(config_path("fleet.yaml").read_text())
    edit(document)
    path = tmp_path / "fleet.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_default_fleet() -> None:
    fleet = load_fleet()
    assert len(fleet) == 100
    assert (fleet[0].id, fleet[0].batch) == ("VAN-001", "2025-T1")
    assert (fleet[41].id, fleet[41].batch) == ("VAN-042", "2025-T2")
    assert (fleet[99].id, fleet[99].batch) == ("VAN-100", "2025-T4")


def test_size_override_keeps_ids_and_batches() -> None:
    small = load_fleet(size=10)
    assert small == load_fleet()[:10]


def test_size_not_covered_by_batches_is_rejected() -> None:
    with pytest.raises(ConfigError, match="exceeds the batch ranges"):
        load_fleet(size=101)


def test_size_that_does_not_fit_the_id_digits_is_rejected(tmp_path: Path) -> None:
    def edit(document):
        document["id_digits"] = 2

    with pytest.raises(ConfigError, match="does not fit 2-digit"):
        load_fleet(path=fleet_copy(tmp_path, edit))


@pytest.mark.parametrize(
    ("batches", "message"),
    [
        ([{"id": "A", "first": 1, "last": 60}, {"id": "B", "first": 50, "last": 100}], "overlaps"),
        ([{"id": "A", "first": 1, "last": 40}, {"id": "B", "first": 51, "last": 100}], "gap"),
        ([{"id": "A", "first": 2, "last": 100}], "gap"),
        ([{"id": "A", "first": 10, "last": 5}], "invalid range"),
        ([{"id": "A", "first": 1}], "needs exactly id, first, last"),
        ([], "at least one production batch"),
    ],
)
def test_bad_batch_ranges_are_rejected(tmp_path: Path, batches: list, message: str) -> None:
    def edit(document):
        document["batches"] = batches

    with pytest.raises(ConfigError, match=message):
        load_fleet(path=fleet_copy(tmp_path, edit))


def test_variation_stays_within_the_configured_spreads() -> None:
    model = van_model()
    values = draw_fleet_parameters(42, load_fleet(), model, fleet_config())
    radius = model.value("tyre_rolling_radius")
    vehicle_spread = model.value("disc_heat_capacity_vehicle_spread")
    wheel_spread = model.value("disc_heat_capacity_wheel_spread")
    front, rear = model.value("disc_heat_capacity_front"), model.value("disc_heat_capacity_rear")
    nominal = (front, front, rear, rear)
    for value in values:
        assert 0 <= value.payload <= model.value("payload_max")
        assert abs(value.tyre_rolling_radius / radius - 1) <= model.value("tyre_radius_spread")
        assert len(value.disc_heat_capacity) == len(WHEELS)
        for capacity, base in zip(value.disc_heat_capacity, nominal, strict=True):
            factor = capacity / base
            assert (1 - vehicle_spread) * (1 - wheel_spread) <= factor
            assert factor <= (1 + vehicle_spread) * (1 + wheel_spread)
        for pressure in value.tyre_pressure_nominal:
            offset = pressure - model.value("tyre_pressure_nominal")
            assert abs(offset) <= model.value("tyre_pressure_wheel_spread")


def test_initial_odometer_stays_within_its_bounds() -> None:
    fleet = fleet_config()
    values = draw_fleet_parameters(42, load_fleet(), van_model(), fleet)
    low, high = fleet.si("odometer_initial_min"), fleet.si("odometer_initial_max")
    assert all(low <= value.odometer_initial <= high for value in values)
    assert len({value.odometer_initial for value in values}) == len(values)


def test_a_custom_fleet_file_gives_its_own_odometer_bounds(tmp_path: Path) -> None:
    def edit(document):
        document["odometer"]["odometer_initial_min"]["value"] = 1
        document["odometer"]["odometer_initial_max"]["value"] = 2

    path = fleet_copy(tmp_path, edit)
    custom = load_fleet_config(path)
    values = draw_fleet_parameters(42, load_fleet(path=path), van_model(), custom)
    assert all(1000 <= value.odometer_initial <= 2000 for value in values)


def test_earlier_draws_are_unchanged_by_the_odometer_draw() -> None:
    """Values drawn before the odometer was added (commit 8739021): it is drawn last."""
    value = draw_fleet_parameters(42, load_fleet(size=1), van_model(), fleet_config())[0]
    assert value.payload == 197.4870954802308
    assert value.tyre_rolling_radius == 0.3476860237095967
    assert value.tyre_pressure_nominal[-1] == 3.8491164472282535


def test_inverted_odometer_bounds_are_rejected(tmp_path: Path) -> None:
    def edit(document):
        document["odometer"]["odometer_initial_min"]["value"] = 200000

    with pytest.raises(ConfigError, match="odometer_initial_min < odometer_initial_max"):
        load_fleet_config(fleet_copy(tmp_path, edit))


def test_left_and_right_discs_differ() -> None:
    """Per-wheel variation exists, so the left/right spread invariant is not trivially zero."""
    value = draw_fleet_parameters(42, load_fleet(size=1), van_model(), fleet_config())[0]
    fl, fr, rl, rr = value.disc_heat_capacity
    assert fl != fr and rl != rr
