"""FR-02: a fixed seed gives identical data, and adding a vehicle does not change the others.

At this stage the per-vehicle data is the drawn parameter set; the same checks are extended to the
simulated signals once the vehicle model exists.
"""

from simulator.config_files import van_model
from simulator.fleet import draw_fleet_parameters, fleet_config, load_fleet


def draw(seed: int, vehicles):
    return draw_fleet_parameters(seed, vehicles, van_model(), fleet_config())


def test_same_seed_gives_identical_parameters() -> None:
    fleet = load_fleet(size=10)
    assert draw(42, fleet) == draw(42, fleet)


def test_adding_a_vehicle_does_not_change_the_others() -> None:
    ten = draw(42, load_fleet(size=10))
    eleven = draw(42, load_fleet(size=11))
    assert eleven[:10] == ten


def test_a_vehicle_does_not_depend_on_its_position_in_the_fleet() -> None:
    fleet = load_fleet(size=10)
    reordered = draw(42, tuple(reversed(fleet)))
    assert tuple(reversed(reordered)) == draw(42, fleet)


def test_another_seed_gives_other_parameters() -> None:
    fleet = load_fleet(size=10)
    a = draw(42, fleet)
    b = draw(43, fleet)
    assert all(x != y for x, y in zip(a, b, strict=True))
