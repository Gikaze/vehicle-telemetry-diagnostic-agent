"""FR-02: a fixed seed gives identical data, and adding a vehicle does not change the others.

At this stage the per-vehicle data is the drawn parameter set; the same checks are extended to the
simulated signals once the vehicle model exists.
"""

from simulator.config_files import van_model
from simulator.fleet import draw_fleet_parameters, load_fleet


def test_same_seed_gives_identical_parameters() -> None:
    fleet, model = load_fleet(size=10), van_model()
    assert draw_fleet_parameters(42, fleet, model) == draw_fleet_parameters(42, fleet, model)


def test_adding_a_vehicle_does_not_change_the_others() -> None:
    model = van_model()
    ten = draw_fleet_parameters(42, load_fleet(size=10), model)
    eleven = draw_fleet_parameters(42, load_fleet(size=11), model)
    assert eleven[:10] == ten


def test_a_vehicle_does_not_depend_on_its_position_in_the_fleet() -> None:
    fleet, model = load_fleet(size=10), van_model()
    reordered = draw_fleet_parameters(42, tuple(reversed(fleet)), model)
    assert tuple(reversed(reordered)) == draw_fleet_parameters(42, fleet, model)


def test_another_seed_gives_other_parameters() -> None:
    fleet, model = load_fleet(size=10), van_model()
    a = draw_fleet_parameters(42, fleet, model)
    b = draw_fleet_parameters(43, fleet, model)
    assert all(x != y for x, y in zip(a, b, strict=True))
