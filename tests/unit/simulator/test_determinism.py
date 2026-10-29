"""FR-02: a fixed seed gives identical data, and adding a vehicle does not change the others.

Checked on the drawn parameter sets and on every emitted sample of the fleet loop, true and
measured values, with the production day plans.
"""

import dataclasses
import hashlib

import numpy as np

from simulator.config_files import van_model
from simulator.fleet import Vehicle, draw_fleet_parameters, fleet_config, load_fleet
from simulator.fleet_state import FleetSimulation

# Seed 16: VAN-011 is the first van to start (06:31), and four of the ten others start within
# its first 10 minutes of driving.
SEED = 16
# Seed 133: vans also driving in that stretch, to compare against.
OTHER_SEED = 133
DRIVING = 6000  # steps: 10 min of VAN-011 driving


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


def _slot_bytes(value, i: int, n: int, out: list[bytes]) -> None:
    """Bytes of slot i of a sample value: per-slot arrays at index i, fleet-wide values whole."""
    if dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            _slot_bytes(getattr(value, f.name), i, n, out)
        return
    array = np.asarray(value)
    if array.ndim and array.shape[0] == n:
        array = array[i]
    out.append(str(array.dtype).encode() + np.ascontiguousarray(array).tobytes())


def emitted(seed: int, vehicles: tuple[Vehicle, ...], until_step: int | None = None):
    """Per vehicle ID, the (step, digest of every true and measured value) of each sample where
    it drives; stops after `until_step`, or once VAN-011 has driven DRIVING steps. Returns the
    records and the last step run."""
    sim = FleetSimulation(seed, vehicles)
    records: dict[str, list[tuple[int, str]]] = {vehicle.id: [] for vehicle in vehicles}
    n, step = len(vehicles), -1
    for sample in sim.run(1):
        step = sample.step
        if until_step is not None and step > until_step:
            break
        for i in np.flatnonzero(sample.active):
            out: list[bytes] = []
            _slot_bytes(sample.true, int(i), n, out)
            for name, value in sample.measured.items():
                out.append(name.encode() + value[i].tobytes())
            records[sample.vehicle_ids[i]].append((step, hashlib.sha256(b"".join(out)).hexdigest()))
        if until_step is None and len(records.get("VAN-011", [])) >= DRIVING:
            break
    return records, step


def test_emitted_samples_do_not_change_when_a_vehicle_is_added_or_the_order_changes() -> None:
    """Ten vans and the same ten plus VAN-011, over VAN-011's first 10 minutes: each of the ten
    has the same samples, step by step and bit for bit, in both runs; so has the reversed fleet
    (a second run of the same seed). Another seed gives other samples."""
    eleven, last = emitted(SEED, load_fleet(size=11))
    assert len(eleven["VAN-011"]) == DRIVING
    driving = [vehicle for vehicle, rows in eleven.items() if rows and vehicle != "VAN-011"]
    assert len(driving) >= 3
    ten, _ = emitted(SEED, load_fleet(size=10), until_step=last)
    reversed_ten, _ = emitted(SEED, tuple(reversed(load_fleet(size=10))), until_step=last)
    for vehicle, rows in ten.items():
        assert rows == eleven[vehicle], vehicle
        assert rows == reversed_ten[vehicle], vehicle
    other, _ = emitted(OTHER_SEED, load_fleet(size=10), until_step=last)
    common = 0
    for vehicle, rows in ten.items():
        theirs = dict(other[vehicle])
        for step, digest in rows:
            if step in theirs:
                common += 1
                assert digest != theirs[step], (vehicle, step)
    assert common > 0
