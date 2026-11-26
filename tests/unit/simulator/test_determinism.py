"""FR-02: a fixed seed gives identical data, and adding a vehicle does not change the others.

Checked on the drawn parameter sets and on every emitted sample of the fleet loop, true and
measured values: in every run with injected short day plans (each vehicle's road, parameters and
sensor noise still come from its own streams), and with the production day plans in the thorough
Hypothesis profile only (HYPOTHESIS_PROFILE=thorough), where the run takes about half a minute.
"""

import dataclasses
import hashlib
import os

import numpy as np
import pytest
from driven import short_plans

from simulator.config_files import van_model
from simulator.fleet import Vehicle, draw_fleet_parameters, fleet_config, load_fleet
from simulator.fleet_state import FleetSimulation, PlanDay, plan_from_duty

# Seed 16: VAN-011 is the first van to start (06:31), and four of the ten others start within
# its first 10 minutes of driving.
SEED = 16
# Seed 133: vans also driving in that stretch, to compare against.
OTHER_SEED = 133
DRIVING = 6000  # steps: 10 min of VAN-011 driving
# Short plans: two trips with a break; VAN-n starts n * 37 s after 06:00 (driven.short_plans), so
# VAN-004 (148 s) starts while VAN-002 and VAN-003 drive their first trip and VAN-001 its second.
SHORT = short_plans([90.0, 60.0], [20.0])
# Steps of the other seed's run, from the first sample (200 s): keeps that run short. With the
# short plans every vehicle drives in it (VAN-003 starts at 111 s).
OTHER_STEPS = 2000


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


def emitted(
    seed: int,
    vehicles: tuple[Vehicle, ...],
    plan_day: PlanDay = plan_from_duty,
    until_step: int | None = None,
    added: str | None = None,
):
    """Per vehicle ID, the (step, digest of every true and measured value) of each sample where
    it drives, over day 0; stops after `until_step`, or once vehicle `added` has driven DRIVING
    steps. Returns the records and the last step run."""
    sim = FleetSimulation(seed, vehicles, plan_day)
    records: dict[str, list[tuple[int, str]]] = {vehicle.id: [] for vehicle in vehicles}
    n, step = len(vehicles), -1
    for sample in sim.run(1):
        if until_step is not None and sample.step > until_step:
            break
        step = sample.step
        for i in np.flatnonzero(sample.active):
            out: list[bytes] = []
            _slot_bytes(sample.true, int(i), n, out)
            for name, value in sample.measured.items():
                out.append(name.encode() + value[i].tobytes())
            records[sample.vehicle_ids[i]].append((step, hashlib.sha256(b"".join(out)).hexdigest()))
        if added is not None and len(records[added]) >= DRIVING:
            break
    return records, step


def check_independence(
    base: tuple[Vehicle, ...], extra: Vehicle, plan_day: PlanDay, added_first: bool
) -> None:
    """`base` with `extra` added (first, so every other vehicle moves to another slot, or last),
    against `base` alone and `base` reversed (a second and a third run of the same seed): each
    vehicle of `base` has the same samples, step by step and bit for bit, in the three runs, and
    at least three drive while `extra` drives. Another seed, run for at most OTHER_STEPS steps
    from the first sample, gives other samples at every step it shares with a vehicle (some
    steps are shared)."""
    larger = (extra, *base) if added_first else (*base, extra)
    production = plan_day is plan_from_duty
    more, last = emitted(SEED, larger, plan_day, added=extra.id if production else None)
    assert more[extra.id]
    if production:
        assert len(more[extra.id]) == DRIVING
    extra_steps = {step for step, _ in more[extra.id]}
    overlapping = [
        vehicle for vehicle in base if extra_steps & {step for step, _ in more[vehicle.id]}
    ]
    assert len(overlapping) >= 3
    alone, _ = emitted(SEED, base, plan_day, until_step=last)
    reversed_alone, _ = emitted(SEED, tuple(reversed(base)), plan_day, until_step=last)
    for vehicle, rows in alone.items():
        assert rows == more[vehicle], vehicle
        assert rows == reversed_alone[vehicle], vehicle
    first = min(rows[0][0] for rows in alone.values() if rows)
    other, _ = emitted(OTHER_SEED, base, plan_day, until_step=min(last, first + OTHER_STEPS))
    common = 0
    for vehicle, rows in alone.items():
        theirs = dict(other[vehicle])
        for step, digest in rows:
            if step in theirs:
                common += 1
                assert digest != theirs[step], (vehicle, step)
    assert common > 0


def test_emitted_samples_do_not_change_when_a_vehicle_is_added_or_the_order_changes() -> None:
    """Short plans, over the whole run: VAN-001 to VAN-003, alone, reversed and with VAN-004
    added in the first slot."""
    fleet = load_fleet(size=4)
    check_independence(fleet[:3], fleet[3], SHORT, added_first=True)


@pytest.mark.skipif(
    os.environ.get("HYPOTHESIS_PROFILE") != "thorough",
    reason="production day plans: thorough profile only (HYPOTHESIS_PROFILE=thorough)",
)
def test_production_plans_do_not_change_when_a_vehicle_is_added_or_the_order_changes() -> None:
    """Production day plans, over VAN-011's first 10 minutes of driving: ten vans, reversed and
    with VAN-011 added in the last slot."""
    fleet = load_fleet(size=11)
    check_independence(fleet[:10], fleet[10], plan_from_duty, added_first=False)
