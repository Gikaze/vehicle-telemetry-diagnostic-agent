"""FR-17 (simulator side): a per-vehicle sequence counter, and gaps, duplicates and out-of-order
delivery injected from each vehicle's TRANSPORT stream; gaps are detectable from the counter.

Expected decisions are read here from the vehicle's own stream (rng.vehicle_generator), drawn in
one block of a different size from the transport's, never from the transport; hand-made draws pin
the hold and release rules.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml
from driven import short_plans

from simulator import stream
from simulator.aggregator import PeriodicRecord
from simulator.config_files import TRANSPORT_RATES, load_stream_config, stream_config
from simulator.fleet import Vehicle, load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.params import ConfigError, config_path
from simulator.rng import Stream, vehicle_generator
from simulator.transport import DRAW_BLOCK, Sequenced, Transport, TransportRates

SEED = 42
# Short trips (30 s and 20 s, 3 and 2 or 3 windows): enough messages per vehicle for the defects.
PLANS = short_plans([30.0, 20.0], [10.0])
RATES = TransportRates(gap=0.1, duplicate=0.1, reorder=0.1)
TRIP = 37  # records per trip of the hand-made stream
RECORDS = 3000  # hand-made records of one vehicle: several draw blocks


def record(vehicle_id: str, k: int, last_of_trip: bool = False) -> PeriodicRecord:
    return PeriodicRecord(vehicle_id, k, 1, last_of_trip, (), ())


def one_vehicle(count: int = RECORDS) -> list[PeriodicRecord]:
    """Records 0 .. count-1 of VAN-001, a trip ending every TRIP records and at the end."""
    return [record("VAN-001", k, (k + 1) % TRIP == 0 or k == count - 1) for k in range(count)]


def draws(vehicle_id: str, count: int) -> np.ndarray:
    """The vehicle's TRANSPORT draws for `count` messages, in one block."""
    return vehicle_generator(SEED, vehicle_id, Stream.TRANSPORT).random((count, 3))


def seqs(delivered: list[Sequenced]) -> list[int]:
    return [m.seq for m in delivered]


def test_seq_counts_every_record_of_each_vehicle_from_zero() -> None:
    """Two vehicles interleaved, trips ending: without defects each vehicle's messages carry
    0, 1, 2, ... in arrival order, across its trips, and each carries its own record."""
    records = [record(f"VAN-00{1 + k % 2}", k, k % 5 == 4) for k in range(40)]
    delivered = Transport(SEED, ("VAN-001", "VAN-002"), TransportRates()).send(records)
    assert [m.record for m in delivered] == records
    for vehicle in ("VAN-001", "VAN-002"):
        assert seqs([m for m in delivered if m.record.vehicle_id == vehicle]) == list(range(20))


def deliver(vehicles, days: int, rates: TransportRates) -> list[Sequenced]:
    return list(stream.messages(FleetSimulation(SEED, vehicles, PLANS), days, rates))


def by_vehicle(delivered: list[Sequenced]) -> dict[str, list[Sequenced]]:
    result: dict[str, list[Sequenced]] = {}
    for message in delivered:
        result.setdefault(message.record.vehicle_id, []).append(message)
    return result


@pytest.fixture(scope="module")
def runs() -> dict[str, list[Sequenced]]:
    """VAN-001 to VAN-003 on short plans: two days without defects, one and two days with
    defects."""
    fleet = load_fleet(size=3)
    return {
        "clean": deliver(fleet, 2, TransportRates()),
        "one_day": deliver(fleet, 1, RATES),
        "two_days": deliver(fleet, 2, RATES),
    }


def test_seq_runs_across_trips_and_days(runs: dict[str, list[Sequenced]]) -> None:
    """Without defects, each vehicle's seq is 0, 1, 2, ... over its four trips on two days, its
    records in the order the windows close."""
    clean = by_vehicle(runs["clean"])
    assert set(clean) == {"VAN-001", "VAN-002", "VAN-003"}
    for own in clean.values():
        assert seqs(own) == list(range(len(own)))
        assert sum(m.record.last_of_trip for m in own) == 4
        assert {m.record.start_step // 864_000 for m in own} == {0, 1}
        closing = [m.record.start_step + m.record.steps for m in own]
        assert closing == sorted(closing)


def test_missing_seqs_are_exactly_the_dropped_messages() -> None:
    """Gaps detectable from the counter: the seqs never delivered are those whose gap draw is
    below the rate. Each delivered seq comes once, or twice in a row when its duplicate draw is
    below the rate, with the same record."""
    delivered = Transport(SEED, ("VAN-001",), RATES).send(one_vehicle())
    u = draws("VAN-001", RECORDS)
    dropped = {k for k in range(RECORDS) if u[k, 0] < RATES.gap}
    assert dropped and RECORDS > 2 * DRAW_BLOCK
    assert set(range(RECORDS)) - set(seqs(delivered)) == dropped
    counts = Counter(seqs(delivered))
    duplicated = {k for k in counts if u[k, 1] < RATES.duplicate}
    assert duplicated
    assert {k for k, n in counts.items() if n == 2} == duplicated
    assert set(counts.values()) == {1, 2}
    for position, message in enumerate(delivered):
        if counts[message.seq] == 2 and seqs(delivered).index(message.seq) == position:
            assert delivered[position + 1] == message
        assert message.record == record("VAN-001", message.seq, message.record.last_of_trip)


def test_out_of_order_messages_come_right_after_the_next_delivered_one() -> None:
    """A message delivered after a higher seq had its reorder draw below the rate, does not end
    its trip, and comes right after the copies of the next delivered seq of its trip."""
    delivered = Transport(SEED, ("VAN-001",), RATES).send(one_vehicle())
    u = draws("VAN-001", RECORDS)
    order = list(dict.fromkeys(seqs(delivered)))  # each seq once, at its first delivery
    late = [k for i, k in enumerate(order) if any(j > k for j in order[:i])]
    assert late
    for k in late:
        assert u[k, 2] < RATES.reorder and (k + 1) % TRIP != 0
        following = min(j for j in order if j > k)
        assert following // TRIP == k // TRIP  # same trip
        assert order[order.index(k) - 1] == following


def test_a_vehicle_s_messages_do_not_depend_on_the_fleet(
    runs: dict[str, list[Sequenced]],
) -> None:
    """With defects on, over one day: VAN-001 to VAN-003 alone, reversed, and with VAN-000
    added in the first slot. VAN-000 starts driving first (short plans start VAN-n n * 37 s after
    06:00), so its messages are the first of the run: a stream shared with the others would hand
    them other draws. Each vehicle gets the same messages in the same order, and defects occur."""
    fleet = load_fleet(size=3)
    first = Vehicle(id="VAN-000", number=0, model=fleet[0].model, batch=fleet[0].batch)
    alone = by_vehicle(runs["one_day"])
    reversed_fleet = by_vehicle(deliver(tuple(reversed(fleet)), 1, RATES))
    extra_run = deliver((first, *fleet), 1, RATES)
    assert extra_run[0].record.vehicle_id == "VAN-000"
    with_extra = by_vehicle(extra_run)
    assert set(alone) == {"VAN-001", "VAN-002", "VAN-003"}
    for vehicle, delivered in alone.items():
        assert delivered == reversed_fleet[vehicle], vehicle
        assert delivered == with_extra[vehicle], vehicle
    # Some vehicle's seqs are not simply 0, 1, 2, ...: a gap, a duplicate or a reordering.
    assert any(seqs(m) != list(range(len(m))) for m in alone.values())


def test_one_day_is_a_prefix_of_two_days(runs: dict[str, list[Sequenced]]) -> None:
    """With defects on, a one-day run delivers the first messages of a two-day run."""
    one, two = runs["one_day"], runs["two_days"]
    assert one and two[: len(one)] == one and len(two) > len(one)


def test_changing_one_rate_does_not_move_another_rate_s_decisions() -> None:
    gaps_only = Transport(SEED, ("VAN-001",), TransportRates(gap=0.2)).send(one_vehicle())
    everything = Transport(
        SEED, ("VAN-001",), TransportRates(gap=0.2, duplicate=0.5, reorder=0.5)
    ).send(one_vehicle())
    assert set(seqs(gaps_only)) == set(seqs(everything))
    assert seqs(gaps_only) != seqs(everything)


class FixedDraws:
    """A generator stand-in returning hand-made rows, then rows without any defect."""

    def __init__(self, rows: list[tuple[float, float, float]]):
        self.rows = rows

    def random(self, shape: tuple[int, int]) -> np.ndarray:
        block = np.full(shape, 0.99)
        block[: len(self.rows)] = self.rows
        self.rows = []
        return block


NONE, DROP, DUP, HOLD = (0.9, 0.9, 0.9), (0.0, 0.9, 0.9), (0.9, 0.0, 0.9), (0.9, 0.9, 0.0)


def send_with(rows: list[tuple[float, float, float]], ends: list[bool]) -> list[Sequenced]:
    transport = Transport(SEED, ("VAN-001",), TransportRates(gap=0.5, duplicate=0.5, reorder=0.5))
    transport.vehicles["VAN-001"].generator = FixedDraws(rows)  # type: ignore[assignment]
    return transport.send([record("VAN-001", k, end) for k, end in enumerate(ends)])


@pytest.mark.parametrize(
    ("rows", "ends", "expected"),
    [
        # Held, then released after the next delivered message.
        ([HOLD, NONE, NONE], [False, False, True], [1, 0, 2]),
        # A dropped message does not release the held one; the next delivered one does.
        ([HOLD, DROP, NONE, NONE], [False, False, False, True], [2, 0, 3]),
        # Held, then every later message of the trip dropped: released at the trip's end.
        ([HOLD, DROP, DROP], [False, False, True], [0]),
        # A held message keeps its duplicate: both copies come after the next one.
        ([(0.9, 0.0, 0.0), NONE, NONE], [False, False, True], [1, 0, 0, 2]),
        # One held message at a time: a second reorder draw is delivered at once.
        ([HOLD, HOLD, NONE], [False, False, True], [1, 0, 2]),
        # The record that ends a trip is delivered at once, even with a reorder draw.
        ([NONE, HOLD], [False, True], [0, 1]),
        # Duplicates come twice in a row.
        ([DUP, NONE], [False, True], [0, 0, 1]),
    ],
)
def test_hold_and_release_rules(rows, ends, expected) -> None:
    assert seqs(send_with(rows, ends)) == expected


def test_rates_default_to_zero() -> None:
    config = stream_config()
    assert {name: config.value(name) for name in TRANSPORT_RATES} == dict.fromkeys(
        TRANSPORT_RATES, 0
    )
    assert TransportRates.from_config(config) == TransportRates()


@pytest.mark.parametrize("value", [-0.1, 1.5])
@pytest.mark.parametrize("name", TRANSPORT_RATES)
def test_a_rate_outside_0_1_is_rejected(tmp_path: Path, name: str, value: float) -> None:
    document = yaml.safe_load(config_path("stream.yaml").read_text())
    document["transport"][name]["value"] = value
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigError, match=name):
        load_stream_config(path)
    with pytest.raises(ValueError):
        TransportRates(**{name.removesuffix("_rate"): value})
