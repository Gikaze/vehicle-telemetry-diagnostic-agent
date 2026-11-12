"""The stream drivers: a run cut at a fleet step (until_step) is the full run's prefix, byte for
byte, and holds exactly the records closed before the cut; `run` publishes what the drivers
deliver, stops after max_messages, and paces each step's messages in real time.

The expected cut is computed from the full run's records by the step each record closes at
(the last sample of a periodic window, the sample of an event), not by the cut logic.
"""

import datetime
import json

import pytest
from driven import short_plans

from simulator import stream
from simulator.aggregator import EventRecord, PeriodicRecord, Record
from simulator.catalogue import catalogue
from simulator.clock import StepClock
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.serialise import Payload, Serialiser
from simulator.transport import TransportRates

SEED = 42
PLANS = short_plans([30.0, 20.0], [10.0])
RATES = TransportRates(gap=0.1, duplicate=0.1, reorder=0.1)
NONE = TransportRates()


def sim() -> FleetSimulation:
    return FleetSimulation(SEED, load_fleet(size=3), PLANS)


def serialiser() -> Serialiser:
    return Serialiser(catalogue(), StepClock(datetime.date(2026, 1, 5)))


def closes_at(record: Record) -> int:
    if isinstance(record, PeriodicRecord):
        return record.start_step + record.steps - 1
    assert isinstance(record, EventRecord)
    return record.step


@pytest.fixture(scope="module")
def full_records() -> list[Record]:
    return list(stream.records(sim(), 1))


@pytest.fixture(scope="module")
def full_payloads() -> list[Payload]:
    return list(stream.payloads(sim(), 1, RATES, serialiser()))


def cuts(records: list[Record]) -> list[int]:
    """Two steps at which a record closes: the first record's, and one in the middle."""
    ends = [closes_at(r) for r in records]
    return [ends[0], ends[len(ends) // 2]]


def test_a_cut_run_holds_exactly_the_records_closed_before_the_cut(full_records) -> None:
    """Each cut is a step where a record closes: that record is left out."""
    assert len(full_records) > 10
    for cut in cuts(full_records):
        expected = [r for r in full_records if closes_at(r) < cut]
        got = list(stream.records(sim(), 1, until_step=cut))
        assert got == expected, cut
        assert len(got) < len(full_records)  # the first cut leaves nothing
        starts = [r.start_step if isinstance(r, PeriodicRecord) else r.step for r in got]
        assert all(start < cut for start in starts)


def test_a_cut_run_is_a_byte_prefix_of_the_full_run(full_records, full_payloads) -> None:
    """With gaps, duplicates and reordering on."""
    middle = cuts(full_records)[1]
    cut = list(stream.payloads(sim(), 1, RATES, serialiser(), until_step=middle))
    assert 0 < len(cut) < len(full_payloads)
    assert cut == full_payloads[: len(cut)]
    joined = b"".join(p.data + b"\n" for p in cut)
    assert b"".join(p.data + b"\n" for p in full_payloads).startswith(joined)
    messages = list(stream.messages(sim(), 1, RATES, until_step=middle))
    assert [serialiser().encode(m) for m in messages] == cut


def test_a_cut_at_the_first_step_gives_nothing(full_records) -> None:
    first = min(r.start_step for r in full_records if isinstance(r, PeriodicRecord))
    assert list(stream.records(sim(), 1, until_step=first)) == []
    assert list(stream.payloads(sim(), 1, RATES, serialiser(), until_step=first)) == []


class Collect:
    def __init__(self) -> None:
        self.batches: list[list[Payload]] = []
        self.closed = False

    def publish(self, batch) -> None:
        self.batches.append(list(batch))

    def close(self) -> None:
        self.closed = True

    def payloads(self) -> list[Payload]:
        return [p for batch in self.batches for p in batch]


def test_run_publishes_the_delivered_payloads(full_records, full_payloads) -> None:
    out = Collect()
    assert stream.run(sim(), 1, RATES, serialiser(), out) == len(full_payloads)
    assert out.payloads() == full_payloads
    assert all(out.batches)
    middle = cuts(full_records)[1]
    out = Collect()
    sent = stream.run(sim(), 1, RATES, serialiser(), out, until_step=middle)
    cut = list(stream.payloads(sim(), 1, RATES, serialiser(), until_step=middle))
    assert sent == len(cut)
    assert out.payloads() == cut


def test_run_stops_after_max_messages(full_payloads) -> None:
    """Small limits, one short of the run, and one inside a step that delivers several messages
    (a duplicate, or a held message released with the next one): the step's batch is cut."""
    sizes = [len(d) for _, d in stream.stepped_messages(sim(), 1, RATES)]
    split = next(sum(sizes[:i]) + 1 for i, size in enumerate(sizes) if size > 1)
    for limit in [1, 2, 3, 7, split, len(full_payloads) - 1]:
        out = Collect()
        assert stream.run(sim(), 1, RATES, serialiser(), out, max_messages=limit) == limit
        assert out.payloads() == full_payloads[:limit]
        assert all(out.batches)  # nothing published after the limit, not even an empty batch


class FakeClock:
    """Wall time that moves only when slept."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds


def test_the_pacer_by_hand() -> None:
    clock = FakeClock()
    pacer = stream.Pacer(clock.sleep, clock.monotonic)
    pacer.wait_until(100)  # the origin: no wait
    assert clock.sleeps == []
    pacer.wait_until(130)  # 30 steps of 0.1 s
    assert clock.sleeps == [pytest.approx(3.0)]
    clock.now += 5.0  # work took longer than the next 20 steps
    pacer.wait_until(150)
    assert clock.sleeps == [pytest.approx(3.0)]
    pacer.wait_until(864_100)  # one day after the origin
    assert clock.now == pytest.approx(1000.0 + 86_400.0)


def window_end_ms(payload: Payload) -> int:
    """Simulated time of the last sample of a periodic message's window, ms."""
    doc = json.loads(payload.data)
    start = datetime.datetime.fromisoformat(doc["window_start"])
    return round(start.timestamp() * 1000) + round(doc["window_s"] * 1000) - 100


def test_run_paced_releases_each_message_at_its_simulated_time() -> None:
    """Without defects each message is published at the step its window closes: its wall time
    after the first publication equals its simulated time after the first window's end."""
    clock = FakeClock()
    published: list[tuple[float, Payload]] = []

    class Timed(Collect):
        def publish(self, batch) -> None:
            published.extend((clock.now, p) for p in batch)

    pacer = stream.Pacer(clock.sleep, clock.monotonic)
    stream.run(sim(), 1, NONE, serialiser(), Timed(), pacer)
    assert len(published) > 10
    assert len({p.partition_key for _, p in published}) == 3
    t0, first = published[0]
    for wall, payload in published:
        expected = (window_end_ms(payload) - window_end_ms(first)) / 1000
        assert wall - t0 == pytest.approx(expected, abs=1e-6)
    assert clock.sleeps
