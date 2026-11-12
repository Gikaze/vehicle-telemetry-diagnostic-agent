"""Telemetry stream driver: the fleet loop's samples through the aggregator and the transport
(spec section 2).

    FleetSample -> Aggregator -> Record (periodic or event) -> Transport -> Sequenced
        -> Serialiser -> Payload

Time is the integer fleet step throughout; the serialiser's clock turns steps into simulated
time stamps. `run` sends the payloads of each step to a publisher, paced or not.

`until_step` cuts a run: the samples at that fleet step and later are never fed, so every record
comes from earlier samples (a periodic record starts and closes before the cut, an event is seen
before it). The windows still open at the cut, and a message the transport holds back, are not
sent. A cut run is therefore a prefix of the full run, byte for byte.
"""

import time
from collections.abc import Callable, Iterator

from simulator.aggregator import Aggregator, Record
from simulator.clock import MS_PER_STEP
from simulator.fleet_state import FleetSample, FleetSimulation
from simulator.publishers import Publisher
from simulator.serialise import Payload, Serialiser
from simulator.transport import Sequenced, Transport, TransportRates


def samples(
    sim: FleetSimulation, days: int, until_step: int | None = None
) -> Iterator[FleetSample]:
    """The fleet samples of `days` days of `sim`, stopping before the first at `until_step`."""
    for sample in sim.run(days):
        if until_step is not None and sample.step >= until_step:
            return
        yield sample


def records(sim: FleetSimulation, days: int, until_step: int | None = None) -> Iterator[Record]:
    """The periodic and event records of `days` days of `sim`, by step, then slot; for each slot,
    its events before the periodic record of a window closing at that step."""
    aggregator = Aggregator(sim.catalogue, sim.vehicle_ids)
    for sample in samples(sim, days, until_step):
        yield from aggregator.feed(sample)


def stepped_messages(
    sim: FleetSimulation, days: int, rates: TransportRates, until_step: int | None = None
) -> Iterator[tuple[int, list[Sequenced]]]:
    """Per fleet step that delivers any, the step and its delivered messages, in delivery
    order."""
    aggregator = Aggregator(sim.catalogue, sim.vehicle_ids)
    transport = Transport(sim.root_seed, sim.vehicle_ids, rates)
    for sample in samples(sim, days, until_step):
        delivered = transport.send(aggregator.feed(sample))
        if delivered:
            yield sample.step, delivered


def messages(
    sim: FleetSimulation, days: int, rates: TransportRates, until_step: int | None = None
) -> Iterator[Sequenced]:
    """The messages delivered for the records of `days` days of `sim`, in delivery order."""
    for _, delivered in stepped_messages(sim, days, rates, until_step):
        yield from delivered


def payloads(
    sim: FleetSimulation,
    days: int,
    rates: TransportRates,
    serialiser: Serialiser,
    until_step: int | None = None,
) -> Iterator[Payload]:
    """The delivered messages of `days` days of `sim` as JSON bytes, in delivery order."""
    for message in messages(sim, days, rates, until_step):
        yield serialiser.encode(message)


class Pacer:
    """Real-time pacing: the messages of fleet step k are released (k - k0) steps of simulated
    time after those of the first paced step k0, in wall time. Parked periods, which the fleet
    loop skips, are slept through. `sleep` and `monotonic` are injected for tests."""

    def __init__(
        self,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.sleep = sleep
        self.monotonic = monotonic
        self.origin: tuple[int, float] | None = None

    def wait_until(self, step: int) -> None:
        now = self.monotonic()
        if self.origin is None:
            self.origin = (step, now)
            return
        first_step, first_time = self.origin
        due = first_time + (step - first_step) * MS_PER_STEP / 1000
        if due > now:
            self.sleep(due - now)


def run(
    sim: FleetSimulation,
    days: int,
    rates: TransportRates,
    serialiser: Serialiser,
    publisher: Publisher,
    pacer: Pacer | None = None,
    max_messages: int | None = None,
    until_step: int | None = None,
) -> int:
    """Publish the delivered messages of `days` days of `sim`, one batch per fleet step, after
    the pacer's wait if paced; stop after `max_messages` messages. The number sent."""
    sent = 0
    for step, delivered in stepped_messages(sim, days, rates, until_step):
        if max_messages is not None:
            delivered = delivered[: max_messages - sent]
        if pacer is not None:
            pacer.wait_until(step)
        publisher.publish([serialiser.encode(message) for message in delivered])
        sent += len(delivered)
        if max_messages is not None and sent >= max_messages:
            break
    return sent
