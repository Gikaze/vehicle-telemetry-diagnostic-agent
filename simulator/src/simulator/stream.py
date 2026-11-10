"""Telemetry stream driver: the fleet loop's samples through the aggregator and the transport
(spec section 2).

    FleetSample -> Aggregator -> Record (periodic or event) -> Transport -> Sequenced
        -> Serialiser -> Payload

Time is the integer fleet step throughout; the serialiser's clock turns steps into simulated
time stamps. The publishers attach here.
"""

from collections.abc import Iterator

from simulator.aggregator import Aggregator, Record
from simulator.fleet_state import FleetSimulation
from simulator.serialise import Payload, Serialiser
from simulator.transport import Sequenced, Transport, TransportRates


def records(sim: FleetSimulation, days: int) -> Iterator[Record]:
    """The periodic and event records of `days` days of `sim`, by step, then slot; for each slot,
    its events before the periodic record of a window closing at that step."""
    aggregator = Aggregator(sim.catalogue, sim.vehicle_ids)
    for sample in sim.run(days):
        yield from aggregator.feed(sample)


def messages(sim: FleetSimulation, days: int, rates: TransportRates) -> Iterator[Sequenced]:
    """The messages delivered for the records of `days` days of `sim`, in delivery order."""
    transport = Transport(sim.root_seed, sim.vehicle_ids, rates)
    for record in records(sim, days):
        yield from transport.send([record])


def payloads(
    sim: FleetSimulation, days: int, rates: TransportRates, serialiser: Serialiser
) -> Iterator[Payload]:
    """The delivered messages of `days` days of `sim` as JSON bytes, in delivery order."""
    for message in messages(sim, days, rates):
        yield serialiser.encode(message)
