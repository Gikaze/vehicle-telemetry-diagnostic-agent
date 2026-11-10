"""Telemetry stream driver: the fleet loop's samples through the aggregator and the transport
(spec section 2).

    FleetSample -> Aggregator -> PeriodicRecord -> Transport -> Sequenced

Time is the integer fleet step throughout. The later stages (serialisation, publishers) attach
here.
"""

from collections.abc import Iterator

from simulator.aggregator import Aggregator, PeriodicRecord
from simulator.fleet_state import FleetSimulation
from simulator.transport import Sequenced, Transport, TransportRates


def records(sim: FleetSimulation, days: int) -> Iterator[PeriodicRecord]:
    """The periodic records of `days` days of `sim`, in the order the windows close (by step,
    then slot)."""
    aggregator = Aggregator(sim.catalogue, sim.vehicle_ids)
    for sample in sim.run(days):
        yield from aggregator.feed(sample)


def messages(sim: FleetSimulation, days: int, rates: TransportRates) -> Iterator[Sequenced]:
    """The messages delivered for the records of `days` days of `sim`, in delivery order."""
    transport = Transport(sim.root_seed, sim.vehicle_ids, rates)
    for record in records(sim, days):
        yield from transport.send([record])
