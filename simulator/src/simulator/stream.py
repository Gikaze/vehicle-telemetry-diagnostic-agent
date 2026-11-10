"""Telemetry stream driver: the fleet loop's samples through the aggregator (spec section 2).

    FleetSample -> Aggregator -> PeriodicRecord

Time is the integer fleet step throughout. The later stages (transport, serialisation,
publishers) attach here.
"""

from collections.abc import Iterator

from simulator.aggregator import Aggregator, PeriodicRecord
from simulator.fleet_state import FleetSimulation


def records(sim: FleetSimulation, days: int) -> Iterator[PeriodicRecord]:
    """The periodic records of `days` days of `sim`, in the order the windows close (by step,
    then slot)."""
    aggregator = Aggregator(sim.catalogue, sim.vehicle_ids)
    for sample in sim.run(days):
        yield from aggregator.feed(sample)
