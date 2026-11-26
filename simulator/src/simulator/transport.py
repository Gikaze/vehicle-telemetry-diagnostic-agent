"""Transport: per-vehicle sequence counter and injected delivery defects (FR-17, spec section 5).

Each vehicle has one sequence counter for the run, starting at 0 and shared by every message type:
`seq` is assigned to each record as it arrives, before any defect, so a lost message leaves a gap
in the counter. Then each message takes three uniform draws in [0, 1) from the vehicle's TRANSPORT
stream (rng.py), always in the order gap, duplicate, reorder and whatever the rates, so changing
one rate never moves another rate's decisions:

- gap: draw < gap_rate, the message is not delivered;
- duplicate: draw < duplicate_rate, a delivered message is delivered twice in a row (same seq, same
  record);
- reorder: draw < reorder_rate, a delivered message is held, unless one is already held, and
  delivered right after the vehicle's next delivered message (with its duplicate, if any). A held
  message is released when the record that ends its trip arrives, delivered or not, so nothing is
  held across a parked period, and a trip's last record is never delivered late: if it is held,
  the same release delivers it at once.

The draws come in blocks of DRAW_BLOCK messages, continuing the vehicle's stream, so the block size
changes no decision, and a vehicle's draws depend only on the root seed, its ID and its own number
of messages: adding a vehicle or changing the fleet order changes nothing for the others. The
stream is not indexed by day, so a run of N days is a prefix of a longer run from the same seed.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from simulator.aggregator import PeriodicRecord, Record
from simulator.params import Config
from simulator.rng import Stream, vehicle_generator

DRAW_BLOCK = 1024  # messages of draws taken at once per vehicle
GAP, DUPLICATE, REORDER = range(3)


@dataclass(frozen=True)
class Sequenced:
    seq: int
    record: Record


@dataclass(frozen=True)
class TransportRates:
    gap: float = 0.0
    duplicate: float = 0.0
    reorder: float = 0.0

    def __post_init__(self) -> None:
        for name in ("gap", "duplicate", "reorder"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"the {name} rate must lie between 0 and 1")

    @classmethod
    def from_config(cls, config: Config) -> "TransportRates":
        return cls(
            gap=float(config.value("gap_rate")),
            duplicate=float(config.value("duplicate_rate")),
            reorder=float(config.value("reorder_rate")),
        )


def ends_trip(record: Record) -> bool:
    """The periodic record of a trip's last window; an event never ends a trip (it comes before
    the periodic record of its window)."""
    return isinstance(record, PeriodicRecord) and record.last_of_trip


class _Vehicle:
    """One vehicle's counter, draws and held message."""

    def __init__(self, root_seed: int, vehicle_id: str):
        self.generator = vehicle_generator(root_seed, vehicle_id, Stream.TRANSPORT)
        self.next_seq = 0
        self.draws = np.empty((0, 3))
        self.cursor = 0
        self.held: list[Sequenced] = []

    def draw(self) -> np.ndarray:
        if self.cursor == len(self.draws):
            self.draws = self.generator.random((DRAW_BLOCK, 3))
            self.cursor = 0
        row = self.draws[self.cursor]
        self.cursor += 1
        return row


class Transport:
    """Sequence numbers and delivery defects for the vehicles of one run."""

    def __init__(self, root_seed: int, vehicle_ids: Iterable[str], rates: TransportRates):
        self.rates = rates
        self.vehicles = {vehicle_id: _Vehicle(root_seed, vehicle_id) for vehicle_id in vehicle_ids}

    def send(self, records: Iterable[Record]) -> list[Sequenced]:
        """The messages delivered for `records`, in delivery order."""
        delivered: list[Sequenced] = []
        for record in records:
            delivered += self._send(record)
        return delivered

    def _send(self, record: Record) -> list[Sequenced]:
        vehicle = self.vehicles[record.vehicle_id]
        message = Sequenced(vehicle.next_seq, record)
        vehicle.next_seq += 1
        u = vehicle.draw()
        out: list[Sequenced] = []
        if u[GAP] >= self.rates.gap:
            copies = [message, message] if u[DUPLICATE] < self.rates.duplicate else [message]
            if not vehicle.held and u[REORDER] < self.rates.reorder:
                vehicle.held = copies
            else:
                out = copies + vehicle.held
                vehicle.held = []
        if ends_trip(record) and vehicle.held:
            out += vehicle.held
            vehicle.held = []
        return out
