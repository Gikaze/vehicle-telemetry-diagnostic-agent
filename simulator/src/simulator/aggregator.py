"""Aggregator: the fleet loop's 10 Hz samples into periodic window records and event records (spec
section 5).

Windows follow each trip: the first starts at the trip's first step, each later one
WINDOW_STEPS steps after the previous one, and the last window of a trip holds the remaining 1 to
WINDOW_STEPS samples (closed by the sample's `ended` flag). A slot without a trip in progress is
never read, so nothing is recorded while a vehicle is parked. Only the emitted (measured) values
of a sample are used.

Per numeric signal, over the window:

- each sample is on the catalogue grid, so it is taken as its whole grid index
  raw = (value - offset) / resolution (catalogue.to_raw, computed for every signal at once by
  Aggregator.grid_indices) and summed exactly in int64;
- `min`, `max` and `last` are samples of the window, mapped back with catalogue.from_raw, so they
  are bit for bit the emitted sample values;
- `mean` is the mean grid index rounded half to even (np.rint), mapped back the same way: it is on
  the grid, and within the range since the range bounds lie on the grid. sum / count is a float
  division of integers below 2**53 with count <= WINDOW_STEPS, so the quotient is correctly
  rounded and never lands on a half-integer it is not exactly equal to;
- a sample equal to the signal's invalid value does not enter `mean`, `min` or `max`; `last` is
  the last sample as read (the invalid value if that sample is invalid); if no sample of the
  window is valid, the four statistics are the invalid value.

Per boolean signal: `last`, and `count`, the rising edges (false to true) between consecutive
samples of the same trip. The value before a window's first sample is the previous window's last
sample in the trip; a trip's first sample has no predecessor and is never counted as an edge.

Event records (spec section 5): a rising edge of a boolean signal marked `event` in the catalogue
(the same edges `count` counts, so never on a trip's first sample) gives an EventRecord at the
step where the edge is seen, at once, without waiting for the window to close.

Records of one step come in slot order; for each slot, its event records (in catalogue order) come
before the periodic record of a window that closes at that step, which contains the edge.
"""

from dataclasses import dataclass

import numpy as np

from simulator.catalogue import Catalogue, from_raw
from simulator.constants import AGGREGATION_WINDOW, PHYSICS_DT
from simulator.fleet_state import FleetSample

STEPS_PER_SECOND = round(1 / PHYSICS_DT)
WINDOW_STEPS = AGGREGATION_WINDOW * STEPS_PER_SECOND
INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min

NumericStats = tuple[float, float, float, float]  # mean, min, max, last
BooleanStats = tuple[bool, int]  # last, count


@dataclass(frozen=True)
class PeriodicRecord:
    """One vehicle over one window, values in catalogue units; no sequence number yet."""

    vehicle_id: str
    start_step: int  # fleet step of the window's first sample
    steps: int  # samples in the window, 1 to WINDOW_STEPS
    last_of_trip: bool
    numeric: tuple[NumericStats, ...]  # numeric signals in catalogue order
    boolean: tuple[BooleanStats, ...]  # boolean signals in catalogue order

    @property
    def window_s(self) -> float:
        """Window length, s, from the whole number of steps: 0.3, where 3 * 0.1 would give
        0.30000000000000004."""
        return self.steps / STEPS_PER_SECOND


@dataclass(frozen=True)
class EventRecord:
    """A rising edge of an event signal, seen at fleet step `step`; no sequence number yet."""

    vehicle_id: str
    step: int  # the sample where the signal is first true
    signal: str
    value: bool  # the new value: true for a rising edge

    @property
    def window_s(self) -> float:
        """The physics step the sample describes (spec section 4), s."""
        return 1 / STEPS_PER_SECOND


Record = PeriodicRecord | EventRecord


class Aggregator:
    """Open windows of every slot of a fleet, fed one FleetSample per step."""

    def __init__(self, catalogue: Catalogue, vehicle_ids: tuple[str, ...]):
        self.vehicle_ids = vehicle_ids
        self.numeric = tuple(s for s in catalogue.signals if s.is_number)
        self.boolean = tuple(s for s in catalogue.signals if not s.is_number)
        self.event_columns = np.array([s.event for s in self.boolean], dtype=bool)
        self.event_names = tuple(s.name for s in self.boolean if s.event)
        self.invalid = np.array([s.invalid for s in self.numeric], dtype=float)
        self.offset = np.array([s.offset for s in self.numeric], dtype=float)
        self.resolution = np.array([s.resolution for s in self.numeric], dtype=float)
        n, m, b = len(vehicle_ids), len(self.numeric), len(self.boolean)
        self.count = np.zeros(n, dtype=np.int64)  # samples in the open window
        self.start = np.zeros(n, dtype=np.int64)
        self.sum = np.zeros((n, m), dtype=np.int64)
        self.valid = np.zeros((n, m), dtype=np.int64)
        self.min = np.full((n, m), INT64_MAX, dtype=np.int64)
        self.max = np.full((n, m), INT64_MIN, dtype=np.int64)
        self.last = np.zeros((n, m))
        self.edges = np.zeros((n, b), dtype=np.int64)
        self.previous = np.zeros((n, b), dtype=bool)
        self.in_trip = np.zeros(n, dtype=bool)  # `previous` holds a sample of the current trip

    def feed(self, sample: FleetSample) -> list[Record]:
        """Add the active slots' samples of one step; the event records of this step and the
        periodic records of the windows it closes."""
        slots = np.flatnonzero(sample.active)
        if slots.size == 0:
            return []
        values = np.stack([sample.measured[s.name][slots] for s in self.numeric], axis=1)
        flags = np.stack([sample.measured[s.name][slots] for s in self.boolean], axis=1)
        raw = self.grid_indices(values).astype(np.int64)
        valid = values != self.invalid
        opening = self.count[slots] == 0
        self.start[slots[opening]] = sample.step
        self.sum[slots] = np.where(opening[:, None], 0, self.sum[slots]) + np.where(valid, raw, 0)
        self.valid[slots] = np.where(opening[:, None], 0, self.valid[slots]) + valid
        low = np.where(opening[:, None], INT64_MAX, self.min[slots])
        high = np.where(opening[:, None], INT64_MIN, self.max[slots])
        self.min[slots] = np.where(valid, np.minimum(low, raw), low)
        self.max[slots] = np.where(valid, np.maximum(high, raw), high)
        self.last[slots] = values
        rising = flags & ~self.previous[slots] & self.in_trip[slots][:, None]
        self.edges[slots] = np.where(opening[:, None], 0, self.edges[slots]) + rising
        self.previous[slots] = flags
        self.in_trip[slots] = True
        self.count[slots] += 1
        ended = sample.ended[slots]
        closes = (self.count[slots] == WINDOW_STEPS) | ended
        events = rising[:, self.event_columns]
        records: list[Record] = []
        for r in np.flatnonzero(closes | events.any(axis=1)):
            i = int(slots[r])
            for j in np.flatnonzero(events[r]):
                records.append(
                    EventRecord(self.vehicle_ids[i], sample.step, self.event_names[j], True)
                )
            if closes[r]:
                records.append(self._record(i, bool(ended[r])))
        self.count[slots[closes]] = 0
        self.in_trip[slots[ended]] = False
        return records

    def grid_indices(self, values: np.ndarray) -> np.ndarray:
        """catalogue.to_raw of every numeric signal at once: `values` (rows, numeric signals in
        catalogue order); the same elementwise arithmetic, (value - offset) / resolution rounded
        half to even, over the whole matrix."""
        return np.round((values - self.offset) / self.resolution)

    def _record(self, i: int, last_of_trip: bool) -> PeriodicRecord:
        valid = self.valid[i]
        some = valid > 0
        mean_raw = np.rint(self.sum[i] / np.where(some, valid, 1))
        numeric = []
        for j, signal in enumerate(self.numeric):
            last = float(self.last[i, j])
            if some[j]:
                mean, low, high = from_raw(
                    np.array([mean_raw[j], self.min[i, j], self.max[i, j]]), signal
                ).tolist()
            else:
                mean = low = high = float(self.invalid[j])
            numeric.append((mean, low, high, last))
        flags = self.previous[i].tolist()
        boolean = tuple(zip(flags, self.edges[i].tolist(), strict=True))
        return PeriodicRecord(
            vehicle_id=self.vehicle_ids[i],
            start_step=int(self.start[i]),
            steps=int(self.count[i]),
            last_of_trip=last_of_trip,
            numeric=tuple(numeric),
            boolean=boolean,
        )
