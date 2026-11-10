"""FR-01: periodic records every 10 s per vehicle while its ignition is on (spec section 5).

Expected values are computed here from the samples of the same run (step, active, ended and the
measured values), never through the aggregator: windows from the trips' steps, means with exact
fractions, rising edges by walking each trip. Hand-made samples cover what a v1.0 run cannot
show on demand (an edge on a window's first sample, invalid samples, ties in the mean).
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest
from driven import short_plans

from simulator import stream
from simulator.aggregator import Aggregator, PeriodicRecord
from simulator.catalogue import Signal, catalogue, to_raw
from simulator.constants import AGGREGATION_WINDOW, PHYSICS_DT
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSample, FleetSimulation

SEED = 42
# Trips of 90 s and 60 s (plus their overrun), so no trip is a whole number of windows.
PLANS = short_plans([90.0, 60.0], [20.0])
WINDOW = round(AGGREGATION_WINDOW / PHYSICS_DT)  # samples in a full window
NUMERIC = tuple(s for s in catalogue().signals if s.is_number)
BOOLEAN = tuple(s for s in catalogue().signals if not s.is_number)


@dataclass(frozen=True)
class Run:
    vehicle_ids: tuple[str, ...]
    samples: list[FleetSample]
    records: list[PeriodicRecord]


@pytest.fixture(scope="module")
def run() -> Run:
    """Three vans over one day of short plans, every sample kept next to the records."""
    sim = FleetSimulation(SEED, load_fleet(size=3), PLANS)
    samples: list[FleetSample] = []
    loop = sim.run

    def recording(days: int):
        for sample in loop(days):
            samples.append(sample)
            yield sample

    sim.run = recording  # type: ignore[method-assign]
    return Run(sim.vehicle_ids, samples, list(stream.records(sim, 1)))


@dataclass(frozen=True)
class Window:
    vehicle_id: str
    steps: list[int]
    last_of_trip: bool
    rows: list[int]  # indices into Run.samples


def trips(run: Run) -> dict[str, list[list[int]]]:
    """Per vehicle, its trips as lists of sample indices, split at each `ended` sample."""
    result: dict[str, list[list[int]]] = {v: [] for v in run.vehicle_ids}
    current: dict[str, list[int]] = {v: [] for v in run.vehicle_ids}
    for index, sample in enumerate(run.samples):
        for slot, vehicle in enumerate(run.vehicle_ids):
            if sample.active[slot]:
                current[vehicle].append(index)
                if sample.ended[slot]:
                    result[vehicle].append(current[vehicle])
                    current[vehicle] = []
    assert all(not rows for rows in current.values())
    return result


def expected_windows(run: Run) -> dict[str, list[Window]]:
    windows: dict[str, list[Window]] = {}
    for vehicle, vehicle_trips in trips(run).items():
        windows[vehicle] = []
        for trip in vehicle_trips:
            for first in range(0, len(trip), WINDOW):
                rows = trip[first : first + WINDOW]
                windows[vehicle].append(
                    Window(
                        vehicle,
                        [run.samples[r].step for r in rows],
                        first + WINDOW >= len(trip),
                        rows,
                    )
                )
    return windows


def records_of(run: Run, vehicle: str) -> list[PeriodicRecord]:
    return [r for r in run.records if r.vehicle_id == vehicle]


def grid_index(value: float, signal: Signal) -> int:
    """Whole grid index of an emitted value, by exact fractions; the value must be on the grid."""
    q = (Fraction(value) - Fraction(str(signal.offset))) / Fraction(str(signal.resolution))
    k = round(q)
    assert abs(q - k) < Fraction(1, 10**6), (signal.name, value)
    return k


def expected_mean(values: list[float], signal: Signal) -> float:
    """Mean grid index rounded half to even (Fraction rounding), back to a value."""
    indices = [grid_index(v, signal) for v in values]
    k = round(Fraction(sum(indices), len(indices)))
    return float(Fraction(str(signal.offset)) + k * Fraction(str(signal.resolution)))


def test_the_window_is_ten_seconds() -> None:
    """FR-01's 10 s cadence, as 100 physics steps."""
    assert AGGREGATION_WINDOW == 10
    assert WINDOW == 100


def test_windows_follow_each_trip(run: Run) -> None:
    """Per vehicle, one record per window of each trip, in order: starts every WINDOW steps from
    the trip's first step, full windows of 10 s, and a shorter last window of 0.1 to 10 s."""
    windows = expected_windows(run)
    lengths = [len(w.steps) for ws in windows.values() for w in ws if w.last_of_trip]
    assert lengths and any(n < WINDOW for n in lengths) and min(lengths) >= 1
    for vehicle, expected in windows.items():
        got = records_of(run, vehicle)
        assert [(r.start_step, r.steps, r.last_of_trip) for r in got] == [
            (w.steps[0], len(w.steps), w.last_of_trip) for w in expected
        ], vehicle
        for record, window in zip(got, expected, strict=True):
            assert window.steps == list(range(window.steps[0], window.steps[0] + len(window.steps)))
            assert record.window_s == len(window.steps) / 10
            assert 0 < record.window_s <= AGGREGATION_WINDOW
            if not window.last_of_trip:
                assert record.window_s == AGGREGATION_WINDOW
        within = [b.start_step - a.start_step for a, b in pairwise(got) if not a.last_of_trip]
        assert within and set(within) == {WINDOW}


def test_no_window_while_parked(run: Run) -> None:
    """Every step a record covers is a step where that vehicle has a trip in progress, and every
    such step is covered by exactly one record."""
    for slot, vehicle in enumerate(run.vehicle_ids):
        active = [s.step for s in run.samples if s.active[slot]]
        covered = [
            step
            for r in records_of(run, vehicle)
            for step in range(r.start_step, r.start_step + r.steps)
        ]
        assert covered == active, vehicle
    parked = [s.step for s in run.samples if not s.active.all()]
    assert parked  # some vehicle is parked during the run


def test_records_come_by_closing_step_then_slot(run: Run) -> None:
    keys = [(r.start_step + r.steps - 1, run.vehicle_ids.index(r.vehicle_id)) for r in run.records]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_numeric_statistics_from_the_samples(run: Run) -> None:
    """min, max and last are the window's samples bit for bit; the mean is the mean grid index
    rounded half to even. Some means fall between the samples' values."""
    between = 0
    for windows in expected_windows(run).values():
        for window, record in zip(windows, records_of(run, windows[0].vehicle_id), strict=True):
            slot = run.vehicle_ids.index(window.vehicle_id)
            for j, signal in enumerate(NUMERIC):
                values = [float(run.samples[r].measured[signal.name][slot]) for r in window.rows]
                assert signal.invalid not in values
                mean, low, high, last = record.numeric[j]
                assert (low, high, last) == (min(values), max(values), values[-1]), signal.name
                assert mean == expected_mean(values, signal), signal.name
                between += mean not in values
    assert between > 0


def test_boolean_statistics_count_rising_edges_within_the_trip(run: Run) -> None:
    """last is the window's last sample; count the false-to-true changes from the previous
    sample of the same trip. Some edges occur in the run."""
    edges = 0
    for vehicle, vehicle_trips in trips(run).items():
        slot = run.vehicle_ids.index(vehicle)
        got = iter(records_of(run, vehicle))
        for trip in vehicle_trips:
            for first in range(0, len(trip), WINDOW):
                record = next(got)
                for j, signal in enumerate(BOOLEAN):
                    series = [bool(run.samples[r].measured[signal.name][slot]) for r in trip]
                    count = sum(
                        1
                        for k in range(max(first, 1), min(first + WINDOW, len(trip)))
                        if series[k] and not series[k - 1]
                    )
                    last = series[min(first + WINDOW, len(trip)) - 1]
                    assert record.boolean[j] == (last, count), (vehicle, signal.name)
                    edges += count
    assert edges > 0


def test_grid_indices_equal_to_raw_signal_by_signal(run: Run) -> None:
    """The aggregator's one-matrix grid indices equal catalogue.to_raw of each signal, bit for
    bit, on every active row of the run."""
    aggregator = Aggregator(catalogue(), run.vehicle_ids)
    rows = 0
    for sample in run.samples:
        slots = np.flatnonzero(sample.active)
        values = np.stack([sample.measured[s.name][slots] for s in NUMERIC], axis=1)
        got = aggregator.grid_indices(values)
        for j, signal in enumerate(NUMERIC):
            assert got[:, j].tobytes() == to_raw(values[:, j], signal).tobytes(), signal.name
        rows += slots.size
    assert rows > 0


def test_a_step_without_an_active_slot_changes_nothing() -> None:
    """A step where every slot is parked yields no record and leaves the open window as it was."""
    aggregator = Aggregator(catalogue(), ("VAN-001",))
    assert aggregator.feed(make_sample(0, {"coolant_temp": [80.0]}, [True])) == []
    assert aggregator.feed(make_sample(1, {"coolant_temp": [math.nan]}, [False])) == []
    (record,) = aggregator.feed(make_sample(2, {"coolant_temp": [82.0]}, [True], [True]))
    j = [s.name for s in NUMERIC].index("coolant_temp")
    assert (record.start_step, record.steps) == (0, 2)
    assert record.numeric[j] == (81.0, 80.0, 82.0, 82.0)


def test_every_value_is_on_the_grid_and_in_the_range(run: Run) -> None:
    for record in run.records:
        for (mean, low, high, last), signal in zip(record.numeric, NUMERIC, strict=True):
            assert signal.range is not None
            for value in (mean, low, high, last):
                assert signal.range[0] <= value <= signal.range[1], signal.name
                grid_index(value, signal)
        for last, count in record.boolean:
            assert isinstance(last, bool) and isinstance(count, int) and count >= 0


# Hand-made samples: one or two slots, every numeric signal at a valid value unless set.


def make_sample(
    step: int,
    values: dict[str, list],
    active: list[bool],
    ended: list[bool] | None = None,
) -> FleetSample:
    n = len(active)
    measured: dict[str, np.ndarray] = {}
    for signal in catalogue().signals:
        if signal.name in values:
            measured[signal.name] = np.array(
                values[signal.name], dtype=float if signal.is_number else bool
            )
        elif signal.is_number:
            assert signal.range is not None
            measured[signal.name] = np.full(n, float(signal.range[0]))
        else:
            measured[signal.name] = np.zeros(n, dtype=bool)
    return FleetSample(
        step=step,
        active=np.array(active),
        ended=np.array(ended if ended is not None else [False] * n),
        vehicle_ids=tuple(f"VAN-{i + 1:03d}" for i in range(n)),
        true=None,  # type: ignore[arg-type]  # the aggregator never reads true states
        measured=measured,
    )


def feed_series(name: str, series: list, ends: set[int] = frozenset()) -> list[PeriodicRecord]:
    """One slot fed `series` of signal `name`, one sample per step from step 0; a trip ends after
    each index in `ends` and after the last sample."""
    aggregator = Aggregator(catalogue(), ("VAN-001",))
    records = []
    for k, value in enumerate(series):
        ended = k in ends or k == len(series) - 1
        records += aggregator.feed(make_sample(k, {name: [value]}, [True], [ended]))
    return records


def test_an_edge_on_a_window_s_first_sample_counts_in_that_window() -> None:
    series = [True] + [False] * 99 + [True] * 50 + [False, True]
    records = feed_series("brake_pedal_pressed", series)
    j = [s.name for s in BOOLEAN].index("brake_pedal_pressed")
    assert [r.boolean[j] for r in records] == [(False, 0), (True, 2)]


def test_a_level_held_across_windows_counts_once() -> None:
    series = [False] * 50 + [True] * 150
    records = feed_series("abs_active", series)
    j = [s.name for s in BOOLEAN].index("abs_active")
    assert [r.boolean[j] for r in records] == [(True, 1), (True, 0)]


def test_a_trip_s_first_sample_is_never_an_edge() -> None:
    """The first trip starts true; the second starts true after the first ended false."""
    series = [True, True, False, True, False]
    records = feed_series("esc_active", series, ends={2})
    j = [s.name for s in BOOLEAN].index("esc_active")
    assert [r.boolean[j] for r in records] == [(False, 0), (False, 0)]


def test_the_mean_rounds_half_to_even_on_the_grid() -> None:
    """accel_long (resolution 0.01): 0.015 rounds to 0.02, 0.025 to 0.02, -0.015 to -0.02."""
    j = [s.name for s in NUMERIC].index("accel_long")
    for series, mean in (
        ([0.01, 0.02], 0.02),
        ([0.02, 0.03], 0.02),
        ([-0.01, -0.02], -0.02),
        ([0.01, 0.02, 0.04], 0.02),
    ):
        (record,) = feed_series("accel_long", series)
        assert record.numeric[j][0] == mean, series


def test_invalid_samples_stay_out_of_mean_min_and_max() -> None:
    invalid = catalogue().signal("coolant_temp").invalid
    j = [s.name for s in NUMERIC].index("coolant_temp")
    cases = (
        ([80.0, invalid, 90.0], (85.0, 80.0, 90.0, 90.0)),
        ([80.0, 90.0, invalid], (85.0, 80.0, 90.0, invalid)),
        ([invalid, invalid], (invalid, invalid, invalid, invalid)),
    )
    for series, expected in cases:
        (record,) = feed_series("coolant_temp", series)
        assert record.numeric[j] == expected, series


def test_window_s_is_exact_for_a_short_window() -> None:
    (record,) = feed_series("abs_active", [False] * 37)
    assert record.steps == 37 and repr(record.window_s) == "3.7"


def test_a_parked_slot_is_never_read() -> None:
    """Slot 1 is parked with NaN values: slot 0's windows are unaffected and no record names
    VAN-002."""
    aggregator = Aggregator(catalogue(), ("VAN-001", "VAN-002"))
    records = []
    for k in range(3):
        sample = make_sample(
            k,
            {"coolant_temp": [80.0 + k, math.nan]},
            [True, False],
            [k == 2, False],
        )
        records += aggregator.feed(sample)
    (record,) = records
    j = [s.name for s in NUMERIC].index("coolant_temp")
    assert record.vehicle_id == "VAN-001" and record.numeric[j] == (81.0, 80.0, 82.0, 82.0)
