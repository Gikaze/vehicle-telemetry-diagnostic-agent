"""One fleet day from a root seed: checks on every vehicle, and one vehicle's day recorded.

run_fleet_day drives the vehicles through the fleet loop (fleet_state.py) for day 0 and keeps,
without storing the samples:

- each trip's start step, ignition-off step (the step after its last step) and planned duration;
- the first step of any NaN in an emitted numeric signal, per vehicle and signal (only the slots
  with a trip in progress are emitted);
- every step of one vehicle in progress: its emitted signals and a few true states, for the
  vehicle-day plot of spec section 7 (simulator/tools/vehicle_day.py).

The checks return their verdict with the lines to print, and exit_status turns them into the
tool's exit status:

- check_day_end: every planned trip was driven and each vehicle's last ignition off is at or
  before shift_end_latest (duty_cycle.yaml), the bound the day plans keep with day_end_margin;
- check_trip_overruns: no trip runs past its planned duration by more than trip_overrun_bound,
  the per-trip bound behind day_end_margin;
- check_no_nan: no NaN in any emitted numeric signal of any vehicle (encoding saturates every
  value to its catalogue range, so no range check is needed).

The zoom windows of the plot follow fixed rules on the recorded day (hottest_disc_window,
delivery_window), so the same seed gives the same figure.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from simulator.constants import PHYSICS_DT
from simulator.fleet import Vehicle
from simulator.fleet_state import BAR, FleetSimulation, PlanDay, TrueState, plan_from_duty
from simulator.model.brakes import held_steps
from simulator.profile.day_plan import DayPlan

KMH = 3.6  # km/h per m/s
# True states recorded next to the emitted signals, in the units of the matching signal.
TRUE_COLUMNS: dict[str, Callable[[TrueState], np.ndarray]] = {
    "true_speed": lambda t: t.drive.speed * KMH,  # km/h
    "true_accel_long": lambda t: t.drive.accel,
    "true_accel_lat": lambda t: t.steering.accel_lat,
    "true_master_cyl_pressure": lambda t: t.master_cyl_pressure / BAR,  # bar
    "gear": lambda t: t.gear,
    "held": lambda t: held_steps(t.drive.speed, t.drive.next_speed),
    "at_delivery": lambda t: t.drive.at_delivery,
}
# The approach before a delivery stop and the restart after it, in a delivery zoom window, s.
APPROACH = 30.0
RESTART = 30.0
# The part of the hottest-disc zoom window after the peak, s.
AFTER_PEAK = 30.0


@dataclass(frozen=True)
class TripTiming:
    vehicle_id: str
    trip: int  # index in the day
    start_step: int
    off_step: int  # ignition off: the step after the trip's last step
    planned_duration: float  # s

    @property
    def overrun(self) -> float:
        """Actual minus planned duration, s."""
        return (self.off_step - self.start_step) * PHYSICS_DT - self.planned_duration


@dataclass(frozen=True)
class Recording:
    """Every step of one vehicle with a trip in progress, in order: one row per step."""

    vehicle_id: str
    columns: tuple[str, ...]  # "step", "trip", the emitted signals, then TRUE_COLUMNS
    data: np.ndarray  # (steps, columns), booleans as 0 and 1

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]

    @property
    def steps(self) -> np.ndarray:
        return self["step"].astype(np.int64)

    def __repr__(self) -> str:
        return f"Recording({self.vehicle_id}, steps={len(self.data)})"


class _Rows:
    """Rows appended one at a time into a buffer that doubles when full."""

    def __init__(self, width: int):
        self.data = np.empty((4096, width))
        self.size = 0

    def append(self, row: np.ndarray) -> None:
        if self.size == len(self.data):
            self.data = np.concatenate([self.data, np.empty_like(self.data)])
        self.data[self.size] = row
        self.size += 1

    def array(self) -> np.ndarray:
        return self.data[: self.size].copy()


@dataclass(frozen=True)
class FleetDay:
    root_seed: int
    plans: dict[str, DayPlan]  # per vehicle ID, in fleet order
    trips: tuple[TripTiming, ...]  # in order of ignition off
    nan_first: dict[tuple[str, str], int]  # (vehicle ID, signal): first step with a NaN
    numeric_signals: int  # emitted numeric signals checked for NaN, per vehicle
    samples: int
    recording: Recording | None
    loop_seconds: float  # compute time of the loop, checks and recording included


def run_fleet_day(
    root_seed: int,
    vehicles: tuple[Vehicle, ...],
    record: str | None = None,
    plan_day: PlanDay = plan_from_duty,
) -> FleetDay:
    """Day 0 of `vehicles` from `root_seed`, recording the vehicle with ID `record`."""
    sim = FleetSimulation(root_seed, vehicles, plan_day)
    ids = sim.vehicle_ids
    numeric = tuple(signal.name for signal in sim.catalogue.signals if signal.is_number)
    emitted = sim.catalogue.names
    columns = ("step", "trip", *emitted, *TRUE_COLUMNS)
    slot = ids.index(record) if record is not None else None
    rows = _Rows(len(columns))
    start = np.full(sim.n, -1, dtype=np.int64)
    trip = np.zeros(sim.n, dtype=int)
    trips: list[TripTiming] = []
    nan_first: dict[tuple[str, str], int] = {}
    samples = 0
    clock = time.perf_counter()
    for sample in sim.run(1):
        samples += 1
        k = sample.step
        start[sample.active & (start < 0)] = k
        values = np.stack([sample.measured[name] for name in numeric], axis=1)
        bad = np.isnan(values) & sample.active[:, None]
        if bad.any():
            for i, j in zip(*np.nonzero(bad), strict=True):
                nan_first.setdefault((ids[i], numeric[j]), k)
        if slot is not None and sample.active[slot]:
            row = [k, trip[slot]]
            row += [sample.measured[name][slot] for name in emitted]
            row += [get(sample.true)[slot] for get in TRUE_COLUMNS.values()]
            rows.append(np.array(row, dtype=float))
        for i in np.flatnonzero(sample.ended):
            plan = sim.plans[i]
            assert plan is not None
            trips.append(
                TripTiming(ids[i], int(trip[i]), int(start[i]), k + 1, plan.trip_durations[trip[i]])
            )
            start[i] = -1
            trip[i] += 1
    loop_seconds = time.perf_counter() - clock
    plans = {vehicle_id: plan for vehicle_id, plan in zip(ids, sim.plans, strict=True) if plan}
    recording = None if record is None else Recording(record, columns, rows.array())
    return FleetDay(
        root_seed=root_seed,
        plans=plans,
        trips=tuple(trips),
        nan_first=nan_first,
        numeric_signals=len(numeric),
        samples=samples,
        recording=recording,
        loop_seconds=loop_seconds,
    )


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    lines: tuple[str, ...]


def clock_time(seconds: float) -> str:
    """hh:mm:ss.s after midnight."""
    tenths = round(seconds * 10)
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    return f"{hours:02d}:{minutes:02d}:{rest / 10:04.1f}"


def check_day_end(day: FleetDay, shift_end_latest: float) -> CheckResult:
    """Every planned trip driven, and each vehicle's last ignition off at or before
    `shift_end_latest` (s after midnight; one-day run)."""
    passed, lines = True, []
    for vehicle_id, plan in day.plans.items():
        trips = [t for t in day.trips if t.vehicle_id == vehicle_id]
        if len(trips) != plan.trips:
            passed = False
            lines.append(f"{vehicle_id}: {len(trips)} of {plan.trips} planned trips driven")
        if not trips:
            continue
        off = trips[-1].off_step * PHYSICS_DT
        within = off <= shift_end_latest
        passed = passed and within
        lines.append(
            f"{vehicle_id}: last ignition off {clock_time(off)}, "
            f"{(shift_end_latest - off) / 60:.1f} min before {clock_time(shift_end_latest)}"
            + ("" if within else " (too late)")
        )
    return CheckResult("last ignition off before shift_end_latest", passed, tuple(lines))


def check_trip_overruns(trips: tuple[TripTiming, ...], bound: float) -> CheckResult:
    """No trip longer than its planned duration plus `bound` (s)."""
    lines, passed = [], True
    for vehicle_id in dict.fromkeys(t.vehicle_id for t in trips):
        own = [t for t in trips if t.vehicle_id == vehicle_id]
        worst = max(own, key=lambda t: t.overrun)
        lines.append(
            f"{vehicle_id}: largest overrun {worst.overrun:.1f} s (trip {worst.trip + 1} of "
            f"{len(own)}), bound {bound:.1f} s"
        )
        for t in own:
            if not t.overrun <= bound:
                passed = False
                lines.append(f"{vehicle_id}: trip {t.trip + 1} overruns by {t.overrun:.1f} s")
    return CheckResult("trip overrun within trip_overrun_bound", passed, tuple(lines))


def check_no_nan(day: FleetDay) -> CheckResult:
    """No NaN in any emitted numeric signal of any vehicle."""
    if not day.nan_first:
        line = (
            f"{day.numeric_signals} numeric signals of {len(day.plans)} vehicles over "
            f"{day.samples} samples"
        )
        return CheckResult("no NaN in an emitted signal", True, (line,))
    lines = tuple(
        f"{vehicle_id}: {name} NaN first at {clock_time(step * PHYSICS_DT)}"
        for (vehicle_id, name), step in sorted(day.nan_first.items())
    )
    return CheckResult("no NaN in an emitted signal", False, lines)


def exit_status(results: list[CheckResult]) -> tuple[int, list[str]]:
    """0 if every check passed, else 1; and the lines to print."""
    lines = []
    for result in results:
        lines.append(f"{'PASS' if result.passed else 'FAIL'} {result.name}")
        lines += [f"  {line}" for line in result.lines]
    return (0 if all(result.passed for result in results) else 1), lines


def _clamp_to_trip(rec: Recording, trip: int, start: int, length: float) -> int:
    """Window start (fleet step) moved so that the window lies in `trip` where it fits."""
    steps = rec.steps[rec["trip"] == trip]
    last_start = int(steps[-1]) + 1 - round(length / PHYSICS_DT)
    return max(int(steps[0]), min(start, last_start))


def hottest_disc_window(rec: Recording, length: float) -> int:
    """Start step of the window ending AFTER_PEAK s after the hottest front disc reading (the
    first if tied), in its trip."""
    front = np.maximum(rec["brake_disc_temp_est_fl"], rec["brake_disc_temp_est_fr"])
    peak = int(np.argmax(front))
    start = int(rec.steps[peak]) + round((AFTER_PEAK - length) / PHYSICS_DT)
    return _clamp_to_trip(rec, int(rec["trip"][peak]), start, length)


def delivery_window(rec: Recording, trip: int, length: float) -> int:
    """Start step of the window from APPROACH s before the van is held at the first delivery
    stop of `trip` that leaves RESTART s after it in the window; failing that, the first such
    stop of the day."""
    longest = round((length - APPROACH - RESTART) / PHYSICS_DT)
    delivery = rec["at_delivery"] > 0
    steps = rec.steps
    # Runs of consecutive delivery steps: (first index, length).
    edges = np.flatnonzero(np.diff(np.concatenate([[0], delivery.astype(int), [0]])))
    runs = [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    fitting = [(i, n) for i, n in runs if n <= longest and steps[i + n - 1] - steps[i] == n - 1]
    candidates = [r for r in fitting if rec["trip"][r[0]] == trip] or fitting
    if not candidates:
        raise ValueError(f"{rec.vehicle_id}: no delivery stop of at most {longest} steps")
    first, _ = candidates[0]
    start = int(steps[first]) - round(APPROACH / PHYSICS_DT)
    return _clamp_to_trip(rec, int(rec["trip"][first]), start, length)


def window_rows(rec: Recording, start: int, length: float) -> np.ndarray:
    """Row indices of the recorded steps in [start, start + length)."""
    steps = rec.steps
    return np.flatnonzero((steps >= start) & (steps < start + round(length / PHYSICS_DT)))


def describe_window(rec: Recording, label: str, start: int, length: float) -> str:
    """One line on a zoom window: its time, trips and held steps, and the hottest front disc."""
    rows = window_rows(rec, start, length)
    held, delivery = rec["held"][rows] > 0, rec["at_delivery"][rows] > 0
    trips = sorted({int(t) + 1 for t in rec["trip"][rows]})
    front = np.maximum(rec["brake_disc_temp_est_fl"], rec["brake_disc_temp_est_fr"])[rows]
    return (
        f"zoom {label}: {clock_time(start * PHYSICS_DT)} + {length:.0f} s, trip {trips}, "
        f"{len(rows)} steps, {int(delivery.sum())} held at a delivery stop, "
        f"{int((held & ~delivery).sum())} held elsewhere, front disc up to {front.max():.0f} degC"
    )
