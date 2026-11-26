"""Plausibility: the disc rise ratio shared by the invariant tests and the stream checks, and the
provisional zero-anomaly check of a telemetry stream (NFR-15, spec section 7; bounds in
config/plausibility.yaml). The stream check stands in for the FR-06 rules until they exist (M3).

The stream check reads parsed messages in delivery order and keeps a bounded state per vehicle
(the previous full window, at most `ratio_windows` window means), never a whole trip. Its rules
hold for the true state (the invariant tests check them there) and are widened here by the error
of the emitted statistics, computed from the catalogue resolutions and sensors.yaml:

- a sample is off its true value by at most error_bound (clipped noise plus half a resolution;
  half a resolution for a noise-free signal such as the disc estimates and the ambient);
- `min`, `max` and `last` are samples; `mean` is the mean of the samples rounded to the grid, so
  off the true window mean by at most error_bound plus half a resolution; a mean whose window has
  min == max equals its samples, so it is off by error_bound only.

Rules (the name an anomaly carries):

- catalogue_range, catalogue_grid, catalogue_invalid: each numeric statistic lies in the signal's
  range, on its grid, and is not its invalid value;
- edge_count: a boolean's rising edges in a window of n samples are at most ceil(n / 2);
- disc_lr_spread_last, disc_lr_spread_mean: on each axle, |left - right| <= rel * (larger rise +
  rise error) + abs + the error of the two readings; the rise above the ambient is taken from
  `last` for the first and from the window `max` (which bounds every sample) for the second;
- disc_front_rear_ratio: over `ratio_windows` consecutive full windows (disc_ratio_window / 10 s,
  a whole number checked at load), with F and R the mean front and rear axle rises of the window
  means, evaluated once the true rear rise surely exceeds disc_ratio_min_rear_rise, i.e. once
  R - d > disc_ratio_min_rear_rise:
  r_min * (R - d) - d <= F <= r_max * (R + d) + d, d the error of an axle mean rise (one disc
  resolution plus the ambient error), i.e. a ratio in [r_min - d (1 + r_min) / R,
  r_max + d (1 + r_max) / R];
- coolant_rate: between consecutive full windows, |mean difference| <= coolant_rate_max * 10 s +
  twice the error of a mean;
- coolant_band: the window `max` <= coolant_band_max + error_bound.

Consecutive means 10 s apart in window_start, both full (window_s 10): a lost message, a trip end
(the last window of a trip is shorter, or a new trip starts later) or a delivery out of order
restarts the windowed state. A message whose seq is not above the vehicle's highest seq so far
(a duplicate, or one delivered late) gets the per-message rules only. Event messages carry no
statistic and are skipped.
"""

import datetime
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulator.catalogue import Catalogue, Signal
from simulator.constants import AGGREGATION_WINDOW, PHYSICS_DT
from simulator.params import Config
from simulator.sensors import SensorParameters, error_bound

DISCS = (
    "brake_disc_temp_est_fl",
    "brake_disc_temp_est_fr",
    "brake_disc_temp_est_rl",
    "brake_disc_temp_est_rr",
)
AMBIENT = "ambient_temp"
COOLANT = "coolant_temp"
STATISTICS = ("mean", "min", "max", "last")
WINDOW_MS = AGGREGATION_WINDOW * 1000
STEPS_PER_SECOND = round(1 / PHYSICS_DT)
SLACK = 1e-9  # float rounding allowance on every comparison of an emitted value with a bound


def disc_rise_ratio(
    temps: np.ndarray, ambient: float, window: int, min_rear_rise: float
) -> np.ndarray:
    """Front/rear ratio of the axle mean disc temperature rises, one value per window.

    `temps` holds one row per sample, four columns in WHEELS order (fl, fr, rl, rr), degC. The
    samples are cut into consecutive windows of `window` samples (a trailing partial window is
    dropped). Per window: the mean over the window of the front axle mean rise above `ambient`,
    over the same for the rear axle; NaN where the rear value is not above `min_rear_rise` (K).
    """
    if window < 1:
        raise ValueError("the window must hold at least one sample")
    count = len(temps) // window
    rise = np.asarray(temps)[: count * window] - ambient
    front = rise[:, :2].mean(axis=1).reshape(count, window).mean(axis=1)
    rear = rise[:, 2:].mean(axis=1).reshape(count, window).mean(axis=1)
    evaluated = rear > min_rear_rise
    return np.where(evaluated, front / np.where(evaluated, rear, 1.0), np.nan)


@dataclass(frozen=True)
class Anomaly:
    rule: str
    vehicle_id: str
    seq: int
    window_start: str
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.rule}: {self.vehicle_id} seq {self.seq} window_start {self.window_start}: "
            f"{self.detail}"
        )


class AnomalyFound(Exception):
    """The first anomaly of a message, with its rule, vehicle, seq and window_start."""

    def __init__(self, anomaly: Anomaly):
        super().__init__(str(anomaly))
        self.anomaly = anomaly


@dataclass(frozen=True)
class StreamBounds:
    """The bounds of the stream rules on emitted values, derived from plausibility.yaml, the
    catalogue resolutions and the sensor noise (module docstring)."""

    spread_rel: float
    spread_abs: float  # K
    disc_error: float  # K, a disc reading (a sample: min, max, last)
    disc_mean_error: float  # K, a disc window mean
    ambient_error: float  # K, an ambient reading; a mean over a varying window adds half a step
    ambient_resolution: float  # K
    ratio_min: float
    ratio_max: float
    ratio_windows: int  # whole 10 s windows per ratio evaluation
    min_rear_rise: float  # K
    coolant_step_max: float  # K, between the means of consecutive full windows
    coolant_band_limit: float  # degC, on a window max

    @classmethod
    def from_configs(
        cls, plausibility: Config, catalogue: Catalogue, sensors: SensorParameters
    ) -> "StreamBounds":
        disc_error = max(error_bound(name, sensors) for name in DISCS)
        disc_step = max(_resolution(catalogue.signal(name)) for name in DISCS)
        coolant_error = error_bound(COOLANT, sensors)
        coolant_mean_error = coolant_error + _resolution(catalogue.signal(COOLANT)) / 2
        return cls(
            spread_rel=plausibility.si("disc_lr_spread_rel_max"),
            spread_abs=plausibility.si("disc_lr_spread_abs_max"),
            disc_error=disc_error,
            disc_mean_error=disc_error + disc_step / 2,
            ambient_error=error_bound(AMBIENT, sensors),
            ambient_resolution=_resolution(catalogue.signal(AMBIENT)),
            ratio_min=plausibility.si("disc_front_rear_ratio_min"),
            ratio_max=plausibility.si("disc_front_rear_ratio_max"),
            ratio_windows=round(plausibility.si("disc_ratio_window") / AGGREGATION_WINDOW),
            min_rear_rise=plausibility.si("disc_ratio_min_rear_rise"),
            coolant_step_max=(
                plausibility.si("coolant_rate_max") * AGGREGATION_WINDOW + 2 * coolant_mean_error
            ),
            coolant_band_limit=plausibility.value("coolant_band_max") + coolant_error,
        )

    def ratio_bounds(self, rear_rise: float, error: float) -> tuple[float, float]:
        """The front/rear ratio bounds widened for a measured rear rise and an axle mean rise
        error: [r_min - d (1 + r_min) / R, r_max + d (1 + r_max) / R]."""
        return (
            self.ratio_min - error * (1 + self.ratio_min) / rear_rise,
            self.ratio_max + error * (1 + self.ratio_max) / rear_rise,
        )


def _resolution(signal: Signal) -> float:
    assert signal.resolution is not None
    return signal.resolution


@dataclass
class _Vehicle:
    highest_seq: int = -1
    coolant: tuple[int, float] | None = None  # start (ms) and coolant mean of the last full window
    # Consecutive full windows: start (ms), the four disc mean rises above the ambient mean (in
    # DISCS order), and the error of that ambient mean.
    run: list[tuple[int, tuple[float, ...], float]] = field(default_factory=list)


class StreamChecker:
    """The zero-anomaly check of a telemetry stream: `check(message)` on each parsed message in
    delivery order raises AnomalyFound on the first anomaly. `evaluated` counts the evaluations
    of each rule group, so a test can see that a rule was live."""

    def __init__(self, bounds: StreamBounds, catalogue: Catalogue):
        self.bounds = bounds
        self.numeric = tuple(s for s in catalogue.signals if s.is_number)
        self.boolean = tuple(s.name for s in catalogue.signals if not s.is_number)
        self.vehicles: dict[str, _Vehicle] = {}
        self.evaluated: Counter[str] = Counter()

    @classmethod
    def from_repository(cls) -> "StreamChecker":
        from simulator.catalogue import catalogue
        from simulator.config_files import plausibility, sensors

        cat = catalogue()
        bounds = StreamBounds.from_configs(
            plausibility(), cat, SensorParameters.from_configs(sensors(), cat)
        )
        return cls(bounds, cat)

    def check(self, message: dict[str, Any]) -> None:
        found = self.anomalies(message)
        if found:
            raise AnomalyFound(found[0])

    def anomalies(self, message: dict[str, Any]) -> list[Anomaly]:
        """Every anomaly of one message (and the vehicle's state updated with it)."""
        if message["message_type"] != "periodic":
            return []
        found: list[Anomaly] = []

        def report(rule: str, detail: str) -> None:
            found.append(
                Anomaly(
                    rule, message["vehicle_id"], message["seq"], message["window_start"], detail
                )
            )

        signals = message["signals"]
        samples = round(message["window_s"] * STEPS_PER_SECOND)
        self._catalogue(signals, samples, report)
        self._spread(signals, report)
        self._coolant_band(signals, report)
        vehicle = self.vehicles.setdefault(message["vehicle_id"], _Vehicle())
        if message["seq"] > vehicle.highest_seq:
            vehicle.highest_seq = message["seq"]
            full = samples == AGGREGATION_WINDOW * STEPS_PER_SECOND
            start = _milliseconds(message["window_start"])
            self._coolant_rate(vehicle, signals, start, full, report)
            self._ratio(vehicle, signals, start, full, report)
        return found

    def _catalogue(self, signals: dict, samples: int, report) -> None:
        self.evaluated["catalogue"] += 1
        for signal in self.numeric:
            low, high = signal.range
            resolution = _resolution(signal)
            stats = signals[signal.name]
            for stat in STATISTICS:
                value = stats[stat]
                what = f"{signal.name} {stat} {value}"
                if value == signal.invalid:
                    report("catalogue_invalid", f"{what} is the invalid value")
                    continue
                if not low - SLACK <= value <= high + SLACK:
                    report("catalogue_range", f"{what} outside [{low}, {high}]")
                index = (value - signal.offset) / resolution
                if abs(index - round(index)) > 1e-6:
                    report("catalogue_grid", f"{what} off its {resolution} grid")
        most = math.ceil(samples / 2)
        for name in self.boolean:
            count = signals[name]["count"]
            if count > most:
                report("edge_count", f"{name} count {count} > {most} in {samples} samples")

    def _spread(self, signals: dict, report) -> None:
        b = self.bounds
        ambient = signals[AMBIENT]
        rise_error = b.disc_error + b.ambient_error
        for left, right in ((DISCS[0], DISCS[1]), (DISCS[2], DISCS[3])):
            a, c = signals[left], signals[right]
            self.evaluated["disc_lr_spread"] += 1
            rise = max(a["last"], c["last"]) - ambient["last"]
            bound = b.spread_rel * (rise + rise_error) + b.spread_abs + 2 * b.disc_error
            spread = abs(a["last"] - c["last"])
            if spread > bound + SLACK:
                report("disc_lr_spread_last", f"{left}/{right} last {spread} K > {bound:.4f} K")
            rise = max(a["max"], c["max"]) - ambient["min"]
            bound = b.spread_rel * (rise + rise_error) + b.spread_abs + 2 * b.disc_mean_error
            spread = abs(a["mean"] - c["mean"])
            if spread > bound + SLACK:
                report("disc_lr_spread_mean", f"{left}/{right} mean {spread} K > {bound:.4f} K")

    def _coolant_band(self, signals: dict, report) -> None:
        self.evaluated["coolant_band"] += 1
        top = signals[COOLANT]["max"]
        if top > self.bounds.coolant_band_limit + SLACK:
            report("coolant_band", f"max {top} degC > {self.bounds.coolant_band_limit} degC")

    def _coolant_rate(self, vehicle: _Vehicle, signals: dict, start: int, full: bool, report):
        mean = signals[COOLANT]["mean"]
        previous = vehicle.coolant
        if full and previous is not None and start == previous[0] + WINDOW_MS:
            self.evaluated["coolant_rate"] += 1
            step = abs(mean - previous[1])
            if step > self.bounds.coolant_step_max + SLACK:
                report(
                    "coolant_rate",
                    f"mean moved {step} K in {AGGREGATION_WINDOW} s > "
                    f"{self.bounds.coolant_step_max:.4f} K",
                )
        vehicle.coolant = (start, mean) if full else None

    def _ratio(self, vehicle: _Vehicle, signals: dict, start: int, full: bool, report) -> None:
        b = self.bounds
        if not full:
            vehicle.run = []
            return
        if vehicle.run and vehicle.run[-1][0] + WINDOW_MS != start:
            vehicle.run = []
        ambient = signals[AMBIENT]
        constant = ambient["min"] == ambient["max"]
        ambient_error = b.ambient_error + (0.0 if constant else b.ambient_resolution / 2)
        rises = tuple(signals[name]["mean"] - ambient["mean"] for name in DISCS)
        vehicle.run.append((start, rises, ambient_error))
        if len(vehicle.run) < b.ratio_windows:
            return
        run, vehicle.run = vehicle.run, []
        temps = np.array([rises for _, rises, _ in run])
        error = b.disc_mean_error + max(e for _, _, e in run)
        # The bounds hold once the TRUE rear rise exceeds the threshold: evaluate only when the
        # measured rise minus its error does (R - d > threshold).
        ratio = disc_rise_ratio(temps, 0.0, b.ratio_windows, b.min_rear_rise + error)[0]
        if np.isnan(ratio):
            return
        self.evaluated["disc_front_rear_ratio"] += 1
        rear = float(temps[:, 2:].mean())
        low, high = b.ratio_bounds(rear, error)
        if not low - SLACK <= ratio <= high + SLACK:
            report(
                "disc_front_rear_ratio",
                f"ratio {ratio:.4f} outside [{low:.4f}, {high:.4f}] at rear rise {rear:.2f} K "
                f"over the {len(run)} windows from {_iso(run[0][0])}",
            )


def _milliseconds(stamp: str) -> int:
    return round(datetime.datetime.fromisoformat(stamp).timestamp() * 1000)


def _iso(milliseconds: int) -> str:
    moment = datetime.datetime.fromtimestamp(milliseconds / 1000, datetime.UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
