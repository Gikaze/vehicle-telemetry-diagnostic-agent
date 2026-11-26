"""NFR-15 (provisional definition): the zero-anomaly check of a telemetry stream.

Two kinds of tests:

- hand-made messages sweep each rule across its bound on the catalogue grid; the verdict of the
  checker equals, at every value, the test's own formula, written here from plausibility.yaml,
  the catalogue resolutions and sensors.yaml (never from StreamBounds);
- the checker fed by the stream (stream drivers, JSON bytes, json.loads, in delivery order): a
  normal run has no anomaly and every rule is evaluated; the same run with transport defects has
  none either; a fault injected into the stream's messages is reported, with the message's
  vehicle, seq and window_start.
"""

import copy
import datetime
import json
import math
from collections.abc import Callable
from typing import Any

import pytest
import yaml
from driven import short_plans

from simulator import stream
from simulator.aggregator import Record
from simulator.catalogue import catalogue
from simulator.clock import StepClock
from simulator.config_files import load_plausibility, load_sensors
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.params import ConfigError, config_path
from simulator.plausibility import AnomalyFound, StreamChecker
from simulator.serialise import Serialiser
from simulator.transport import Transport, TransportRates

FL, FR, RL, RR = (f"brake_disc_temp_est_{w}" for w in ("fl", "fr", "rl", "rr"))
AMBIENT, COOLANT = "ambient_temp", "coolant_temp"

# The rule constants, by hand from the configuration files.
P = load_plausibility()
REL, ABS = P.si("disc_lr_spread_rel_max"), P.si("disc_lr_spread_abs_max")
R_MIN, R_MAX = P.si("disc_front_rear_ratio_min"), P.si("disc_front_rear_ratio_max")
MIN_REAR = P.si("disc_ratio_min_rear_rise")
WINDOWS = round(P.si("disc_ratio_window") / 10)  # 10 s telemetry windows (spec section 5)
DISC_RES = catalogue().signal(FL).resolution
AMBIENT_RES = catalogue().signal(AMBIENT).resolution
COOLANT_RES = catalogue().signal(COOLANT).resolution
S = load_sensors()
COOLANT_ERROR = S.value("noise_clip_sigma") * S.value("coolant_temp") + COOLANT_RES / 2
# Disc estimates and the ambient are noise-free: a sample is off by half a resolution; a window
# mean by one resolution (sample plus mean rounding); a constant ambient window mean is a sample.
DISC_ERROR, DISC_MEAN_ERROR, AMBIENT_ERROR = DISC_RES / 2, DISC_RES, AMBIENT_RES / 2


def test_the_ratio_window_is_six_telemetry_windows() -> None:
    assert P.value("disc_ratio_window") == 60 and WINDOWS == 6
    assert StreamChecker.from_repository().bounds.ratio_windows == WINDOWS


@pytest.mark.parametrize("seconds", [65, 5, 0])
def test_a_ratio_window_of_no_whole_telemetry_windows_is_refused(tmp_path, seconds) -> None:
    document = yaml.safe_load(config_path("plausibility.yaml").read_text())
    document["disc_temperature"]["disc_ratio_window"]["value"] = seconds
    path = tmp_path / "plausibility.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigError, match="whole number of 10 s telemetry windows"):
        load_plausibility(path)


# --- hand-made messages -----------------------------------------------------------------------

SIM = FleetSimulation(42, load_fleet(size=2), short_plans([600.0], []))
SERIALISER = Serialiser(catalogue(), StepClock(datetime.date(2026, 1, 5)))


@pytest.fixture(scope="module")
def records() -> list[Record]:
    """2 vans, one 600 s trip: the shortest plan found whose rear discs pass the ratio
    threshold (4 evaluations)."""
    return list(stream.records(SIM, 1))


def documents(records: list[Record], rates: TransportRates) -> list[dict[str, Any]]:
    transport = Transport(SIM.root_seed, SIM.vehicle_ids, rates)
    return [json.loads(SERIALISER.encode(m).data) for r in records for m in transport.send([r])]


@pytest.fixture(scope="module")
def normal(records) -> list[dict[str, Any]]:
    return documents(records, TransportRates())


@pytest.fixture(scope="module")
def template(normal) -> dict[str, Any]:
    """A full periodic window of the run, ambient and coolant set constant."""
    doc = copy.deepcopy(next(d for d in normal if d["window_s"] == 10.0))
    for name, value in ((AMBIENT, 10.0), (COOLANT, 85.0)):
        doc["signals"][name] = constant(value)
    return doc


def constant(value: float) -> dict[str, float]:
    return {"mean": value, "min": value, "max": value, "last": value}


BASE = datetime.datetime(2026, 1, 5, 8, 0, tzinfo=datetime.UTC)


def message(template: dict, index: int, signals: dict[str, dict], seconds: float = 10.0) -> dict:
    """The template as window `index` of a vehicle (seq index, 10 s apart), signals replaced."""
    doc = copy.deepcopy(template)
    doc["seq"] = index
    start = BASE + datetime.timedelta(seconds=10 * index)
    doc["window_start"] = start.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    doc["window_s"] = seconds
    doc["signals"].update(copy.deepcopy(signals))
    return doc


def rules(checker: StreamChecker, doc: dict) -> set[str]:
    return {a.rule for a in checker.anomalies(doc)}


def grid(low: float, high: float, step: float) -> list[float]:
    return [round(low + k * step, 6) for k in range(round((high - low) / step) + 1)]


def test_the_left_right_spread_on_last_follows_its_bound(template) -> None:
    """Right disc at 60 degC, ambient 10: the left one swept over the grid on `last` only."""
    right = 60.0
    for left in grid(40.0, 80.0, DISC_RES):
        rise = max(left, right) - 10.0
        bound = REL * (rise + DISC_ERROR + AMBIENT_ERROR) + ABS + 2 * DISC_ERROR
        stats = {"mean": right, "min": min(left, right), "max": max(left, right), "last": left}
        doc = message(template, 0, {FL: stats, FR: constant(right)})
        flagged = "disc_lr_spread_last" in rules(StreamChecker.from_repository(), doc)
        assert flagged == (abs(left - right) > bound), left


def test_the_left_right_spread_on_mean_follows_its_bound(template) -> None:
    """Right disc at 60 degC: the left one's mean (and max) swept, `last` kept equal."""
    right = 60.0
    for left in grid(40.0, 80.0, DISC_RES):
        rise = max(left, right) - 10.0  # from the window max
        bound = REL * (rise + DISC_ERROR + AMBIENT_ERROR) + ABS + 2 * DISC_MEAN_ERROR
        stats = {"mean": left, "min": min(left, right), "max": max(left, right), "last": right}
        doc = message(template, 0, {FL: stats, FR: constant(right)})
        flagged = "disc_lr_spread_mean" in rules(StreamChecker.from_repository(), doc)
        assert flagged == (abs(left - right) > bound), left


def ratio_signals(front: tuple[float, float], rear: tuple[float, float]) -> dict[str, dict]:
    """Constant windows: the two front and the two rear discs, ambient 10 degC."""
    return {
        FL: constant(front[0]),
        FR: constant(front[1]),
        RL: constant(rear[0]),
        RR: constant(rear[1]),
        AMBIENT: constant(10.0),
    }


def ratio_run(template: dict, front: tuple[float, float], rear: tuple[float, float]) -> list[dict]:
    return [message(template, i, ratio_signals(front, rear)) for i in range(WINDOWS)]


# The error of an axle mean rise on window means with a constant ambient, and the measured rear
# rise above which the true one surely exceeds the threshold.
D = DISC_MEAN_ERROR + AMBIENT_ERROR
START = MIN_REAR + D


def ratio_flagged(docs: list[dict]) -> tuple[bool, int]:
    checker = StreamChecker.from_repository()
    found = set()
    for doc in docs:
        found |= rules(checker, doc)
    return "disc_front_rear_ratio" in found, checker.evaluated["disc_front_rear_ratio"]


def test_the_front_rear_ratio_follows_its_widened_bounds(template) -> None:
    """Rear discs 22 K above a constant ambient (the first whole rise above START), the front
    axle mean swept over its 0.5 K grid: the verdict follows
    r_min (R - d) - d <= F <= r_max (R + d) + d, d = one disc resolution plus half an ambient
    resolution. Both edges are crossed by the sweep."""
    rear_rise = math.floor(START / DISC_RES + 1) * DISC_RES
    assert rear_rise - D > MIN_REAR
    low, high = R_MIN * (rear_rise - D) - D, R_MAX * (rear_rise + D) + D
    rear = (10.0 + rear_rise, 10.0 + rear_rise)
    seen = set()
    for left in grid(35.0, 60.0, DISC_RES):
        for right in (left, left + DISC_RES):
            front_rise = (left + right) / 2 - 10.0
            flagged, evaluated = ratio_flagged(ratio_run(template, (left, right), rear))
            assert evaluated == 1
            expected = not (low <= front_rise <= high)
            assert flagged == expected, (left, right)
            seen.add(expected)
    assert seen == {True, False}


def test_the_ratio_is_evaluated_only_once_the_true_rear_rise_passes_the_threshold(
    template,
) -> None:
    """Measured rear rises on the 0.5 K grid of an axle mean, from one step below the threshold
    to two above START: evaluated exactly when R - d > threshold; a measured R between the
    threshold and START (the true rise may still be below it) is not evaluated, even with the
    front out of any bound."""
    seen = set()
    for rise in grid(MIN_REAR - DISC_RES, START + DISC_RES, DISC_RES / 2):
        rear = (10.0 + math.floor(rise), 10.0 + math.floor(rise) + 2 * (rise % 1))
        flagged, evaluated = ratio_flagged(ratio_run(template, (90.0, 90.0), rear))
        expected = rise - D > MIN_REAR
        assert (flagged, evaluated) == (expected, int(expected)), rise
        seen.add((MIN_REAR < rise <= START, expected))
    assert (True, False) in seen  # a rise above the threshold yet not evaluated


def test_the_ratio_needs_whole_consecutive_full_windows(template) -> None:
    """Out of bounds everywhere (front rise 80 K, rear 25 K), yet never evaluated when a window
    is short, when a window is missing (a 20 s jump), or with one window too few."""
    run = ratio_run(template, (90.0, 90.0), (35.0, 35.0))
    assert ratio_flagged(run) == (True, 1)
    short = [*run[:3], {**run[3], "window_s": 9.9}, *run[4:]]
    assert ratio_flagged(short)[1] == 0
    jump = [message(template, i + (i >= 3), run[i]["signals"]) for i in range(WINDOWS)]
    assert ratio_flagged(jump)[1] == 0
    assert ratio_flagged(run[:-1])[1] == 0


def test_the_coolant_rate_follows_its_bound(template) -> None:
    """Two consecutive full windows; the second mean swept from the first on the 1 K grid."""
    first = 60.0
    bound = P.si("coolant_rate_max") * 10 + 2 * (COOLANT_ERROR + COOLANT_RES / 2)
    for second in grid(45.0, 75.0, COOLANT_RES):
        checker = StreamChecker.from_repository()
        checker.check(message(template, 0, {COOLANT: constant(first)}))
        found = rules(checker, message(template, 1, {COOLANT: constant(second)}))
        assert checker.evaluated["coolant_rate"] == 1
        assert ("coolant_rate" in found) == (abs(second - first) > bound), second


def test_the_coolant_rate_needs_consecutive_full_windows(template) -> None:
    far = constant(85.0 - 30 * COOLANT_RES)
    for second in (
        message(template, 2, {COOLANT: far}),  # 20 s later: a lost message
        message(template, 1, {COOLANT: far}, seconds=5.0),  # a trip's short last window
    ):
        checker = StreamChecker.from_repository()
        checker.check(message(template, 0, {}))
        assert "coolant_rate" not in rules(checker, second)
        assert checker.evaluated["coolant_rate"] == 0


def test_a_late_window_keeps_the_runs_of_the_windows_around_it(template) -> None:
    """Windows delivered 1, 3, 2, 4, 5, 6, 7, 8 (window i: seq i, starting 10 i s after BASE).
    Window 2 comes after 3, below the highest seq so far: it gets the per-message rules only, so
    3 -> 4 stays consecutive. Coolant rate: 3->4 after the first four, then 4->5 ... 7->8, 5 in
    all (were 2 taken as the previous window, 4 would follow nothing: 0, then 4 in all). Ratio:
    the run 3, 4, ..., 8 holds 6 windows, 1 evaluation (else 4 ... 8, 5 windows, none)."""
    signals = ratio_signals((54.0, 54.0), (35.0, 35.0))  # F 44 K, R 25 K: ratio 1.76
    checker = StreamChecker.from_repository()
    for count, index in enumerate([1, 3, 2, 4, 5, 6, 7, 8], start=1):
        checker.check(message(template, index, signals))
        if count == 4:
            assert checker.evaluated["coolant_rate"] == 1
    assert checker.evaluated["coolant_rate"] == 5
    assert checker.evaluated["disc_front_rear_ratio"] == 1


def test_the_coolant_band_follows_its_bound(template) -> None:
    limit = P.value("coolant_band_max") + COOLANT_ERROR
    for top in grid(90.0, 100.0, COOLANT_RES):
        stats = {"mean": 90.0, "min": 90.0, "max": top, "last": 90.0}
        found = rules(StreamChecker.from_repository(), message(template, 0, {COOLANT: stats}))
        assert ("coolant_band" in found) == (top > limit), top


def test_the_catalogue_rules(template) -> None:
    """The pedal position (0 to 100 %, 0.1 % steps, invalid 102.3): the range edges pass, one
    step beyond fails, half a step is off the grid, the invalid value is reported as such."""
    name = "accel_pedal_position"
    pedal = catalogue().signal(name)
    low, high = pedal.range
    cases: list[tuple[dict, str | None]] = [
        (constant(high), None),
        (constant(low), None),
        (constant(round(high + pedal.resolution, 6)), "catalogue_range"),
        (constant(round(low - pedal.resolution, 6)), "catalogue_range"),
        (constant(round(low + pedal.resolution / 2, 6)), "catalogue_grid"),
        (constant(pedal.invalid), "catalogue_invalid"),
    ]
    for stats, rule in cases:
        found = rules(StreamChecker.from_repository(), message(template, 0, {name: stats}))
        assert found == ({rule} if rule else set()), stats


def test_the_rising_edge_count_is_at_most_half_the_samples(template) -> None:
    for seconds in (10.0, 0.1, 0.5, 3.7):
        samples = round(seconds * 10)
        for count in range(samples + 2):
            flags = {"abs_active": {"last": True, "count": count}}
            doc = message(template, 0, flags, seconds=seconds)
            flagged = "edge_count" in rules(StreamChecker.from_repository(), doc)
            assert flagged == (count > math.ceil(samples / 2)), (seconds, count)


def test_event_messages_are_skipped() -> None:
    checker = StreamChecker.from_repository()
    event = {"message_type": "event", "vehicle_id": "VAN-001", "seq": 0}
    assert checker.anomalies(event) == []
    assert checker.evaluated == {}


# --- the checker fed by the stream ------------------------------------------------------------


def feed(checker: StreamChecker, docs: list[dict]) -> None:
    for doc in docs:
        try:
            checker.check(doc)
        except AnomalyFound as found:
            assert (found.anomaly.vehicle_id, found.anomaly.seq) == (doc["vehicle_id"], doc["seq"])
            assert found.anomaly.window_start == doc["window_start"]
            raise


def consecutive_full_pairs(docs: list[dict]) -> int:
    """Per vehicle, messages following a full window of the same vehicle 10 s earlier."""
    previous: dict[str, tuple[datetime.datetime, float]] = {}
    pairs = 0
    for doc in docs:
        start = datetime.datetime.fromisoformat(doc["window_start"])
        last = previous.get(doc["vehicle_id"])
        if doc["window_s"] == 10.0 and last and last[1] == 10.0:
            pairs += start - last[0] == datetime.timedelta(seconds=10)
        previous[doc["vehicle_id"]] = (start, doc["window_s"])
    return pairs


def test_a_normal_stream_has_no_anomaly_and_every_rule_is_live(normal) -> None:
    checker = StreamChecker.from_repository()
    feed(checker, normal)
    n = len(normal)
    assert {d["vehicle_id"] for d in normal} == {"VAN-001", "VAN-002"}
    assert checker.evaluated["catalogue"] == n
    assert checker.evaluated["disc_lr_spread"] == 2 * n
    assert checker.evaluated["coolant_band"] == n
    assert checker.evaluated["coolant_rate"] == consecutive_full_pairs(normal) > 0
    assert checker.evaluated["disc_front_rear_ratio"] > 0


def test_a_stream_with_delivery_defects_has_no_anomaly(records) -> None:
    docs = documents(records, TransportRates(gap=0.2, duplicate=0.2, reorder=0.2))
    seqs = [(d["vehicle_id"], d["seq"]) for d in docs]
    assert len(seqs) > len(set(seqs))  # duplicates delivered
    checker = StreamChecker.from_repository()
    feed(checker, docs)
    assert checker.evaluated["catalogue"] == len(docs)


def shifted(docs: list[dict], vehicle: str, change: Callable[[dict], None]) -> list[dict]:
    out = copy.deepcopy(docs)
    for doc in out:
        if doc["vehicle_id"] == vehicle:
            change(doc["signals"])
    return out


def test_one_disc_shifted_in_the_stream_is_reported(normal) -> None:
    """The left/right allowance at zero rise is abs + two mean errors (4 K on means, 3 K on
    `last`): a 3 K shift can stay inside it. One resolution more than the mean allowance is
    reported on the first window of the trip."""
    shift = ABS + 2 * DISC_MEAN_ERROR + DISC_RES

    def hotter(signals: dict) -> None:
        for stat in ("mean", "min", "max", "last"):
            signals[FL][stat] += shift

    with pytest.raises(AnomalyFound) as found:
        feed(StreamChecker.from_repository(), shifted(normal, "VAN-002", hotter))
    assert found.value.anomaly.rule.startswith("disc_lr_spread")
    assert found.value.anomaly.vehicle_id == "VAN-002"


def test_a_front_axle_hotter_than_the_ratio_allows_is_reported(normal) -> None:
    """Front rises scaled by 1.3 (rounded to the grid): ratio about 2.3, beyond the widened
    upper bound at any rear rise above the threshold."""

    def front_hotter(signals: dict) -> None:
        ambient = signals[AMBIENT]["mean"]
        for name in (FL, FR):
            for stat in ("mean", "min", "max", "last"):
                rise = signals[name][stat] - ambient
                signals[name][stat] = round(ambient + 1.3 * rise / DISC_RES) * DISC_RES

    checker = StreamChecker.from_repository()
    found = set()
    for doc in shifted(normal, "VAN-001", front_hotter):
        found |= {a.rule for a in checker.anomalies(doc)}
    assert "disc_front_rear_ratio" in found


def test_a_coolant_jump_in_the_stream_is_reported(normal) -> None:
    """From the 20th message of VAN-001 on, its coolant reads lower by more than the step bound;
    the anomaly is reported at that message."""
    vehicle = [d for d in normal if d["vehicle_id"] == "VAN-001"]
    target = vehicle[20]
    drop = math.ceil((P.si("coolant_rate_max") * 10 + 2 * (COOLANT_ERROR + COOLANT_RES / 2)) + 1)
    docs = copy.deepcopy(normal)
    after = False
    for doc in docs:
        if doc["vehicle_id"] == "VAN-001":
            after = after or doc["seq"] == target["seq"]
            if after:
                for stat in ("mean", "min", "max", "last"):
                    doc["signals"][COOLANT][stat] -= drop
    with pytest.raises(AnomalyFound) as found:
        feed(StreamChecker.from_repository(), docs)
    assert found.value.anomaly.rule == "coolant_rate"
    assert (found.value.anomaly.seq, found.value.anomaly.window_start) == (
        target["seq"],
        target["window_start"],
    )
