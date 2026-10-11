"""Road of a trip (FR-02, NFR-15): structure, curve geometry, event rates and independence."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from simulator.profile.road import Road, StopKind, heading_change, ramps_heading
from simulator.profile_config import CURVES, drive_profile

seeds = st.integers(min_value=0, max_value=2**32 - 1)
trips = st.integers(min_value=0, max_value=5)
LENGTH = 30_000  # m of road per generated example


def road(seed: int = 42, trip: int = 2, vehicle: str = "VAN-001", day: int = 0) -> Road:
    return Road(seed, vehicle, day, trip, drive_profile())


def snapshot(r: Road) -> list:
    return [(plan.segment, plan.curves, plan.stops, plan.gaps) for plan in r.plans]


# Independence -----------------------------------------------------------------------------------


def test_lookahead_does_not_change_the_road() -> None:
    """Generating trip 2 in 100 m steps or in one 30 km step gives the same road."""
    stepwise = road()
    for distance in np.arange(100, LENGTH, 100):
        stepwise.ensure(float(distance))
    at_once = road()
    at_once.ensure(stepwise.generated_until)
    assert snapshot(stepwise) == snapshot(at_once)


def test_trip_road_depends_only_on_seed_vehicle_day_and_trip() -> None:
    """Consuming trip 1's road further does not move trip 2's road."""
    trip_1 = road(trip=1)
    trip_1.ensure(LENGTH)
    alone = road(trip=2)
    alone.ensure(LENGTH)
    trip_1.ensure(5 * LENGTH)
    after = road(trip=2)
    after.ensure(LENGTH)
    assert snapshot(alone) == snapshot(after)
    other = road(trip=3)
    other.ensure(LENGTH)
    assert snapshot(other) != snapshot(alone)
    other_vehicle = road(vehicle="VAN-002")
    other_vehicle.ensure(LENGTH)
    assert snapshot(other_vehicle) != snapshot(alone)


# Structure --------------------------------------------------------------------------------------


@given(seed=seeds, trip=trips)
def test_segments_are_contiguous_with_nominal_lengths(seed: int, trip: int) -> None:
    r = road(seed, trip)
    r.ensure(LENGTH)
    profile = drive_profile()
    position = 0.0
    for plan in r.plans:
        segment = plan.segment
        assert segment.start == pytest.approx(position)
        assert segment.length == pytest.approx(segment.target_speed * segment.nominal_duration)
        kind = segment.kind
        assert profile.si(f"{kind}_target_speed_min") <= segment.target_speed
        assert segment.target_speed <= profile.si(f"{kind}_target_speed_max")
        position = segment.end


@given(seed=seeds, trip=trips)
def test_events_stay_in_their_segment_and_never_overlap(seed: int, trip: int) -> None:
    r = road(seed, trip)
    r.ensure(LENGTH)
    for plan in r.plans:
        segment = plan.segment
        intervals = [(c.start, c.end) for c in plan.curves]
        for start, end in intervals:
            assert segment.start <= start < end <= segment.end
        for (_, end_a), (start_b, _) in zip(intervals, intervals[1:], strict=False):
            assert end_a <= start_b
        for stop in plan.stops:
            assert segment.start <= stop.position <= segment.end
            assert not any(start < stop.position < end for start, end in intervals)


@given(seed=seeds, trip=trips)
def test_each_curve_turns_exactly_its_angle(seed: int, trip: int) -> None:
    """Integrated curvature equals the angle; the ramps alone never exceed it."""
    r = road(seed, trip)
    r.ensure(LENGTH)
    for curve in r.curves:
        assert abs(heading_change(curve)) == pytest.approx(curve.angle, rel=1e-6)
        assert ramps_heading(curve) <= curve.angle + 1e-12
        assert np.sign(heading_change(curve)) == curve.direction


@given(seed=seeds, trip=trips)
def test_planned_curve_speed_keeps_lateral_acceleration_bounded(seed: int, trip: int) -> None:
    profile = drive_profile()
    r = road(seed, trip)
    r.ensure(LENGTH)
    for curve in r.curves:
        assert curve.speed**2 / curve.radius <= profile.si("curve_lateral_accel") + 1e-9


def test_review_case_junction_ramp_is_shortened() -> None:
    """A 10 m, 90 deg junction at 4.47 m/s with a 4 s ramp would turn 102 deg on its ramps."""
    for curve in (c for seed in range(20) for c in _curves(seed) if c.radius < 11):
        assert ramps_heading(curve) <= math.pi / 2 + 1e-12


def _curves(seed: int):
    r = road(seed)
    r.ensure(LENGTH)
    return r.curves


# Final stop ---------------------------------------------------------------------------------------


@given(seed=seeds, position=st.floats(min_value=50.0, max_value=20_000.0))
def test_final_stop_truncates_the_road(seed: int, position: float) -> None:
    r = road(seed)
    r.ensure(position)
    in_curve = [c for c in r.curves if c.start < position < c.end]
    final = r.end_at(position)
    expected = in_curve[0].end if in_curve else position
    assert final.position == pytest.approx(expected)
    assert final.kind is StopKind.FINAL
    assert r.plans[-1].segment.end == pytest.approx(final.position)
    assert all(c.end <= final.position for c in r.curves)
    assert all(s.position <= final.position for s in r.stops)
    assert r.stops[-1] is final
    r.ensure(10 * position)  # no generation after the end
    assert r.plans[-1].segment.end == pytest.approx(final.position)


# Event rates --------------------------------------------------------------------------------------


def configured_rates(kind: str) -> tuple[float, float]:
    """(curve, traffic stop) rates per metre of free road, read from the profile in the test."""
    profile = drive_profile()
    curve_rate = profile.si(CURVES[kind][4])
    stop_rate = profile.si("urban_traffic_stops_per_km") if kind == "urban" else 0.0
    return curve_rate, stop_rate


def _long_roads(count: int = 60):
    for seed in range(count):
        r = road(seed)
        r.ensure(200_000)
        yield r


def test_drawn_gaps_follow_the_configured_rate() -> None:
    """Mean drawn gap per segment kind is 1 / rate within 4 standard errors (exponential)."""
    gaps: dict[str, list[float]] = {"urban": [], "rural": [], "downhill": []}
    for r in _long_roads():
        for plan in r.plans:
            gaps[plan.segment.kind].extend(plan.gaps)
    for kind, values in gaps.items():
        curve_rate, stop_rate = configured_rates(kind)
        mean = 1.0 / (curve_rate + stop_rate)
        assert len(values) > 500
        assert np.mean(values) == pytest.approx(mean, abs=4 * mean / math.sqrt(len(values)))


def test_events_per_km_of_road_are_below_the_free_road_rate() -> None:
    """Events take road length and overflowing curves are dropped (documented convention)."""
    for kind in ("rural", "downhill"):
        length = events = 0.0
        for r in _long_roads(20):
            for plan in r.plans:
                if plan.segment.kind == kind:
                    length += plan.segment.length
                    events += len(plan.curves)
        curve_rate, _ = configured_rates(kind)
        assert events / length < curve_rate


def test_segment_kinds_follow_their_weights() -> None:
    profile = drive_profile()
    kinds = [plan.segment.kind for r in _long_roads() for plan in r.plans]
    for kind in ("urban", "rural", "downhill"):
        share = kinds.count(kind) / len(kinds)
        weight = profile.value(f"weight_{kind}")
        assert share == pytest.approx(weight, abs=4 * math.sqrt(weight * (1 - weight) / len(kinds)))
