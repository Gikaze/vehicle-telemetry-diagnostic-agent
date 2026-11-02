"""Road of one trip, in distance: segments, curves and stops (docs/simulator-spec.md, section 2).

A trip's road is drawn from the vehicle's PROFILE stream of that day and trip, one segment at a
time, in a fixed order that is part of the reproducibility contract. Generation is lazy (segments
are added as the van approaches them) but sequential, so the road never depends on how far ahead
it was generated, nor on any other trip.

Per segment the draws are: kind (by weight), target speed, nominal duration, grade; then the event
process; then, for urban segments, the delivery stop draws. The segment length is target speed
times nominal duration. Events follow a Poisson process in distance: each gap from the end of the
previous event is event_min_spacing plus an exponential draw at the per-km-of-free-road rate; an
event closer than event_min_spacing to the segment end ends the segment's events, so the delivery
stop at the segment end is also spaced from the last event. Urban turns and traffic stops form
one merged process. A curve that would overflow its segment is dropped (its draws are still
consumed), so events never cross a segment boundary, never overlap, and stops never fall inside
a curve.

A curve has entry and exit ramps where curvature changes linearly, and a constant-curvature arc.
The ramp time is drawn, then shortened so that both ramps together turn no more than the curve
angle (profile_config.curve_ramp_time); the arc covers the rest, so the heading changes by
exactly the angle.
"""

from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np

from simulator.params import Config
from simulator.profile_config import CURVES, SEGMENT_TYPES, curve_ramp_time, curve_speed
from simulator.rng import Stream, vehicle_generator


class StopKind(StrEnum):
    TRAFFIC = "traffic"
    DELIVERY = "delivery"
    FINAL = "final"


@dataclass(frozen=True)
class Segment:
    kind: str
    start: float  # m from the trip start
    length: float  # m
    target_speed: float  # m/s
    grade: float  # rise over run, negative downhill
    nominal_duration: float  # s

    @property
    def end(self) -> float:
        return self.start + self.length


@dataclass(frozen=True)
class Curve:
    start: float  # m
    radius: float  # m
    angle: float  # rad, heading change
    direction: int  # +1 left, -1 right (ISO 8855)
    speed: float  # m/s, planned curve speed v_c
    ramp_length: float  # m, each of the entry and exit ramps
    arc_length: float  # m, constant curvature

    @property
    def end(self) -> float:
        return self.start + 2 * self.ramp_length + self.arc_length

    def curvature(self, s: np.ndarray | float) -> np.ndarray:
        """Signed curvature at road position s (1/m); zero outside the curve."""
        position = np.asarray(s, dtype=float) - self.start
        peak = self.direction / self.radius
        ramp, arc = self.ramp_length, self.arc_length
        rising = position / ramp if ramp > 0 else np.ones_like(position)
        falling = (2 * ramp + arc - position) / ramp if ramp > 0 else np.ones_like(position)
        shape = np.clip(np.minimum(rising, falling), 0.0, 1.0)
        inside = (position >= 0) & (position <= 2 * ramp + arc)
        return np.where(inside, peak * shape, 0.0)


@dataclass(frozen=True)
class Stop:
    position: float  # m
    duration: float  # s
    kind: StopKind


@dataclass(frozen=True)
class SegmentPlan:
    segment: Segment
    curves: tuple[Curve, ...]
    stops: tuple[Stop, ...]
    gaps: tuple[float, ...]  # every gap drawn, including the one that overflowed (for tests)


@dataclass
class Road:
    """Lazily generated road of one trip."""

    root_seed: int
    vehicle_id: str
    day: int
    trip: int
    profile: Config
    plans: list[SegmentPlan] = field(default_factory=list)
    final_stop: Stop | None = None

    def __post_init__(self) -> None:
        self._rng = vehicle_generator(
            self.root_seed, self.vehicle_id, Stream.PROFILE, self.day, self.trip
        )

    @property
    def generated_until(self) -> float:
        return self.plans[-1].segment.end if self.plans else 0.0

    def ensure(self, distance: float) -> None:
        """Generate segments until the road covers `distance` (unless the trip has ended)."""
        while self.final_stop is None and self.generated_until < distance:
            self.plans.append(_draw_segment(self._rng, self.profile, self.generated_until))

    @property
    def curves(self) -> list[Curve]:
        return [curve for plan in self.plans for curve in plan.curves]

    @property
    def stops(self) -> list[Stop]:
        stops = [stop for plan in self.plans for stop in plan.stops]
        return stops + ([self.final_stop] if self.final_stop else [])

    def segment_at(self, s: float) -> Segment:
        self.ensure(s)
        for plan in self.plans:
            if s < plan.segment.end:
                return plan.segment
        return self.plans[-1].segment

    def end_at(self, position: float) -> Stop:
        """Insert the final stop at `position`, or at the end of the curve it falls in.

        Stops and curves beyond the final stop and segments starting after it are removed; the
        segment containing it is shortened to end there.
        """
        self.ensure(position)
        for curve in self.curves:
            if curve.start < position < curve.end:
                position = curve.end
                break
        kept: list[SegmentPlan] = []
        for plan in self.plans:
            if plan.segment.start >= position:
                break
            segment = plan.segment
            if segment.end > position:
                segment = replace(segment, length=position - segment.start)
            kept.append(
                SegmentPlan(
                    segment=segment,
                    curves=tuple(c for c in plan.curves if c.end <= position),
                    stops=tuple(stop for stop in plan.stops if stop.position < position),
                    gaps=plan.gaps,
                )
            )
        self.plans = kept
        self.final_stop = Stop(position=position, duration=0.0, kind=StopKind.FINAL)
        return self.final_stop


def _draw_segment(rng: np.random.Generator, profile: Config, start: float) -> SegmentPlan:
    weights = [profile.value(f"weight_{kind}") for kind in SEGMENT_TYPES]
    kind = SEGMENT_TYPES[int(rng.choice(len(SEGMENT_TYPES), p=weights))]
    speed = float(
        rng.uniform(profile.si(f"{kind}_target_speed_min"), profile.si(f"{kind}_target_speed_max"))
    )
    duration = float(
        rng.uniform(profile.si(f"{kind}_duration_min"), profile.si(f"{kind}_duration_max"))
    )
    grade = _draw_grade(rng, profile, kind)
    segment = Segment(
        kind=kind,
        start=start,
        length=speed * duration,
        target_speed=speed,
        grade=grade,
        nominal_duration=duration,
    )
    curves, stops, gaps = _draw_events(rng, profile, segment)
    if kind == "urban":
        deliver = rng.random() < profile.value("delivery_stop_probability")
        stop_duration = float(
            rng.uniform(profile.si("delivery_stop_min"), profile.si("delivery_stop_max"))
        )
        if deliver:
            stops.append(Stop(position=segment.end, duration=stop_duration, kind=StopKind.DELIVERY))
    return SegmentPlan(segment=segment, curves=tuple(curves), stops=tuple(stops), gaps=tuple(gaps))


def _draw_grade(rng: np.random.Generator, profile: Config, kind: str) -> float:
    if kind == "downhill":
        return -float(
            rng.uniform(profile.si("downhill_grade_min"), profile.si("downhill_grade_max"))
        )
    limit = profile.si(f"{kind}_grade_max")
    return float(rng.uniform(-limit, limit))


def event_rates(profile: Config, kind: str) -> tuple[float, float]:
    """(curve rate, traffic stop rate) per metre of free road for a segment kind."""
    curve_rate = profile.si(CURVES[kind][4])
    stop_rate = profile.si("urban_traffic_stops_per_km") if kind == "urban" else 0.0
    return curve_rate, stop_rate


def _draw_events(
    rng: np.random.Generator, profile: Config, segment: Segment
) -> tuple[list[Curve], list[Stop], list[float]]:
    curve_rate, stop_rate = event_rates(profile, segment.kind)
    total_rate = curve_rate + stop_rate
    curves: list[Curve] = []
    stops: list[Stop] = []
    gaps: list[float] = []
    if total_rate <= 0:
        return curves, stops, gaps
    spacing = profile.si("event_min_spacing")
    cursor = segment.start
    while True:
        gap = spacing + float(rng.exponential(1.0 / total_rate))
        gaps.append(gap)
        position = cursor + gap
        if position > segment.end - spacing:
            return curves, stops, gaps
        if rng.random() < curve_rate / total_rate:
            curve = _draw_curve(rng, profile, segment, position)
            if curve.end > segment.end - spacing:
                return curves, stops, gaps
            curves.append(curve)
            cursor = curve.end
        else:
            duration = float(
                rng.uniform(profile.si("traffic_stop_min"), profile.si("traffic_stop_max"))
            )
            stops.append(Stop(position=position, duration=duration, kind=StopKind.TRAFFIC))
            cursor = position


def _draw_curve(rng: np.random.Generator, profile: Config, segment: Segment, start: float) -> Curve:
    radius_min, radius_max, angle_min, angle_max, _ = CURVES[segment.kind]
    radius = float(rng.uniform(profile.si(radius_min), profile.si(radius_max)))
    angle = float(rng.uniform(profile.si(angle_min), profile.si(angle_max)))
    direction = 1 if rng.random() < 0.5 else -1
    drawn_ramp = float(
        rng.uniform(
            profile.si("curve_transition_time_min"), profile.si("curve_transition_time_max")
        )
    )
    speed = curve_speed(radius, segment.target_speed, profile.si("curve_lateral_accel"))
    ramp_time = curve_ramp_time(radius, angle, speed, drawn_ramp)
    ramp_length = speed * ramp_time
    arc_length = max(radius * angle - ramp_length, 0.0)
    return Curve(
        start=start,
        radius=radius,
        angle=angle,
        direction=direction,
        speed=speed,
        ramp_length=ramp_length,
        arc_length=arc_length,
    )


def heading_change(curve: Curve, samples: int = 20001) -> float:
    """Heading change through a curve by numerical integration of its curvature (for tests)."""
    s = np.linspace(curve.start, curve.end, samples)
    return float(np.trapezoid(curve.curvature(s), s))


def ramps_heading(curve: Curve) -> float:
    """Heading turned by the two ramps alone: L / R."""
    return curve.ramp_length / curve.radius
