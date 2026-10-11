"""Planned speed of the fleet along each trip's road, vectorised across vehicles.

The driver plans a reference speed along the road and the van tracks it with feed-forward
(drive_profile.yaml):

    v_ref(s) = min(segment speed, curve speed inside a curve,
                   sqrt(v_c^2 + 2 approach_decel (s_c - s)) for every constraint ahead,
                   sqrt(v_0^2 + 2 pull_away_accel (s - s_0)) since the last constraint passed)

On an approach the command is the constant deceleration that meets the active constraint from the
current state, -(v^2 - v_c^2) / (2 (s_c - s)); otherwise a = a_ref + speed_tracking_gain
(v_ref - v) (see SpeedPlanner.command). A feed-forward plus proportional law on approaches either
overshoots the plan at the ramp-to-approach transition (one 0.1 s step at +pull_away_accel) or,
with the 0.5 1/s gain, lags it near standstill, because the approach to a stop collapses to 0
within 2 to 3 s; a constant deceleration is integrated exactly, so stops are met. Constraints
are curve starts (curve speed), stops (0) and segment boundaries towards a lower speed. Each
vehicle keeps the constraints within its braking horizon in a window; the window is refreshed in
Python only when a vehicle passes a constraint, leaves a curve or a segment, or a constraint enters
its horizon. Per-step work is numpy over the fleet.

Constraints only ever grow at the end of the list (the road is generated sequentially) and the
final stop never removes a constraint already passed (a curve in progress is kept, the stop moves
to its end), so the index of the next constraint stays valid when the list is rebuilt.

Stops: below stop_speed_threshold with a stop just ahead (or just passed) the speed is set to 0
and held for the stop duration. The trip ends with a final stop inserted when the remaining planned
time equals the time to stop at approach_decel, at the stopping distance; ignition goes off when
the van is stopped there. If the time runs out while the van is stopped at a traffic or delivery
stop, or with such a stop before the stopping point, that stop becomes the final stop.
"""

from dataclasses import dataclass, field

import numpy as np

from simulator.constants import PHYSICS_DT
from simulator.params import Config
from simulator.profile.road import Road
from simulator.profile_config import SEGMENT_TYPES

STOP = 0
CURVE = 1
BOUNDARY = 2

# Columns of the candidate matrix in reference(): segment, curve, ramp, then the window.
_ENVELOPE_COLUMN = 3
_RAMP_COLUMN = 2


@dataclass(frozen=True)
class DriverParameters:
    approach_decel: float
    pull_away_accel: float
    speed_tracking_gain: float
    stop_speed_threshold: float
    horizon: float  # m: braking distance from the top segment speed at approach_decel

    @classmethod
    def from_configs(cls, profile: Config, model: Config) -> "DriverParameters":
        top = max(profile.si(f"{kind}_target_speed_max") for kind in SEGMENT_TYPES)
        decel = profile.si("approach_decel")
        return cls(
            approach_decel=decel,
            pull_away_accel=profile.si("pull_away_accel"),
            speed_tracking_gain=model.si("speed_tracking_gain"),
            stop_speed_threshold=profile.si("stop_speed_threshold"),
            horizon=top * top / (2 * decel),
        )


@dataclass(frozen=True)
class Constraint:
    position: float
    speed: float
    kind: int
    duration: float = 0.0  # stops only
    final: bool = False


@dataclass
class _Trip:
    """Python-side state of one vehicle's trip, touched only on events."""

    road: Road
    constraints: list[Constraint] = field(default_factory=list)
    built_until: float = -1.0
    built_final: bool = False
    next_index: int = 0  # first constraint not yet passed
    segment_index: int = 0


class SpeedPlanner:
    """Reference speed, stop logic and trip end for trips driven at the same time."""

    def __init__(self, roads: list[Road], durations: list[float], driver: DriverParameters):
        self.driver = driver
        self.trips = [_Trip(road=road) for road in roads]
        n = len(self.trips)
        self.n = n
        self.duration = np.asarray(durations, dtype=float)
        self.time = np.zeros(n)  # s since the trip start
        self.segment_speed = np.zeros(n)
        self.grade = np.zeros(n)
        self.segment_end = np.zeros(n)
        self.curve_speed = np.full(n, np.inf)
        self.curve_end = np.full(n, np.inf)
        self.ramp_position = np.zeros(n)  # start of the pull-away ramp
        self.ramp_speed = np.zeros(n)
        self.stopped = np.zeros(n, dtype=bool)
        self.stop_until = np.zeros(n)
        self.final_inserted = np.zeros(n, dtype=bool)
        self.ended = np.zeros(n, dtype=bool)
        self.end_time = np.full(n, np.nan)
        self.next_stop_position = np.full(n, np.inf)
        self.next_stop_duration = np.zeros(n)
        self.next_stop_final = np.zeros(n, dtype=bool)
        self.window_k = 4
        self.window_position = np.full((n, self.window_k), np.inf)
        self.window_speed = np.full((n, self.window_k), np.inf)
        self.window_reach = np.zeros(n)  # the window holds every constraint before this position
        for i in range(n):
            self._enter_segment(i, 0.0, 0)
            self._refresh(i, 0.0)

    # Python-side events -------------------------------------------------------------------------

    def _build(self, i: int, until: float) -> None:
        trip = self.trips[i]
        road = trip.road
        road.ensure(until)
        final = road.final_stop is not None
        if trip.built_until == road.generated_until and trip.built_final == final:
            return
        constraints: list[Constraint] = []
        plans = road.plans
        for index, plan in enumerate(plans):
            # A curve no slower than its segment needs no approach; as a constraint it would ask
            # a zero-length approach (a deceleration spike) for the tracking error at its start.
            constraints += [
                Constraint(c.start, c.speed, CURVE)
                for c in plan.curves
                if c.speed < plan.segment.target_speed
            ]
            constraints += [Constraint(st.position, 0.0, STOP, st.duration) for st in plan.stops]
            if index + 1 < len(plans):
                following = plans[index + 1].segment
                if following.target_speed < plan.segment.target_speed:
                    constraints.append(
                        Constraint(following.start, following.target_speed, BOUNDARY)
                    )
        if road.final_stop is not None:
            constraints.append(Constraint(road.final_stop.position, 0.0, STOP, final=True))
        constraints.sort(key=lambda c: (c.position, c.kind))
        trip.constraints = constraints
        trip.built_until = road.generated_until
        trip.built_final = final

    def _refresh(self, i: int, s: float) -> None:
        """Rebuild vehicle i's window with every constraint ahead within its horizon."""
        horizon_end = s + self.driver.horizon
        self._build(i, horizon_end + self.driver.horizon)
        trip = self.trips[i]
        pending = trip.constraints[trip.next_index :]
        ahead = [c for c in pending if c.position < horizon_end]
        if len(ahead) > self.window_k:
            self._grow(len(ahead))
        self.window_position[i] = np.inf
        self.window_speed[i] = np.inf
        for k, constraint in enumerate(ahead):
            self.window_position[i, k] = constraint.position
            self.window_speed[i, k] = constraint.speed
        rest = pending[len(ahead) :]
        # Refresh again when the horizon reaches the next constraint or the end of the
        # constraints built so far (the road beyond may hold more).
        known_until = np.inf if trip.built_final else trip.built_until
        self.window_reach[i] = min(rest[0].position if rest else np.inf, known_until)
        stop = next((c for c in ahead if c.kind == STOP), None)
        self.next_stop_position[i] = stop.position if stop else np.inf
        self.next_stop_duration[i] = stop.duration if stop else 0.0
        self.next_stop_final[i] = stop.final if stop else False

    def _grow(self, k: int) -> None:
        pad = np.full((self.n, k - self.window_k), np.inf)
        self.window_position = np.hstack([self.window_position, pad])
        self.window_speed = np.hstack([self.window_speed, pad.copy()])
        self.window_k = k

    def _enter_segment(self, i: int, s: float, index: int) -> None:
        trip = self.trips[i]
        trip.road.ensure(s + 1.0)
        index = min(index, len(trip.road.plans) - 1)
        trip.segment_index = index
        segment = trip.road.plans[index].segment
        self.segment_speed[i] = segment.target_speed
        self.grade[i] = segment.grade
        self.segment_end[i] = segment.end

    def _pass_constraints(self, i: int, s: float) -> None:
        trip = self.trips[i]
        while trip.next_index < len(trip.constraints):
            constraint = trip.constraints[trip.next_index]
            if constraint.position > s or constraint.kind == STOP:
                break
            if constraint.kind == CURVE:
                curve = next(c for c in trip.road.curves if c.start == constraint.position)
                self.curve_speed[i] = curve.speed
                self.curve_end[i] = curve.end
            trip.next_index += 1
        self._refresh(i, s)

    def _end(self, i: int) -> None:
        self.ended[i] = True
        self.end_time[i] = self.time[i]

    # Per-step update ----------------------------------------------------------------------------

    def command(self, s: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(v_ref, commanded acceleration) for every vehicle at positions s and speeds v.

        On an approach (a constraint envelope is the lowest term) the command is the constant
        deceleration that meets the active constraint exactly from the current state,
        -(v^2 - v_c^2) / (2 (s_c - s)): it equals -approach_decel on the plan, is gentler when the
        van is behind it and slightly stronger right after a ramp-to-approach transition, and a
        constant deceleration is integrated exactly, so the van reaches a stop at its position.
        Otherwise a = a_ref + gain (v_ref - v), with a_ref = +pull_away_accel * min(v / v_ref, 1)
        on the pull-away ramp and 0 on a constant limit.
        """
        d = self.driver
        rows = np.arange(self.n)
        distance = self.window_position - s[:, None]
        finite = np.isfinite(self.window_position)

        def envelope_at(gap: np.ndarray) -> np.ndarray:
            speed = np.sqrt(np.maximum(self.window_speed**2 + 2 * d.approach_decel * gap, 0.0))
            return np.where(finite, speed, np.inf)

        def ramp_at(position: np.ndarray) -> np.ndarray:
            travelled = position - self.ramp_position
            return np.sqrt(np.maximum(self.ramp_speed**2 + 2 * d.pull_away_accel * travelled, 0.0))

        ramp = ramp_at(s)
        # Term chosen one step ahead: an approach shorter than one step's travel would start late.
        step = v * PHYSICS_DT
        ahead = envelope_at(distance - step[:, None])
        choice = np.argmin(
            np.column_stack([self.segment_speed, self.curve_speed, ramp_at(s + step), ahead]),
            axis=1,
        )
        candidates = np.column_stack(
            [self.segment_speed, self.curve_speed, ramp, envelope_at(distance)]
        )
        v_ref = candidates[rows, choice]
        approaching = choice >= _ENVELOPE_COLUMN
        column = np.where(approaching, choice - _ENVELOPE_COLUMN, 0)
        gap = distance[rows, column]
        target = self.window_speed[rows, column]
        target = np.where(np.isfinite(target), target, 0.0)
        # Constraint already passed (a stop overrun by less than a step): stop within this step.
        required = np.divide(
            -(v * v - target * target),
            2 * gap,
            out=-v / PHYSICS_DT,
            where=gap > 0,
        )
        # A van below its target speed never closes the gap faster than it pulls away.
        required = np.minimum(required, d.pull_away_accel)
        # Ramp feed-forward is dv_ref/dt = pull_away_accel * v / v_ref; capped so a van above the
        # ramp (power limit released) never accelerates faster than planned.
        scale = np.divide(v, v_ref, out=np.ones_like(v), where=v_ref > 0)
        ramp_rate = np.where(
            choice == _RAMP_COLUMN, d.pull_away_accel * np.minimum(scale, 1.0), 0.0
        )
        tracking = ramp_rate + d.speed_tracking_gain * (v_ref - v)
        # At rest short of a constraint the exact law asks 0 and would hold the van there for good;
        # not shown unreachable (rounding), so pull away towards it instead.
        at_rest_short = approaching & (v == 0) & (gap > 0) & (required <= 0)
        approach = np.where(at_rest_short, d.pull_away_accel, required)
        return v_ref, np.where(approaching, approach, tracking)

    def update_events(self, s: np.ndarray, v: np.ndarray) -> None:
        """Bookkeeping for the vehicles whose road situation changed at this step."""
        active = ~self.ended
        first_is_stop = self.window_speed[:, 0] == 0
        crossed = active & (s >= self.window_position[:, 0]) & ~first_is_stop
        entering = active & (s + self.driver.horizon >= self.window_reach)
        left_curve = active & (s >= self.curve_end)
        new_segment = active & (s >= self.segment_end)
        for i in np.flatnonzero(left_curve):
            self.curve_speed[i] = np.inf
            self.curve_end[i] = np.inf
            self.ramp_position[i], self.ramp_speed[i] = s[i], v[i]
        for i in np.flatnonzero(new_segment):
            final = self.trips[i].road.final_stop
            if final is not None and s[i] >= final.position:
                self.segment_end[i] = np.inf
                continue
            self._enter_segment(i, s[i], self.trips[i].segment_index + 1)
            self.ramp_position[i], self.ramp_speed[i] = s[i], v[i]
        for i in np.flatnonzero(crossed | entering):
            self._pass_constraints(i, s[i])

    def insert_final_stops(self, s: np.ndarray, v: np.ndarray) -> None:
        """Insert the final stop when the remaining time equals the time to stop."""
        remaining = self.duration - self.time
        due = ~self.final_inserted & ~self.ended & (remaining <= v / self.driver.approach_decel)
        for i in np.flatnonzero(due):
            trip = self.trips[i]
            self.final_inserted[i] = True
            if self.stopped[i]:
                # Time runs out during a traffic or delivery stop: that stop becomes the final
                # stop, at the same position, and ignition goes off now.
                trip.road.end_at(self.next_stop_position[i])
                self._refresh(i, s[i])
                self._end(i)
                continue
            position = s[i] + v[i] ** 2 / (2 * self.driver.approach_decel)
            # A planned stop before that point is where the van stops anyway: it becomes the
            # final stop (otherwise it would stay an intermediate stop right before the end).
            position = min(position, self.next_stop_position[i])
            stop = trip.road.end_at(position)
            self._refresh(i, s[i])
            if v[i] == 0 and stop.position <= s[i]:
                self._end(i)

    def apply_stops(self, s: np.ndarray, v: np.ndarray, v_start: np.ndarray) -> np.ndarray:
        """Set the speed to 0 when arriving at a stop and release stops that are over.

        A van arrives when its speed at the start of the step is below stop_speed_threshold and
        the stop is at most one step's travel at that speed ahead (or has been passed). Setting
        the speed to 0 then decelerates by at most stop_speed_threshold / DT. Returns the
        corrected speed: 0 for stopped and ended vehicles.
        """
        d = self.driver
        arriving = (
            ~self.stopped
            & ~self.ended
            & (v_start < d.stop_speed_threshold)
            & (self.next_stop_position - s <= d.stop_speed_threshold * PHYSICS_DT)
        )
        for i in np.flatnonzero(arriving):
            if self.next_stop_final[i]:
                self._end(i)
            else:
                self.stopped[i] = True
                self.stop_until[i] = self.time[i] + self.next_stop_duration[i]
        release = self.stopped & ~self.ended & (self.time >= self.stop_until)
        for i in np.flatnonzero(release):
            trip = self.trips[i]
            while trip.next_index < len(trip.constraints):
                passed = trip.constraints[trip.next_index]
                trip.next_index += 1
                if passed.kind == STOP:
                    break
            self.stopped[i] = False
            self.ramp_position[i], self.ramp_speed[i] = s[i], 0.0
            self._refresh(i, s[i])
        return np.where(self.stopped | self.ended | arriving, 0.0, v)

    def advance_time(self) -> None:
        self.time = np.where(self.ended, self.time, self.time + PHYSICS_DT)
