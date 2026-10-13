"""Longitudinal driving of a set of trips at 10 Hz: speed plan, power limit, stops and trip end.

This loop drives the speed only; the full vehicle model (steering, wheels, thermal states,
sensors) builds on it. Per step, for every vehicle at once (`step`):

    a = clip(command, lowest, highest)   # speed plan; service braking, accel_max, engine power
    v' = max(v + a * DT, 0), s' = s + (v + v') / 2 * DT
    then road events, final-stop insertion and stops (which may set v' = 0).

The recorded acceleration is the actual (v' - v) / DT, after stops are applied.

A slot with no trip in progress (not started yet, or ended) is held: its speed stays 0, its
position stays where it is (0, or where its trip ended), and its acceleration, reference speed,
grade and forces are 0; no output of the step is NaN.
"""

from dataclasses import dataclass

import numpy as np

from simulator.constants import PHYSICS_DT
from simulator.model.longitudinal import (
    LongitudinalParameters,
    acceleration_limits,
    split_forces,
)
from simulator.profile.road import Road
from simulator.profile.speed_plan import DriverParameters, SpeedPlanner


@dataclass(frozen=True)
class Trace:
    """Time series of one trip, one entry per 0.1 s step, from ignition on to ignition off."""

    time: np.ndarray  # s since the trip start
    position: np.ndarray  # m
    next_position: np.ndarray  # m, at the end of the step (as moved by the model)
    speed: np.ndarray  # m/s at the start of the step
    accel: np.ndarray  # m/s2, actual over the step
    speed_ref: np.ndarray  # m/s
    grade: np.ndarray
    engine_force: np.ndarray  # N
    brake_force: np.ndarray  # N
    stopped: np.ndarray  # bool, held at a stop during the step
    at_delivery: np.ndarray  # bool, held at a delivery stop (not a traffic stop) during the step
    end_time: float  # s, ignition off
    duration: float  # s, planned
    start_step: int  # fleet step at which the trip started


@dataclass(frozen=True)
class Step:
    """One 0.1 s step of the whole fleet; arrays have one entry per slot."""

    active: np.ndarray  # bool, a trip was in progress at the start of the step
    time: np.ndarray  # s since the trip start, at the start of the step
    position: np.ndarray  # m, at the start of the step
    speed: np.ndarray  # m/s, at the start of the step
    accel: np.ndarray  # m/s2, actual over the step
    speed_ref: np.ndarray  # m/s
    grade: np.ndarray
    engine_force: np.ndarray  # N
    brake_force: np.ndarray  # N
    stopped: np.ndarray  # bool, held at a stop during the step
    at_delivery: np.ndarray  # bool, held at a delivery stop during the step
    next_position: np.ndarray  # m
    next_speed: np.ndarray  # m/s


def step(
    planner: SpeedPlanner,
    s: np.ndarray,
    v: np.ndarray,
    mass: np.ndarray,
    density: float,
    longitudinal: LongitudinalParameters,
) -> Step:
    """Advance every slot by one step; the caller keeps (next_position, next_speed)."""
    active = ~planner.ended
    v_ref, command = planner.command(s, v)
    grade = planner.grade.copy()
    lowest, highest = acceleration_limits(v, grade, mass, density, longitudinal)
    accel = np.clip(command, lowest, highest)
    held = planner.stopped | planner.ended
    accel = np.where(held, 0.0, accel)
    forces = split_forces(accel, v, grade, mass, density, longitudinal)
    v_next = np.maximum(v + accel * PHYSICS_DT, 0.0)
    s_next = s + (v + v_next) / 2 * PHYSICS_DT
    stopped_now = planner.stopped.copy()
    delivery_now = planner.at_delivery.copy()
    time_now = planner.time.copy()
    planner.advance_time()
    planner.update_events(s_next, v_next)
    planner.insert_final_stops(s_next, v_next)
    v_next = planner.apply_stops(s_next, v_next, v)
    return Step(
        active=active,
        time=time_now,
        position=s.copy(),
        speed=v.copy(),
        accel=(v_next - v) / PHYSICS_DT,
        speed_ref=np.where(active, v_ref, 0.0),
        grade=np.where(active, grade, 0.0),
        engine_force=np.where(held, 0.0, forces.engine),
        brake_force=np.where(held, 0.0, forces.brake),
        stopped=stopped_now,
        at_delivery=delivery_now,
        next_position=s_next,
        next_speed=v_next,
    )


def drive_trips(
    roads: list[Road],
    durations: list[float],
    masses: list[float],
    density: float,
    longitudinal: LongitudinalParameters,
    driver: DriverParameters,
    time_limit: float,
    start_steps: list[int] | None = None,
) -> list[Trace]:
    """Drive trip i in slot i, starting at fleet step start_steps[i] (default 0), until every
    trip has ended (a trip lasting more than `time_limit` is an error)."""
    n = len(roads)
    starts = [0] * n if start_steps is None else list(start_steps)
    planner = SpeedPlanner(n, driver)
    mass = np.asarray(masses, dtype=float)
    s = np.zeros(n)
    v = np.zeros(n)
    steps: list[Step] = []
    k = 0
    while k <= max(starts) or not planner.ended.all():
        for i in np.flatnonzero(np.asarray(starts) == k):
            planner.start_trip(int(i), roads[i], durations[i])
            s[i], v[i] = 0.0, 0.0
        if planner.time.max() > time_limit:
            raise RuntimeError("trips did not end within the time limit")
        result = step(planner, s, v, mass, density, longitudinal)
        steps.append(result)
        s, v = result.next_position, result.next_speed
        k += 1
    fields = (
        "time",
        "position",
        "next_position",
        "speed",
        "accel",
        "speed_ref",
        "grade",
        "engine_force",
        "brake_force",
        "stopped",
        "at_delivery",
        "active",
    )
    stacked = {name: np.array([getattr(result, name) for result in steps]) for name in fields}
    traces = []
    for i in range(n):
        rows = stacked["active"][:, i]
        traces.append(
            Trace(
                time=stacked["time"][rows, i],
                position=stacked["position"][rows, i],
                next_position=stacked["next_position"][rows, i],
                speed=stacked["speed"][rows, i],
                accel=stacked["accel"][rows, i],
                speed_ref=stacked["speed_ref"][rows, i],
                grade=stacked["grade"][rows, i],
                engine_force=stacked["engine_force"][rows, i],
                brake_force=stacked["brake_force"][rows, i],
                stopped=stacked["stopped"][rows, i],
                at_delivery=stacked["at_delivery"][rows, i],
                end_time=float(planner.end_time[i]),
                duration=float(durations[i]),
                start_step=int(np.argmax(rows)),
            )
        )
    return traces
