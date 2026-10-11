"""Longitudinal driving of a set of trips at 10 Hz: speed plan, power limit, stops and trip end.

This loop drives the speed only; the full vehicle model (steering, wheels, thermal states,
sensors) builds on it. Per step, for every vehicle at once:

    a = clip(command, lowest, highest)   # speed plan; service braking, accel_max, engine power
    v' = max(v + a * DT, 0), s' = s + (v + v') / 2 * DT
    then road events, final-stop insertion and stops (which may set v' = 0).

The recorded acceleration is the actual (v' - v) / DT, after stops are applied.
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
    speed: np.ndarray  # m/s at the start of the step
    accel: np.ndarray  # m/s2, actual over the step
    speed_ref: np.ndarray  # m/s
    grade: np.ndarray
    engine_force: np.ndarray  # N
    brake_force: np.ndarray  # N
    stopped: np.ndarray  # bool, held at a stop during the step
    end_time: float  # s, ignition off
    duration: float  # s, planned


def drive_trips(
    roads: list[Road],
    durations: list[float],
    masses: list[float],
    density: float,
    longitudinal: LongitudinalParameters,
    driver: DriverParameters,
    time_limit: float,
) -> list[Trace]:
    """Drive the trips side by side until every trip has ended (or `time_limit`, an error)."""
    planner = SpeedPlanner(roads, durations, driver)
    n = len(roads)
    mass = np.asarray(masses, dtype=float)
    s = np.zeros(n)
    v = np.zeros(n)
    columns: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "time",
            "position",
            "speed",
            "accel",
            "speed_ref",
            "grade",
            "engine",
            "brake",
            "stopped",
            "active",
        )
    }
    while not planner.ended.all():
        if planner.time.max() > time_limit:
            raise RuntimeError("trips did not end within the time limit")
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
        time_now = planner.time.copy()
        planner.advance_time()
        planner.update_events(s_next, v_next)
        planner.insert_final_stops(s_next, v_next)
        v_next = planner.apply_stops(s_next, v_next, v)
        for key, value in (
            ("time", time_now),
            ("position", s.copy()),
            ("speed", v.copy()),
            ("accel", (v_next - v) / PHYSICS_DT),
            ("speed_ref", v_ref),
            ("grade", grade),
            ("engine", np.where(held, 0.0, forces.engine)),
            ("brake", np.where(held, 0.0, forces.brake)),
            ("stopped", stopped_now),
            ("active", active),
        ):
            columns[key].append(value)
        s, v = s_next, v_next
    stacked = {key: np.array(values) for key, values in columns.items()}
    traces = []
    for i in range(n):
        rows = stacked["active"][:, i]
        traces.append(
            Trace(
                time=stacked["time"][rows, i],
                position=stacked["position"][rows, i],
                speed=stacked["speed"][rows, i],
                accel=stacked["accel"][rows, i],
                speed_ref=stacked["speed_ref"][rows, i],
                grade=stacked["grade"][rows, i],
                engine_force=stacked["engine"][rows, i],
                brake_force=stacked["brake"][rows, i],
                stopped=stacked["stopped"][rows, i],
                end_time=float(planner.end_time[i]),
                duration=float(durations[i]),
            )
        )
    return traces
