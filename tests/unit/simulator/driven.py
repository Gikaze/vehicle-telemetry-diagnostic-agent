"""Shared test helper: drive a few trips and compute their steering, for model invariant tests."""

from dataclasses import dataclass

import numpy as np

from simulator.config_files import van_model
from simulator.constants import PHYSICS_DT
from simulator.model.drive import Trace, drive_trips
from simulator.model.forces import air_density
from simulator.model.longitudinal import LongitudinalParameters
from simulator.model.steering import CurveTracker, SteeringParameters, SteeringState, steering_state
from simulator.profile.road import Road
from simulator.profile.speed_plan import DriverParameters
from simulator.profile_config import drive_profile, trip_overrun_bound


@dataclass(frozen=True)
class DrivenTrip:
    road: Road
    trace: Trace
    mass: float  # kg
    density: float  # kg/m3, air density of the day
    positions: np.ndarray  # m, at the start of each step, then after the last step
    curvature: np.ndarray  # 1/m, at `positions`
    steering: SteeringState  # one entry per step


def drive_with_steering(
    seed: int, trip: int, duration: float, ambient: float, vans: int = 4
) -> list[DrivenTrip]:
    """`vans` vans from empty to laden drive trip `trip` of day 0 side by side."""
    model, profile = van_model(), drive_profile()
    vehicles = [f"VAN-{n:03d}" for n in range(1, vans + 1)]
    roads = [Road(seed, vehicle, 0, trip, profile) for vehicle in vehicles]
    masses = list(np.linspace(2300, 3300, len(roads)))
    density = air_density(101300, ambient)
    traces = drive_trips(
        roads,
        [duration] * len(roads),
        masses,
        density,
        LongitudinalParameters.from_model(model),
        DriverParameters.from_configs(profile, model),
        time_limit=duration + 2 * trip_overrun_bound(profile),
    )
    p = SteeringParameters.from_model(model)
    results = []
    for road, trace, mass in zip(roads, traces, masses, strict=True):
        tracker = CurveTracker(1)
        tracker.start_trip(0, road)
        # Positions at the start of each step, then the position after the last step.
        next_speed = trace.speed + trace.accel * PHYSICS_DT
        end = trace.position[-1] + (trace.speed[-1] + next_speed[-1]) / 2 * PHYSICS_DT
        positions = np.append(trace.position, end)
        kappa = np.empty_like(positions)
        for k, s in enumerate(positions):
            tracker.update(np.array([s]), np.array([True]))
            kappa[k] = tracker.curvature(np.array([s]))[0]
        state = steering_state(kappa[:-1], kappa[1:], trace.speed, p)
        results.append(DrivenTrip(road, trace, float(mass), density, positions, kappa, state))
    return results
