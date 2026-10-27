"""Shared test helper: drive a few trips and compute their steering, for model invariant tests;
the slip bound the configuration allows."""

from dataclasses import dataclass

import numpy as np

from simulator.config_files import van_model
from simulator.fleet import Vehicle, VehicleParameters, draw_vehicle_parameters, fleet_config
from simulator.model.drive import Trace, drive_trips
from simulator.model.forces import air_density
from simulator.model.longitudinal import (
    LongitudinalParameters,
    acceleration_limits,
    split_forces,
)
from simulator.model.steering import CurveTracker, SteeringParameters, SteeringState, steering_state
from simulator.model.wheels import WheelParameters, axle_slip
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


def drawn(seed: int, number: int) -> VehicleParameters:
    """Parameters drawn for VAN-<number> (fleet.py), the van in slot number - 1 of
    drive_with_steering."""
    vehicle = Vehicle(id=f"VAN-{number:03d}", number=number, model="nordwind-lv1", batch="test")
    return draw_vehicle_parameters(seed, vehicle, van_model(), fleet_config())


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
        positions = np.append(trace.position, trace.next_position[-1])
        kappa = np.empty_like(positions)
        for k, s in enumerate(positions):
            tracker.update(np.array([s]), np.array([True]))
            kappa[k] = tracker.curvature(np.array([s]))[0]
        state = steering_state(kappa[:-1], kappa[1:], trace.speed, p)
        results.append(DrivenTrip(road, trace, float(mass), density, positions, kappa, state))
    return results


def slip_bound(masses: np.ndarray) -> float:
    """Largest |slip| the model can produce, from the configuration: service braking on the
    steepest descent and the largest traction on the steepest climb, at the lowest speed (no drag
    helping), over the given masses, with the model's stiffness and axle loads."""
    model, profile = van_model(), drive_profile()
    longitudinal, p = LongitudinalParameters.from_model(model), WheelParameters.from_model(model)
    speed = np.full_like(masses, 0.1)
    descent = np.full_like(masses, -profile.si("downhill_grade_max"))
    climb = np.full_like(
        masses, max(profile.si(f"{kind}_grade_max") for kind in ("urban", "rural"))
    )
    bound = 0.0
    braking = np.full_like(masses, -longitudinal.decel_service_max)
    forces = split_forces(braking, speed, descent, masses, 1.2, longitudinal)
    bound = max(
        bound, *(np.max(np.abs(s)) for s in axle_slip(forces.engine, forces.brake, masses, p))
    )
    _, highest = acceleration_limits(speed, climb, masses, 1.2, longitudinal)
    forces = split_forces(highest, speed, climb, masses, 1.2, longitudinal)
    bound = max(
        bound, *(np.max(np.abs(s)) for s in axle_slip(forces.engine, forces.brake, masses, p))
    )
    return float(bound)
