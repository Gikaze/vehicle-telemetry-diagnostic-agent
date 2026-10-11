"""Longitudinal dynamics of the fleet, vectorised: acceleration limits and the force split.

Arrays have one entry per vehicle. The driver's acceleration command is limited by the service
limits and by engine power: traction force F_trac <= engine_power_max / max(v, v_min), with v_min
= pedal_mapping_min_speed, so the traction power never exceeds engine_power_max and the pedal
saturates at high speed.
The force needed for the commanded acceleration is split into traction (pedal) or, when it is
negative beyond engine braking, brake force.
"""

from dataclasses import dataclass

import numpy as np

from simulator.model.forces import drag_force, engine_braking_force, grade_force
from simulator.model.forces import rolling_resistance_force as rolling_force
from simulator.params import Config


@dataclass(frozen=True)
class LongitudinalParameters:
    """Model constants used by the longitudinal dynamics, in SI units."""

    drag_area: float
    rolling_resistance: float
    engine_braking_decel: float
    engine_power_max: float
    pedal_min_speed: float
    accel_max: float
    decel_service_max: float

    @classmethod
    def from_model(cls, model: Config) -> "LongitudinalParameters":
        return cls(
            drag_area=model.si("drag_area"),
            rolling_resistance=model.si("rolling_resistance"),
            engine_braking_decel=model.si("engine_braking_decel"),
            engine_power_max=model.si("engine_power_max"),
            pedal_min_speed=model.si("pedal_mapping_min_speed"),
            accel_max=model.si("accel_max"),
            decel_service_max=model.si("decel_service_max"),
        )


@dataclass(frozen=True)
class Forces:
    required: np.ndarray  # N, force the command needs beyond gravity and resistances
    engine: np.ndarray  # N, net engine force: traction if > 0, partial or full engine drag if < 0
    brake: np.ndarray  # N, >= 0, delivered by the brakes; engine - brake == required

    @property
    def traction(self) -> np.ndarray:
        """Positive engine force (pedal pressed), N."""
        return np.maximum(self.engine, 0.0)


def road_forces(
    speed: np.ndarray,
    grade: np.ndarray,
    mass: np.ndarray,
    density: float,
    p: LongitudinalParameters,
) -> np.ndarray:
    """Net external force without traction or brakes (gravity, drag, rolling resistance), N."""
    moving = speed > 0
    resistance = drag_force(speed, density, p.drag_area) + rolling_force(
        mass, grade, p.rolling_resistance
    )
    return grade_force(mass, grade) - np.where(moving, resistance, 0.0)


def traction_force_limit(speed: np.ndarray, p: LongitudinalParameters) -> np.ndarray:
    """Largest traction force: engine power over speed, with a speed floor for pulling away."""
    return p.engine_power_max / np.maximum(speed, p.pedal_min_speed)


def acceleration_limits(
    speed: np.ndarray,
    grade: np.ndarray,
    mass: np.ndarray,
    density: float,
    p: LongitudinalParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """(lowest, highest) achievable acceleration: service braking, and accel_max or power."""
    external = road_forces(speed, grade, mass, density, p)
    power_limited = (traction_force_limit(speed, p) + external) / mass
    highest = np.minimum(p.accel_max, power_limited)
    lowest = np.full_like(highest, -p.decel_service_max)
    return lowest, np.maximum(highest, lowest)


def split_forces(
    accel: np.ndarray,
    speed: np.ndarray,
    grade: np.ndarray,
    mass: np.ndarray,
    density: float,
    p: LongitudinalParameters,
) -> Forces:
    """Traction and brake forces that produce `accel` given gravity and resistances.

    With the pedal fully released the engine drags with engine_braking_decel; a smaller negative
    force is a partly released pedal. The brakes supply only what drag, rolling resistance and
    full engine braking do not.
    """
    required = mass * accel - road_forces(speed, grade, mass, density, p)
    engine_braking = np.where(speed > 0, engine_braking_force(mass, p.engine_braking_decel), 0.0)
    engine = np.maximum(required, -engine_braking)
    brake = np.maximum(-required - engine_braking, 0.0)
    return Forces(required=required, engine=engine, brake=brake)
