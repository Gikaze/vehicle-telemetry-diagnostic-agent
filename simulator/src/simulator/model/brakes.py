"""Brake pressure, brake switch and the braking power reaching each disc (spec section 4).

The master cylinder pressure follows the friction brake force F_brake of the longitudinal model
(model/longitudinal.py, split_forces), with a the signed acceleration (negative when slowing):

    F_brake = max(-m * a + grade force - engine drag - rolling resistance - air drag, 0)

where the grade force is positive downhill and the engine drag (engine_braking_decel * m),
rolling resistance and air drag count only while v > 0. Then (van_model.yaml, brakes section):

    p = brake_pressure_gain * F_brake / brake_reference_mass

A step is held when the speed is 0 at both its ends: at a traffic stop, on the unmarked step at rest
before a stop captures the van, and at the final stop before ignition off. The driver then presses
the pedal enough to hold the van, at least standstill_brake_pressure (drive_profile.yaml):

    p = max(standstill_brake_pressure, brake_pressure_gain * F_hold / brake_reference_mass)

with F_hold = m * g * |sin(atan(grade))| (rolling resistance ignored); the brakes do no work. The
unmarked rest step can be the last step at a final stop on a descent of up to 10 %, where 10 bar
alone would not hold a laden van (11.5 bar needed). At a delivery stop, which is always held, the
driver leaves the cab with the engine running: the pedal is released (p = 0) and the van is held
by the parking brake, which is not modelled (spec section 8).
The brake light switch reports brake_pedal_pressed exactly when p > brake_switch_pressure.

Braking energy over a step is F_brake times the distance the model moved the van in that step
(model/drive.py), so the step on which the van comes to rest is counted with its actual
distance. Per wheel in WHEELS order (fl, fr, rl, rr): front_brake_share / 2 to each front disc,
(1 - front_brake_share) / 2 to each rear disc, of which brake_energy_to_disc heats the disc.
"""

from dataclasses import dataclass

import numpy as np

from simulator.constants import PHYSICS_DT
from simulator.model.forces import grade_force
from simulator.params import Config


@dataclass(frozen=True)
class BrakeParameters:
    pressure_gain: float  # Pa/(m/s2) at the reference mass
    reference_mass: float  # kg
    switch_pressure: float  # Pa
    standstill_pressure: float  # Pa
    front_share: float
    energy_to_disc: float

    @classmethod
    def from_configs(cls, model: Config, profile: Config) -> "BrakeParameters":
        return cls(
            pressure_gain=model.si("brake_pressure_gain"),
            reference_mass=model.si("brake_reference_mass"),
            switch_pressure=model.si("brake_switch_pressure"),
            standstill_pressure=profile.si("standstill_brake_pressure"),
            front_share=model.si("front_brake_share"),
            energy_to_disc=model.si("brake_energy_to_disc"),
        )


def _check_brake_force(brake_force: np.ndarray) -> None:
    if not np.all(brake_force >= 0):  # also rejects NaN
        raise ValueError("the friction brake force must be non-negative and not NaN")


def held_steps(speed: np.ndarray, next_speed: np.ndarray) -> np.ndarray:
    """Steps with the van at rest at both ends."""
    return (speed == 0) & (next_speed == 0)


def hold_force(mass: np.ndarray, grade: np.ndarray) -> np.ndarray:
    """Force that holds a van at rest on a grade, N (rolling resistance ignored)."""
    return np.abs(grade_force(mass, grade))


def master_cylinder_pressure(
    brake_force: np.ndarray,
    held: np.ndarray,
    at_delivery_stop: np.ndarray,
    holding: np.ndarray,
    p: BrakeParameters,
) -> np.ndarray:
    """Pressure, Pa: from the friction brake force while moving; on a held step, at least the
    standstill pressure and enough for the holding force `holding` (N); 0 at a delivery stop
    (pedal released). A delivery stop is always held."""
    _check_brake_force(brake_force)
    if np.any(at_delivery_stop & ~held):
        raise ValueError("a van at a delivery stop must be at rest over the whole step")
    moving = p.pressure_gain * brake_force / p.reference_mass
    hold = np.maximum(p.standstill_pressure, p.pressure_gain * holding / p.reference_mass)
    return np.where(at_delivery_stop, 0.0, np.where(held, hold, moving))


def brake_pedal_pressed(pressure: np.ndarray, p: BrakeParameters) -> np.ndarray:
    return pressure > p.switch_pressure


def disc_heating_power(
    brake_force: np.ndarray, step_distance: np.ndarray, p: BrakeParameters
) -> np.ndarray:
    """Power heating each disc over the step, W, shape (n, 4) in WHEELS order: the energy
    brake_energy_to_disc * F_brake * step distance, spread over DT."""
    _check_brake_force(brake_force)
    power = brake_force * step_distance / PHYSICS_DT * p.energy_to_disc
    front, rear = p.front_share / 2, (1 - p.front_share) / 2
    return power[..., None] * np.array([front, front, rear, rear])
