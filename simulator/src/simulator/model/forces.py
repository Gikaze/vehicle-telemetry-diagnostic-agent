"""Longitudinal force laws, in SI units; shared by the vehicle model and the configuration checks.

Sign convention (ISO 8855): forces along the vehicle x axis, positive forward. A grade is the
road slope as a fraction (rise over run), negative downhill.
"""

import numpy as np
from numpy.typing import ArrayLike

from simulator.constants import GRAVITY, R_DRY_AIR, ZERO_CELSIUS_K


def air_density(pressure_pa: float, ambient_temp_c: float) -> float:
    """Ideal-gas density of dry air, kg/m3."""
    return pressure_pa / (R_DRY_AIR * (ambient_temp_c + ZERO_CELSIUS_K))


def grade_force(mass: ArrayLike, grade: ArrayLike) -> np.ndarray:
    """Gravity component along the road; positive forward when going downhill (grade < 0)."""
    return -np.asarray(mass) * GRAVITY * np.sin(np.arctan(grade))


def rolling_resistance_force(mass: ArrayLike, grade: ArrayLike, coefficient: float) -> np.ndarray:
    """Magnitude of the rolling resistance (acts against motion)."""
    return np.asarray(mass) * GRAVITY * coefficient * np.cos(np.arctan(grade))


def drag_force(speed: ArrayLike, density: float, drag_area: float) -> np.ndarray:
    """Magnitude of the aerodynamic drag (acts against motion)."""
    v = np.asarray(speed)
    return 0.5 * density * drag_area * v * v


def engine_braking_force(mass: ArrayLike, decel: float) -> np.ndarray:
    """Magnitude of the engine drag with the accelerator released."""
    return np.asarray(mass) * decel


def brake_force_to_hold_speed(
    mass: ArrayLike,
    speed: ArrayLike,
    grade: ArrayLike,
    *,
    density: float,
    drag_area: float,
    rolling_resistance: float,
    engine_braking_decel: float,
) -> np.ndarray:
    """Brake force needed to hold a constant speed; positive when the brakes must work."""
    return (
        grade_force(mass, grade)
        - engine_braking_force(mass, engine_braking_decel)
        - rolling_resistance_force(mass, grade, rolling_resistance)
        - drag_force(speed, density, drag_area)
    )
