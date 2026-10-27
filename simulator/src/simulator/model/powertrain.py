"""Accelerator pedal, gear schedule and engine speed of the fleet, vectorised (spec section 4).

Accelerator pedal (van_model.yaml, engine_power_max): with F_trac = max(engine force, 0) and v the
start-of-step speed, the speed at which the traction limit P_max / max(v, v_min) was applied
(model/longitudinal.py),

    demand = F_trac * max(v, v_min) / P_max,    pedal = 100 * clip(demand, 0, 1)

so the demand never exceeds 1 beyond rounding, and engine drag (a negative engine force) reads 0 %.

Engine speed in gear g (1 = first), with the engine turning with the driven rear wheels:

    n_g = v * (1 + sigma_rear) / r * ratio_g * 60 / (2 pi)

with sigma_rear the rear axle slip (model/wheels.py) and r the vehicle's drawn tyre radius. Each
step, before the engine speed is output, the gear schedule selects first gear at rest (v = 0) and
otherwise shifts at most once: up when n_g > upshift_speed (below top gear), down when
n_g < downshift_speed (above first gear). The engine speed is idle_speed at rest and
max(idle_speed, n_g) when moving: the clutch slips while the van pulls away, which binds in first
gear only since downshift_speed > idle_speed. The engine runs for the whole trip, delivery stops
included (ignition on), so it idles at every stop. The loader requires downshift_speed <=
upshift_speed * (smallest ratio step) - shift_margin, so a shift never lands beyond the opposite
threshold (no gear hunting).
"""

from dataclasses import dataclass

import numpy as np

from simulator.constants import RAD_S_TO_RPM
from simulator.params import Config


@dataclass(frozen=True)
class PowertrainParameters:
    power_max: float  # W
    pedal_min_speed: float  # m/s
    ratios: tuple[float, ...]  # overall ratio per gear, first gear first
    upshift: float  # rpm
    downshift: float  # rpm
    idle: float  # rpm

    @classmethod
    def from_model(cls, model: Config) -> "PowertrainParameters":
        return cls(
            power_max=model.si("engine_power_max"),
            pedal_min_speed=model.si("pedal_mapping_min_speed"),
            ratios=tuple(float(r) for r in model.value("gear_overall_ratios")),
            upshift=float(model.value("upshift_speed")),
            downshift=float(model.value("downshift_speed")),
            idle=float(model.value("idle_speed")),
        )

    @property
    def top_gear(self) -> int:
        return len(self.ratios)


def pedal_demand(
    engine_force: np.ndarray, speed: np.ndarray, p: PowertrainParameters
) -> np.ndarray:
    """Traction power over engine_power_max, unclipped, at the start-of-step speed (m/s)."""
    return np.maximum(engine_force, 0.0) * np.maximum(speed, p.pedal_min_speed) / p.power_max


def accelerator_pedal(demand: np.ndarray) -> np.ndarray:
    """Pedal position, %, from the demand of pedal_demand."""
    return 100 * np.clip(demand, 0.0, 1.0)


def gear_engine_speed(
    gear: np.ndarray,
    speed: np.ndarray,
    slip_rear: np.ndarray,
    radius: np.ndarray,
    p: PowertrainParameters,
) -> np.ndarray:
    """Engine speed, rpm, turning with the rear wheels in `gear` (no clutch slip)."""
    ratio = np.asarray(p.ratios)[np.asarray(gear) - 1]
    return speed * (1 + slip_rear) / radius * ratio * RAD_S_TO_RPM


def select_gear(
    gear: np.ndarray,
    speed: np.ndarray,
    slip_rear: np.ndarray,
    radius: np.ndarray,
    p: PowertrainParameters,
) -> np.ndarray:
    """Gear for this step from the previous step's gear: first at rest, else at most one shift."""
    n = gear_engine_speed(gear, speed, slip_rear, radius, p)
    up = (n > p.upshift) & (gear < p.top_gear)
    down = (n < p.downshift) & (gear > 1)
    return np.where(speed == 0, 1, gear + up.astype(int) - down.astype(int))


def engine_speed(
    gear: np.ndarray,
    speed: np.ndarray,
    slip_rear: np.ndarray,
    radius: np.ndarray,
    p: PowertrainParameters,
) -> np.ndarray:
    """Engine speed, rpm, in the gear selected for the step: idle at rest, never below idle."""
    n = gear_engine_speed(gear, speed, slip_rear, radius, p)
    return np.where(speed == 0, p.idle, np.maximum(p.idle, n))
