"""Steering, yaw rate and lateral acceleration of the fleet, vectorised (spec section 4).

Reference point: the centre of the rear axle. The van's speed v, the road curvature kappa and the
curve speed law are taken there. Kinematic single-track model (low lateral acceleration, no
tyre slip angles):

    road-wheel angle      delta = atan(wheelbase * kappa)
    steering-wheel angle  theta = steering_ratio * delta
    yaw rate              psi_dot = v * tan(delta) / wheelbase = v * kappa
    lateral acceleration  a_lat = v * psi_dot (sensor at the reference point, spec section 8)
    steering-wheel rate   (theta(s_(k+1)) - theta(s_k)) / DT, over the step

Angles, yaw rate and lateral acceleration are values at the start of the step, like the speed.
The rate is taken over the step, like the longitudinal acceleration (v_(k+1) - v_k) / DT: its sum
times DT telescopes to the angle change, and it is exactly 0 when the van does not move during
the step. (The instantaneous derivative kappa'(s) * v misses the part of a step spent on a curve
ramp: integrated over a trip it drifted up to 25 deg from the angle. A difference with the
previous step would carry the previous step's motion into the first step at rest.)

Steering torques (van_model.yaml, steering section), with fade(v) = exp(-v / eps_assist_fade_speed):

    driver = gain_rate * rate + gain_lat * a_lat + gain_park * theta * fade(v)
    assist = (ratio_high + (ratio_standstill - ratio_high) * fade(v)) * driver

Angles are in radians and rates in rad/s here; the sensor layer converts to catalogue units.
"""

from dataclasses import dataclass

import numpy as np

from simulator.constants import PHYSICS_DT
from simulator.params import Config
from simulator.profile.road import Road


@dataclass(frozen=True)
class SteeringParameters:
    wheelbase: float  # m
    steering_ratio: float
    gain_rate: float  # Nm/(rad/s)
    gain_lat: float  # Nm/(m/s2)
    gain_park: float  # Nm/rad
    ratio_standstill: float
    ratio_high: float
    fade_speed: float  # m/s

    @classmethod
    def from_model(cls, model: Config) -> "SteeringParameters":
        return cls(
            wheelbase=model.si("wheelbase"),
            steering_ratio=model.si("steering_ratio"),
            gain_rate=model.si("driver_torque_gain_rate"),
            gain_lat=model.si("driver_torque_gain_lat"),
            gain_park=model.si("driver_torque_gain_park"),
            ratio_standstill=model.si("eps_assist_ratio_standstill"),
            ratio_high=model.si("eps_assist_ratio_high_speed"),
            fade_speed=model.si("eps_assist_fade_speed"),
        )


@dataclass(frozen=True)
class SteeringState:
    """True steering and lateral states of every slot for one step (SI units)."""

    curvature: np.ndarray  # 1/m, of the rear axle centre's path
    road_wheel_angle: np.ndarray  # rad
    steering_wheel_angle: np.ndarray  # rad
    steering_wheel_rate: np.ndarray  # rad/s
    yaw_rate: np.ndarray  # rad/s
    accel_lat: np.ndarray  # m/s2
    driver_torque: np.ndarray  # Nm
    assist_torque: np.ndarray  # Nm


def steering_wheel_angle(curvature: np.ndarray, p: SteeringParameters) -> np.ndarray:
    """theta = steering_ratio * atan(wheelbase * kappa), rad."""
    return p.steering_ratio * np.arctan(p.wheelbase * curvature)


def steering_state(
    curvature: np.ndarray,
    next_curvature: np.ndarray,
    speed: np.ndarray,
    p: SteeringParameters,
) -> SteeringState:
    """Steering states from the curvature and speed at the start of the step, and the curvature
    at the end of the step (for the rate over the step)."""
    delta = np.arctan(p.wheelbase * curvature)
    theta = p.steering_ratio * delta
    rate = (steering_wheel_angle(next_curvature, p) - theta) / PHYSICS_DT
    yaw = speed * curvature
    accel_lat = speed * yaw
    fade = np.exp(-speed / p.fade_speed)
    driver = p.gain_rate * rate + p.gain_lat * accel_lat + p.gain_park * theta * fade
    assist = (p.ratio_high + (p.ratio_standstill - p.ratio_high) * fade) * driver
    return SteeringState(
        curvature=curvature,
        road_wheel_angle=delta,
        steering_wheel_angle=theta,
        steering_wheel_rate=rate,
        yaw_rate=yaw,
        accel_lat=accel_lat,
        driver_torque=driver,
        assist_torque=assist,
    )


class CurveTracker:
    """The current or next curve of each slot's road, for a vectorised curvature lookup.

    Each slot holds one curve: the one the van is in, or the next one ahead. The slot is refreshed
    in Python only when the van reaches that curve's start (which also confirms the curve still
    exists: a final stop can remove curves beyond it), its end, or the end of the road generated so
    far, so per-step work is numpy over the fleet. A slot without a curve ahead holds a curve
    starting at infinity.
    """

    def __init__(self, n: int):
        self.n = n
        self.roads: list[Road | None] = [None] * n
        self.start = np.full(n, np.inf)
        self.ramp = np.ones(n)
        self.arc = np.zeros(n)
        self.peak = np.zeros(n)  # signed curvature of the arc, 1/m
        self.check_at = np.full(n, np.inf)  # refresh when the position reaches this

    def start_trip(self, i: int, road: Road) -> None:
        self.roads[i] = road
        self._refresh(i, 0.0)

    def _refresh(self, i: int, s: float) -> None:
        road = self.roads[i]
        if road is None:
            raise RuntimeError(f"slot {i} has no road")
        road.ensure(s + 1.0)
        curve = next((c for c in road.curves if c.end > s), None)
        if curve is None:
            self.start[i], self.peak[i] = np.inf, 0.0
            self.ramp[i], self.arc[i] = 1.0, 0.0
            self.check_at[i] = np.inf if road.final_stop is not None else road.generated_until
            return
        self.start[i] = curve.start
        self.ramp[i], self.arc[i] = curve.ramp_length, curve.arc_length
        self.peak[i] = curve.direction / curve.radius
        self.check_at[i] = curve.start if s < curve.start else curve.end

    def curvature(self, s: np.ndarray) -> np.ndarray:
        """Signed curvature at positions s, one per slot (same law as Curve.curvature)."""
        position = s - self.start
        rising = position / self.ramp
        falling = (2 * self.ramp + self.arc - position) / self.ramp
        shape = np.clip(np.minimum(rising, falling), 0.0, 1.0)
        inside = (position >= 0) & (position <= 2 * self.ramp + self.arc)
        return np.where(inside, self.peak * shape, 0.0)

    def update(self, s: np.ndarray, active: np.ndarray) -> None:
        """Refresh the slots of active vehicles that reached their check position."""
        for i in np.flatnonzero(active & (s >= self.check_at)):
            self._refresh(int(i), float(s[i]))
