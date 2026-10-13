"""Wheel speeds, longitudinal slip and odometer of the fleet, vectorised (spec section 4).

Wheel order is fl, fr, rl, rr (fleet.WHEELS). Reference point: the rear axle centre, which moves at
v with yaw rate psi_dot (model/steering.py). Kinematic single track, left turn positive (ISO 8855):

    rear wheels    v -/+ psi_dot * track / 2                           (left, right)
    front wheels   sqrt((v -/+ psi_dot * track / 2)^2 + (psi_dot * wheelbase)^2)

so in a curve the outer wheels are faster and the front axle runs on a wider radius.

Longitudinal slip, linear far below the force peak, per axle (van_model.yaml, tyre_slip_stiffness):
sigma = F_x / (stiffness * F_z), with static axle loads from the payload (unladen_rear_axle_share,
payload_rear_axle_share; no load transfer). F_x is front_brake_share * F_brake (braking) on the
front axle; the engine force (traction, or engine drag when negative) minus
(1 - front_brake_share) * F_brake on the rear (driven) axle. The brake force always acts backward
here, which is right while moving and at rest on a descent. At rest on a climb the brakes hold
the van against rolling back and push forward, so the axle forces sum to F_grade instead of
-F_grade (off by 2 * |F_grade|); at a delivery stop the parking brake holds the van and the axle
split is not physical. Neither affects the wheel speed readings, exactly 0 at rest whatever the
slip. The odometer increment is 0 on a held step too, except where a stop cancels a commanded
pull-away and the model still moves the van up to about 6 mm (spec section 8, issue #7): that
distance is counted, scaled by the slip from the holding force. A wheel's rolling speed
omega * r = ground speed * (1 + sigma): faster than the ground when driving, slower when braking.

The brake ECU converts rotation to speed with the nominal tyre radius, so a wheel speed reading is
omega * r_nominal = rolling speed * r_nominal / r_true: a van with larger tyres reads low. Readings
are 0 exactly when v = 0 (no rotation). The odometer counts the distance read from the two rear
wheels: per step, the distance travelled times (1 + sigma_rear) * r_nominal / r_true (in a curve the
rear wheels' mean speed is v: the inner and outer offsets cancel).
"""

from dataclasses import dataclass

import numpy as np

from simulator.constants import GRAVITY
from simulator.params import Config


@dataclass(frozen=True)
class WheelParameters:
    track: float  # m
    wheelbase: float  # m
    slip_stiffness: float  # F_x / F_z per unit slip
    front_brake_share: float
    unladen_rear_share: float
    payload_rear_share: float
    mass_empty: float  # kg
    radius_nominal: float  # m, used by the brake ECU

    @classmethod
    def from_model(cls, model: Config) -> "WheelParameters":
        return cls(
            track=model.si("track_width"),
            wheelbase=model.si("wheelbase"),
            slip_stiffness=model.si("tyre_slip_stiffness"),
            front_brake_share=model.si("front_brake_share"),
            unladen_rear_share=model.si("unladen_rear_axle_share"),
            payload_rear_share=model.si("payload_rear_axle_share"),
            mass_empty=model.si("mass_empty"),
            radius_nominal=model.si("tyre_rolling_radius"),
        )


def axle_loads(mass: np.ndarray, p: WheelParameters) -> tuple[np.ndarray, np.ndarray]:
    """(front, rear) static axle loads, N, for vehicles of total mass `mass` (empty + payload)."""
    payload = mass - p.mass_empty
    rear = GRAVITY * (p.unladen_rear_share * p.mass_empty + p.payload_rear_share * payload)
    return GRAVITY * mass - rear, rear


def axle_forces(
    engine_force: np.ndarray, brake_force: np.ndarray, p: WheelParameters
) -> tuple[np.ndarray, np.ndarray]:
    """(front, rear) longitudinal tyre forces, N, positive forward; their sum is engine - brake."""
    front = -p.front_brake_share * brake_force
    rear = engine_force - (1 - p.front_brake_share) * brake_force
    return front, rear


def axle_slip(
    engine_force: np.ndarray,
    brake_force: np.ndarray,
    mass: np.ndarray,
    p: WheelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """(front, rear) longitudinal slip, positive when driving, negative when braking."""
    front_load, rear_load = axle_loads(mass, p)
    front_force, rear_force = axle_forces(engine_force, brake_force, p)
    return (
        front_force / (p.slip_stiffness * front_load),
        rear_force / (p.slip_stiffness * rear_load),
    )


def ground_speeds(speed: np.ndarray, yaw_rate: np.ndarray, p: WheelParameters) -> np.ndarray:
    """Speed over ground of each wheel centre, (n, 4) in WHEELS order, m/s."""
    half = yaw_rate * p.track / 2
    lateral = yaw_rate * p.wheelbase
    left, right = speed - half, speed + half
    return np.stack(
        [np.hypot(left, lateral), np.hypot(right, lateral), left, right],
        axis=-1,
    )


def wheel_speed_readings(
    speed: np.ndarray,
    yaw_rate: np.ndarray,
    slip_front: np.ndarray,
    slip_rear: np.ndarray,
    radius: np.ndarray,
    p: WheelParameters,
) -> np.ndarray:
    """omega * r_nominal of each wheel, (n, 4) in WHEELS order, m/s, before sensor noise."""
    slip = np.stack([slip_front, slip_front, slip_rear, slip_rear], axis=-1)
    scale = (p.radius_nominal / radius)[..., None]
    return ground_speeds(speed, yaw_rate, p) * (1 + slip) * scale


def odometer_increment(
    distance: np.ndarray, slip_rear: np.ndarray, radius: np.ndarray, p: WheelParameters
) -> np.ndarray:
    """Distance read from the rear wheels over a step that travelled `distance` (m)."""
    return distance * (1 + slip_rear) * p.radius_nominal / radius
