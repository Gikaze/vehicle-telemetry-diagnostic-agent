"""Brake ECU logic of the fleet, vectorised: reference speed and ABS, TCS and ESC flags (spec
section 4).

The ECU works on measured signals only: the four wheel speed readings w (omega * r_nominal,
WHEELS order fl, fr, rl, rr, with sensor noise and quantisation), the yaw rate psi_dot, the
steering wheel angle theta and the brake switch (brake_pedal_pressed). Two kinematic corrections
bring every wheel to the speed of the rear axle centre, in this order (van_model.yaml,
ecu_slip_speed_min):

    1. front axle projection   w_front * cos(delta),  delta = theta / steering_ratio
    2. yaw term                + psi_dot * track / 2 (left wheels), - psi_dot * track / 2 (right)

Without slip, the rear wheels then read the rear axle speed exactly. The front wheels each run
at their own road-wheel angle, but the ECU uses the single angle delta for both: in the 10 m
curve at its curve speed the inner front wheel keeps +1.01 % and the outer -0.94 % (spec
section 8). Slip and the tyre radius bias scale each reading, including its yaw term, which the
ECU adds back unscaled: a wheel slipping by sigma keeps sigma * psi_dot * track / 2 (sigma times
8.5 % of the speed in the 10 m curve). A wheel that reads 0 (no sensor pulses) stays 0, so the
corrected speeds and the reference speed are 0 at standstill.

Reference speed: the fastest corrected wheel while the brake pedal is pressed (braking slows the
wheels), the mean of the corrected front wheels otherwise (not driven: no traction slip).

Slip of a wheel against the reference speed v_ref, as a difference d and a ratio
d / max(v_ref, ecu_reference_speed_min) (the guard keeps noise at walking pace from reading as a
large slip):

    ABS   pedal pressed, some wheel with v_ref - c_i above both ecu_slip_speed_min and
          abs_slip_threshold
    TCS   pedal released, some rear (driven) wheel with c_i - v_ref above both ecu_slip_speed_min
          and tcs_slip_threshold
    ESC   v_ref >= ecu_reference_speed_min and |psi_dot - v_ref * tan(delta) / wheelbase| above
          esc_yaw_error_threshold

None of these is reached by the v1.0 drive profiles (spec section 8). Inputs and outputs are in
SI units (m/s, rad, rad/s); the sensor layer converts to catalogue units.
"""

from dataclasses import dataclass

import numpy as np

from simulator.params import Config

FRONT = np.array([True, True, False, False])
# Left wheels read v - psi_dot * track / 2 (left turn positive, ISO 8855): the yaw term adds it
# back.
SIDE = np.array([1.0, -1.0, 1.0, -1.0])


@dataclass(frozen=True)
class EcuParameters:
    track: float  # m
    wheelbase: float  # m
    steering_ratio: float
    abs_threshold: float  # braking slip ratio
    tcs_threshold: float  # traction slip ratio
    esc_threshold: float  # rad/s, yaw rate error
    reference_speed_min: float  # m/s
    slip_speed_min: float  # m/s

    @classmethod
    def from_model(cls, model: Config) -> "EcuParameters":
        return cls(
            track=model.si("track_width"),
            wheelbase=model.si("wheelbase"),
            steering_ratio=model.si("steering_ratio"),
            abs_threshold=model.si("abs_slip_threshold"),
            tcs_threshold=model.si("tcs_slip_threshold"),
            esc_threshold=model.si("esc_yaw_error_threshold"),
            reference_speed_min=model.si("ecu_reference_speed_min"),
            slip_speed_min=model.si("ecu_slip_speed_min"),
        )


def road_wheel_angle(steering_wheel_angle: np.ndarray, p: EcuParameters) -> np.ndarray:
    """delta = theta / steering_ratio, rad."""
    return np.asarray(steering_wheel_angle) / p.steering_ratio


def corrected_wheel_speeds(
    wheel_speeds: np.ndarray,
    yaw_rate: np.ndarray,
    steering_wheel_angle: np.ndarray,
    p: EcuParameters,
) -> np.ndarray:
    """Wheel speeds brought to the rear axle centre, (n, 4), m/s: front axle projection, then
    the yaw term; a wheel reading 0 stays 0."""
    cos_delta = np.cos(road_wheel_angle(steering_wheel_angle, p))[..., None]
    projected = np.where(FRONT, wheel_speeds * cos_delta, wheel_speeds)
    corrected = projected + SIDE * (np.asarray(yaw_rate) * p.track / 2)[..., None]
    return np.where(wheel_speeds == 0, 0.0, corrected)


def reference_speed(corrected: np.ndarray, braking: np.ndarray) -> np.ndarray:
    """Vehicle reference speed, m/s: fastest corrected wheel while braking, front mean otherwise."""
    return np.where(braking, corrected.max(axis=-1), corrected[..., :2].mean(axis=-1))


def _ratio(difference: np.ndarray, reference: np.ndarray, p: EcuParameters) -> np.ndarray:
    return difference / np.maximum(reference, p.reference_speed_min)[..., None]


def braking_slip(
    corrected: np.ndarray, reference: np.ndarray, p: EcuParameters
) -> tuple[np.ndarray, np.ndarray]:
    """(difference v_ref - c_i, m/s, ratio) of the four wheels, (n, 4)."""
    difference = reference[..., None] - corrected
    return difference, _ratio(difference, reference, p)


def traction_slip(
    corrected: np.ndarray, reference: np.ndarray, p: EcuParameters
) -> tuple[np.ndarray, np.ndarray]:
    """(difference c_i - v_ref, m/s, ratio) of the two rear (driven) wheels, (n, 2)."""
    difference = corrected[..., 2:] - reference[..., None]
    return difference, _ratio(difference, reference, p)


def yaw_rate_error(
    yaw_rate: np.ndarray, steering_wheel_angle: np.ndarray, reference: np.ndarray, p: EcuParameters
) -> np.ndarray:
    """Measured yaw rate minus the single-track reference v_ref * tan(delta) / wheelbase, rad/s."""
    expected = reference * np.tan(road_wheel_angle(steering_wheel_angle, p)) / p.wheelbase
    return yaw_rate - expected


def abs_active(
    corrected: np.ndarray, reference: np.ndarray, braking: np.ndarray, p: EcuParameters
) -> np.ndarray:
    difference, ratio = braking_slip(corrected, reference, p)
    slipping = (difference > p.slip_speed_min) & (ratio > p.abs_threshold)
    return braking & np.any(slipping, axis=-1)


def tcs_active(
    corrected: np.ndarray, reference: np.ndarray, braking: np.ndarray, p: EcuParameters
) -> np.ndarray:
    difference, ratio = traction_slip(corrected, reference, p)
    spinning = (difference > p.slip_speed_min) & (ratio > p.tcs_threshold)
    return ~braking & np.any(spinning, axis=-1)


def esc_active(
    yaw_rate: np.ndarray, steering_wheel_angle: np.ndarray, reference: np.ndarray, p: EcuParameters
) -> np.ndarray:
    error = yaw_rate_error(yaw_rate, steering_wheel_angle, reference, p)
    return (reference >= p.reference_speed_min) & (np.abs(error) > p.esc_threshold)
