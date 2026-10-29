"""Brake disc, tyre and coolant temperatures, and tyre pressures, of the fleet, vectorised (spec
section 4).

Each disc i (WHEELS order fl, fr, rl, rr) follows a first-order law towards the day's ambient
T_a, heated by the braking power P_i that reaches it (model/brakes.py, disc_heating_power) and
cooled with a time constant tau that shortens with speed (van_model.yaml, brakes section):

    C_i dT_i/dt = P_i - C_i (T_i - T_a) / tau(v)
    1/tau(v) = (1 / tau_standstill) * (1 + v / v_c)
    v_c = reference_speed / (tau_standstill / tau_reference - 1)

C_i is the vehicle's drawn heat capacity of that disc (fleet.py). Over a 0.1 s step, P_i and tau
are held constant and the update is the exact solution (zero-order hold, stable for any tau):

    T' = T_a + (T - T_a) e^(-DT/tau) + (P_i / C_i) tau (1 - e^(-DT/tau))

P_i comes from the step's friction brake force and the distance the model moved the van, with
the held mask computed here from the speeds at both ends of the step: a held step heats nothing,
even where a stop cancels a commanded pull-away and the van still moves up to about 6 mm (spec
section 8). tau uses the step's mean speed (v + v') / 2, which is 0 on a held step, so a held step
cools exactly as at standstill.

The same tau on all four discs makes each rise T_i - T_a proportional to its power share over
C_i when the discs start at ambient: the ratio of the front and rear axle mean rises is then
constant for a vehicle.

The tyres follow a first-order law towards a target that rises with the step's mean speed v
(van_model.yaml, tyres section), with one time constant for warm-up and cool-down:

    T_target = T_a + warmup_rise * min(1, v / warmup_reference_speed)
    T' = T_target + (T - T_target) e^(-DT/tau)

The four tyres share one temperature (no load, brake heat or sun effect in v1.0); the state keeps
one column per wheel. Each tyre's gauge pressure follows the ideal gas at constant volume on
absolute pressure, from its cold pressure p_cold (drawn per wheel, fleet.py) set at
tyre_pressure_reference_temp and the day's atmospheric pressure p_atm (duty_cycle.yaml):

    p = (p_cold + p_atm) * (T + 273.15) / (T_ref + 273.15) - p_atm

The engine coolant follows a first-order law towards the regulated temperature, which rises with
the load given by the accelerator pedal (model/powertrain.py), with one time constant for the
warm-up and for tracking within the band (van_model.yaml, engine section):

    T_target = thermostat + load_rise * pedal / 100
    T' = T_target + (T - T_target) e^(-DT/tau_warmup)

The engine runs for the whole trip, so this law holds on every step of a trip.

While parked (ignition off), discs, tyres and coolant cool towards the ambient in closed form,
exact for any parked duration t (parked_temperature):

    T' = T_a + (T - T_a) e^(-t/tau)

with the time constant of each law at rest: tau_standstill for the discs (no brake power at
standstill), the tyre tau (the tyre target at v = 0 is T_a) and coolant_cooldown_tau for the
coolant (engine off). The ambient is fixed per day and switches at midnight, so a parked period
that crosses midnight cools piecewise, towards each day's ambient for the time spent in that day.
The discs and the tyres at rest follow the same law step by step (disc_step, tyre_step), so
parking for N steps gives the result of N steps at rest. The fleet loop (fleet_state.py) applies
it at every trip start; only the first trip of a simulation starts at the ambient temperature.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from simulator.constants import PHYSICS_DT, SECONDS_PER_DAY, ZERO_CELSIUS_K
from simulator.model.brakes import BrakeParameters, disc_heating_power, held_steps
from simulator.params import Config


@dataclass(frozen=True)
class DiscParameters:
    tau_standstill: float  # s
    cooling_speed: float  # m/s, v_c of the cooling law

    @classmethod
    def from_model(cls, model: Config) -> "DiscParameters":
        standstill = model.si("disc_cooling_tau_standstill")
        reference = model.si("disc_cooling_tau_reference")
        speed = model.si("disc_cooling_reference_speed")
        return cls(tau_standstill=standstill, cooling_speed=speed / (standstill / reference - 1))


def disc_cooling_tau(mean_speed: np.ndarray, p: DiscParameters) -> np.ndarray:
    """Cooling time constant, s, at the given speed (m/s)."""
    return p.tau_standstill / (1 + np.asarray(mean_speed) / p.cooling_speed)


def disc_step(
    temp: np.ndarray,
    brake_force: np.ndarray,
    speed: np.ndarray,
    next_speed: np.ndarray,
    step_distance: np.ndarray,
    ambient: float,
    capacity: np.ndarray,
    p: DiscParameters,
    brake: BrakeParameters,
) -> np.ndarray:
    """Disc temperatures, degC, (n, 4) in WHEELS order, at the end of a step from `temp` at its
    start; `capacity` is each disc's heat capacity, J/K, (n, 4)."""
    held = held_steps(speed, next_speed)
    power = disc_heating_power(brake_force, step_distance, held, brake)
    tau = disc_cooling_tau((speed + next_speed) / 2, p)[..., None]
    decay = -np.expm1(-PHYSICS_DT / tau)  # 1 - e^(-DT/tau)
    return temp - (temp - ambient) * decay + power / capacity * tau * decay


@dataclass(frozen=True)
class TyreParameters:
    warmup_rise: float  # K
    warmup_speed: float  # m/s
    tau: float  # s
    reference_temp: float  # degC, at which the cold pressures are set

    @classmethod
    def from_model(cls, model: Config) -> "TyreParameters":
        return cls(
            warmup_rise=model.si("tyre_warmup_rise"),
            warmup_speed=model.si("tyre_warmup_reference_speed"),
            tau=model.si("tyre_thermal_tau"),
            reference_temp=model.value("tyre_pressure_reference_temp"),
        )


def tyre_target_temperature(
    mean_speed: np.ndarray, ambient: float, p: TyreParameters
) -> np.ndarray:
    """Temperature, degC, that the tyres approach at the given speed (m/s)."""
    return ambient + p.warmup_rise * np.minimum(1.0, np.asarray(mean_speed) / p.warmup_speed)


def tyre_step(
    temp: np.ndarray,
    speed: np.ndarray,
    next_speed: np.ndarray,
    ambient: float,
    p: TyreParameters,
) -> np.ndarray:
    """Tyre temperatures, degC, (n, 4) in WHEELS order, at the end of a step from `temp`."""
    target = tyre_target_temperature((speed + next_speed) / 2, ambient, p)[..., None]
    return target + (temp - target) * np.exp(-PHYSICS_DT / p.tau)


def tyre_pressure(
    temp: np.ndarray, cold_pressure: np.ndarray, atmospheric: float, p: TyreParameters
) -> np.ndarray:
    """Gauge pressure, Pa, of tyres at `temp` (degC) whose gauge pressure is `cold_pressure` (Pa)
    at the reference temperature; `atmospheric` is the absolute ambient pressure, Pa."""
    scale = (temp + ZERO_CELSIUS_K) / (p.reference_temp + ZERO_CELSIUS_K)
    return (cold_pressure + atmospheric) * scale - atmospheric


@dataclass(frozen=True)
class CoolantParameters:
    thermostat: float  # degC
    load_rise: float  # K, at full pedal
    tau: float  # s, warm-up and tracking in the band
    cooldown_tau: float  # s, engine off

    @classmethod
    def from_model(cls, model: Config) -> "CoolantParameters":
        return cls(
            thermostat=model.value("coolant_thermostat_temp"),
            load_rise=model.si("coolant_load_rise"),
            tau=model.si("coolant_warmup_tau"),
            cooldown_tau=model.si("coolant_cooldown_tau"),
        )


def coolant_target_temperature(pedal: np.ndarray, p: CoolantParameters) -> np.ndarray:
    """Regulated coolant temperature, degC, at the given accelerator pedal position (%)."""
    return p.thermostat + p.load_rise * np.asarray(pedal) / 100


def coolant_step(temp: np.ndarray, pedal: np.ndarray, p: CoolantParameters) -> np.ndarray:
    """Coolant temperature, degC, (n,), at the end of a step from `temp`, engine running with the
    step's accelerator pedal position (%)."""
    target = coolant_target_temperature(pedal, p)
    return target + (temp - target) * np.exp(-PHYSICS_DT / p.tau)


def parked_temperature(
    temp: np.ndarray,
    off_time: np.ndarray,
    on_time: np.ndarray,
    ambients: Sequence[float],
    tau: float,
) -> np.ndarray:
    """Temperatures, degC, at `on_time` of a component parked from `off_time` at `temp` (times in
    s since the midnight starting day 0, broadcast against `temp`), cooling with time constant
    `tau` towards ambients[d] for the time spent in day d, days in order."""
    off, on = np.asarray(off_time, dtype=float), np.asarray(on_time, dtype=float)
    if np.any(on < off) or np.any(off < 0):
        raise ValueError("parked times: expected 0 <= off_time <= on_time")
    temp = np.asarray(temp, dtype=float)
    # Days overlapping [off, on): a period ending exactly at midnight needs no ambient of the next.
    first, last = int(off.min() // SECONDS_PER_DAY), int(np.ceil(on.max() / SECONDS_PER_DAY))
    for day in range(first, last):
        start, end = day * SECONDS_PER_DAY, (day + 1) * SECONDS_PER_DAY
        duration = np.clip(on, start, end) - np.clip(off, start, end)
        temp = ambients[day] + (temp - ambients[day]) * np.exp(-duration / tau)
    return temp


def parked_discs(
    temp: np.ndarray,
    off_time: np.ndarray,
    on_time: np.ndarray,
    ambients: Sequence[float],
    p: DiscParameters,
) -> np.ndarray:
    """Disc temperatures, degC, (n, 4), after parking; times (n,) as in parked_temperature."""
    off, on = np.asarray(off_time)[:, None], np.asarray(on_time)[:, None]
    return parked_temperature(temp, off, on, ambients, p.tau_standstill)


def parked_tyres(
    temp: np.ndarray,
    off_time: np.ndarray,
    on_time: np.ndarray,
    ambients: Sequence[float],
    p: TyreParameters,
) -> np.ndarray:
    """Tyre temperatures, degC, (n, 4), after parking; times (n,) as in parked_temperature."""
    off, on = np.asarray(off_time)[:, None], np.asarray(on_time)[:, None]
    return parked_temperature(temp, off, on, ambients, p.tau)


def parked_coolant(
    temp: np.ndarray,
    off_time: np.ndarray,
    on_time: np.ndarray,
    ambients: Sequence[float],
    p: CoolantParameters,
) -> np.ndarray:
    """Coolant temperatures, degC, (n,), engine off; times (n,) as in parked_temperature."""
    return parked_temperature(temp, off_time, on_time, ambients, p.cooldown_tau)
