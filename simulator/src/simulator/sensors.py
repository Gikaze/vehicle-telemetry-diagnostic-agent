"""Sensor layer: true states to measured signals in catalogue units (spec section 2).

A true value comes in SI units (the accelerator pedal as a fraction, temperatures in degC and the
engine speed in rpm, as the model computes them) and is converted to the unit of its catalogue
signal with UNIT_FACTORS. A noisy signal (sensors.yaml) then gets Gaussian noise, clipped at
noise_clip_sigma:

    measured = encode(true + clip(z, -noise_clip_sigma, noise_clip_sigma) * sigma)

with z a standard normal draw and sigma in the catalogue unit; a noise-free signal (NOISE_FREE) is
encode(true). encode quantises to the catalogue resolution (catalogue.quantise) and saturates to the
catalogue range. The range bounds lie on the grid, so saturation never moves a reading away from a
true value inside the range, and every error is at most error_bound = noise_clip_sigma * sigma +
resolution / 2. A wheel speed sensor reads exactly 0 when the wheel does not turn (no pulses): no
noise is added there, but its draw is still consumed.

Noise is drawn per vehicle from its SENSOR_NOISE stream of the day and trip (SensorNoise), in
blocks of noise_block_duration: every active step takes one draw per channel (CHANNELS order), so
the draws of one trip never depend on another trip or on another vehicle, and the block size
changes no value.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from simulator.catalogue import Catalogue, Signal, quantise
from simulator.config_files import NOISE_FREE, NOISY_SIGNALS
from simulator.constants import PHYSICS_DT
from simulator.params import UNIT_FACTORS, Config
from simulator.rng import Stream, vehicle_generator

# Noisy signals in the order of their draws: part of the reproducibility contract.
CHANNELS = tuple(name for names in NOISY_SIGNALS.values() for name in names)
GROUP = {name: group for group, names in NOISY_SIGNALS.items() for name in names}
# Wheel speed sensors read exactly 0 at standstill (sensors.yaml, wheel_speed).
ZERO_AT_REST = frozenset(NOISY_SIGNALS["wheel_speed"])


@dataclass(frozen=True)
class SensorParameters:
    clip: float  # in sigmas
    sigma: Mapping[str, float]  # per noisy signal, catalogue unit
    signals: Mapping[str, Signal]  # numeric catalogue signals, noisy and noise-free
    block_steps: int  # steps of noise drawn at once per vehicle

    @classmethod
    def from_configs(cls, sensors: Config, catalogue: Catalogue) -> "SensorParameters":
        return cls(
            clip=float(sensors.value("noise_clip_sigma")),
            sigma={name: float(sensors.value(GROUP[name])) for name in CHANNELS},
            signals={name: catalogue.signal(name) for name in (*CHANNELS, *NOISE_FREE)},
            block_steps=round(sensors.si("noise_block_duration") / PHYSICS_DT),
        )


def to_catalogue_unit(values: np.ndarray, signal: Signal) -> np.ndarray:
    """SI values in the catalogue unit of `signal` (unchanged for degC and rpm)."""
    factor = UNIT_FACTORS[signal.unit]
    values = np.asarray(values, dtype=float)
    return values if factor is None else values / factor


def from_catalogue_unit(values: np.ndarray, signal: Signal) -> np.ndarray:
    """Values in the catalogue unit of `signal` back in SI (unchanged for degC and rpm)."""
    factor = UNIT_FACTORS[signal.unit]
    values = np.asarray(values, dtype=float)
    return values if factor is None else values * factor


def encode(values: np.ndarray, signal: Signal) -> np.ndarray:
    """Values in the catalogue unit onto the signal's grid, saturated to its range."""
    assert signal.range is not None
    return np.clip(quantise(values, signal), *signal.range)


def error_bound(name: str, p: SensorParameters) -> float:
    """Largest |measured - true| of a signal, catalogue unit, for a true value in its range."""
    resolution = p.signals[name].resolution
    assert resolution is not None
    return p.clip * p.sigma.get(name, 0.0) + resolution / 2


def measure_signal(name: str, true: np.ndarray, z: np.ndarray, p: SensorParameters) -> np.ndarray:
    """Measured value of one noisy signal, catalogue unit, from its SI true value and draws z."""
    signal = p.signals[name]
    value = to_catalogue_unit(true, signal)
    noise = np.clip(z, -p.clip, p.clip) * p.sigma[name]
    if name in ZERO_AT_REST:
        noise = np.where(value == 0, 0.0, noise)
    return encode(value + noise, signal)


def measure(
    true: Mapping[str, np.ndarray], z: np.ndarray, p: SensorParameters
) -> dict[str, np.ndarray]:
    """Measured values, catalogue units, of every noisy signal (all of CHANNELS required, draws
    z[..., j] for CHANNELS[j]) and of the noise-free signals given in `true` (SI units). The keys
    come in a fixed order, CHANNELS then NOISE_FREE, independent of the order of `true`."""
    missing = sorted(set(CHANNELS) - set(true))
    unknown = sorted(set(true) - set(CHANNELS) - set(NOISE_FREE))
    if missing or unknown:
        raise ValueError(f"true values: missing {missing}, not measurable {unknown}")
    measured = {
        name: measure_signal(name, true[name], z[..., j], p) for j, name in enumerate(CHANNELS)
    }
    for name in (name for name in NOISE_FREE if name in true):
        measured[name] = encode(to_catalogue_unit(true[name], p.signals[name]), p.signals[name])
    return measured


class SensorNoise:
    """Standard normal draws of a fleet, one row of len(CHANNELS) per slot and active step.

    Each slot draws from the SENSOR_NOISE stream of its vehicle's current trip (day and trip
    indices), block_steps steps at a time; an inactive slot does not advance.
    """

    def __init__(self, n: int, p: SensorParameters):
        self.n = n
        self.block_steps = p.block_steps
        self.buffer = np.zeros((n, p.block_steps, len(CHANNELS)))
        self.cursor = np.full(n, p.block_steps)
        self.generators: list[np.random.Generator | None] = [None] * n

    def start_trip(self, i: int, root_seed: int, vehicle_id: str, day: int, trip: int) -> None:
        self.generators[i] = vehicle_generator(
            root_seed, vehicle_id, Stream.SENSOR_NOISE, day, trip
        )
        self._refill(i)

    def _refill(self, i: int) -> None:
        generator = self.generators[i]
        if generator is None:
            raise RuntimeError(f"slot {i} has no trip")
        self.buffer[i] = generator.standard_normal((self.block_steps, len(CHANNELS)))
        self.cursor[i] = 0

    def draw(self, active: np.ndarray) -> np.ndarray:
        """Draws of this step, (n, len(CHANNELS)); 0 for inactive slots."""
        for i in np.flatnonzero(active & (self.cursor == self.block_steps)):
            self._refill(int(i))
        rows = self.buffer[np.arange(self.n), np.minimum(self.cursor, self.block_steps - 1)]
        self.cursor += active.astype(int)
        return np.where(active[:, None], rows, 0.0)
