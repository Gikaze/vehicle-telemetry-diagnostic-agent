"""Ambient conditions of a simulated day, common to the whole fleet (spec section 3)."""

from dataclasses import dataclass

from simulator.model.forces import air_density
from simulator.params import Config
from simulator.rng import Stream, fleet_generator


@dataclass(frozen=True)
class Ambient:
    temperature: float  # degC, fixed for the day
    pressure: float  # Pa
    density: float  # kg/m3, ideal gas at this pressure and temperature


def draw_ambient(root_seed: int, day: int, duty: Config) -> Ambient:
    """Ambient of day `day`, from the fleet AMBIENT stream of that day (never a vehicle stream)."""
    rng = fleet_generator(root_seed, Stream.AMBIENT, day)
    temperature = float(rng.uniform(duty.value("ambient_temp_min"), duty.value("ambient_temp_max")))
    pressure = duty.si("atmospheric_pressure")
    return Ambient(
        temperature=temperature, pressure=pressure, density=air_density(pressure, temperature)
    )
