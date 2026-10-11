"""Constants fixed by definition or by the specification: not configurable parameters."""

# Standard acceleration of gravity, m/s2 (3rd CGPM, 1901; ISO 80000-3).
GRAVITY = 9.80665

# Temperature of 0 degC in kelvin, by definition of the degree Celsius (SI Brochure, 9th edition).
ZERO_CELSIUS_K = 273.15

# Specific gas constant of dry air, J/(kg K), as used by the ISO 2533 standard atmosphere
# (value quoted from secondary sources; the standard's text is not freely available).
# Air density of the day: rho = atmospheric pressure / (R_DRY_AIR * ambient temperature in K).
R_DRY_AIR = 287.0528

# Physics time step, s: 10 Hz, fixed by docs/simulator-spec.md, section 4.
PHYSICS_DT = 0.1
