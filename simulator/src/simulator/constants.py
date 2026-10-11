"""Constants fixed by definition or by the specification: not configurable parameters."""

# Standard acceleration of gravity, m/s2 (3rd CGPM, 1901; ISO 80000-3).
GRAVITY = 9.80665

# Temperature of 0 degC in kelvin, by definition of the degree Celsius (SI Brochure, 9th edition).
ZERO_CELSIUS_K = 273.15

# Specific gas constant of dry air, J/(kg K), as cited for the ISO 2533 standard atmosphere:
# 8314.32 J/(kmol K) / 28.964420 kg/kmol = 287.05287 (with the molar mass rounded to 28.9644 the
# quotient would be 287.05307). Quoted from secondary sources; the standard's text is not freely
# available. Air density of the day: rho = atmospheric pressure / (R_DRY_AIR * ambient in K).
R_DRY_AIR = 287.05287

# Physics time step, s: 10 Hz, fixed by docs/simulator-spec.md, section 4.
PHYSICS_DT = 0.1
