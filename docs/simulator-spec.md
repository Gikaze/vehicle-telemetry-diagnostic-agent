# Fleet simulator — specification

The simulator plays the role of a fleet of connected Nordwind vans. Its job is to exercise the pipeline and the
agent with data that is **physically plausible and internally consistent**, so that anomaly rules, diagnostic
reasoning and fleet statistics are tested against something that behaves like real vehicles.

Requirements covered: FR-01, FR-02, FR-03, FR-16, FR-17, FR-18, NFR-15.

## 1. Fidelity principle

**Faithful for what we test, simple for everything else.** The simulator focuses on chassis signals (braking,
vehicle dynamics, steering, tyres) with simplified powertrain signals. Every model is low-order (kinematics,
single-track "bicycle" model, first-order thermal models). Values are **plausible, not calibrated**: they rely on
general public knowledge and are documented as assumptions (section 8). No real manufacturer data is used.

Out of scope for v1.0: multibody dynamics, detailed tyre models (e.g. Pacejka), load transfer, suspension,
parking brake, gearbox, lighting, body, battery, GNSS.

## 2. Architecture

```text
 drive profile ──► vehicle model (10 Hz) ──► sensor layer ──► fault overlay ──► aggregator ──► transport
 (segments,         speed, accel, yaw,       noise,            scenario          10 s windows    sequence,
  seeded RNG)       thermal states            quantisation      faults            + events        gaps, dups
```

- **Drive profile:** a trip is a sequence of segments (urban, rural, downhill, stop), each with a target speed and
  duration drawn from random streams derived from one root seed: each vehicle has its own streams, keyed
  by its ID, so the same seed gives the same trip and adding a vehicle does not change the other
  vehicles' trips (FR-02).
- **Vehicle model:** true physical states updated every 0.1 s.
- **Sensor layer:** converts true states into measured signals in catalogue units: Gaussian noise clipped at
  `noise_clip_sigma` for the noisy signals (`simulator/config/sensors.yaml`), then quantisation to the catalogue
  resolution and saturation to its range. Noise comes from each vehicle's own stream per day and trip, drawn in
  blocks.
- **Fault overlay:** modifies states or signals when a scenario is active.
- **Aggregator:** summarises each 10 s window and emits immediate event messages.
- **Transport:** adds sequence numbers and realistic delivery defects, then publishes (Kinesis locally,
  IoT Core in the demo).

## 3. Signal catalogue (FR-16)

The catalogue lives in `simulator/catalogue.yaml` and is the **single source of truth**: the telemetry JSON
schema is generated from it, and tests check every emitted value against it. Each entry defines: name, unit,
physical range, resolution, source ECU, invalid value, and whether it is modelled dynamically in v1.0.

| Group (source ECU) | Signals | v1.0 behaviour |
| --- | --- | --- |
| Braking (ABS/ESC) | `wheel_speed_{fl,fr,rl,rr}`, `vehicle_ref_speed`, `brake_pedal_pressed`, `master_cyl_pressure`, `abs_active`, `esc_active`, `tcs_active`, `brake_disc_temp_est_{fl,fr,rl,rr}` | Dynamic |
| | `brake_pad_wear_warning`, `brake_fluid_level_warning` | Static unless a scenario sets them |
| Vehicle dynamics (ESC sensor cluster) | `accel_long`, `accel_lat`, `yaw_rate` | Dynamic |
| Steering (angle sensor, EPS) | `steering_wheel_angle`, `steering_wheel_rate`, `driver_torque`, `eps_assist_torque` | Dynamic |
| | `eps_temperature`, `eps_derating_active`, `sas_calibrated` | Nominal in v1.0 (reserved for future scenarios) |
| Tyres (TPMS) | `tyre_pressure_{fl,fr,rl,rr}`, `tyre_temp_{fl,fr,rl,rr}` | Dynamic |
| Powertrain (simplified) | `engine_speed`, `coolant_temp`, `accel_pedal_position` | Dynamic |
| Context | `odometer` | Dynamic |
| | `ambient_temp` | Fixed per simulated day |
| Diagnostics | `dtcs[]` per ECU: code + status bits (`test_failed`, `pending`, `confirmed`, `warning_indicator`) | Set by fault overlay, with debounce (section 6) |

Disc temperature is an **estimate** computed by the braking ECU from braking energy, as in real vehicles —
not a direct measurement. Powertrain fault codes use the public, standardised OBD-II P-codes (e.g. P0117);
chassis codes are manufacturer-specific in practice, so the simulator uses **fictional Nordwind codes**.

## 4. Vehicle model and couplings

Updated every `DT = 0.1 s`. Parameters (wheelbase, track, steering ratio, thermal constants) live in a per-model
configuration, with small per-vehicle variation so no two vans are identical.

**Reference point:** the centre of the rear axle. The speed `v`, the road curvature `κ` and the curve speed law
`v_c = sqrt(a_lat,curve · R)` are taken there; the kinematic single-track model then gives the road-wheel angle
`δ = atan(L · κ)` (no tyre slip angles at low lateral acceleration).

| Signal | Law | Coupled to |
| --- | --- | --- |
| Speed `v` | Approaches the segment target with bounded acceleration and deceleration; `v ≥ 0` | Drive profile |
| Longitudinal accel | Derivative of `v` | Speed |
| Brake pressure, pedal | Non-zero when deceleration exceeds what drag and engine braking provide | Longitudinal accel |
| Steering wheel angle and rate | Angle `i_s · δ` (`i_s` steering ratio); rate = angle change over the step / `DT`, like the longitudinal acceleration | Road curvature |
| Yaw rate | Single-track model: `v · tan(δ) / L = v · κ` at low lateral acceleration (`δ` road-wheel angle, `L` wheelbase) | Speed, steering angle |
| Lateral accel | `v · yaw_rate`, sensor at the reference point (section 8) | Speed, yaw rate |
| Wheel speeds | Kinematic single track: rear wheels `v ∓ yaw_rate · track / 2`, front wheels `sqrt((v ∓ yaw_rate · track / 2)² + (yaw_rate · L)²)` (outer wheels faster, front axle on a wider radius); times `1 + slip`, with a linear slip per axle `F_x / (C · F_z)` (static axle loads from the payload; braking split by `front_brake_share`, traction and engine drag on the rear axle); read by the ECU with the nominal tyre radius (a van's tyre radius bias shows in every reading); plus sensor noise | Speed, yaw rate, brake and engine forces, payload |
| Odometer | Distance read from the two rear wheels (their mean speed is `v` in a curve), with the rear slip and the tyre radius bias; starts from a per-vehicle reading | Speed, slip |
| Reference speed | **Estimated from measured wheel speeds** (not copied from true `v`), as an ESC does: each wheel brought to the rear axle centre (front wheels × `cos δ`, `δ` from the measured steering angle, then `± yaw_rate · track / 2` from the measured yaw rate); the fastest corrected wheel while the brake pedal is pressed, the front (non-driven) mean otherwise | Wheel speeds, yaw rate, steering angle, brake switch |
| ABS, TCS, ESC flags | Computed by the brake ECU from the measured signals: ABS (pedal pressed) when a wheel is below the reference speed, TCS (pedal released) when a rear wheel is above it, both by more than `ecu_slip_speed_min` **and** a slip ratio over `max(v_ref, ecu_reference_speed_min)` above its threshold; ESC when `|yaw_rate − v_ref · tan δ / L|` exceeds `esc_yaw_error_threshold` above `ecu_reference_speed_min` | Reference speed, wheel speeds, yaw rate, steering angle |
| Disc temperature (×4) | First-order: heats in proportion to braking energy shared per axle (none on a step at rest), cools exponentially towards ambient with `1/τ` affine in the step's mean speed, the same `τ` on the four discs; exact update over each step with power and `τ` held | Brake force, speed, ambient |
| Tyre temperature and pressure (×4) | Temperature: first-order towards `T_a + rise · min(1, v / v_ref)` (`v` the step's mean speed), exact update over each step, one temperature for the four tyres. Pressure: ideal gas at constant volume on absolute pressure, from each wheel's cold pressure set at 20 °C | Speed, ambient |
| Coolant temperature | First-order towards the regulated temperature `thermostat + load_rise · pedal / 100` (90 to 94 °C), one time constant for warm-up and tracking in the band, exact update over each step; the engine runs for the whole trip | Engine running time, load (pedal) |
| Engine speed | Rear wheel rotation (speed, rear slip, the vehicle's tyre radius) times the overall ratio of the gear; the schedule selects first gear at rest and shifts at most once per step, up above `upshift_speed`, down below `downshift_speed`; idle at rest, never below idle (clutch slipping when pulling away) | Speed, slip, gear |
| Accelerator pedal | Traction power over the rated power, `F_trac · max(v, v_min) / P_max`, 0 with the engine dragging | Traction force, speed |
| Steering torques | Driver torque and assist torque grow with steering rate and lateral acceleration | Steering, lateral accel |

## 5. Telemetry format and transmission (FR-01, FR-17)

**Edge aggregation.** Physics runs at 10 Hz; sending raw samples to the cloud would be costly and unrealistic.
Every 10 seconds the simulator emits one **periodic message** per vehicle with, per signal, `mean`, `min`, `max`
and `last` (counters such as ABS activations for brief events). Urgent changes — an ESC intervention, a new
DTC — are sent immediately as **event messages**.

**Message envelope.** Every message carries `vehicle_id`, `seq` (monotonic sequence counter), `window_start`,
`window_s` and `schema_version`, similar in spirit to end-to-end protection counters on vehicle networks.

**Delivery realism.** The transport layer injects, at configurable rates: gaps (e.g. tunnels), duplicates and
out-of-order delivery. The pipeline must ingest idempotently: a duplicate never creates a second measurement,
and a gap is detectable from the sequence counter.

## 6. Fault models (FR-02)

Faults are overlays on the normal model, activated by scenario files in `simulator/scenarios/` with a start time,
target vehicles and parameters.

| Scenario (v1.0) | Signature | Use case |
| --- | --- | --- |
| `caliper_drag` | One wheel's disc temperature rises more at each braking and cools less; other wheels unaffected | 1 |
| `coolant_sensor_short` | Coolant sensor circuit shorted to ground: low voltage on an NTC sensor reads as a very high temperature, so the value jumps to a stuck high reading faster than physically possible; DTC P0117 (circuit low) set | 2 |
| `batch_defect` | `caliper_drag` occurs with a higher probability on one production batch | 3 |

Reserved for after v1.0: steering angle sensor offset, EPS thermal derating, slow tyre leak, pad wear.

### Scenario files

Faults act at one of two levels, and the scenario file says which:

- **physical:** a model parameter changes (e.g. extra drag torque on one brake), and the symptoms emerge
  through the vehicle model — temperature, slight deceleration, everything stays consistent;
- **sensor:** the measured value is replaced (stuck, offset, short) and a DTC is set, while the true physical
  state continues unchanged.

```yaml
# simulator/scenarios/caliper_drag.yaml
name: caliper_drag
level: physical
targets: { vehicles: [VAN-042] }        # or { batch: 2025-T3, probability: 0.6 }
start: "08:00"
params: { wheel: fl, drag_torque_nm: 40 }
expected_signature: brake_disc_temp_est_fl rises above the other wheels at each braking
```

Parameters of each scenario are tuned so its signature appears within the scenario window; the scenario
test checks it does.

### Fault code status (FR-18)

Each fault code carries a subset of the UDS (ISO 14229) DTC status byte. The simulator does **not** implement
UDS services (no request/response, sessions or freeze frames); it only adopts the status semantics.

Structure, similar in spirit to a diagnostic event manager:

1. **Monitors** check fault conditions at each simulation step and report *failed* or *passed*.
2. A **fault manager** receives these reports, applies the cycle-based debounce and maintains each code's
   status bits.
3. Every status change is sent as an event message; periodic messages carry the current status of all codes.

The pipeline stores codes with their status, and the agent's vehicle data tool returns them with the signals
(code, ECU, status bits, first occurrence, number of failed cycles). This lets the agent distinguish a new
fault (pending), a persistent fault (confirmed and currently failing) and an intermittent fault (confirmed but
not currently failing).

| Bit | Name | Simulator behaviour |
| --- | --- | --- |
| 0 | `testFailed` | Set while the fault condition is present in the current monitoring sample |
| 2 | `pendingDTC` | Set when the fault occurred in the current or previous operation cycle; cleared after one full cycle without failure |
| 3 | `confirmedDTC` | Set after the fault occurred in N operation cycles (configurable, default 2); stays set until cleared |
| 7 | `warningIndicatorRequested` | Set with `confirmedDTC` for faults configured to request a warning lamp |

An operation cycle is one trip (ignition on to ignition off). Status changes are sent as event messages.
Confirmation thresholds are simulator parameters, not values taken from any real vehicle.

## 7. Verification (NFR-15)

1. **Catalogue conformity:** every emitted value respects its range and resolution; periodic messages respect
   the 10 s cadence (unit tests).
2. **Physical invariants** (property-based tests with Hypothesis, over many random trips), for example:
   - stationary vehicle: all wheel speeds and yaw rate are zero;
   - no braking and stationary: disc temperatures never increase;
   - normal operation: coolant temperature rate of change stays below a defined bound;
   - acceleration stays within the bounds of the vehicle model;
   - lateral acceleration ≈ speed × yaw rate, within tolerance;
   - without an injected fault, the four disc temperatures stay within a defined spread.
3. **Normal-stream test:** 100 vehicles × 24 simulated hours produce **zero** anomalies (false-positive check,
   shared with FR-06).
4. **Scenario tests:** each fault model produces its expected signature, and only that one.
5. **Expert review:** a plot of one simulated vehicle-day (speed, accelerations, temperatures, pressures) is
   published in `docs/assets/` and reviewed by a domain expert before v1.0.

## 8. Assumptions and limits

- Parameter values are indicative orders of magnitude for a light commercial van, chosen from general public
  knowledge; they are listed with their rationale in `simulator/config/` and are not calibrated against any
  real vehicle.
- The single-track model is used at low lateral acceleration only; the simulator does not reproduce limit handling.
- The inertial sensor cluster (`accel_long`, `accel_lat`, `yaw_rate`) sits at the reference point, the rear axle
  centre, so `accel_lat = v · yaw_rate` and `accel_long = dv/dt` hold exactly. A real cluster sits near the
  centre of gravity, a distance `x` ahead of the rear axle (about 1.5 m laden to 2.0 m empty with the model's
  axle loads), and would read `a_y = v · ψ̇ + x · ψ̈` and `a_x = dv/dt − x · ψ̇²`. The worst case of the drive
  profile is the tightest curve (`urban_turn_radius_min` = 10 m at its curve speed 4.47 m/s, `ψ̇` = 0.447 rad/s)
  with the shortest curvature ramp: `curve_transition_time_min` = 2 s (the cap `R · angle / v` is never shorter,
  at least 2.47 s for downhill bends), so `ψ̈ = v / (R · t)` = 0.224 rad/s². The extra terms then reach 0.34 to
  0.44 m/s² on `a_y` on the ramps and −0.30 to −0.40 m/s² on `a_x` in the arc.
- Sensor noise is Gaussian, white and unbiased, clipped at 3 σ, so every measurement error of a true value inside the
  catalogue range is at most 3 σ plus half the resolution. There is no offset, drift or filtering (the coolant σ
  stands for the ECU's filtered reading). A wheel speed sensor reads exactly 0 at standstill. Readings saturate at
  the range bounds: a released brake (0 bar) reads 0 to 0.6 bar and a released accelerator 0 to 0.6 %, exactly 0
  in about 60 % of the samples; `brake_pedal_pressed` comes from the brake light switch on the true pressure, not
  from this reading.
- `accel_long` reads the derivative of the speed, `dv/dt`. A real accelerometer on a grade also reads the gravity
  component along the road, `g · sin(atan(grade))` (about 1 m/s² on a 10 % grade); the model leaves it out.
- Brake pedal at stops: whenever the van is at rest (speed 0 at both ends of a step) outside a delivery stop,
  the driver holds `max(standstill_brake_pressure, brake_pressure_gain · F_hold / brake_reference_mass)`, with
  `F_hold = m · g · |sin(atan(grade))|`. `standstill_brake_pressure` (10 bar) is checked at load to hold the laden
  van on `urban_grade_max`, the steepest grade of the urban segments where traffic stops lie. At rest also covers
  the unmarked step when the approach law brings the van to 0 one step before a stop captures it, which can be
  the last step at a final stop on a descent of up to 10 % (11.5 bar for a laden van): holding the pedal there
  keeps `brake_pedal_pressed` from dropping for 0.1 s, which would count as a false press. At a delivery stop
  the driver leaves the cab with the engine running: the pedal is released (0 bar, `brake_pedal_pressed` false)
  and the van is held by the parking brake, which is not modelled and has no signal. A delivery stop sits at the
  end of an urban segment, so the van may stand on the next segment's grade (rural up to 3 %, descent up to
  10 %). On every step at rest, delivery stops included, the model records no traction and a friction brake
  force of `F_hold` (supplied by the parking brake at a delivery, with the pedal released); the pressure still
  follows the rule above, not this force.
- When the van comes to rest at a stop, its speed is set to 0 from below `stop_speed_threshold` without a force:
  at most 0.1 m/s, i.e. at most ½ · 3300 kg · (0.1 m/s)² = 16.5 J per stop that no brake absorbs (below 0.01 K on
  a disc).
- On a step where a stop cancels a pull-away commanded from rest, the position still advances with the commanded
  speed: the van and the odometer can advance up to about 6 mm on a step at rest (seen on 9 of 320 five-minute
  trips, with no missed stop). The brakes do no work on that step (no disc heating). The odometer also gets up to
  0.014 mm of rear slip from `F_hold` there, which is not physical at rest (static friction, like the disc
  power). Keeping the van in place shifts later positions and speeds and can leave it more than 1 cm short of a
  final stop; see issue #7.
- No pull-away happens inside a curve: trip starts and intermediate stops are at least `event_min_spacing`
  (50 m) away from any curve; the final stop can sit at a curve's end, but ignition goes off there. A van
  at rest therefore never steers. This is a property of the drive profile, not of the steering law, which only
  guarantees a zero steering rate in a step where the van does not move: a van pulling away inside a curve ramp
  would turn the wheel from rest, and a real driver may steer at standstill, which the profile never does.
- Brake disc temperature is an uncalibrated first-order estimate. The discs, the tyres and the
  coolant start at the ambient temperature; carrying their heat over parked and ignition-off periods
  (closed-form cooling, `coolant_cooldown_tau` for the coolant) comes with the fleet state and day
  loop.
- The coolant uses one time constant for the warm-up and for tracking within the regulated band; a
  real thermostat regulates faster once open.
- The engine runs for the whole trip and idles at every stop, delivery stops included (ignition on).
  The clutch is modelled only as the idle floor of the engine speed when pulling away in first
  gear; there are no shift times or torque interruptions.
- The four tyres of a van share one temperature (no load, brake heat or sun effect); their pressures
  differ only by the per-wheel cold pressure. Cold pressures are set at 20 °C whatever the day's
  ambient, so a nominal tyre reads 3.39 bar cold on a −5 °C day. Pressure follows the ideal gas on
  absolute pressure (gauge plus the day's atmospheric pressure).
- The reference speed is a simple estimate from wheel speeds. The brake ECU uses one road-wheel angle for both
  front wheels, so its front axle correction leaves +1.01 % on the inner and −0.94 % on the outer front wheel in
  the 10 m curve at its curve speed; slip and the tyre radius bias also scale the yaw term it adds back.
- ABS, TCS and ESC never intervene in v1.0: in the worst case of the drive profiles, with the measurement errors
  at their bounds, the lowest margins are 2.13 (ABS), 1.86 (TCS) and 3.93 (ESC), checked from the
  configuration by a unit test (`van_model.yaml`, `ecu_slip_speed_min` and `esc_yaw_error_threshold`).
- Fault codes carry a subset of the UDS status bits with a simple cycle-based debounce; UDS services, freeze
  frames and DTC ageing are not simulated.
- Telemetry uses fixed 10 s aggregation windows plus event messages.
- The simulator demonstrates an architecture and a diagnostic workflow, not a validated vehicle model.
