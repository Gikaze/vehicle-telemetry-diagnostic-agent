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
  duration drawn from a seeded random generator. Same seed, same trip (FR-02).
- **Vehicle model:** true physical states updated every 0.1 s.
- **Sensor layer:** converts true states into measured signals (noise, resolution, estimation).
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

| Signal | Law | Coupled to |
| --- | --- | --- |
| Speed `v` | Approaches the segment target with bounded acceleration and deceleration; `v ≥ 0` | Drive profile |
| Longitudinal accel | Derivative of `v` | Speed |
| Brake pressure, pedal | Non-zero when deceleration exceeds what drag and engine braking provide | Longitudinal accel |
| Yaw rate | Single-track model: `v · δ / (i_s · L)` at low lateral acceleration (`δ` road-wheel angle from steering wheel angle, `i_s` steering ratio, `L` wheelbase) | Speed, steering angle |
| Lateral accel | `v · yaw_rate` | Speed, yaw rate |
| Wheel speeds | `v` plus a cornering offset proportional to `yaw_rate · track / 2` (outer wheels faster), plus slip under hard braking, plus sensor noise | Speed, yaw rate, brake pressure |
| Reference speed | **Estimated from wheel speeds** (not copied from true `v`), as an ESC does | Wheel speeds |
| Disc temperature (×4) | First-order: heats in proportion to braking energy shared per axle, cools exponentially, faster with speed | Brake pressure, speed, ambient |
| Tyre temperature and pressure (×4) | Tyre temperature rises with driving; pressure follows temperature (ideal-gas behaviour) around the model's nominal value | Speed, ambient |
| Coolant temperature | First-order warm-up towards the regulated operating band (thermostat) | Engine running time, load |
| Engine speed | From vehicle speed and a simple gear schedule; idle when stopped | Speed |
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
- Brake disc temperature is an uncalibrated first-order estimate.
- The reference speed is a simple estimate from wheel speeds.
- Fault codes carry a subset of the UDS status bits with a simple cycle-based debounce; UDS services, freeze
  frames and DTC ageing are not simulated.
- Telemetry uses fixed 10 s aggregation windows plus event messages.
- The simulator demonstrates an architecture and a diagnostic workflow, not a validated vehicle model.
