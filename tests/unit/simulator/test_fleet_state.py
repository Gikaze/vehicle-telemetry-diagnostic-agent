"""Fleet loop (fleet_state.py): call sites, carried state, trip schedule and loop invariants
(spec section 4, FR-16, NFR-15)."""

import math
from dataclasses import dataclass, field, fields, is_dataclass

import numpy as np
import pytest
from driven import lateral_tolerance, short_plans
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.catalogue import catalogue
from simulator.config_files import NOISE_FREE, van_model
from simulator.constants import PHYSICS_DT, RAD_S_TO_RPM, SECONDS_PER_DAY
from simulator.fleet import load_fleet
from simulator.fleet_day import TripTiming, check_trip_overruns
from simulator.fleet_state import STATIC, FleetSample, FleetSimulation, PlanDay
from simulator.model import ecu
from simulator.model.powertrain import select_gear
from simulator.model.thermal import coolant_target_temperature, disc_cooling_tau
from simulator.profile.day_plan import DayPlan
from simulator.profile_config import trip_overrun_bound
from simulator.sensors import encode, error_bound, from_catalogue_unit, to_catalogue_unit

HOUR = 3600.0
WHEEL_SPEEDS = ("wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr")
SUFFIXES = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class EndState:
    """A slot's carried state after the last step of a trip (not emitted: ignition off)."""

    disc: np.ndarray
    tyre: np.ndarray
    coolant: float
    odometer: float
    off_time: float


@dataclass
class Trip:
    slot: int
    rows: list[FleetSample] = field(default_factory=list)
    end: EndState | None = None

    def true(self, get) -> np.ndarray:
        return np.array([get(sample.true)[self.slot] for sample in self.rows])

    def measured(self, name: str) -> np.ndarray:
        return np.array([sample.measured[name][self.slot] for sample in self.rows])

    @property
    def steps(self) -> np.ndarray:
        return np.array([sample.step for sample in self.rows])

    def __repr__(self) -> str:  # the full repr runs to hundreds of MB in failure reports
        return f"Trip(slot={self.slot}, steps={len(self.rows)})"


@dataclass(frozen=True)
class Run:
    sim: FleetSimulation
    samples: list[FleetSample]
    trips: list[list[Trip]]  # per slot, in order

    def __repr__(self) -> str:
        return f"Run(samples={len(self.samples)}, trips={[len(t) for t in self.trips]})"


def drive_fleet(seed: int, vans: int, plan_day: PlanDay, days: int = 1) -> Run:
    sim = FleetSimulation(seed, load_fleet(size=vans), plan_day)
    samples: list[FleetSample] = []
    trips: list[list[Trip]] = [[] for _ in range(vans)]
    current: list[Trip | None] = [None] * vans
    for sample in sim.run(days):
        samples.append(sample)
        for i in np.flatnonzero(sample.active):
            if current[i] is None:
                current[i] = Trip(slot=int(i))
                trips[i].append(current[i])
            current[i].rows.append(sample)
        for i in np.flatnonzero(sample.ended):
            current[i].end = EndState(
                disc=sim.disc_temp[i].copy(),
                tyre=sim.tyre_temp[i].copy(),
                coolant=float(sim.coolant_temp[i]),
                odometer=float(sim.odometer[i]),
                off_time=float(sim.off_time[i]),
            )
            current[i] = None
    assert all(trip is None for trip in current), "a trip did not end"
    return Run(sim, samples, trips)


def all_trips(run: Run) -> list[Trip]:
    return [trip for trips in run.trips for trip in trips]


# Seed 14, four vans, four trips of 300 s: a moving hold (a stop cancels a commanded pull-away
# and the van still moves) and top gear both occur (test_moving_hold_and_top_gear_occur), and
# descents at a true pressure near the brake switch (test_true_pressure_reaches_the_switch_band).
FOUR_TRIPS = short_plans([300.0] * 4, [60.0] * 3)


@pytest.fixture(scope="module")
def run() -> Run:
    return drive_fleet(14, 4, FOUR_TRIPS)


def test_moving_hold_and_top_gear_occur(run: Run) -> None:
    moving_holds = top = 0
    top_gear = run.sim.p.powertrain.top_gear
    for trip in all_trips(run):
        speed, next_speed = (
            trip.true(lambda t: t.drive.speed),
            trip.true(lambda t: t.drive.next_speed),
        )
        distance = trip.true(lambda t: t.drive.next_position - t.drive.position)
        moving_holds += int(np.sum((speed == 0) & (next_speed == 0) & (distance > 0)))
        top += int(np.sum(trip.true(lambda t: t.gear) == top_gear))
    assert moving_holds > 0 and top > 0


def test_every_sample_has_a_driving_vehicle_and_consecutive_steps_within_a_trip(run: Run) -> None:
    assert all(sample.active.any() for sample in run.samples)
    for trip in all_trips(run):
        assert np.all(np.diff(trip.steps) == 1)


def sample_bytes(value) -> bytes:
    """Every array and number of a sample: dataclasses by field, the measured dict in order."""
    if is_dataclass(value):
        return b"".join(sample_bytes(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, dict):
        return b"".join(name.encode() + sample_bytes(item) for name, item in value.items())
    if isinstance(value, tuple):
        return b"".join(sample_bytes(item) for item in value)
    array = np.asarray(value)
    return str(array.dtype).encode() + np.ascontiguousarray(array).tobytes()


def test_samples_do_not_change_once_emitted() -> None:
    """The bytes of each sample, taken when it is emitted, are unchanged at the end of the run:
    later steps and trip starts (another trip of the same van, another van) never write into it."""
    sim = FleetSimulation(3, load_fleet(size=2), short_plans([60.0, 60.0], [30.0]))
    emitted = [(sample, sample_bytes(sample)) for sample in sim.run(1)]
    assert sum(sample.ended.sum() for sample, _ in emitted) == 4
    for sample, data in emitted:
        assert sample_bytes(sample) == data, sample.step


def test_disc_temperatures_follow_the_exact_step_law(run: Run) -> None:
    """T' = T_a + (T - T_a) e^(-DT/tau) + (P/C) tau (1 - e^(-DT/tau)) between consecutive
    samples and into the end state, with P = brake_energy_to_disc * share * F_brake * (s' - s) / DT
    (0 on a held step) and tau at the step's mean speed; a held step never heats."""
    p, brakes = run.sim.p.discs, run.sim.p.brakes
    share = np.array([brakes.front_share / 2] * 2 + [(1 - brakes.front_share) / 2] * 2)
    held_count = 0
    for trip in all_trips(run):
        speed, next_speed = (
            trip.true(lambda t: t.drive.speed),
            trip.true(lambda t: t.drive.next_speed),
        )
        distance = trip.true(lambda t: t.drive.next_position - t.drive.position)
        force = trip.true(lambda t: t.drive.brake_force)
        ambient = np.array([sample.true.ambient.temperature for sample in trip.rows])
        temp = trip.true(lambda t: t.disc_temp)
        held = (speed == 0) & (next_speed == 0)
        power = np.where(held, 0.0, brakes.energy_to_disc * force * distance / PHYSICS_DT)
        tau = disc_cooling_tau((speed + next_speed) / 2, p)[:, None]
        decay = -np.expm1(-PHYSICS_DT / tau)
        capacity = run.sim.capacity[trip.slot]
        expected = (
            temp
            - (temp - ambient[:, None]) * decay
            + power[:, None] * share / capacity * tau * decay
        )
        after = np.vstack([temp[1:], trip.end.disc])
        np.testing.assert_allclose(after, expected, rtol=1e-12, atol=0)
        assert np.all(after[held] <= temp[held])
        held_count += int(np.sum(held & (distance > 0)))
    assert held_count > 0


def test_tyre_temperatures_follow_the_mean_speed_target(run: Run) -> None:
    """T' = T_target + (T - T_target) e^(-DT/tau), T_target from the step's mean speed."""
    p = run.sim.p.tyres
    for trip in all_trips(run):
        speed, next_speed = (
            trip.true(lambda t: t.drive.speed),
            trip.true(lambda t: t.drive.next_speed),
        )
        ambient = np.array([sample.true.ambient.temperature for sample in trip.rows])
        mean = (speed + next_speed) / 2
        target = (ambient + p.warmup_rise * np.minimum(1.0, mean / p.warmup_speed))[:, None]
        temp = trip.true(lambda t: t.tyre_temp)
        after = np.vstack([temp[1:], trip.end.tyre])
        expected = target + (temp - target) * math.exp(-PHYSICS_DT / p.tau)
        np.testing.assert_allclose(after, expected, rtol=1e-12, atol=0)


def test_coolant_follows_the_target_of_the_step_pedal(run: Run) -> None:
    p = run.sim.p.coolant
    for trip in all_trips(run):
        temp = trip.true(lambda t: t.coolant_temp)
        target = coolant_target_temperature(trip.true(lambda t: t.accel_pedal), p)
        after = np.append(temp[1:], trip.end.coolant)
        expected = target + (temp - target) * math.exp(-PHYSICS_DT / p.tau)
        np.testing.assert_allclose(after, expected, rtol=1e-12, atol=0)


def test_pedal_demand_uses_the_start_of_step_speed(run: Run) -> None:
    p = run.sim.p.powertrain
    for trip in all_trips(run):
        speed = trip.true(lambda t: t.drive.speed)
        engine = trip.true(lambda t: t.drive.engine_force)
        demand = trip.true(lambda t: t.pedal_demand)
        expected = np.maximum(engine, 0.0) * np.maximum(speed, p.pedal_min_speed) / p.power_max
        np.testing.assert_allclose(demand, expected, rtol=1e-12, atol=0)
        assert np.all(demand <= 1 + 1e-12)
        assert np.array_equal(trip.true(lambda t: t.accel_pedal), 100 * np.clip(demand, 0, 1))


def test_engine_speed_turns_with_the_rear_wheels_at_the_vehicle_radius(run: Run) -> None:
    """Above idle, n * 2 pi / 60 / ratio_g * r_vehicle / (1 + sigma_rear) == v_k; idle at rest."""
    p = run.sim.p.powertrain
    ratios = np.asarray(p.ratios)
    for trip in all_trips(run):
        speed = trip.true(lambda t: t.drive.speed)
        rpm, gear = trip.true(lambda t: t.engine_speed), trip.true(lambda t: t.gear)
        slip = trip.true(lambda t: t.slip_rear)
        radius = run.sim.radius[trip.slot]
        above = rpm > p.idle
        assert above.any()
        recovered = rpm / RAD_S_TO_RPM / ratios[gear - 1] * radius / (1 + slip)
        np.testing.assert_allclose(recovered[above], speed[above], rtol=1e-12, atol=0)
        assert np.all(rpm[speed == 0] == p.idle) and np.all(gear[speed == 0] == 1)


def test_gear_is_carried_from_step_to_step(run: Run) -> None:
    """Each gear is select_gear of the previous step's emitted gear (first gear before a trip),
    so at most one shift per step; top gear is reached."""
    p, top = run.sim.p.powertrain, 0
    for trip in all_trips(run):
        speed, slip = trip.true(lambda t: t.drive.speed), trip.true(lambda t: t.slip_rear)
        gear = trip.true(lambda t: t.gear)
        previous = np.append(1, gear[:-1])
        radius = np.full(len(gear), run.sim.radius[trip.slot])
        assert np.array_equal(gear, select_gear(previous, speed, slip, radius, p))
        assert np.all(np.abs(np.diff(gear)) <= 1)
        top = max(top, int(gear.max()))
    assert top == p.top_gear


def ecu_from_emitted(sample: FleetSample, p: ecu.EcuParameters) -> dict[str, np.ndarray]:
    signal = catalogue().signal
    wheels = np.stack(
        [from_catalogue_unit(sample.measured[name], signal(name)) for name in WHEEL_SPEEDS],
        axis=-1,
    )
    yaw = from_catalogue_unit(sample.measured["yaw_rate"], signal("yaw_rate"))
    angle = from_catalogue_unit(
        sample.measured["steering_wheel_angle"], signal("steering_wheel_angle")
    )
    braking = sample.measured["brake_pedal_pressed"]
    corrected = ecu.corrected_wheel_speeds(wheels, yaw, angle, p)
    reference = ecu.reference_speed(corrected, braking)
    ref_signal = signal("vehicle_ref_speed")
    return {
        "vehicle_ref_speed": encode(to_catalogue_unit(reference, ref_signal), ref_signal),
        "abs_active": ecu.abs_active(corrected, reference, braking, p),
        "tcs_active": ecu.tcs_active(corrected, reference, braking, p),
        "esc_active": ecu.esc_active(yaw, angle, reference, p),
    }


def test_true_pressure_reaches_the_switch_band(run: Run) -> None:
    """Steps whose true pressure lies within error_bound of brake_switch_pressure (1.35 to
    2.65 bar), where a switch from the measured pressure could differ from the true one: steady
    descents of about 7 to 9.5 % (empty van) or 6.5 to 8 % (laden)."""
    p = run.sim.p
    signal = catalogue().signal("master_cyl_pressure")
    bound = from_catalogue_unit(error_bound("master_cyl_pressure", p.sensors), signal)
    pressure = np.concatenate(
        [trip.true(lambda t: t.master_cyl_pressure) for trip in all_trips(run)]
    )
    assert np.sum(np.abs(pressure - p.brakes.switch_pressure) < bound) > 0


def test_reference_speed_and_flags_come_from_the_emitted_measured_signals(run: Run) -> None:
    """Recomputed from the emitted (encoded) wheel speeds, yaw rate, steering angle and brake
    switch, the reference speed and the flags equal the emitted ones; the switch is the true
    pressure above brake_switch_pressure."""
    p = run.sim.p
    for sample in run.samples:
        active = sample.active
        for name, value in ecu_from_emitted(sample, p.ecu).items():
            assert np.array_equal(value[active], sample.measured[name][active]), name
        pressed = sample.true.master_cyl_pressure > p.brakes.switch_pressure
        assert np.array_equal(sample.measured["brake_pedal_pressed"][active], pressed[active])


def test_odometer_counts_every_step_and_carries_over_between_trips(run: Run) -> None:
    """odometer_(k+1) = odometer_k + (s' - s) (1 + sigma_rear) r_nom / r, into the end state; the
    next trip starts from the end state."""
    p = run.sim.p.wheels
    for trips in run.trips:
        for trip, following_trip in zip(trips, [*trips[1:], None], strict=True):
            odometer = trip.true(lambda t: t.odometer)
            distance = trip.true(lambda t: t.drive.next_position - t.drive.position)
            slip = trip.true(lambda t: t.slip_rear)
            increment = distance * (1 + slip) * p.radius_nominal / run.sim.radius[trip.slot]
            after = np.append(odometer[1:], trip.end.odometer)
            np.testing.assert_allclose(after, odometer + increment, rtol=1e-15, atol=0)
            if following_trip is not None:
                assert following_trip.true(lambda t: t.odometer)[0] == trip.end.odometer


def test_parked_cooling_between_trips(run: Run) -> None:
    """The first sample of a trip is the end state of the previous trip cooled in closed form
    over the break, T_a + (T - T_a) e^(-t / tau) with tau 600 s (discs), 1200 s (tyres) and
    3600 s (coolant) from van_model.yaml; the first trip of the day starts at ambient."""
    model = van_model()
    taus = (
        model.si("disc_cooling_tau_standstill"),
        model.si("tyre_thermal_tau"),
        model.si("coolant_cooldown_tau"),
    )
    assert taus == (600, 1200, 3600)
    for trips in run.trips:
        first = trips[0].rows[0].true
        ambient = first.ambient.temperature
        slot = trips[0].slot
        assert np.all(first.disc_temp[slot] == ambient) and first.coolant_temp[slot] == ambient
        for previous, trip in zip(trips, trips[1:], strict=False):
            t = trip.rows[0].step * PHYSICS_DT - previous.end.off_time
            start = trip.rows[0].true
            for value, end, tau in (
                (start.disc_temp[slot], previous.end.disc, taus[0]),
                (start.tyre_temp[slot], previous.end.tyre, taus[1]),
                (start.coolant_temp[slot], previous.end.coolant, taus[2]),
            ):
                expected = ambient + (end - ambient) * math.exp(-t / tau)
                np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0)
                assert np.all(np.abs(value - ambient) < np.abs(end - ambient))


def test_trips_start_on_the_grid_from_the_actual_ignition_off(run: Run) -> None:
    """First trip at floor(shift_start / DT); each next one at the step after the previous
    ignition off plus floor(break / DT)."""
    for trips in run.trips:
        slot = trips[0].slot
        plan = FOUR_TRIPS(run.sim.root_seed, run.sim.vehicle_ids[slot], 0)
        assert trips[0].steps[0] == math.floor(plan.shift_start / PHYSICS_DT)
        for index, (previous, trip) in enumerate(zip(trips, trips[1:], strict=False)):
            off_step = previous.steps[-1] + 1
            assert previous.end.off_time == off_step * PHYSICS_DT
            assert trip.steps[0] == off_step + math.floor(plan.breaks[index] / PHYSICS_DT)
        assert len(trips) == plan.trips


def test_parked_cooling_across_midnight() -> None:
    """One trip at 23:00 on day 0 and one at 01:00 on day 1: the parked state at 01:00 cools
    towards day 0's ambient until midnight, then towards day 1's."""
    late, early = short_plans([120.0], [], start=23 * HOUR), short_plans([120.0], [], start=HOUR)

    def plan_day(root_seed: int, vehicle_id: str, day: int) -> DayPlan:
        return (late if day == 0 else early)(root_seed, vehicle_id, day)

    run = drive_fleet(5, 1, plan_day, days=2)
    (day0, day1) = run.trips[0]
    assert day1.steps[0] == math.floor((SECONDS_PER_DAY + HOUR + 37) / PHYSICS_DT)
    a0, a1 = (a.temperature for a in run.sim.ambients)
    assert abs(a0 - a1) > 1
    midnight = SECONDS_PER_DAY - day0.end.off_time
    after = day1.steps[0] * PHYSICS_DT - SECONDS_PER_DAY
    start = day1.rows[0].true
    for value, end, tau in (
        (start.disc_temp[0], day0.end.disc, 600.0),
        (start.tyre_temp[0], day0.end.tyre, 1200.0),
        (start.coolant_temp[0], day0.end.coolant, 3600.0),
    ):
        at_midnight = a0 + (end - a0) * math.exp(-midnight / tau)
        expected = a1 + (at_midnight - a1) * math.exp(-after / tau)
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=0)
    assert start.ambient.temperature == a1


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        # A 120 s trip starting at 23:59: still running at midnight.
        (short_plans([120.0], [], start=SECONDS_PER_DAY - 60 - 37), "still running at midnight"),
        # The second trip after a 2 h break from 23:00: scheduled on the next day.
        (short_plans([60.0, 60.0], [2 * HOUR], start=23 * HOUR), "scheduled after midnight"),
    ],
)
def test_a_day_must_end_before_midnight(plan: PlanDay, message: str) -> None:
    """The day plans keep every trip before shift_end_latest (day_end_margin); the loop refuses
    a plan that does not."""
    with pytest.raises(RuntimeError, match=message):
        drive_fleet(5, 1, plan)


def check_invariants(run: Run) -> None:
    """Invariants of every emitted sample of a run, on its active slots."""
    p = run.sim.p
    cat = catalogue()
    active = np.array([sample.active for sample in run.samples])

    def true(get) -> np.ndarray:
        """A true state over the run, its active entries in order."""
        return np.array([get(sample.true) for sample in run.samples])[active]

    m = {name: np.array([s.measured[name] for s in run.samples])[active] for name in cat.names}
    ambient = true(lambda t: np.full(run.sim.n, t.ambient.temperature))
    speed, next_speed = true(lambda t: t.drive.speed), true(lambda t: t.drive.next_speed)
    # Catalogue conformity, no NaN: numbers in range and on the grid, booleans bool.
    for signal in cat.signals:
        value = m[signal.name]
        if signal.is_number:
            low, high = signal.range
            assert np.all((low <= value) & (value <= high)), signal.name
            steps = (value - signal.offset) / signal.resolution
            assert np.all(np.abs(steps - np.round(steps)) <= 1e-6), signal.name
        else:
            assert value.dtype == bool, signal.name
    for name, value in STATIC.items():
        assert np.all(m[name] == value)
    # Noise-free signals are the encoded true values.
    disc = true(lambda t: t.disc_temp)
    true_noise_free = {
        **{f"brake_disc_temp_est_{w}": disc[:, j] for j, w in enumerate(SUFFIXES)},
        "engine_speed": true(lambda t: t.engine_speed),
        "odometer": true(lambda t: t.odometer),
        "ambient_temp": ambient,
        "eps_temperature": ambient,
    }
    assert set(true_noise_free) | {"vehicle_ref_speed"} == set(NOISE_FREE)
    for name, value in true_noise_free.items():
        signal = cat.signal(name)
        assert np.array_equal(m[name], encode(to_catalogue_unit(value, signal), signal)), name
    # accel_long is the actual speed change over the step, bit for bit.
    assert np.array_equal(true(lambda t: t.drive.accel), (next_speed - speed) / PHYSICS_DT)
    # At rest: wheel speeds read exactly 0, true yaw 0, measured yaw within its bound, and the
    # reference speed is 0.
    rest = speed == 0
    yaw = true(lambda t: t.steering.yaw_rate)
    assert rest.any()
    for name in WHEEL_SPEEDS:
        assert np.all(m[name][rest] == 0)
    assert np.all(yaw[rest] == 0)
    assert np.all(np.abs(m["yaw_rate"][rest]) <= error_bound("yaw_rate", p.sensors))
    assert np.all(m["vehicle_ref_speed"][rest] == 0)
    # No step with the accelerator pedal and the brake pedal both applied (true values).
    pedal, pressure = true(lambda t: t.accel_pedal), true(lambda t: t.master_cyl_pressure)
    assert not np.any((pedal > 0) & (pressure > 0))
    # No ABS, TCS or ESC flag on the v1.0 profiles.
    for flag in ("abs_active", "tcs_active", "esc_active"):
        assert not m[flag].any(), flag
    # a_lat on measured signals within the propagated tolerance.
    kmh, deg_s = cat.signal("wheel_speed_rl"), cat.signal("yaw_rate")
    v_m = (
        from_catalogue_unit(m["wheel_speed_rl"], kmh)
        + from_catalogue_unit(m["wheel_speed_rr"], kmh)
    ) / 2
    psi_m = from_catalogue_unit(m["yaw_rate"], deg_s)
    radius = np.broadcast_to(run.sim.radius, active.shape)[active]
    bias = np.abs((1 + true(lambda t: t.slip_rear)) * p.wheels.radius_nominal / radius - 1)
    tolerance = lateral_tolerance(speed, yaw, bias, p.sensors)
    assert np.all(np.abs(m["accel_lat"] - v_m * psi_m) <= tolerance + 1e-9)
    # Heating only with braking: on a step without brake work the discs move towards ambient.
    for trip in all_trips(run):
        temp = trip.true(lambda t: t.disc_temp)
        after = np.vstack([temp[1:], trip.end.disc])
        speed, next_speed = (
            trip.true(lambda t: t.drive.speed),
            trip.true(lambda t: t.drive.next_speed),
        )
        force = trip.true(lambda t: t.drive.brake_force)
        idle = ((speed == 0) & (next_speed == 0)) | (force == 0)
        ambient = np.array([sample.true.ambient.temperature for sample in trip.rows])[:, None]
        rise, next_rise = (temp - ambient)[idle], (after - ambient)[idle]
        assert np.all(np.abs(next_rise) <= np.abs(rise)) and np.all(rise * next_rise >= 0)
    # No trip runs past its planned duration by more than trip_overrun_bound (one-day runs).
    timings = tuple(
        TripTiming(
            run.sim.vehicle_ids[trip.slot],
            index,
            int(trip.steps[0]),
            int(trip.steps[-1]) + 1,
            run.sim.plans[trip.slot].trip_durations[index],
        )
        for trips in run.trips
        for index, trip in enumerate(trips)
    )
    result = check_trip_overruns(timings, trip_overrun_bound(run.sim.profile))
    assert result.passed, result.lines


def test_loop_invariants_on_the_four_trip_run(run: Run) -> None:
    check_invariants(run)


@settings(max_examples=max(5, settings.default.max_examples // 20))
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    durations=st.lists(st.floats(min_value=60.0, max_value=180.0), min_size=2, max_size=2),
    pause=st.floats(min_value=30.0, max_value=90.0),
)
def test_loop_invariants(seed: int, durations: list[float], pause: float) -> None:
    check_invariants(drive_fleet(seed, 3, short_plans(durations, [pause])))
