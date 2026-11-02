"""Fleet loop: every vehicle's state carried across steps, trips and days, one sample per step.

Time is counted in fleet steps k of PHYSICS_DT from the midnight starting day 0; day d covers
steps [d * STEPS_PER_DAY, (d + 1) * STEPS_PER_DAY). At each midnight the loop draws the day's
ambient (fleet AMBIENT stream) and each vehicle's day plan (its SCHEDULE stream). A vehicle's
first trip starts at floor(shift_start / DT); each later trip at the step after the previous
ignition off plus floor(break / DT), so the grid never starts a trip later than the continuous
plan (the day_end_margin bound holds) and the break runs from the actual ignition off. Steps in
which every vehicle is parked are skipped: no sample, no work.

Sampling (spec section 4): sample k describes t_k, the start of step k. Instantaneous states are
their values at t_k (speed, wheel speeds, yaw rate, lateral acceleration, steering angle,
temperatures, pressures, odometer); quantities over the step are held over [t_k, t_k + DT)
(accel_long = (v_(k+1) - v_k) / DT, steering rate, master cylinder pressure and brake switch,
accelerator pedal, steering torques, gear and engine speed, the gear selected from v_k with
step k's rear slip). The thermal laws return end-of-step values: the sample takes their input,
and the value after a trip's last step goes into the parked state (ignition off, not emitted).

Per step, for the slots with a trip in progress (an ended slot keeps its state):

    1. trips starting at k: parked cooling from the ignition off (model/thermal.py), new road,
       curve tracker and sensor noise stream of the trip, s = v = 0
    2. drive.step with the curvature kappa_k of the previous step
    3. curve tracker at s_(k+1), kappa_(k+1); steering_state(kappa_k, kappa_(k+1), v_k)
    4. axle slip from the step's forces; wheel speed readings at v_k and the yaw rate
    5. held step, master cylinder pressure, brake switch
    6. pedal demand at v_k, accelerator pedal
    7. gear from the carried gear at v_k, engine speed
    8. sample of the temperatures, tyre pressures and odometer at t_k, then their updates
    9. sensor layer on the true values, then the brake ECU on the encoded measured wheel speeds,
       yaw rate and steering angle with the brake switch (reference speed, ABS, TCS, ESC)
   10. the sample; slots whose trip ended record their ignition off and schedule the next trip

Measured values are in catalogue units, in catalogue order; booleans are numpy bool arrays.
Signals not modelled in v1.0 hold their nominal value (STATIC); eps_temperature reads the day's
ambient (no EPS thermal model). dtcs come with the fault overlay.
"""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from simulator.catalogue import Catalogue, catalogue
from simulator.config_files import sensors, van_model
from simulator.constants import PHYSICS_DT, SECONDS_PER_DAY
from simulator.fleet import Vehicle, draw_vehicle_parameters, fleet_config
from simulator.model import drive, ecu
from simulator.model.brakes import (
    BrakeParameters,
    brake_pedal_pressed,
    held_steps,
    hold_force,
    master_cylinder_pressure,
)
from simulator.model.longitudinal import LongitudinalParameters
from simulator.model.powertrain import (
    PowertrainParameters,
    accelerator_pedal,
    engine_speed,
    pedal_demand,
    select_gear,
)
from simulator.model.steering import CurveTracker, SteeringParameters, SteeringState, steering_state
from simulator.model.thermal import (
    CoolantParameters,
    DiscParameters,
    TyreParameters,
    coolant_step,
    disc_step,
    parked_coolant,
    parked_discs,
    parked_tyres,
    tyre_pressure,
    tyre_step,
)
from simulator.model.wheels import (
    WheelParameters,
    axle_slip,
    odometer_increment,
    wheel_speed_readings,
)
from simulator.params import UNIT_FACTORS, Config
from simulator.profile.ambient import Ambient, draw_ambient
from simulator.profile.day_plan import DayPlan, draw_day_plan
from simulator.profile.road import Road
from simulator.profile.speed_plan import DriverParameters, SpeedPlanner
from simulator.profile_config import drive_profile, duty_cycle, effective_shift_end_latest
from simulator.sensors import (
    SensorNoise,
    SensorParameters,
    encode,
    from_catalogue_unit,
    measure,
    to_catalogue_unit,
)

STEPS_PER_DAY = round(SECONDS_PER_DAY / PHYSICS_DT)
BAR = UNIT_FACTORS["bar"]
WHEEL_SPEEDS = ("wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr")
SUFFIXES = ("fl", "fr", "rl", "rr")
# Not modelled in v1.0 (spec section 3): nominal unless a scenario sets them.
STATIC = {
    "brake_pad_wear_warning": False,
    "brake_fluid_level_warning": False,
    "eps_derating_active": False,
    "sas_calibrated": True,
}

PlanDay = Callable[[int, str, int], DayPlan]


@dataclass(frozen=True)
class ModelParameters:
    longitudinal: LongitudinalParameters
    driver: DriverParameters
    steering: SteeringParameters
    wheels: WheelParameters
    brakes: BrakeParameters
    powertrain: PowertrainParameters
    discs: DiscParameters
    tyres: TyreParameters
    coolant: CoolantParameters
    ecu: ecu.EcuParameters
    sensors: SensorParameters

    @classmethod
    def from_configs(
        cls, model: Config, profile: Config, sensors_: Config, catalogue_: Catalogue
    ) -> "ModelParameters":
        return cls(
            longitudinal=LongitudinalParameters.from_model(model),
            driver=DriverParameters.from_configs(profile, model),
            steering=SteeringParameters.from_model(model),
            wheels=WheelParameters.from_model(model),
            brakes=BrakeParameters.from_configs(model, profile),
            powertrain=PowertrainParameters.from_model(model),
            discs=DiscParameters.from_model(model),
            tyres=TyreParameters.from_model(model),
            coolant=CoolantParameters.from_model(model),
            ecu=ecu.EcuParameters.from_model(model),
            sensors=SensorParameters.from_configs(sensors_, catalogue_),
        )


@dataclass(frozen=True)
class TrueState:
    """True states of every slot at step k (SI units, temperatures in degC, pedal in %)."""

    drive: drive.Step
    steering: SteeringState
    slip_front: np.ndarray
    slip_rear: np.ndarray
    wheel_speeds: np.ndarray  # (n, 4), m/s, omega * r_nominal before noise
    master_cyl_pressure: np.ndarray  # Pa
    brake_pedal_pressed: np.ndarray  # bool, from the true pressure
    pedal_demand: np.ndarray  # unclipped
    accel_pedal: np.ndarray  # %
    gear: np.ndarray
    engine_speed: np.ndarray  # rpm
    disc_temp: np.ndarray  # (n, 4)
    tyre_temp: np.ndarray  # (n, 4)
    tyre_pressure: np.ndarray  # (n, 4), Pa gauge
    coolant_temp: np.ndarray
    odometer: np.ndarray  # m
    ambient: Ambient


@dataclass(frozen=True)
class FleetSample:
    step: int  # fleet step since the midnight starting day 0
    active: np.ndarray  # bool per slot: a trip in progress at the start of the step
    ended: np.ndarray  # bool per slot: the trip ended in this step (ignition off at t_(k+1))
    vehicle_ids: tuple[str, ...]
    true: TrueState
    measured: dict[str, np.ndarray]  # catalogue units and order; only `active` slots are emitted


def plan_from_duty(root_seed: int, vehicle_id: str, day: int) -> DayPlan:
    """The production day plan (day_plan.py) with the repository's configuration."""
    profile, duty = drive_profile(), duty_cycle()
    latest = effective_shift_end_latest(profile, duty)
    return draw_day_plan(root_seed, vehicle_id, day, duty, latest)


class FleetSimulation:
    """The vehicles of `vehicles`, one slot each, driven day after day from one root seed.

    `plan_day(root_seed, vehicle_id, day)` gives each vehicle's day plan: plan_from_duty in
    production; tests inject short plans through it.
    """

    def __init__(
        self, root_seed: int, vehicles: tuple[Vehicle, ...], plan_day: PlanDay = plan_from_duty
    ):
        model, profile, duty = van_model(), drive_profile(), duty_cycle()
        self.root_seed = root_seed
        self.vehicle_ids = tuple(vehicle.id for vehicle in vehicles)
        self.plan_day = plan_day
        self.profile, self.duty = profile, duty
        self.catalogue = catalogue()
        self.p = ModelParameters.from_configs(model, profile, sensors(), self.catalogue)
        drawn = [draw_vehicle_parameters(root_seed, v, model, fleet_config()) for v in vehicles]
        n = self.n = len(vehicles)
        self.mass = np.array([model.si("mass_empty") + d.payload for d in drawn])
        self.radius = np.array([d.tyre_rolling_radius for d in drawn])
        self.capacity = np.array([d.disc_heat_capacity for d in drawn])
        self.cold_pressure = np.array([d.tyre_pressure_nominal for d in drawn]) * BAR
        self.planner = SpeedPlanner(n, self.p.driver)
        self.tracker = CurveTracker(n)
        self.noise = SensorNoise(n, self.p.sensors)
        self.s, self.v, self.kappa = np.zeros(n), np.zeros(n), np.zeros(n)
        self.ambients: list[Ambient] = []
        # Carried state; temperatures are set to the first day's ambient by the first day.
        self.disc_temp = np.full((n, 4), np.nan)
        self.tyre_temp = np.full((n, 4), np.nan)
        self.coolant_temp = np.full(n, np.nan)
        self.gear = np.ones(n, dtype=int)
        self.odometer = np.array([d.odometer_initial for d in drawn])
        self.off_time = np.zeros(n)  # s, last ignition off (0: parked since the start)
        # Day plan in progress: next trip start (fleet step) and index of that trip.
        self.plans: list[DayPlan | None] = [None] * n
        self.next_start = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        self.next_trip = np.zeros(n, dtype=int)

    def run(self, days: int) -> Iterator[FleetSample]:
        """Samples of `days` days, one per step in which some vehicle drives."""
        none = np.iinfo(np.int64).max
        for day in range(days):
            self._start_day(day)
            day_end = (day + 1) * STEPS_PER_DAY
            k = day * STEPS_PER_DAY
            while True:
                if self.planner.ended.all():
                    k = int(self.next_start.min())
                    if k == none:
                        break
                    if k >= day_end:
                        raise RuntimeError("a trip was scheduled after midnight")
                yield self._step(k)
                k += 1
                if k >= day_end:
                    raise RuntimeError("a trip was still running at midnight")

    def _start_day(self, day: int) -> None:
        ambient = draw_ambient(self.root_seed, day, self.duty)
        self.ambients.append(ambient)
        if day == 0:
            self.disc_temp[:] = ambient.temperature
            self.tyre_temp[:] = ambient.temperature
            self.coolant_temp[:] = ambient.temperature
        for i, vehicle_id in enumerate(self.vehicle_ids):
            plan = self.plan_day(self.root_seed, vehicle_id, day)
            self.plans[i] = plan
            self.next_trip[i] = 0
            self.next_start[i] = day * STEPS_PER_DAY + math.floor(plan.shift_start / PHYSICS_DT)

    def _start_trips(self, k: int) -> None:
        starting = np.flatnonzero(self.next_start == k)
        if starting.size == 0:
            return
        # Copies: the carried arrays may be the previous sample's (samples are never changed).
        self.s, self.v = self.s.copy(), self.v.copy()
        self.disc_temp, self.tyre_temp = self.disc_temp.copy(), self.tyre_temp.copy()
        self.coolant_temp = self.coolant_temp.copy()
        on = np.full(starting.size, k * PHYSICS_DT)
        off = self.off_time[starting]
        temps = [a.temperature for a in self.ambients]
        self.disc_temp[starting] = parked_discs(
            self.disc_temp[starting], off, on, temps, self.p.discs
        )
        self.tyre_temp[starting] = parked_tyres(
            self.tyre_temp[starting], off, on, temps, self.p.tyres
        )
        self.coolant_temp[starting] = parked_coolant(
            self.coolant_temp[starting], off, on, temps, self.p.coolant
        )
        day = k // STEPS_PER_DAY
        mask = np.zeros(self.n, dtype=bool)
        for i in starting:
            plan, trip = self.plans[i], int(self.next_trip[i])
            assert plan is not None
            road = Road(self.root_seed, self.vehicle_ids[i], day, trip, self.profile)
            self.planner.start_trip(int(i), road, plan.trip_durations[trip])
            self.tracker.start_trip(int(i), road)
            self.noise.start_trip(int(i), self.root_seed, self.vehicle_ids[i], day, trip)
            self.s[i], self.v[i] = 0.0, 0.0
            self.next_start[i] = np.iinfo(np.int64).max
            mask[i] = True
        self.tracker.update(self.s, mask)
        self.kappa = np.where(mask, self.tracker.curvature(self.s), self.kappa)

    def _step(self, k: int) -> FleetSample:
        p = self.p
        self._start_trips(k)
        ambient = self.ambients[k // STEPS_PER_DAY]
        # 2. Longitudinal step.
        result = drive.step(
            self.planner, self.s, self.v, self.mass, ambient.density, p.longitudinal
        )
        active, v, v_next = result.active, result.speed, result.next_speed
        # 3. Curvature at the end of the step, steering over the step.
        self.tracker.update(result.next_position, active)
        kappa_next = np.where(active, self.tracker.curvature(result.next_position), 0.0)
        steer = steering_state(self.kappa, kappa_next, v, p.steering)
        # 4. Slip and wheel speeds.
        slip_front, slip_rear = axle_slip(
            result.engine_force, result.brake_force, self.mass, p.wheels
        )
        wheels = wheel_speed_readings(
            v, steer.yaw_rate, slip_front, slip_rear, self.radius, p.wheels
        )
        # 5. Brake pressure and switch.
        held = held_steps(v, v_next)
        holding = hold_force(self.mass, result.grade)
        pressure = master_cylinder_pressure(
            result.brake_force, held, result.at_delivery, holding, p.brakes
        )
        pressed = brake_pedal_pressed(pressure, p.brakes)
        # 6. Accelerator pedal at the start-of-step speed.
        demand = pedal_demand(result.engine_force, v, p.powertrain)
        pedal = accelerator_pedal(demand)
        # 7. Gear carried from the previous step, engine speed.
        self.gear = np.where(
            active, select_gear(self.gear, v, slip_rear, self.radius, p.powertrain), self.gear
        )
        rpm = engine_speed(self.gear, v, slip_rear, self.radius, p.powertrain)
        # 8. States at t_k, then their end-of-step updates on active slots.
        disc, tyre, coolant = self.disc_temp, self.tyre_temp, self.coolant_temp
        odometer = self.odometer
        tyre_p = tyre_pressure(tyre, self.cold_pressure, ambient.pressure, p.tyres)
        distance = result.next_position - result.position
        column = active[:, None]
        self.disc_temp = np.where(
            column,
            disc_step(
                disc,
                result.brake_force,
                v,
                v_next,
                distance,
                ambient.temperature,
                self.capacity,
                p.discs,
                p.brakes,
            ),
            disc,
        )
        self.tyre_temp = np.where(
            column, tyre_step(tyre, v, v_next, ambient.temperature, p.tyres), tyre
        )
        self.coolant_temp = np.where(active, coolant_step(coolant, pedal, p.coolant), coolant)
        self.odometer = odometer + np.where(
            active, odometer_increment(distance, slip_rear, self.radius, p.wheels), 0.0
        )
        true = TrueState(
            drive=result,
            steering=steer,
            slip_front=slip_front,
            slip_rear=slip_rear,
            wheel_speeds=wheels,
            master_cyl_pressure=pressure,
            brake_pedal_pressed=pressed,
            pedal_demand=demand,
            accel_pedal=pedal,
            gear=self.gear,
            engine_speed=rpm,
            disc_temp=disc,
            tyre_temp=tyre,
            tyre_pressure=tyre_p,
            coolant_temp=coolant,
            odometer=odometer,
            ambient=ambient,
        )
        # 9. Sensor layer, then the brake ECU on measured signals.
        measured = self._measure(true, self.noise.draw(active))
        # 10. Carry the position and speed; record ignition off and schedule the next trip.
        self.s, self.v, self.kappa = result.next_position, v_next, kappa_next
        ended = active & self.planner.ended
        for i in np.flatnonzero(ended):
            self.off_time[i] = (k + 1) * PHYSICS_DT
            plan, trip = self.plans[i], int(self.next_trip[i])
            assert plan is not None
            if trip + 1 < plan.trips:
                self.next_trip[i] = trip + 1
                self.next_start[i] = k + 1 + math.floor(plan.breaks[trip] / PHYSICS_DT)
        return FleetSample(
            step=k,
            active=active,
            ended=ended,
            vehicle_ids=self.vehicle_ids,
            true=true,
            measured=measured,
        )

    def _measure(self, true: TrueState, z: np.ndarray) -> dict[str, np.ndarray]:
        p, steer = self.p, true.steering
        values = {
            **{name: true.wheel_speeds[:, j] for j, name in enumerate(WHEEL_SPEEDS)},
            "accel_long": true.drive.accel,
            "accel_lat": steer.accel_lat,
            "yaw_rate": steer.yaw_rate,
            "steering_wheel_angle": steer.steering_wheel_angle,
            "steering_wheel_rate": steer.steering_wheel_rate,
            "driver_torque": steer.driver_torque,
            "eps_assist_torque": steer.assist_torque,
            "master_cyl_pressure": true.master_cyl_pressure,
            **{f"tyre_pressure_{w}": true.tyre_pressure[:, j] for j, w in enumerate(SUFFIXES)},
            **{f"tyre_temp_{w}": true.tyre_temp[:, j] for j, w in enumerate(SUFFIXES)},
            "coolant_temp": true.coolant_temp,
            "accel_pedal_position": true.accel_pedal / 100,
            **{f"brake_disc_temp_est_{w}": true.disc_temp[:, j] for j, w in enumerate(SUFFIXES)},
            "engine_speed": true.engine_speed,
            "odometer": true.odometer,
            "ambient_temp": np.full(self.n, true.ambient.temperature),
            "eps_temperature": np.full(self.n, true.ambient.temperature),
        }
        measured = measure(values, z, p.sensors)
        signals = p.sensors.signals
        # Brake ECU on the encoded measured signals, back in SI.
        wheels = np.stack(
            [from_catalogue_unit(measured[name], signals[name]) for name in WHEEL_SPEEDS], axis=-1
        )
        yaw = from_catalogue_unit(measured["yaw_rate"], signals["yaw_rate"])
        angle = from_catalogue_unit(
            measured["steering_wheel_angle"], signals["steering_wheel_angle"]
        )
        braking = true.brake_pedal_pressed
        corrected = ecu.corrected_wheel_speeds(wheels, yaw, angle, p.ecu)
        reference = ecu.reference_speed(corrected, braking)
        reference_signal = signals["vehicle_ref_speed"]
        measured["vehicle_ref_speed"] = encode(
            to_catalogue_unit(reference, reference_signal), reference_signal
        )
        measured["brake_pedal_pressed"] = braking
        measured["abs_active"] = ecu.abs_active(corrected, reference, braking, p.ecu)
        measured["tcs_active"] = ecu.tcs_active(corrected, reference, braking, p.ecu)
        measured["esc_active"] = ecu.esc_active(yaw, angle, reference, p.ecu)
        for name, value in STATIC.items():
            measured[name] = np.full(self.n, value)
        return {name: measured[name] for name in self.catalogue.names}
