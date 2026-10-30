"""One fleet day (fleet_day.py): trip timings, recording, the day-end, overrun and NaN checks, the
exit status and the zoom window rules, on injected short plans."""

import math
from dataclasses import replace

import numpy as np
import pytest
from driven import short_plans

from simulator.constants import PHYSICS_DT
from simulator.fleet import load_fleet
from simulator.fleet_day import (
    TRUE_COLUMNS,
    CheckResult,
    FleetDay,
    Recording,
    _Rows,
    check_day_end,
    check_no_nan,
    check_trip_overruns,
    clock_time,
    delivery_window,
    describe_window,
    exit_status,
    hottest_disc_window,
    run_fleet_day,
)
from simulator.fleet_state import FleetSimulation
from simulator.profile_config import drive_profile, duty_cycle, trip_overrun_bound

SEED = 5
TWO_TRIPS = short_plans([120.0, 120.0], [30.0])


@pytest.fixture(scope="module")
def day() -> FleetDay:
    return run_fleet_day(SEED, load_fleet(size=2), record="VAN-001", plan_day=TWO_TRIPS)


def test_trip_timings_follow_the_loop(day: FleetDay) -> None:
    """Per van, the first trip starts at floor(shift_start / DT) and the second at the first's
    ignition off plus floor(break / DT); each trip's ignition off is the step after its last
    recorded step."""
    rec = day.recording
    assert rec is not None
    assert list(day.plans) == ["VAN-001", "VAN-002"]
    for vehicle_id, plan in day.plans.items():
        first, second = (t for t in day.trips if t.vehicle_id == vehicle_id)
        assert (first.trip, second.trip) == (0, 1)
        assert first.start_step == math.floor(plan.shift_start / PHYSICS_DT)
        assert second.start_step == first.off_step + math.floor(plan.breaks[0] / PHYSICS_DT)
        assert first.planned_duration == second.planned_duration == 120.0
    for t in (t for t in day.trips if t.vehicle_id == "VAN-001"):
        steps = rec.steps[rec["trip"] == t.trip]
        assert (steps[0], steps[-1] + 1) == (t.start_step, t.off_step)


def test_recording_holds_every_emitted_value_of_one_vehicle(day: FleetDay) -> None:
    """Row by row, the recording equals the samples of VAN-001 with a trip in progress: every
    emitted signal, and the true speed (km/h), accelerations, pressure (bar), gear, held step and
    delivery hold at the start of the step, bit for bit."""
    rec = day.recording
    assert rec is not None
    sim = FleetSimulation(SEED, load_fleet(size=2), TWO_TRIPS)
    rows = [sample for sample in sim.run(1) if sample.active[0]]
    assert np.array_equal(rec.steps, [sample.step for sample in rows])
    for name in sim.catalogue.names:
        assert np.array_equal(rec[name], [float(s.measured[name][0]) for s in rows]), name
    true = {
        "true_speed": lambda t: t.drive.speed[0] * 3.6,
        "true_accel_long": lambda t: t.drive.accel[0],
        "true_accel_lat": lambda t: t.steering.accel_lat[0],
        "true_master_cyl_pressure": lambda t: t.master_cyl_pressure[0] / 1e5,
        "gear": lambda t: t.gear[0],
        "held": lambda t: t.drive.speed[0] == 0 and t.drive.next_speed[0] == 0,
        "at_delivery": lambda t: t.drive.at_delivery[0],
    }
    assert set(true) == set(TRUE_COLUMNS)
    for name, get in true.items():
        assert np.array_equal(rec[name], [float(get(s.true)) for s in rows]), name
    assert rec["held"].any() and rec["true_master_cyl_pressure"].max() > 0


def test_day_end_passes_up_to_the_bound_and_fails_one_step_before(day: FleetDay) -> None:
    last = max(t.off_step for t in day.trips) * PHYSICS_DT
    assert check_day_end(day, duty_cycle().si("shift_end_latest")).passed
    assert check_day_end(day, last).passed
    result = check_day_end(day, last - PHYSICS_DT)
    assert not result.passed
    assert [line for line in result.lines if line.endswith("(too late)")]


def test_day_end_fails_when_a_planned_trip_was_not_driven(day: FleetDay) -> None:
    missing = replace(day, trips=day.trips[:-1])
    result = check_day_end(missing, duty_cycle().si("shift_end_latest"))
    assert not result.passed
    assert f"{day.trips[-1].vehicle_id}: 1 of 2 planned trips driven" in result.lines
    none = check_day_end(replace(day, trips=()), duty_cycle().si("shift_end_latest"))
    assert none.lines == (
        "VAN-001: 0 of 2 planned trips driven",
        "VAN-002: 0 of 2 planned trips driven",
    )


def test_trip_overruns_pass_up_to_the_bound_and_fail_beyond_it(day: FleetDay) -> None:
    worst = max(t.overrun for t in day.trips)
    assert check_trip_overruns(day.trips, trip_overrun_bound(drive_profile())).passed
    assert check_trip_overruns(day.trips, worst).passed
    result = check_trip_overruns(day.trips, worst - PHYSICS_DT / 2)
    assert not result.passed
    assert [line for line in result.lines if " overruns by " in line]


def nan_at(monkeypatch: pytest.MonkeyPatch, slot: int, active: bool) -> None:
    """Make coolant_temp of `slot` NaN once: 10 s into its first trip if `active`, else on the
    first step with the slot parked."""
    original = FleetSimulation._measure
    done = []

    def measure(self, true, z):
        measured = original(self, true, z)
        moving = bool(true.drive.active[slot])
        due = (
            moving and abs(true.drive.time[slot] - 10.0) < PHYSICS_DT / 2 if active else not moving
        )
        if due and not done:
            done.append(True)
            measured["coolant_temp"] = measured["coolant_temp"].copy()
            measured["coolant_temp"][slot] = np.nan
        return measured

    monkeypatch.setattr(FleetSimulation, "_measure", measure)


def test_a_nan_in_an_emitted_signal_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NaN in VAN-002's coolant_temp 10 s into its first trip: reported with its step, and the
    check fails; the clean run passes."""
    plan = short_plans([60.0], [])
    assert check_no_nan(run_fleet_day(SEED, load_fleet(size=2), plan_day=plan)).passed
    nan_at(monkeypatch, slot=1, active=True)
    day = run_fleet_day(SEED, load_fleet(size=2), plan_day=plan)
    step = math.floor(day.plans["VAN-002"].shift_start / PHYSICS_DT) + 100
    assert day.nan_first == {("VAN-002", "coolant_temp"): step}
    result = check_no_nan(day)
    assert not result.passed
    assert result.lines == (f"VAN-002: coolant_temp NaN first at {clock_time(step * PHYSICS_DT)}",)


def test_a_nan_in_a_parked_slot_is_not_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAN-002 starts 37 s after VAN-001: a NaN in its slot before that is never emitted."""
    nan_at(monkeypatch, slot=1, active=False)
    day = run_fleet_day(SEED, load_fleet(size=2), plan_day=short_plans([60.0], []))
    assert day.nan_first == {} and check_no_nan(day).passed


def test_exit_status_is_1_if_any_check_fails() -> None:
    ok, failed = CheckResult("a", True, ("x",)), CheckResult("b", False, ("y",))
    assert exit_status([ok]) == (0, ["PASS a", "  x"])
    assert exit_status([ok, failed]) == (1, ["PASS a", "  x", "FAIL b", "  y"])


def test_clock_time() -> None:
    assert clock_time(18.5 * 3600) == "18:30:00.0"
    assert clock_time(6 * 3600 + 37.25) == "06:00:37.2"


def test_recording_rows_grow_past_the_first_buffer() -> None:
    """The recording buffer (4096 rows at first) doubles when full and keeps every row."""
    rows = _Rows(2)
    expected = np.arange(10000.0).reshape(5000, 2)
    for row in expected:
        rows.append(row)
    assert np.array_equal(rows.array(), expected)


def synthetic(
    peak: int, deliveries: list[tuple[int, int]], other_holds: list[tuple[int, int]] = ()
) -> Recording:
    """Two trips, steps 0..9999 and 20000..29999; the front-right disc peaks at step `peak`;
    delivery stops and other holds as (first step, steps)."""
    steps = np.concatenate([np.arange(10000), np.arange(20000, 30000)])
    trip = (steps >= 20000).astype(float)
    disc = np.where(steps == peak, 300.0, 50.0)
    delivery, held = np.zeros(steps.size), np.zeros(steps.size)
    for first, length in deliveries:
        delivery[(steps >= first) & (steps < first + length)] = 1.0
    for first, length in [*deliveries, *other_holds]:
        held[(steps >= first) & (steps < first + length)] = 1.0
    columns = (
        "step",
        "trip",
        "brake_disc_temp_est_fl",
        "brake_disc_temp_est_fr",
        "at_delivery",
        "held",
    )
    data = np.column_stack([steps, trip, np.full(steps.size, 60.0), disc, delivery, held])
    return Recording("VAN-001", columns, data)


def test_describe_window() -> None:
    rec = synthetic(5000, [(6000, 600)], other_holds=[(6700, 100)])
    assert describe_window(rec, "b", 5700, 180.0) == (
        "zoom b: 00:09:30.0 + 180 s, trip [1], 1800 steps, 600 held at a delivery stop, "
        "100 held elsewhere, front disc up to 60 degC"
    )


@pytest.mark.parametrize(
    ("peak", "start"),
    [
        (5000, 3500),  # the window ends 30 s after the peak
        (100, 0),  # moved to the trip start
        (9990, 8200),  # moved to end at the trip end
        (25000, 23500),  # in the second trip
    ],
)
def test_hottest_disc_window(peak: int, start: int) -> None:
    assert hottest_disc_window(synthetic(peak, []), 180.0) == start


def test_delivery_window_is_the_first_stop_of_the_trip_that_fits() -> None:
    """At most 180 - 30 - 30 s at the stop: a 130 s stop is skipped; the window starts 30 s
    before the van is held; the trip's own stop first, else the day's first that fits."""
    rec = synthetic(5000, [(2000, 1300), (6000, 600), (25000, 1200)])
    assert delivery_window(rec, 0, 180.0) == 5700
    assert delivery_window(rec, 1, 180.0) == 24700
    assert delivery_window(synthetic(5000, [(6000, 600)]), 1, 180.0) == 5700
    with pytest.raises(ValueError, match="no delivery stop"):
        delivery_window(synthetic(5000, [(2000, 1300)]), 0, 180.0)
