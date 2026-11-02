"""Day plan and ambient (FR-02, FR-18 operation cycles): bounds, structure and independence."""

import math
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from simulator.config_files import van_model
from simulator.params import ConfigError, config_path
from simulator.profile.ambient import draw_ambient
from simulator.profile.day_plan import draw_day_plan
from simulator.profile_config import (
    check_profile_against_model,
    day_end_margin,
    drive_profile,
    duty_cycle,
    effective_shift_end_latest,
    load_duty_cycle,
    longest_curve_time,
    trip_overrun_bound,
)

seeds = st.integers(min_value=0, max_value=2**32 - 1)
vehicles = st.integers(min_value=1, max_value=999).map(lambda n: f"VAN-{n:03d}")
days = st.integers(min_value=0, max_value=365)


def plan(seed: int, vehicle: str, day: int):
    duty = duty_cycle()
    return draw_day_plan(
        seed, vehicle, day, duty, effective_shift_end_latest(drive_profile(), duty)
    )


@given(seed=seeds, vehicle=vehicles, day=days)
def test_day_plan_respects_its_bounds(seed: int, vehicle: str, day: int) -> None:
    duty = duty_cycle()
    day_plan = plan(seed, vehicle, day)
    assert duty.si("shift_start_earliest") <= day_plan.shift_start <= duty.si("shift_start_latest")
    assert duty.si("shift_end_earliest") <= day_plan.shift_end
    assert day_plan.shift_end <= effective_shift_end_latest(drive_profile(), duty)
    assert duty.value("trips_min") <= day_plan.trips <= duty.value("trips_max")
    assert len(day_plan.breaks) == day_plan.trips - 1
    for duration in day_plan.trip_durations:
        assert duty.si("trip_duration_min") <= duration <= duty.si("trip_duration_max")


@given(seed=seeds, vehicle=vehicles, day=days)
def test_trips_and_breaks_fill_the_shift(seed: int, vehicle: str, day: int) -> None:
    day_plan = plan(seed, vehicle, day)
    total = sum(day_plan.trip_durations) + sum(day_plan.breaks)
    assert total == pytest.approx(day_plan.shift_end - day_plan.shift_start, abs=1e-6)
    last_start = day_plan.planned_trip_starts()[-1]
    assert last_start + day_plan.trip_durations[-1] == pytest.approx(day_plan.shift_end, abs=1e-6)


@given(seed=seeds, vehicle=vehicles, day=days)
def test_one_lunch_break_after_the_middle_trip(seed: int, vehicle: str, day: int) -> None:
    duty = duty_cycle()
    day_plan = plan(seed, vehicle, day)
    assert day_plan.lunch_after_trip == math.ceil(day_plan.trips / 2)
    for after_trip, duration in enumerate(day_plan.breaks, start=1):
        if after_trip == day_plan.lunch_after_trip:
            assert duty.si("lunch_break_min") <= duration <= duty.si("lunch_break_max")
        else:
            assert duty.si("short_break_min") <= duration <= duty.si("short_break_max")


@given(seed=seeds, day=days)
def test_worst_case_overrun_still_ends_by_the_latest_shift_end(seed: int, day: int) -> None:
    """Every trip overrunning by the bound still ends the day by shift_end_latest."""
    duty = duty_cycle()
    day_plan = plan(seed, "VAN-001", day)
    worst_end = day_plan.shift_end + day_plan.trips * trip_overrun_bound(drive_profile())
    assert worst_end <= duty.si("shift_end_latest") + 1e-6


def test_overrun_bound_figures() -> None:
    """600 m * 90 deg at 60 km/h = 56.5 s; + 25 m/s / 1.5 m/s2 + one 0.1 s step = 73.3 s."""
    profile = drive_profile()
    assert longest_curve_time(profile) == pytest.approx(600 * 3.14159265 / 2 / (60 / 3.6), abs=0.01)
    assert trip_overrun_bound(profile) == pytest.approx(56.549 + 25 / 1.5 + 0.1, abs=0.01)
    assert day_end_margin(profile, duty_cycle()) / 60 == pytest.approx(6 * 73.32 / 60, abs=0.01)


def test_no_planned_end_fits_when_the_window_is_too_narrow(tmp_path: Path) -> None:
    """Latest end 16:05 minus a 7.3 min margin is before the 16:00 earliest end."""
    document = yaml.safe_load(config_path("duty_cycle.yaml").read_text())
    document["shift"]["shift_end_latest"]["value"] = 16.08
    path = tmp_path / "duty_cycle.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigError, match="no planned end in the shift end window"):
        check_profile_against_model(drive_profile(), van_model(), load_duty_cycle(path))


def test_day_plan_is_reproducible_and_independent() -> None:
    assert plan(42, "VAN-001", 0) == plan(42, "VAN-001", 0)
    assert plan(42, "VAN-001", 0) != plan(42, "VAN-002", 0)
    assert plan(42, "VAN-001", 0) != plan(42, "VAN-001", 1)
    assert plan(42, "VAN-001", 0) != plan(43, "VAN-001", 0)


@given(seed=seeds, day=days)
def test_ambient_is_in_range_and_shared_by_the_fleet(seed: int, day: int) -> None:
    duty = duty_cycle()
    ambient = draw_ambient(seed, day, duty)
    assert duty.value("ambient_temp_min") <= ambient.temperature <= duty.value("ambient_temp_max")
    assert ambient == draw_ambient(seed, day, duty)
    assert 1.15 < ambient.density < 1.33


def test_ambient_changes_with_the_day() -> None:
    duty = duty_cycle()
    assert draw_ambient(42, 0, duty).temperature != draw_ambient(42, 1, duty).temperature
