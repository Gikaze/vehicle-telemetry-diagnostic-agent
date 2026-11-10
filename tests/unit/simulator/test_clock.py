"""The simulated clock: fleet steps to ISO 8601 UTC time stamps from the configured start date.

Expected time stamps come from the standard library (datetime plus a timedelta of whole
milliseconds), never from the clock's own arithmetic.
"""

import datetime
import re
from pathlib import Path

import pytest
import yaml

from simulator.clock import StepClock
from simulator.config_files import load_stream_config, start_date, stream_config
from simulator.constants import PHYSICS_DT
from simulator.params import ConfigError, config_path
from simulator.schema import TIMESTAMP_PATTERN

STEPS_PER_DAY = round(86_400 / PHYSICS_DT)


def expected(start: datetime.date, step: int) -> str:
    midnight = datetime.datetime(start.year, start.month, start.day, tzinfo=datetime.UTC)
    moment = midnight + datetime.timedelta(milliseconds=step * round(PHYSICS_DT * 1000))
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def test_every_step_of_a_day_by_the_standard_library() -> None:
    """All 864 000 steps of day 0, and the steps around each midnight of days 1 to 3."""
    start = datetime.date(2026, 1, 5)
    clock = StepClock(start)
    steps = [*range(STEPS_PER_DAY)]
    steps += [d * STEPS_PER_DAY + k for d in range(1, 4) for k in (-1, 0, 1)]
    for step in steps:
        assert clock.timestamp(step) == expected(start, step), step


@pytest.mark.parametrize(
    ("start", "step", "stamp"),
    [
        (datetime.date(2026, 1, 5), 0, "2026-01-05T00:00:00.000Z"),
        (datetime.date(2026, 1, 5), 216_370, "2026-01-05T06:00:37.000Z"),
        (datetime.date(2026, 1, 5), 216_373, "2026-01-05T06:00:37.300Z"),
        # Day 30, 07:00:12.3: past the end of January.
        (datetime.date(2026, 1, 5), 30 * 864_000 + 252_123, "2026-02-04T07:00:12.300Z"),
        # The last step of a year, then the first of the next.
        (datetime.date(2026, 12, 31), 863_999, "2026-12-31T23:59:59.900Z"),
        (datetime.date(2026, 12, 31), 864_000, "2027-01-01T00:00:00.000Z"),
        # A leap day.
        (datetime.date(2028, 2, 28), 864_000 + 5, "2028-02-29T00:00:00.500Z"),
    ],
)
def test_time_stamps_by_hand(start: datetime.date, step: int, stamp: str) -> None:
    assert StepClock(start).timestamp(step) == stamp
    assert re.fullmatch(TIMESTAMP_PATTERN, stamp)


def test_the_start_date_moves_every_stamp() -> None:
    assert StepClock(datetime.date(2026, 3, 2)).timestamp(216_370) == "2026-03-02T06:00:37.000Z"


def test_a_negative_step_is_refused() -> None:
    with pytest.raises(ValueError):
        StepClock(datetime.date(2026, 1, 5)).timestamp(-1)


def test_the_configured_start_date() -> None:
    assert start_date(stream_config()) == datetime.date(2026, 1, 5)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("2026-13-01", "is not a date"),
        ("2026-02-30", "is not a date"),
        ("05/01/2026", "must be YYYY-MM-DD"),
        ("2026-1-5", "must be YYYY-MM-DD"),
        ("20260105", "must be YYYY-MM-DD"),  # ISO basic format: date.fromisoformat accepts it
        (datetime.date(2026, 1, 5), "expected str"),  # an unquoted YAML date
    ],
)
def test_a_malformed_start_date_is_rejected(tmp_path: Path, value: object, message: str) -> None:
    document = yaml.safe_load(config_path("stream.yaml").read_text())
    document["start_date"] = value
    path = tmp_path / "stream.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigError, match=message):
        load_stream_config(path)
