"""Simulated clock: fleet steps to simulated UTC time stamps (spec section 5).

Time is the integer fleet step everywhere in the simulator; step 0 is midnight (UTC) at the
start of the clock's start date (stream.yaml carries one, `start_date`; config_files.start_date
reads it). A step becomes a time stamp in
integer milliseconds, step * MS_PER_STEP, never through float seconds, so step 216 370 is exactly
06:00:37.000 and no rounding accumulates over days. The format is ISO 8601 UTC with
milliseconds, the schema's TIMESTAMP_PATTERN: 2026-01-05T06:00:37.000Z.
"""

import datetime

from simulator.constants import PHYSICS_DT

MS_PER_STEP = round(PHYSICS_DT * 1000)
MS_PER_DAY = 86_400_000


class StepClock:
    """Time stamps of fleet steps from a start date (step 0 = its midnight, UTC)."""

    def __init__(self, start_date: datetime.date):
        self.start_date = start_date

    def timestamp(self, step: int) -> str:
        if step < 0:
            raise ValueError("a fleet step is never negative")
        days, ms = divmod(step * MS_PER_STEP, MS_PER_DAY)
        date = self.start_date + datetime.timedelta(days=days)
        seconds, millis = divmod(ms, 1000)
        minutes, second = divmod(seconds, 60)
        hour, minute = divmod(minutes, 60)
        return f"{date.isoformat()}T{hour:02d}:{minute:02d}:{second:02d}.{millis:03d}Z"
