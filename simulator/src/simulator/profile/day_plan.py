"""Day plan of one vehicle: shift window, trips (operation cycles) and the breaks between them.

Drawn from the vehicle's SCHEDULE stream of that day, in a fixed order that is part of the
reproducibility contract: shift start, shift end, number of trips, the breaks in order (the lunch
break follows trip ceil(n / 2), the others are short), then one weight per trip. The driving time
(end - start - breaks) is split between the trips in proportion to the weights.

Durations are planned: a trip may overrun its planned duration by at most
profile_config.trip_overrun_bound, and the shift end is drawn early enough for the day to end by
shift_end_latest even if every trip overruns.
"""

import math
from dataclasses import dataclass

from simulator.params import Config
from simulator.rng import Stream, vehicle_generator


@dataclass(frozen=True)
class DayPlan:
    vehicle_id: str
    day: int
    shift_start: float  # s after midnight
    shift_end: float  # s after midnight, planned end of the last trip
    trip_durations: tuple[float, ...]  # s, planned
    breaks: tuple[float, ...]  # s, between consecutive trips (one fewer than trips)
    lunch_after_trip: int  # 1-based number of the trip followed by the lunch break

    @property
    def trips(self) -> int:
        return len(self.trip_durations)

    def planned_trip_starts(self) -> tuple[float, ...]:
        """Start of each trip if every trip lasts exactly its planned duration, s after midnight."""
        starts, time = [], self.shift_start
        for index, duration in enumerate(self.trip_durations):
            starts.append(time)
            time += duration + (self.breaks[index] if index < len(self.breaks) else 0.0)
        return tuple(starts)


def lunch_after(trips: int) -> int:
    return math.ceil(trips / 2)


def draw_day_plan(
    root_seed: int, vehicle_id: str, day: int, duty: Config, shift_end_latest: float
) -> DayPlan:
    """Day plan of one vehicle; `shift_end_latest` is the effective latest planned end, s."""
    rng = vehicle_generator(root_seed, vehicle_id, Stream.SCHEDULE, day)
    start = float(rng.uniform(duty.si("shift_start_earliest"), duty.si("shift_start_latest")))
    end = float(rng.uniform(duty.si("shift_end_earliest"), shift_end_latest))
    trips = int(rng.integers(duty.value("trips_min"), duty.value("trips_max") + 1))
    lunch = lunch_after(trips)
    breaks = []
    for after_trip in range(1, trips):
        if after_trip == lunch:
            low, high = duty.si("lunch_break_min"), duty.si("lunch_break_max")
        else:
            low, high = duty.si("short_break_min"), duty.si("short_break_max")
        breaks.append(float(rng.uniform(low, high)))
    spread = duty.value("trip_weight_spread")
    weights = rng.uniform(1 - spread, 1 + spread, trips)
    driving = end - start - sum(breaks)
    durations = tuple(float(driving * weight / weights.sum()) for weight in weights)
    return DayPlan(
        vehicle_id=vehicle_id,
        day=day,
        shift_start=start,
        shift_end=end,
        trip_durations=durations,
        breaks=tuple(breaks),
        lunch_after_trip=lunch if trips > 1 else 0,
    )
