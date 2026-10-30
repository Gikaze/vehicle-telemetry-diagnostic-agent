"""Run one fleet day from a seed, check it and plot one vehicle's day (spec section 7, NFR-15).

    uv run python simulator/tools/vehicle_day.py --seed 42 --vehicles 10 --vehicle VAN-001 \
        --out docs/assets/m1-vehicle-day.png

The run, the checks and the exit status come from simulator.fleet_day; this script parses the
arguments and draws the figure: a full-day overview of the emitted signals (parked periods
shaded, delivery stops marked) and two 3-minute zooms, around the hottest front disc and around
a delivery stop of that trip, with true values where the noise hides a coupling. The PNG carries
no metadata (Agg writes no date; the Software entry is removed), so the same seed and lockfile
give the same bytes. Exit status 1 if a check fails; the PNG is written either way.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from simulator.constants import PHYSICS_DT  # noqa: E402
from simulator.fleet import load_fleet  # noqa: E402
from simulator.fleet_day import (  # noqa: E402
    Recording,
    check_day_end,
    check_no_nan,
    check_trip_overruns,
    clock_time,
    delivery_window,
    describe_window,
    exit_status,
    hottest_disc_window,
    run_fleet_day,
    window_rows,
)
from simulator.profile_config import drive_profile, duty_cycle, trip_overrun_bound  # noqa: E402

ZOOM = 180.0  # s
SUFFIXES = ("fl", "fr", "rl", "rr")
WHEEL_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--vehicle", default="VAN-001")
    parser.add_argument("--out", type=Path, default=Path("docs/assets/m1-vehicle-day.png"))
    for zoom in ("a", "b"):
        parser.add_argument(
            f"--zoom-{zoom}-start",
            type=float,
            help=f"start of zoom {zoom}, s after midnight (default: the rule in fleet_day)",
        )
    return parser.parse_args(argv)


def broken(x: np.ndarray, y: np.ndarray, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x and y with a NaN inserted wherever the steps are not consecutive (no line across a
    parked period)."""
    gaps = np.flatnonzero(np.diff(steps) > 1) + 1
    return np.insert(x, gaps, np.nan), np.insert(y.astype(float), gaps, np.nan)


def shade(ax: Axes, x: np.ndarray, mask: np.ndarray, colour: str, label: str | None) -> None:
    ax.fill_between(
        x,
        0,
        1,
        where=mask,
        step="post",
        color=colour,
        alpha=0.35,
        lw=0,
        label=label,
        transform=ax.get_xaxis_transform(),
    )


def overview(fig, gs, rec: Recording) -> None:
    steps = rec.steps
    hours = steps * PHYSICS_DT / 3600
    trips = rec["trip"].astype(int)
    delivery = rec["at_delivery"] > 0
    rest = np.flatnonzero(np.diff(np.concatenate([[0], delivery.astype(int)])) == 1)
    axes = [fig.add_subplot(gs[i, :]) for i in range(4)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for ax in axes:
        for trip in range(trips.max()):
            end, start = hours[trips == trip][-1], hours[trips == trip + 1][0]
            ax.axvspan(end, start, color="0.85", lw=0)
        ax.grid(alpha=0.3)
    ax = axes[0]
    ax.plot(*broken(hours, rec["vehicle_ref_speed"], steps), lw=0.4, color="k")
    ax.plot(
        hours[rest], np.full(rest.size, -6.0), "^", ms=5, color="tab:orange", label="delivery stop"
    )
    for trip in range(trips.max() + 1):
        ax.annotate(
            f"trip {trip + 1}",
            (hours[trips == trip][0], 1.02),
            xycoords=("data", "axes fraction"),
            fontsize=8,
        )
    ax.set_ylabel("vehicle_ref_speed\nkm/h")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{rec.vehicle_id}: one day (emitted values; parked periods shaded)", pad=14)
    ax = axes[1]
    for suffix, colour in zip(SUFFIXES, WHEEL_COLOURS, strict=True):
        ax.plot(
            *broken(hours, rec[f"brake_disc_temp_est_{suffix}"], steps),
            lw=0.6,
            color=colour,
            label=f"disc {suffix}",
        )
    ax.plot(*broken(hours, rec["coolant_temp"], steps), lw=0.6, color="tab:purple", label="coolant")
    ax.plot(*broken(hours, rec["ambient_temp"], steps), lw=0.8, color="0.4", label="ambient")
    ax.set_ylabel("temperature\ndegC")
    ax.legend(loc="upper right", fontsize=8, ncol=6)
    ax = axes[2]
    right = ax.twinx()
    for suffix, colour in zip(SUFFIXES, WHEEL_COLOURS, strict=True):
        ax.plot(
            *broken(hours, rec[f"tyre_temp_{suffix}"], steps),
            lw=0.4,
            color=colour,
            label=f"tyre temp {suffix}",
        )
        right.plot(
            *broken(hours, rec[f"tyre_pressure_{suffix}"], steps),
            lw=0.4,
            color=colour,
            ls="--",
            label=f"pressure {suffix}",
        )
    ax.set_ylabel("tyre temperature\ndegC (solid)")
    right.set_ylabel("tyre pressure\nbar (dashed)")
    ax = axes[3]
    ax.plot(*broken(hours, rec["odometer"], steps), lw=0.8, color="k")
    ax.set_ylabel("odometer\nkm")
    ax.set_xlabel("time of day, h")


def zoom(fig, gs, column: int, rec: Recording, start: int, title: str) -> None:
    rows = window_rows(rec, start, ZOOM)
    t = (rec.steps[rows] - start) * PHYSICS_DT

    def get(name: str) -> np.ndarray:
        return rec[name][rows]

    axes = [fig.add_subplot(gs[4 + i, column]) for i in range(7)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    delivery, held = get("at_delivery") > 0, get("held") > 0
    for index, ax in enumerate(axes):
        first = index == 0
        shade(ax, t, delivery, "tab:orange", "held, delivery" if first else None)
        shade(ax, t, held & ~delivery, "0.6", "held, other" if first else None)
        ax.grid(alpha=0.3)
    ax = axes[0]
    for suffix, colour in zip(SUFFIXES, WHEEL_COLOURS, strict=True):
        ax.plot(t, get(f"wheel_speed_{suffix}"), lw=0.5, color=colour, label=f"wheel {suffix}")
    ax.plot(t, get("vehicle_ref_speed"), lw=1.0, color="k", label="vehicle_ref_speed")
    ax.plot(t, get("true_speed"), lw=1.0, color="tab:purple", ls="--", label="true speed")
    ax.set_ylabel("speed\nkm/h")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.set_title(title)
    ax = axes[1]
    ax.plot(t, get("accel_long"), lw=0.5, color="tab:blue", label="accel_long")
    ax.plot(t, get("true_accel_long"), lw=0.8, color="navy", label="true accel_long")
    ax.plot(t, get("accel_lat"), lw=0.5, color="tab:red", label="accel_lat")
    ax.plot(t, get("true_accel_lat"), lw=0.8, color="darkred", label="true accel_lat")
    ax.set_ylabel("acceleration\nm/s2")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax = axes[2]
    shade(ax, t, get("brake_pedal_pressed") > 0, "tab:green", "brake_pedal_pressed")
    ax.plot(t, get("master_cyl_pressure"), lw=0.6, color="k", label="master_cyl_pressure")
    ax.plot(
        t, get("true_master_cyl_pressure"), lw=0.8, color="tab:red", ls="--", label="true pressure"
    )
    ax.axhline(2.0, lw=0.8, color="tab:green", ls=":", label="switch 2 bar")
    ax.set_ylabel("brake pressure\nbar")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax = axes[3]
    ax.plot(t, get("accel_pedal_position"), lw=0.6, color="k")
    ax.set_ylabel("accel_pedal_position\n%")
    ax = axes[4]
    ax.plot(t, get("engine_speed"), lw=0.8, color="k", label="engine_speed")
    ax.set_ylabel("engine_speed\nrpm")
    right = ax.twinx()
    right.step(t, get("gear"), where="post", lw=0.8, color="tab:blue")
    right.set_ylabel("gear (true)", color="tab:blue")
    right.set_yticks(range(1, 7))
    ax = axes[5]
    ax.plot(t, get("steering_wheel_angle"), lw=0.8, color="k")
    ax.set_ylabel("steering_wheel_angle\ndeg (black)")
    right = ax.twinx()
    right.plot(t, get("yaw_rate"), lw=0.5, color="tab:red")
    right.set_ylabel("yaw_rate\ndeg/s (red)", color="tab:red")
    ax = axes[6]
    for suffix, colour in zip(SUFFIXES, WHEEL_COLOURS, strict=True):
        ax.plot(
            t, get(f"brake_disc_temp_est_{suffix}"), lw=0.8, color=colour, label=f"disc {suffix}"
        )
    ax.set_ylabel("disc temperature\ndegC")
    ax.legend(loc="upper left", fontsize=7, ncol=4)
    ax.set_xlabel("s after the window start")


def plot(rec: Recording, zoom_a: int, zoom_b: int, out: Path, seed: int) -> None:
    matplotlib.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(16, 26), dpi=100)
    gs = GridSpec(
        11,
        2,
        figure=fig,
        height_ratios=[1.2, 1, 1, 0.6] + [1] * 7,
        hspace=0.45,
        wspace=0.25,
        top=0.965,
        bottom=0.025,
        left=0.07,
        right=0.93,
    )
    overview(fig, gs, rec)
    for column, (label, start) in enumerate((("a", zoom_a), ("b", zoom_b))):
        when = clock_time(start * PHYSICS_DT)
        rule = "hottest front disc" if label == "a" else "delivery stop"
        zoom(fig, gs, column, rec, start, f"zoom {label} ({rule}): {when} + 3 min")
    fig.suptitle(f"Simulated vehicle-day, seed {seed}", y=0.99)
    fig.savefig(out, dpi=100, metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    day = run_fleet_day(args.seed, load_fleet(size=args.vehicles), record=args.vehicle)
    profile, duty = drive_profile(), duty_cycle()
    results = [
        check_day_end(day, duty.si("shift_end_latest")),
        check_trip_overruns(day.trips, trip_overrun_bound(profile)),
        check_no_nan(day),
    ]
    status, lines = exit_status(results)
    print(
        f"seed {args.seed}, {args.vehicles} vehicles: {day.samples} samples, "
        f"loop {day.loop_seconds:.1f} s"
    )
    print("\n".join(lines))
    rec = day.recording
    assert rec is not None
    if args.zoom_a_start is None:
        zoom_a = hottest_disc_window(rec, ZOOM)
    else:
        zoom_a = round(args.zoom_a_start / PHYSICS_DT)
    if args.zoom_b_start is None:
        trip = int(rec["trip"][np.searchsorted(rec.steps, zoom_a)])
        zoom_b = delivery_window(rec, trip, ZOOM)
    else:
        zoom_b = round(args.zoom_b_start / PHYSICS_DT)
    for label, start in (("a", zoom_a), ("b", zoom_b)):
        print(describe_window(rec, label, start, ZOOM) + f" (start {start * PHYSICS_DT:.1f} s)")
    clock = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(rec, zoom_a, zoom_b, args.out, args.seed)
    print(f"plot {time.perf_counter() - clock:.1f} s: {args.out}, {args.out.stat().st_size} bytes")
    return status


if __name__ == "__main__":
    sys.exit(main())
