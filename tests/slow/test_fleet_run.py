"""FR-01, NFR-15: the fleet runs for 24 simulated hours (or day 0 up to a cut), and every message
is valid against the committed schema and free of anomalies.

The run goes through the production stream drivers (stream.stepped_messages, then the
serialiser's JSON bytes), and each payload is parsed back as a consumer would. Every delivered
message is validated with jsonschema against simulator/schema/telemetry.schema.json, the
committed contract (paths.schema_path, never a schema regenerated here), and checked by
plausibility.StreamChecker in delivery order. The first invalid message or anomaly fails the
test with its index, vehicle, seq and window_start. Nothing is kept but counters, the running
SHA-256 of the payload bytes and the checker's bounded state.

Settings (make test-24h sets them):

    TEST_24H_VEHICLES  fleet size, default 100
    TEST_24H_UNTIL     HH:MM of day 0: stop before the samples at that time (stream.samples);
                       unset or empty: the whole day
    TEST_24H_SUMMARY   file the summary is written to once the run has passed

The seed is 42; the start date and transport rates come from stream.yaml. The test prints one
JSON summary line (run with -s): commit, settings, counts, digest, stage times, peak RSS.
"""

import dataclasses
import datetime
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import time
from collections import Counter

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from simulator import stream
from simulator.catalogue import catalogue
from simulator.clock import StepClock
from simulator.config_files import start_date, stream_config
from simulator.constants import PHYSICS_DT
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.paths import schema_path
from simulator.plausibility import AnomalyFound, StreamChecker
from simulator.serialise import Serialiser
from simulator.transport import TransportRates

pytestmark = pytest.mark.slow

SEED = 42


def until_step(text: str) -> int | None:
    """The fleet step of HH:MM on day 0 (0.1 s steps): 09:00 is step 324 000."""
    if not text:
        return None
    match = re.fullmatch(r"([0-9]{2}):([0-9]{2})", text)
    assert match, f"TEST_24H_UNTIL must be HH:MM, got {text!r}"
    hours, minutes = int(match[1]), int(match[2])
    assert hours < 24 and minutes < 60, text
    return round((hours * 3600 + minutes * 60) / PHYSICS_DT)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def test_the_fleet_run_is_valid_and_free_of_anomalies() -> None:
    vehicles = int(os.environ.get("TEST_24H_VEHICLES") or 100)
    until = os.environ.get("TEST_24H_UNTIL", "")
    cut = until_step(until)
    config = stream_config()
    start = start_date(config)
    clock = StepClock(start)
    if cut is not None:
        # The cut is the simulated time asked for: the clock (tested on its own) stamps it so.
        assert clock.timestamp(cut) == f"{start.isoformat()}T{until}:00.000Z"
        cut_stamp = clock.timestamp(cut)

    sim = FleetSimulation(SEED, load_fleet(size=vehicles))
    serialiser = Serialiser(catalogue(), clock)
    validator = Draft202012Validator(json.loads(schema_path().read_text()))
    checker = StreamChecker.from_repository()
    rates = TransportRates.from_config(config)

    seconds = dict.fromkeys(["simulate", "serialise", "parse", "validate", "check"], 0.0)
    digest = hashlib.sha256()
    kinds: Counter[str] = Counter()
    delivered = validated = payload_bytes = 0
    seen: set[str] = set()
    last_start = ""
    wall = time.perf_counter()
    batches = stream.stepped_messages(sim, 1, rates, until_step=cut)
    while True:
        t0 = time.perf_counter()
        batch = next(batches, None)
        t1 = time.perf_counter()
        seconds["simulate"] += t1 - t0
        if batch is None:
            break
        _, messages = batch
        payloads = [serialiser.encode(message) for message in messages]
        t2 = time.perf_counter()
        documents = [json.loads(payload.data) for payload in payloads]
        t3 = time.perf_counter()
        for payload, doc in zip(payloads, documents, strict=True):
            vehicle, seq, window = doc["vehicle_id"], doc["seq"], doc["window_start"]
            where = f"message {delivered} ({vehicle} seq {seq} {window})"
            error = best_match(validator.iter_errors(doc))  # the error inside oneOf at fault
            if error is not None:
                pytest.fail(f"{where} is invalid at {error.json_path}: {error.message[:200]}")
            validated += 1
            delivered += 1
            digest.update(payload.data + b"\n")
            payload_bytes += len(payload.data)
            kinds[doc["message_type"]] += 1
            seen.add(doc["vehicle_id"])
            last_start = max(last_start, doc["window_start"])
        t4 = time.perf_counter()
        for doc in documents:
            try:
                checker.check(doc)
            except AnomalyFound as found:
                pytest.fail(f"anomaly: {found.anomaly}")
        t5 = time.perf_counter()
        seconds["serialise"] += t2 - t1
        seconds["parse"] += t3 - t2
        seconds["validate"] += t4 - t3
        seconds["check"] += t5 - t4
    wall_seconds = time.perf_counter() - wall

    assert delivered > 0
    assert validated == delivered  # every delivered message, none sampled
    assert checker.evaluated["catalogue"] == kinds["periodic"]
    assert checker.evaluated["disc_front_rear_ratio"] > 0  # the ratio rule was live
    if cut is not None:
        assert last_start < cut_stamp
    summary = {
        "test": "fleet run",
        "commit": git("rev-parse", "HEAD").strip(),
        "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "vehicles": vehicles,
        "vehicles_seen": len(seen),
        "until": until or None,
        "until_step": cut,
        "start_date": start.isoformat(),
        "rates": dataclasses.asdict(rates),
        "messages": delivered,
        "validated": validated,
        "kinds": dict(kinds),
        "payload_bytes": payload_bytes,
        "anomalies": 0,
        "evaluated": dict(checker.evaluated),
        "stream_sha256": digest.hexdigest(),
        "last_window_start": last_start,
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "wall_seconds": round(wall_seconds, 1),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    print("\nFLEET RUN SUMMARY " + json.dumps(summary))
    # Written last, once every assertion has passed: make test-24h fails without it (a skipped
    # or deselected test would otherwise leave pytest's exit code at 0).
    path = os.environ.get("TEST_24H_SUMMARY")
    if path:
        with open(path, "w") as out:
            out.write(json.dumps(summary, indent=2) + "\n")
