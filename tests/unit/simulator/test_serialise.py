"""FR-01, FR-16, FR-02: the serialised messages validate against the committed schema, every
emitted value respects the catalogue, checked on the messages themselves, and a seed gives the
same bytes on every run (on one machine and numpy build).

The run uses short plans and transport defects, so duplicates and reordered messages are
serialised too. Hand-made records pin the exact bytes.
"""

import datetime
import json
import math
from fractions import Fraction
from functools import cache
from itertools import pairwise
from typing import Any

import pytest
from driven import short_plans
from jsonschema import Draft202012Validator

from simulator import stream
from simulator.aggregator import EventRecord, PeriodicRecord
from simulator.catalogue import Signal, catalogue
from simulator.clock import StepClock
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.paths import schema_path
from simulator.serialise import Payload, Serialiser
from simulator.transport import Sequenced, TransportRates

SEED = 42
PLANS = short_plans([30.0, 20.0], [10.0])
RATES = TransportRates(gap=0.1, duplicate=0.1, reorder=0.1)
START = datetime.date(2026, 1, 5)
ENVELOPE = ["message_type", "schema_version", "vehicle_id", "seq", "window_start", "window_s"]


@cache
def validator() -> Draft202012Validator:
    """The committed schema, as consumers load it."""
    return Draft202012Validator(json.loads(schema_path().read_text()))


def run_payloads() -> tuple[list[Sequenced], list[Payload]]:
    """The payloads of a run from the driver (stream.payloads), and the sequenced messages of a
    second run of the same seed, to compare them with."""
    serialiser = Serialiser(catalogue(), StepClock(START))
    sim = FleetSimulation(SEED, load_fleet(size=3), PLANS)
    payloads = list(stream.payloads(sim, 1, RATES, serialiser))
    messages = list(stream.messages(FleetSimulation(SEED, load_fleet(size=3), PLANS), 1, RATES))
    return messages, payloads


@pytest.fixture(scope="module")
def run() -> tuple[list[Sequenced], list[Payload]]:
    return run_payloads()


def documents(run) -> list[dict[str, Any]]:
    return [json.loads(p.data) for p in run[1]]


def test_every_message_validates_against_the_committed_schema(run) -> None:
    docs = documents(run)
    assert {doc["vehicle_id"] for doc in docs} == {"VAN-001", "VAN-002", "VAN-003"}
    assert any(doc["window_s"] < 10 for doc in docs)
    for doc in docs:
        validator().validate(doc)


def on_grid(value: float, signal: Signal) -> bool:
    q = (Fraction(value) - Fraction(str(signal.offset))) / Fraction(str(signal.resolution))
    return abs(q - round(q)) < Fraction(1, 10**6)


def test_every_emitted_value_respects_the_catalogue(run) -> None:
    """FR-16 on the messages: each numeric statistic a finite number in the signal's range (or
    exactly its invalid value) on its grid; each boolean `last` true or false and each `count` an
    integer from 0 to the window's samples."""
    checked = 0
    for doc in documents(run):
        if doc["message_type"] != "periodic":
            continue
        samples = round(doc["window_s"] * 10)
        for signal in catalogue().signals:
            stats = doc["signals"][signal.name]
            if signal.is_number:
                assert signal.range is not None
                for value in stats.values():
                    assert isinstance(value, int | float) and not isinstance(value, bool)
                    assert math.isfinite(value), signal.name
                    in_range = signal.range[0] <= value <= signal.range[1]
                    assert in_range or value == signal.invalid, (signal.name, value)
                    assert on_grid(value, signal), (signal.name, value)
                    checked += 1
            else:
                assert isinstance(stats["last"], bool)
                assert isinstance(stats["count"], int) and not isinstance(stats["count"], bool)
                assert 0 <= stats["count"] <= samples
                checked += 2
    assert checked > 0


def written_decimals(text: str) -> int:
    """Decimals of a JSON number as written, trailing zeros stripped (20.330 -> 2, 9.0 -> 0); an
    exponent counts as too many."""
    if "e" in text.lower():
        return 99
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def resolution_decimals(resolution: float) -> int:
    """Decimals of a resolution, from its decimal text (0.01 -> 2, 0.5 -> 1, 1 -> 0)."""
    return written_decimals(repr(float(resolution)))


def test_every_number_is_written_with_its_resolution_s_decimals(run) -> None:
    """FR-16 on the text of the messages: every numeric statistic is written with no more
    decimals than its signal's resolution has (20.33, never 20.330000000000002, which the
    1e-6 grid tolerance would accept), and window_s with at most one."""
    checked = 0
    for payload in run[1]:
        doc = json.loads(payload.data, parse_float=str, parse_int=str)
        assert written_decimals(doc["window_s"]) <= 1, doc["window_s"]
        for signal in catalogue().signals:
            if not signal.is_number:
                continue
            assert signal.resolution is not None
            allowed = resolution_decimals(signal.resolution)
            for text in doc["signals"][signal.name].values():
                assert written_decimals(text) <= allowed, (signal.name, text)
                checked += 1
    assert checked > 0


def test_keys_come_in_the_fixed_order(run) -> None:
    for doc in documents(run):
        assert list(doc) == [*ENVELOPE, "signals"]
        assert list(doc["signals"]) == list(catalogue().names)
        for signal in catalogue().signals:
            expected = ["mean", "min", "max", "last"] if signal.is_number else ["last", "count"]
            assert list(doc["signals"][signal.name]) == expected


def test_the_envelope_carries_the_message_and_its_time(run) -> None:
    """vehicle_id and seq as sequenced; window_start the start step's time from the start date
    (by the standard library); the partition key the vehicle ID; one line of ASCII."""
    messages, payloads = run
    midnight = datetime.datetime(START.year, START.month, START.day, tzinfo=datetime.UTC)
    for message, payload in zip(messages, payloads, strict=True):
        doc = json.loads(payload.data)
        record = message.record
        assert isinstance(record, PeriodicRecord)
        assert (doc["vehicle_id"], doc["seq"]) == (record.vehicle_id, message.seq)
        moment = midnight + datetime.timedelta(milliseconds=100 * record.start_step)
        assert doc["window_start"] == moment.isoformat(timespec="milliseconds")[:-6] + "Z"
        assert doc["window_s"] == record.steps / 10
        assert payload.partition_key == record.vehicle_id
        assert b"\n" not in payload.data and payload.data.isascii()


def test_a_duplicate_has_the_same_bytes(run) -> None:
    messages, payloads = run
    sent = list(zip(messages, payloads, strict=True))
    pairs = [(a, b) for (m, a), (n, b) in pairwise(sent) if m == n]
    assert pairs and all(a == b for a, b in pairs)


def test_the_same_seed_gives_the_same_bytes(run) -> None:
    """A second run of the same seed, in the same process: byte-identical messages."""
    _, again = run_payloads()
    assert [p.data for p in again] == [p.data for p in run[1]]


def numeric_stats() -> tuple:
    values = []
    for signal in catalogue().signals:
        if signal.is_number:
            assert signal.range is not None
            values.append((float(signal.range[0]),) * 4)
    return tuple(values)


def boolean_stats() -> tuple:
    return tuple((False, 0) for s in catalogue().signals if not s.is_number)


def test_window_s_is_written_from_the_steps() -> None:
    """Every window length from 1 to 100 steps is written as its decimal, 0.1 to 10.0 (3 * 0.1
    would print 0.30000000000000004)."""
    serialiser = Serialiser(catalogue(), StepClock(START))
    for steps in range(1, 101):
        record = PeriodicRecord("VAN-001", 0, steps, True, numeric_stats(), boolean_stats())
        text = f"{steps // 10}.{steps % 10}".encode()
        assert b'"window_s":' + text + b"," in serialiser.encode(Sequenced(0, record)).data


def test_an_event_message_byte_for_byte() -> None:
    serialiser = Serialiser(catalogue(), StepClock(START))
    message = Sequenced(17, EventRecord("VAN-042", 270_123, "abs_active", True))
    payload = serialiser.encode(message)
    assert payload == Payload(
        "VAN-042",
        b'{"message_type":"event","schema_version":1,"vehicle_id":"VAN-042","seq":17,'
        b'"window_start":"2026-01-05T07:30:12.300Z","window_s":0.1,'
        b'"trigger":{"signal":"abs_active","value":true}}',
    )
    validator().validate(json.loads(payload.data))


def test_a_nan_is_refused() -> None:
    numeric = list(numeric_stats())
    numeric[0] = (math.nan, 0.0, 0.0, 0.0)
    record = PeriodicRecord("VAN-001", 0, 100, False, tuple(numeric), boolean_stats())
    with pytest.raises(ValueError):
        Serialiser(catalogue(), StepClock(START)).encode(Sequenced(0, record))


def test_only_records_are_serialised() -> None:
    with pytest.raises(TypeError):
        Serialiser(catalogue(), StepClock(START)).encode(Sequenced(0, "VAN-001"))  # type: ignore[arg-type]


def test_the_start_date_applies() -> None:
    record = PeriodicRecord("VAN-001", 216_370, 100, False, numeric_stats(), boolean_stats())
    stamps = [
        json.loads(Serialiser(catalogue(), StepClock(day)).encode(Sequenced(0, record)).data)[
            "window_start"
        ]
        for day in (START, datetime.date(2027, 6, 1))
    ]
    assert stamps == ["2026-01-05T06:00:37.000Z", "2027-06-01T06:00:37.000Z"]
