"""Serialisation: sequenced messages to JSON bytes (spec section 5).

One message is one JSON object, ASCII, with no newline (a JSON Lines writer adds one per
message). The layout is fixed, so the same messages always give the same bytes:

- envelope keys in the order message_type, schema_version, vehicle_id, seq, window_start,
  window_s, then `signals` (periodic) or `trigger` (event);
- signals in catalogue order; numeric statistics in the order mean, min, max, last, and boolean
  statistics last, count;
- compact separators, ASCII only, and no NaN or infinity (allow_nan=False raises instead of
  writing a token JSON does not have);
- numbers as Python's shortest round-trip repr of the float64 (0.1, not the 17 significant
  digits 0.10000000000000001); window_s from the whole number of steps (0.3, not
  0.30000000000000004) and window_start from the integer step (clock.py).

The partition key of a message is its vehicle ID, so one vehicle's messages stay in one shard of
a Kinesis stream.
"""

import json
from typing import Any, NamedTuple

from simulator.aggregator import EventRecord, PeriodicRecord
from simulator.catalogue import Catalogue
from simulator.clock import StepClock
from simulator.transport import Sequenced


class Payload(NamedTuple):
    partition_key: str
    data: bytes


class Serialiser:
    def __init__(self, catalogue: Catalogue, clock: StepClock):
        self.clock = clock
        self.schema_version = catalogue.schema_version
        self.numeric = tuple(s.name for s in catalogue.signals if s.is_number)
        self.boolean = tuple(s.name for s in catalogue.signals if not s.is_number)
        self.order = catalogue.names

    def to_dict(self, message: Sequenced) -> dict[str, Any]:
        record = message.record
        if isinstance(record, PeriodicRecord):
            kind, start = "periodic", record.start_step
        elif isinstance(record, EventRecord):
            kind, start = "event", record.step
        else:
            raise TypeError(f"not a record: {record!r}")
        document: dict[str, Any] = {
            "message_type": kind,
            "schema_version": self.schema_version,
            "vehicle_id": record.vehicle_id,
            "seq": message.seq,
            "window_start": self.clock.timestamp(start),
            "window_s": record.window_s,
        }
        if isinstance(record, PeriodicRecord):
            stats: dict[str, dict[str, Any]] = {}
            for name, (mean, low, high, last) in zip(self.numeric, record.numeric, strict=True):
                stats[name] = {"mean": mean, "min": low, "max": high, "last": last}
            for name, (flag, count) in zip(self.boolean, record.boolean, strict=True):
                stats[name] = {"last": flag, "count": count}
            document["signals"] = {name: stats[name] for name in self.order}
        else:
            document["trigger"] = {"signal": record.signal, "value": record.value}
        return document

    def encode(self, message: Sequenced) -> Payload:
        data = json.dumps(
            self.to_dict(message), separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        return Payload(message.record.vehicle_id, data)
