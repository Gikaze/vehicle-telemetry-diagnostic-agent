"""FR-03, FR-17 against LocalStack: the Kinesis publisher delivers a defective stream as published,
and a vehicle's lost messages show as gaps in its sequence counter on the consumer's side.

The test builds its own environment (the LocalStack endpoint from AWS_ENDPOINT_URL, default
http://localhost:4566, and the dummy `test` credentials), so it never reads a profile or the
caller's keys. It creates a uniquely named stream with the publisher's own guarded client, reads
every shard back from TRIM_HORIZON and deletes the stream.
"""

import json
import os
import uuid
from collections import Counter, defaultdict
from itertools import islice

import pytest

from simulator import stream
from simulator.catalogue import catalogue
from simulator.clock import StepClock
from simulator.config_files import start_date, stream_config
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.publishers import KinesisPublisher
from simulator.serialise import Payload, Serialiser
from simulator.transport import TransportRates

pytestmark = pytest.mark.integration

MESSAGES = 400
RATES = TransportRates(gap=0.1, duplicate=0.1, reorder=0.1)
ENVIRON = {
    "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"),
    "AWS_REGION": "eu-central-1",
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
}


def published_payloads() -> list[Payload]:
    """The first MESSAGES delivered messages of 3 vans on day 0, with transport defects."""
    sim = FleetSimulation(42, load_fleet(size=3))
    serialiser = Serialiser(catalogue(), StepClock(start_date(stream_config())))
    return list(islice(stream.payloads(sim, 1, RATES, serialiser), MESSAGES))


def read_all(client, name: str) -> list[tuple[str, bytes]]:
    records: list[tuple[str, bytes]] = []
    for shard in client.list_shards(StreamName=name)["Shards"]:
        iterator = client.get_shard_iterator(
            StreamName=name, ShardId=shard["ShardId"], ShardIteratorType="TRIM_HORIZON"
        )["ShardIterator"]
        for _ in range(1000):
            response = client.get_records(ShardIterator=iterator, Limit=1000)
            records += [(r["PartitionKey"], r["Data"]) for r in response["Records"]]
            if not response["Records"] and response.get("MillisBehindLatest", 0) == 0:
                break
            iterator = response["NextShardIterator"]
    return records


def gaps(pairs: list[tuple[str, bytes]]) -> dict[str, set[int]]:
    """Per vehicle, the sequence numbers missing below its highest one."""
    seqs: dict[str, set[int]] = defaultdict(set)
    for key, data in pairs:
        seqs[key].add(json.loads(data)["seq"])
    return {key: set(range(max(s) + 1)) - s for key, s in seqs.items()}


def out_of_order(pairs: list[tuple[str, bytes]]) -> set[str]:
    """The vehicles with a message whose seq is below the seq of one delivered before it."""
    highest: dict[str, int] = {}
    late: set[str] = set()
    for key, data in pairs:
        seq = json.loads(data)["seq"]
        if seq < highest.get(key, -1):
            late.add(key)
        highest[key] = max(seq, highest.get(key, -1))
    return late


def test_a_defective_stream_arrives_as_published_with_detectable_gaps() -> None:
    payloads = published_payloads()
    published = [(p.partition_key, p.data) for p in payloads]
    # The published stream carries all three defects.
    assert sum(len(missing) for missing in gaps(published).values()) > 0, "no message was lost"
    assert len(published) > len(set(published)), "no duplicate was published"
    assert out_of_order(published), "no message was published out of order"
    name = f"vtda-test-{uuid.uuid4().hex[:12]}"
    publisher = KinesisPublisher(name, ENVIRON)
    client = publisher.client
    client.create_stream(StreamName=name, ShardCount=2)
    try:
        client.get_waiter("stream_exists").wait(StreamName=name)
        publisher.publish(payloads)
        received = read_all(client, name)
    finally:
        client.delete_stream(StreamName=name, EnforceConsumerDeletion=True)
        publisher.close()

    assert Counter(received) == Counter(published)
    assert gaps(received) == gaps(published)
