"""FR-03: the publishers. JSON Lines on a stream and in a file; the Kinesis publisher's two guards
(a local endpoint only, credentials from the environment only), checked before any client exists;
no profile or ~/.aws file ever read, proven with a fake ~/.aws under a temporary HOME; PutRecords
requests within the documented limits; failed records retried, then the run fails.

The Kinesis limits are typed here from the Kinesis API reference (PutRecords and
PutRecordsRequestEntry), not read from the module, so a wrong constant fails.
"""

import io
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from driven import short_plans

from simulator import stream
from simulator.catalogue import catalogue
from simulator.clock import StepClock
from simulator.config_files import start_date, stream_config
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.publishers import (
    CredentialsMissing,
    EndpointRefused,
    FilePublisher,
    KinesisPublisher,
    PublishFailed,
    StreamPublisher,
    kinesis_client,
    local_endpoint,
)
from simulator.serialise import Payload, Serialiser
from simulator.transport import TransportRates

# Kinesis API reference: PutRecords "can support up to 500 records. Each record in the request can
# be as large as 10 MiB, up to a limit of 10 MiB for the entire request, including partition
# keys"; PutRecordsRequestEntry: Data at most 10485760 bytes, PartitionKey at most 256 characters.
MAX_RECORDS = 500
MAX_REQUEST_BYTES = 10_485_760
MAX_PARTITION_KEY = 256
# Spec section 5, Publishing: failed records are sent again, up to 5 attempts in all.
MAX_ATTEMPTS = 5

LOCAL = {
    "AWS_ENDPOINT_URL": "http://localhost:4566",
    "AWS_REGION": "eu-central-1",
    "AWS_ACCESS_KEY_ID": "env-key",
    "AWS_SECRET_ACCESS_KEY": "env-secret",
}


@pytest.fixture(scope="module")
def payloads() -> list[Payload]:
    """Payloads of a short 3-van run with transport defects, from the driver."""
    sim = FleetSimulation(42, load_fleet(size=3), short_plans([20.0], [10.0]))
    serialiser = Serialiser(catalogue(), StepClock(start_date(stream_config())))
    rates = TransportRates(gap=0.1, duplicate=0.1, reorder=0.1)
    return list(stream.payloads(sim, 1, rates, serialiser))


def test_the_stream_publisher_writes_one_message_per_line(payloads) -> None:
    out = io.BytesIO()
    publisher = StreamPublisher(out)
    publisher.publish(payloads[:7])
    publisher.publish(payloads[7:])
    publisher.close()
    lines = out.getvalue().split(b"\n")
    assert lines[-1] == b""
    assert lines[:-1] == [p.data for p in payloads]
    assert all(json.loads(line)["vehicle_id"] for line in lines[:-1])


def test_the_stream_publisher_defaults_to_standard_output() -> None:
    assert StreamPublisher().stream is sys.stdout.buffer


def test_the_file_publisher_writes_json_lines(tmp_path: Path, payloads) -> None:
    path = tmp_path / "out.jsonl"
    path.write_bytes(b"a previous run\n")
    publisher = FilePublisher(path)
    publisher.publish(payloads)
    publisher.close()
    assert path.read_bytes() == b"".join(p.data + b"\n" for p in payloads)


class Flushes(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


@pytest.mark.parametrize(("flush", "per_batch"), [(True, 1), (False, 0)])
def test_the_stream_publisher_flushes_each_batch_when_asked(payloads, flush, per_batch) -> None:
    out = Flushes()
    publisher = StreamPublisher(out, flush)
    for i in range(3):
        publisher.publish(payloads[i : i + 1])
        assert out.flushes == per_batch * (i + 1)
    publisher.close()
    assert out.flushes == 3 * per_batch + 1


def test_the_file_publisher_flushes_each_batch_when_asked(tmp_path: Path, payloads) -> None:
    """One message is smaller than the file buffer: on disk at once only if flushed."""
    path = tmp_path / "out.jsonl"
    publisher = FilePublisher(path, flush=True)
    for i in range(3):
        publisher.publish(payloads[i : i + 1])
        assert path.read_bytes() == b"".join(p.data + b"\n" for p in payloads[: i + 1])
    publisher.close()


def test_close_writes_the_last_buffered_message(tmp_path: Path, payloads) -> None:
    """One message is smaller than the file buffer: it is on disk only once the file is closed."""
    path = tmp_path / "out.jsonl"
    publisher = FilePublisher(path)
    publisher.publish(payloads[:1])
    publisher.close()
    assert path.read_bytes() == payloads[0].data + b"\n"


# --- endpoint and credentials guards ---------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:4566",
        "http://127.0.0.1:4566",
        "https://localhost:4566/",
        "http://localhost",
    ],
)
def test_a_local_endpoint_is_accepted(url: str) -> None:
    assert local_endpoint(url) == url


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://kinesis.eu-central-1.amazonaws.com",
        "http://localhost.localstack.cloud:4566",
        "http://localhost.evil.example:4566",
        "http://localhost@evil.example:4566",
        "http://user:pw@localhost:4566",
        "http://evil.example/?host=localhost",
        "http://127.0.0.2:4566",
        "http://[::1]:4566",
        "ftp://localhost:4566",
        "localhost:4566",
        "http://localhost:4566/kinesis",
        "http://localhost:4566/?x=1",
        "http://localhost:99999",
    ],
)
def test_any_other_endpoint_is_refused(url: str | None) -> None:
    with pytest.raises(EndpointRefused):
        local_endpoint(url)


class Factory:
    """A client factory that records its calls and returns a fake client."""

    def __init__(self, client: Any = None):
        self.calls: list[tuple[str, ...]] = []
        self.client = client

    def __call__(self, *args: str) -> Any:
        self.calls.append(args)
        return self.client


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"AWS_ENDPOINT_URL": None}, EndpointRefused),
        ({"AWS_ENDPOINT_URL": ""}, EndpointRefused),
        ({"AWS_ENDPOINT_URL": "https://kinesis.eu-central-1.amazonaws.com"}, EndpointRefused),
        ({"AWS_ACCESS_KEY_ID": None}, CredentialsMissing),
        ({"AWS_SECRET_ACCESS_KEY": ""}, CredentialsMissing),
        ({"AWS_ACCESS_KEY_ID": None, "AWS_SECRET_ACCESS_KEY": None}, CredentialsMissing),
        ({"AWS_REGION": None}, ValueError),
    ],
)
def test_a_refused_setting_creates_no_client(change: dict, error: type) -> None:
    environ = {k: v for k, v in {**LOCAL, **change}.items() if v is not None}
    factory = Factory()
    with pytest.raises(error):
        KinesisPublisher("telemetry", environ, factory)
    assert factory.calls == []


def test_the_client_gets_the_environment_settings() -> None:
    factory = Factory()
    KinesisPublisher("telemetry", LOCAL, factory)
    assert factory.calls == [("http://localhost:4566", "eu-central-1", "env-key", "env-secret")]


def test_a_missing_credential_is_named_never_shown() -> None:
    with pytest.raises(CredentialsMissing) as refused:
        KinesisPublisher("telemetry", {**LOCAL, "AWS_ACCESS_KEY_ID": ""}, Factory())
    assert "AWS_ACCESS_KEY_ID" in str(refused.value)
    assert "AWS_SECRET_ACCESS_KEY" not in str(refused.value)
    assert "env-secret" not in str(refused.value)


# --- no profile and no ~/.aws file, with the real boto3 client --------------------------------

WATCHED: dict[str, Any] = {"dir": None, "opened": []}


def audit(event: str, args: tuple) -> None:
    """Records every file opened under WATCHED["dir"] while it is set (an audit hook stays for
    the life of the interpreter, so it is gated)."""
    watched = WATCHED["dir"]
    if watched and event == "open" and isinstance(args[0], str | bytes | os.PathLike):
        path = os.fsdecode(args[0])
        if path.startswith(watched):
            WATCHED["opened"].append(path)


sys.addaudithook(audit)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[str]]:
    """HOME in a temporary directory with a ~/.aws holding a profile with other keys and region;
    AWS_PROFILE, AWS_DEFAULT_PROFILE, AWS_CONFIG_FILE and AWS_SHARED_CREDENTIALS_FILE point at
    it, and AWS_ENDPOINT_URL_KINESIS at another endpoint. Returns the files opened under it."""
    aws = tmp_path / ".aws"
    aws.mkdir()
    (aws / "credentials").write_text(
        "[default]\naws_access_key_id = DEFAULTKEY\naws_secret_access_key = DEFAULTSECRET\n"
        "[fake]\naws_access_key_id = PROFILEKEY\naws_secret_access_key = PROFILESECRET\n"
    )
    (aws / "config").write_text(
        "[default]\nregion = us-east-1\n[profile fake]\nregion = us-west-2\n"
        "endpoint_url = http://profile.example\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AWS_PROFILE", "fake")
    monkeypatch.setenv("AWS_DEFAULT_PROFILE", "fake")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(aws / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(aws / "credentials"))
    monkeypatch.setenv("AWS_ENDPOINT_URL_KINESIS", "http://env.example")
    for name, value in LOCAL.items():
        monkeypatch.setenv(name, value)
    # The detector is live: an open under the fake ~/.aws is recorded.
    WATCHED.update(dir=str(aws), opened=[])
    (aws / "config").read_text()
    assert WATCHED["opened"] == [str(aws / "config")]
    WATCHED["opened"] = []
    yield lambda: list(WATCHED["opened"])
    WATCHED.update(dir=None, opened=[])


def test_the_client_reads_no_profile_and_no_aws_file(fake_home) -> None:
    publisher = KinesisPublisher("telemetry")
    assert fake_home() == []
    signer = publisher.client._request_signer
    assert signer._credentials.access_key == "env-key"
    assert signer._credentials.secret_key == "env-secret"
    assert publisher.client.meta.region_name == "eu-central-1"
    assert publisher.client.meta.endpoint_url == "http://localhost:4566"
    publisher.close()


def test_missing_keys_never_fall_back_to_the_profile(
    fake_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    with pytest.raises(CredentialsMissing):
        KinesisPublisher("telemetry")
    assert fake_home() == []


def test_the_client_factory_alone_reads_no_aws_file(fake_home) -> None:
    client = kinesis_client("http://127.0.0.1:4566", "eu-central-1", "k", "s")
    assert fake_home() == []
    assert client._request_signer._credentials.access_key == "k"
    client.close()


# --- PutRecords requests ------------------------------------------------------------------------


class FakeKinesis:
    """Records each PutRecords request; `fail(attempt, record)` says which records fail."""

    def __init__(self, fail: Callable[[int, dict], str | None] = lambda attempt, record: None):
        self.requests: list[list[dict]] = []
        self.fail = fail
        self.closed = False

    def put_records(self, StreamName: str, Records: list[dict]) -> dict:  # noqa: N803
        assert StreamName == "telemetry"
        attempt = len(self.requests) + 1
        self.requests.append(Records)
        results = []
        for record in Records:
            code = self.fail(attempt, record)
            results.append(
                {"ErrorCode": code, "ErrorMessage": "x"} if code else {"SequenceNumber": "1"}
            )
        return {"FailedRecordCount": sum("ErrorCode" in r for r in results), "Records": results}

    def close(self) -> None:
        self.closed = True


def kinesis(fake: FakeKinesis, sleeps: list[float] | None = None) -> KinesisPublisher:
    record_sleep = sleeps.append if sleeps is not None else lambda seconds: None
    return KinesisPublisher("telemetry", LOCAL, Factory(fake), record_sleep)


def synthetic(count: int, size: int) -> list[Payload]:
    return [Payload(f"VAN-{i % 7:03d}", str(i).encode().ljust(size, b"x")) for i in range(count)]


def sent(fake: FakeKinesis) -> list[tuple[str, bytes]]:
    return [(r["PartitionKey"], r["Data"]) for request in fake.requests for r in request]


def test_requests_hold_at_most_the_documented_record_count() -> None:
    fake = FakeKinesis()
    batch = synthetic(2 * MAX_RECORDS + 1, 10)
    kinesis(fake).publish(batch)
    assert [len(r) for r in fake.requests] == [MAX_RECORDS, MAX_RECORDS, 1]
    assert sent(fake) == [(p.partition_key, p.data) for p in batch]


def test_requests_hold_at_most_the_documented_bytes() -> None:
    """Records of a third of the limit (key included): three fit in a request, a fourth does
    not; a record of exactly the limit goes alone."""
    key = len(b"VAN-000")
    third = MAX_REQUEST_BYTES // 3 - key
    batch = synthetic(7, third) + [Payload("VAN-000", b"y" * (MAX_REQUEST_BYTES - key))]
    fake = FakeKinesis()
    kinesis(fake).publish(batch)
    sizes = [sum(len(r["PartitionKey"]) + len(r["Data"]) for r in req) for req in fake.requests]
    assert [len(r) for r in fake.requests] == [3, 3, 1, 1]
    assert all(size <= MAX_REQUEST_BYTES for size in sizes)
    assert sent(fake) == [(p.partition_key, p.data) for p in batch]


def test_a_record_over_the_limit_or_a_long_key_is_refused() -> None:
    with pytest.raises(ValueError):
        kinesis(FakeKinesis()).publish([Payload("VAN-000", b"y" * MAX_REQUEST_BYTES)])
    with pytest.raises(ValueError):
        kinesis(FakeKinesis()).publish([Payload("V" * (MAX_PARTITION_KEY + 1), b"{}")])
    fake = FakeKinesis()
    kinesis(fake).publish([Payload("V" * MAX_PARTITION_KEY, b"{}")])
    assert len(fake.requests) == 1


def test_only_the_failed_records_are_sent_again_in_order() -> None:
    """Records 1 and 3 fail on the first attempt, record 3 on the second as well."""
    batch = synthetic(5, 4)
    failing = {1: {batch[1].data, batch[3].data}, 2: {batch[3].data}}

    def fail(attempt: int, record: dict) -> str | None:
        return "InternalFailure" if record["Data"] in failing.get(attempt, set()) else None

    fake, sleeps = FakeKinesis(fail), []
    kinesis(fake, sleeps).publish(batch)
    assert [[r["Data"] for r in request] for request in fake.requests] == [
        [p.data for p in batch],
        [batch[1].data, batch[3].data],
        [batch[3].data],
    ]
    assert len(sleeps) == 2
    assert sleeps[1] == 2 * sleeps[0] > 0


def test_records_failing_every_attempt_fail_the_run() -> None:
    batch = synthetic(3, 4)

    def fail(attempt: int, record: dict) -> str | None:
        return "ProvisionedThroughputExceededException" if record["Data"] == batch[2].data else None

    fake, sleeps = FakeKinesis(fail), []
    with pytest.raises(PublishFailed, match="1 records .* ProvisionedThroughputExceeded"):
        kinesis(fake, sleeps).publish(batch)
    assert len(fake.requests) == MAX_ATTEMPTS
    assert all(request == fake.requests[1] for request in fake.requests[1:])
    assert len(sleeps) == MAX_ATTEMPTS - 1
    assert all(b == 2 * a for a, b in zip(sleeps, sleeps[1:], strict=False))


def test_close_closes_the_client() -> None:
    fake = FakeKinesis()
    publisher = kinesis(fake)
    publisher.close()
    assert fake.closed
