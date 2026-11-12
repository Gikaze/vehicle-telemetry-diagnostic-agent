"""Publishers: serialised messages to standard output, a JSON Lines file or a Kinesis stream (spec
section 5, FR-03).

A publisher takes batches of payloads in delivery order, `publish(batch)`, and `close()` at the
end of a run. Standard output and the file get one JSON object per line.

The Kinesis publisher only ever talks to a local emulator, and only with credentials it is given:

- endpoint guard: AWS_ENDPOINT_URL must be an http or https URL whose host is `localhost` or
  `127.0.0.1` (LocalStack); unset, empty or any other host is refused (EndpointRefused);
- credentials guard: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are read from the environment and
  passed to the client; if either is unset or empty the run is refused (CredentialsMissing);
- both guards run before any client exists. The client's botocore session has no profile (the
  AWS_PROFILE and AWS_DEFAULT_PROFILE lookup is removed) and reads its config and credentials
  files from os.devnull, so neither ~/.aws/config nor ~/.aws/credentials (nor the files named by
  AWS_CONFIG_FILE and AWS_SHARED_CREDENTIALS_FILE) is ever opened.

PutRecords takes at most KINESIS_MAX_RECORDS records and KINESIS_MAX_REQUEST_BYTES (partition keys
plus data) per request (Kinesis API reference, PutRecords). A record the service fails
(ProvisionedThroughputExceededException or InternalFailure) is sent again, alone with the other
failed records of its request, up to KINESIS_MAX_ATTEMPTS attempts in all, then the run fails
(PublishFailed). PutRecords does not keep order, and a retried record lands after records sent
later: a consumer must order a vehicle's messages by `seq`.
"""

import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from urllib.parse import urlsplit

import boto3
import botocore.session

from simulator.serialise import Payload

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1"})
KINESIS_MAX_RECORDS = 500
KINESIS_MAX_REQUEST_BYTES = 10 * 1024 * 1024  # 10 MiB, also the limit of one record
KINESIS_MAX_PARTITION_KEY = 256  # characters
KINESIS_MAX_ATTEMPTS = 5
KINESIS_FIRST_BACKOFF = 0.1  # s, doubled after each failed attempt


class Publisher(Protocol):
    def publish(self, batch: Sequence[Payload]) -> None: ...

    def close(self) -> None: ...


class EndpointRefused(ValueError):
    """The Kinesis endpoint is not a local emulator."""


class CredentialsMissing(ValueError):
    """AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not set."""


class PublishFailed(RuntimeError):
    """Records still failed after KINESIS_MAX_ATTEMPTS attempts."""


class StreamPublisher:
    """JSON Lines on a binary stream (standard output by default); the stream stays open.
    `flush`: flush after each batch, so a reader of a pipe sees each batch when it is published
    (paced mode), not in buffer-sized bursts."""

    def __init__(self, stream: BinaryIO | None = None, flush: bool = False):
        self.stream = stream if stream is not None else sys.stdout.buffer
        self.flush = flush

    def publish(self, batch: Sequence[Payload]) -> None:
        self.stream.write(b"".join(payload.data + b"\n" for payload in batch))
        if self.flush:
            self.stream.flush()

    def close(self) -> None:
        self.stream.flush()


class FilePublisher:
    """JSON Lines in a file, created or overwritten. `flush`: flush after each batch, so a reader
    tailing the file sees each batch when it is published (paced mode)."""

    def __init__(self, path: Path, flush: bool = False):
        self.file = path.open("wb")
        self.flush = flush

    def publish(self, batch: Sequence[Payload]) -> None:
        self.file.write(b"".join(payload.data + b"\n" for payload in batch))
        if self.flush:
            self.file.flush()

    def close(self) -> None:
        self.file.close()


def local_endpoint(url: str | None) -> str:
    """`url` if it is an http(s) URL of a local emulator (host localhost or 127.0.0.1, no user
    info, path empty or /); EndpointRefused otherwise."""
    if not url:
        raise EndpointRefused("AWS_ENDPOINT_URL is not set: the Kinesis publisher needs LocalStack")
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError as error:
        raise EndpointRefused(f"AWS_ENDPOINT_URL {url!r} is not a URL") from error
    if (
        parts.scheme not in ("http", "https")
        or parts.hostname not in LOCAL_HOSTS
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise EndpointRefused(
            f"AWS_ENDPOINT_URL {url!r} is refused: only http(s)://localhost or 127.0.0.1"
        )
    return url


def credentials(environ: Mapping[str, str]) -> tuple[str, str]:
    """The access key ID and secret access key from the environment; CredentialsMissing names the
    variables that are unset or empty (never a value)."""
    names = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in names if not environ.get(name)]
    if missing:
        raise CredentialsMissing(
            f"{' and '.join(missing)} not set: the Kinesis publisher takes its credentials from "
            "the environment only, never from a profile"
        )
    return environ[names[0]], environ[names[1]]


def kinesis_client(endpoint: str, region: str, access_key: str, secret_key: str) -> Any:
    """A Kinesis client with these settings only: no profile, no config or credentials file."""
    session = botocore.session.Session(session_vars={"profile": (None, None, None, None)})
    session.set_config_variable("config_file", os.devnull)
    session.set_config_variable("credentials_file", os.devnull)
    return boto3.session.Session(botocore_session=session).client(
        "kinesis",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


ClientFactory = Callable[[str, str, str, str], Any]


def requests(batch: Sequence[Payload]) -> Iterator[list[Payload]]:
    """`batch` cut, in order, into PutRecords requests within the record and byte limits."""
    request: list[Payload] = []
    size = 0
    for payload in batch:
        if len(payload.partition_key) > KINESIS_MAX_PARTITION_KEY:
            raise ValueError(f"partition key longer than {KINESIS_MAX_PARTITION_KEY} characters")
        record = len(payload.partition_key.encode()) + len(payload.data)
        if record > KINESIS_MAX_REQUEST_BYTES:
            raise ValueError(f"a record of {record} bytes is over the Kinesis limit")
        if request and (
            len(request) == KINESIS_MAX_RECORDS or size + record > KINESIS_MAX_REQUEST_BYTES
        ):
            yield request
            request, size = [], 0
        request.append(payload)
        size += record
    if request:
        yield request


class KinesisPublisher:
    """PutRecords to a Kinesis stream of a local emulator, settings from the environment."""

    def __init__(
        self,
        stream_name: str,
        environ: Mapping[str, str] = os.environ,
        client_factory: ClientFactory = kinesis_client,
        sleep: Callable[[float], None] = time.sleep,
    ):
        endpoint = local_endpoint(environ.get("AWS_ENDPOINT_URL"))
        access_key, secret_key = credentials(environ)
        region = environ.get("AWS_REGION")
        if not region:
            raise ValueError("AWS_REGION is not set")
        self.stream_name = stream_name
        self.sleep = sleep
        self.client = client_factory(endpoint, region, access_key, secret_key)

    def publish(self, batch: Sequence[Payload]) -> None:
        for request in requests(batch):
            self._put(request)

    def _put(self, request: list[Payload]) -> None:
        pending = request
        backoff = KINESIS_FIRST_BACKOFF
        for attempt in range(1, KINESIS_MAX_ATTEMPTS + 1):
            response = self.client.put_records(
                StreamName=self.stream_name,
                Records=[{"Data": p.data, "PartitionKey": p.partition_key} for p in pending],
            )
            results = response["Records"]
            failed = [p for p, r in zip(pending, results, strict=True) if "ErrorCode" in r]
            if not failed:
                return
            if attempt == KINESIS_MAX_ATTEMPTS:
                codes = sorted({r["ErrorCode"] for r in results if "ErrorCode" in r})
                raise PublishFailed(
                    f"{len(failed)} records still failed after {attempt} attempts: "
                    + ", ".join(codes)
                )
            pending = failed
            self.sleep(backoff)
            backoff *= 2

    def close(self) -> None:
        self.client.close()
