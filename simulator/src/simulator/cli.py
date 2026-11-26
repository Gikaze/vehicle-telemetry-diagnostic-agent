"""vtda-simulator: run the fleet and publish its telemetry (spec section 5, FR-03).

Every setting is an argument with an environment variable; the argument wins over the variable,
the variable over the default. Both go through the same conversion and checks, so the target,
like every other setting, is chosen by configuration with no code change.

    --seed          SIM_SEED            42
    --vehicles      SIM_VEHICLES        the fleet size of fleet.yaml
    --days          SIM_DAYS            1
    --target        SIM_TARGET          stdout (stdout, file or kinesis)
    --out           SIM_OUT_FILE        the JSON Lines file of the file target
    --stream-name   SIM_KINESIS_STREAM  the Kinesis stream of the kinesis target
    --mode          SIM_MODE            batch (batch, or paced: real time)
    --start-date    SIM_START_DATE      start_date of stream.yaml (YYYY-MM-DD)
    --max-messages  SIM_MAX_MESSAGES    none: stop after this many messages

The transport rates are read from stream.yaml only. The Kinesis target reads AWS_ENDPOINT_URL,
AWS_REGION, AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from the environment, and refuses any
endpoint but localhost or 127.0.0.1 (publishers.py). In paced mode the stdout and file targets
are flushed after each step's messages.

Exit codes: 0 when the run is complete; 2 for a refused setting or configuration, before any
message is made (argparse's code for a usage error); 1 when publication fails during the run
(records Kinesis still refuses after the retries), so a caller can tell "misconfigured" from
"target down".
"""

import argparse
import datetime
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from simulator import stream
from simulator.catalogue import catalogue
from simulator.clock import StepClock
from simulator.config_files import load_stream_config, parse_date, start_date
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.params import ConfigError
from simulator.publishers import (
    ClientFactory,
    CredentialsMissing,
    EndpointRefused,
    FilePublisher,
    KinesisPublisher,
    Publisher,
    PublishFailed,
    StreamPublisher,
    kinesis_client,
)
from simulator.serialise import Serialiser
from simulator.transport import TransportRates

TARGETS = ("stdout", "file", "kinesis")
MODES = ("batch", "paced")


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise ValueError(f"must be a positive whole number, got {text!r}")
    return value


def whole(text: str) -> int:
    value = int(text)
    if value < 0:
        raise ValueError(f"must be a whole number, got {text!r}")
    return value


def choice(options: tuple[str, ...]) -> Callable[[str], str]:
    def check(text: str) -> str:
        if text not in options:
            raise ValueError(f"must be one of {', '.join(options)}, got {text!r}")
        return text

    return check


@dataclass(frozen=True)
class Option:
    flag: str
    env: str
    convert: Callable[[str], Any]
    default: Any
    help: str


OPTIONS = {
    "seed": Option("--seed", "SIM_SEED", whole, 42, "root seed"),
    "vehicles": Option("--vehicles", "SIM_VEHICLES", positive, None, "fleet size"),
    "days": Option("--days", "SIM_DAYS", positive, 1, "simulated days"),
    "target": Option("--target", "SIM_TARGET", choice(TARGETS), "stdout", "where to publish"),
    "out": Option("--out", "SIM_OUT_FILE", Path, None, "JSON Lines file (file target)"),
    "stream_name": Option("--stream-name", "SIM_KINESIS_STREAM", str, None, "Kinesis stream"),
    "mode": Option("--mode", "SIM_MODE", choice(MODES), "batch", "batch or paced (real time)"),
    "start_date": Option(
        "--start-date", "SIM_START_DATE", parse_date, None, "simulated date of day 0"
    ),
    "max_messages": Option(
        "--max-messages", "SIM_MAX_MESSAGES", positive, None, "stop after this many messages"
    ),
}


@dataclass(frozen=True)
class Settings:
    seed: int
    vehicles: int | None
    days: int
    target: str
    out: Path | None
    stream_name: str | None
    mode: str
    start_date: datetime.date | None  # None: start_date of stream.yaml
    max_messages: int | None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vtda-simulator",
        description="Run the fleet simulator and publish its telemetry.",
        epilog="Each option can also be set by the environment variable shown; the option wins.",
    )
    for name, option in OPTIONS.items():
        p.add_argument(option.flag, dest=name, metavar=option.env, help=option.help)
    return p


def settings(argv: Sequence[str], environ: Mapping[str, str]) -> Settings:
    """The settings of a run: each from its argument, else its environment variable (unset or
    empty: not given), else its default; both converted and checked the same way."""
    p = parser()
    args = p.parse_args(argv)
    values: dict[str, Any] = {}
    for name, option in OPTIONS.items():
        given = getattr(args, name)
        source = option.flag
        if given is None and environ.get(option.env):
            given, source = environ[option.env], option.env
        if given is None:
            values[name] = option.default
            continue
        try:
            values[name] = option.convert(given)
        except ValueError as error:
            p.error(f"{source}: {error}")
    result = Settings(**values)
    if result.target == "file" and result.out is None:
        p.error("the file target needs --out or SIM_OUT_FILE")
    if result.target == "kinesis" and not result.stream_name:
        p.error("the kinesis target needs --stream-name or SIM_KINESIS_STREAM")
    return result


def publisher(
    run: Settings,
    environ: Mapping[str, str],
    stdout: BinaryIO | None,
    kinesis_factory: ClientFactory,
) -> Publisher:
    flush = run.mode == "paced"
    if run.target == "file":
        assert run.out is not None
        return FilePublisher(run.out, flush)
    if run.target == "kinesis":
        assert run.stream_name is not None
        return KinesisPublisher(run.stream_name, environ, kinesis_factory)
    return StreamPublisher(stdout, flush)


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    stdout: BinaryIO | None = None,
    kinesis_factory: ClientFactory = kinesis_client,
    pacer: stream.Pacer | None = None,
) -> int:
    """Run the simulator; 0 on success, 2 on a refused setting or configuration, 1 on a
    publication that fails during the run."""
    environ = os.environ if environ is None else environ
    run = settings(sys.argv[1:] if argv is None else argv, environ)
    try:
        config = load_stream_config()
        clock = StepClock(run.start_date or start_date(config))
        sim = FleetSimulation(run.seed, load_fleet(size=run.vehicles))
        out = publisher(run, environ, stdout, kinesis_factory)
    except (ConfigError, EndpointRefused, CredentialsMissing, ValueError, OSError) as error:
        print(f"vtda-simulator: error: {error}", file=sys.stderr)
        return 2
    paced = (pacer or stream.Pacer()) if run.mode == "paced" else None
    try:
        stream.run(
            sim,
            run.days,
            TransportRates.from_config(config),
            Serialiser(catalogue(), clock),
            out,
            paced,
            run.max_messages,
        )
    except PublishFailed as error:
        print(f"vtda-simulator: error: {error}", file=sys.stderr)
        return 1
    finally:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
