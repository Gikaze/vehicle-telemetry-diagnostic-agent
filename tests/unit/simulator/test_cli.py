"""FR-03: the command line. The target, like every setting, is chosen by configuration (argument
or environment variable, the argument winning) with no code change; the Kinesis target reaches
the publisher's guards; the start date comes from the argument, the environment or stream.yaml,
and moves every window_start; --max-messages stops the run; paced mode waits, batch mode never.

Expected bytes come from the stream drivers with the settings the test names (seed, fleet size,
start date), not from the command line. The environment passed to the command line is built by
each test, so the caller's variables never leak in.
"""

import datetime
import io
import json
import re
import shutil
from functools import cache
from itertools import islice
from pathlib import Path

import pytest
import yaml

from simulator import stream
from simulator.catalogue import catalogue
from simulator.cli import main, settings
from simulator.clock import StepClock
from simulator.fleet import load_fleet
from simulator.fleet_state import FleetSimulation
from simulator.params import config_path
from simulator.paths import simulator_home
from simulator.serialise import Payload, Serialiser
from simulator.transport import TransportRates

VANS = 2
COUNT = 5
# The committed stream.yaml, read as plain YAML (stream_config() is cached, and some tests point
# SIM_HOME at an edited copy): its start date and transport rates.
COMMITTED = yaml.safe_load(config_path("stream.yaml").read_text())
YAML_DATE = datetime.date.fromisoformat(COMMITTED["start_date"])
YAML_RATES = TransportRates(
    gap=COMMITTED["transport"]["gap_rate"]["value"],
    duplicate=COMMITTED["transport"]["duplicate_rate"]["value"],
    reorder=COMMITTED["transport"]["reorder_rate"]["value"],
)
LOCAL_AWS = {
    "AWS_ENDPOINT_URL": "http://localhost:4566",
    "AWS_REGION": "eu-central-1",
    "AWS_ACCESS_KEY_ID": "env-key",
    "AWS_SECRET_ACCESS_KEY": "env-secret",
}


def expected(seed: int = 42, start: datetime.date = YAML_DATE, count: int = COUNT) -> list[bytes]:
    return list(expected_run(seed, start, count))


@cache
def expected_run(seed: int, start: datetime.date, count: int) -> tuple[bytes, ...]:
    """The first `count` payloads of the stream drivers, computed once per setting."""
    sim = FleetSimulation(seed, load_fleet(size=VANS))
    serialiser = Serialiser(catalogue(), StepClock(start))
    out = tuple(p.data for p in islice(stream.payloads(sim, 1, YAML_RATES, serialiser), count))
    assert len(out) == count, "the run is shorter than the test expects"
    return out


def lines(data: bytes) -> list[bytes]:
    assert data.endswith(b"\n")
    return data[:-1].split(b"\n")


class FakeKinesis:
    def __init__(self) -> None:
        self.records: list[Payload] = []
        self.closed = False

    def put_records(self, StreamName: str, Records: list[dict]) -> dict:  # noqa: N803
        assert StreamName == "telemetry"
        self.records += [Payload(r["PartitionKey"], r["Data"]) for r in Records]
        return {"FailedRecordCount": 0, "Records": [{"SequenceNumber": "1"} for _ in Records]}

    def close(self) -> None:
        self.closed = True


class Factory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.client = FakeKinesis()

    def __call__(self, *args: str) -> FakeKinesis:
        self.calls.append(args)
        return self.client


def run(argv: list[str], environ: dict[str, str], factory: Factory | None = None, pacer=None):
    """(exit code, standard output bytes) of the command line."""
    out = io.BytesIO()
    code = main(argv, environ, out, factory or Factory(), pacer)
    return code, out.getvalue()


BASE_ARGS = ["--vehicles", str(VANS), "--max-messages", str(COUNT)]
BASE_ENV = {"SIM_VEHICLES": str(VANS), "SIM_MAX_MESSAGES": str(COUNT)}


# --- the target, by argument or environment ----------------------------------------------------


@pytest.mark.parametrize("by", ["argument", "environment"])
def test_stdout_target(by: str) -> None:
    if by == "argument":
        argv, env = BASE_ARGS + ["--target", "stdout"], {}
    else:
        argv, env = [], {**BASE_ENV, "SIM_TARGET": "stdout"}
    code, out = run(argv, env)
    assert code == 0
    assert lines(out) == expected()


@pytest.mark.parametrize("by", ["argument", "environment"])
def test_file_target(by: str, tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    if by == "argument":
        argv, env = BASE_ARGS + ["--target", "file", "--out", str(path)], {}
    else:
        argv, env = [], {**BASE_ENV, "SIM_TARGET": "file", "SIM_OUT_FILE": str(path)}
    code, out = run(argv, env)
    assert (code, out) == (0, b"")
    assert lines(path.read_bytes()) == expected()


@pytest.mark.parametrize("by", ["argument", "environment"])
def test_kinesis_target(by: str) -> None:
    if by == "argument":
        argv, env = BASE_ARGS + ["--target", "kinesis", "--stream-name", "telemetry"], LOCAL_AWS
    else:
        argv = []
        env = {**BASE_ENV, **LOCAL_AWS, "SIM_TARGET": "kinesis", "SIM_KINESIS_STREAM": "telemetry"}
    factory = Factory()
    code, out = run(argv, env, factory)
    assert (code, out) == (0, b"")
    assert factory.calls == [("http://localhost:4566", "eu-central-1", "env-key", "env-secret")]
    assert [p.data for p in factory.client.records] == expected()
    assert {p.partition_key for p in factory.client.records} <= {"VAN-001", "VAN-002"}
    assert factory.client.closed


def test_the_argument_wins_over_the_environment(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    env = {"SIM_TARGET": "stdout", "SIM_SEED": "42", "SIM_OUT_FILE": str(tmp_path / "other")}
    code, out = run(BASE_ARGS + ["--target", "file", "--out", str(path), "--seed", "7"], env)
    assert (code, out) == (0, b"")
    assert lines(path.read_bytes()) == expected(seed=7)
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize(
    ("name", "flag", "env", "arg_text", "env_text", "arg_value", "env_value", "default"),
    [
        ("seed", "--seed", "SIM_SEED", "7", "9", 7, 9, 42),
        ("vehicles", "--vehicles", "SIM_VEHICLES", "3", "4", 3, 4, None),
        ("days", "--days", "SIM_DAYS", "2", "3", 2, 3, 1),
        ("target", "--target", "SIM_TARGET", "stdout", "stdout", "stdout", "stdout", "stdout"),
        ("out", "--out", "SIM_OUT_FILE", "a.jsonl", "b.jsonl", Path("a.jsonl"), Path("b.jsonl"),
         None),
        ("stream_name", "--stream-name", "SIM_KINESIS_STREAM", "s1", "s2", "s1", "s2", None),
        ("mode", "--mode", "SIM_MODE", "paced", "batch", "paced", "batch", "batch"),
        ("start_date", "--start-date", "SIM_START_DATE", "2027-03-01", "2028-02-29",
         datetime.date(2027, 3, 1), datetime.date(2028, 2, 29), None),
        ("max_messages", "--max-messages", "SIM_MAX_MESSAGES", "5", "6", 5, 6, None),
    ],
)  # fmt: skip
def test_each_setting_from_argument_environment_or_default(
    name, flag, env, arg_text, env_text, arg_value, env_value, default
) -> None:
    assert getattr(settings([flag, arg_text], {env: env_text}), name) == arg_value
    assert getattr(settings([], {env: env_text}), name) == env_value
    assert getattr(settings([], {env: ""}), name) == default  # empty: not given
    assert getattr(settings([], {}), name) == default


@pytest.mark.parametrize(
    ("argv", "environ", "named"),
    [
        (["--seed", "x"], {}, "--seed"),
        ([], {"SIM_SEED": "x"}, "SIM_SEED"),
        ([], {"SIM_SEED": "-1"}, "SIM_SEED"),
        ([], {"SIM_VEHICLES": "0"}, "SIM_VEHICLES"),
        ([], {"SIM_DAYS": "-1"}, "SIM_DAYS"),
        ([], {"SIM_TARGET": "s3"}, "SIM_TARGET"),
        ([], {"SIM_MODE": "fast"}, "SIM_MODE"),
        ([], {"SIM_MAX_MESSAGES": "0"}, "SIM_MAX_MESSAGES"),
        ([], {"SIM_START_DATE": "20260105"}, "SIM_START_DATE"),
        (["--start-date", "2026-02-30"], {}, "--start-date"),
        (["--target", "file"], {}, "SIM_OUT_FILE"),
        ([], {"SIM_TARGET": "kinesis"}, "SIM_KINESIS_STREAM"),
    ],
)
def test_a_bad_setting_is_refused_by_name(argv, environ, named, capsys) -> None:
    with pytest.raises(SystemExit) as exit_:
        settings(argv, environ)
    assert exit_.value.code == 2
    assert named in capsys.readouterr().err


# --- the Kinesis guards through the command line -----------------------------------------------


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ({"AWS_ENDPOINT_URL": None}, "AWS_ENDPOINT_URL"),
        ({"AWS_ENDPOINT_URL": "https://kinesis.eu-central-1.amazonaws.com"}, "AWS_ENDPOINT_URL"),
        ({"AWS_ACCESS_KEY_ID": None}, "AWS_ACCESS_KEY_ID"),
        ({"AWS_SECRET_ACCESS_KEY": ""}, "AWS_SECRET_ACCESS_KEY"),
    ],
)
def test_the_kinesis_guards_refuse_through_the_command_line(change, named, capsys) -> None:
    env = {k: v for k, v in {**LOCAL_AWS, **change}.items() if v}
    factory = Factory()
    code, out = run(BASE_ARGS + ["--target", "kinesis", "--stream-name", "telemetry"], env, factory)
    assert (code, out) == (2, b"")
    assert named in capsys.readouterr().err
    assert factory.calls == []


def test_a_failed_publication_exits_with_1(capsys) -> None:
    """Every attempt of every record fails (the publisher's real backoff, 1.5 s in all): exit
    code 1, not the 2 of a refused setting."""
    factory = Factory()
    factory.client.put_records = lambda StreamName, Records: {  # noqa: N803
        "FailedRecordCount": len(Records),
        "Records": [{"ErrorCode": "InternalFailure", "ErrorMessage": "x"} for _ in Records],
    }
    argv = BASE_ARGS + ["--target", "kinesis", "--stream-name", "telemetry"]
    code, _ = run(argv, LOCAL_AWS, factory)
    assert code == 1
    assert "InternalFailure" in capsys.readouterr().err
    assert factory.client.closed


def test_an_unwritable_output_file_is_refused(tmp_path: Path, capsys) -> None:
    code, out = run(BASE_ARGS + ["--target", "file", "--out", str(tmp_path)], {})
    assert (code, out) == (2, b"")
    assert str(tmp_path) in capsys.readouterr().err


# --- the start date ----------------------------------------------------------------------------


def window_dates(data: list[bytes]) -> set[str]:
    return {json.loads(line)["window_start"][:10] for line in data}


def sim_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edits: dict[str, str]) -> None:
    """SIM_HOME on a copy of the simulator data files, stream.yaml edited by `edits`: each regex
    (one line, matched exactly once) replaced by its text."""
    home = tmp_path / "sim-home"
    source = simulator_home()
    home.mkdir()
    shutil.copy(source / "catalogue.yaml", home / "catalogue.yaml")
    shutil.copytree(source / "config", home / "config")
    shutil.copytree(source / "schema", home / "schema")
    path = home / "config" / "stream.yaml"
    text = path.read_text()
    for pattern, replacement in edits.items():
        text, count = re.subn(pattern, replacement, text, flags=re.M)
        assert count == 1, pattern
    path.write_text(text)
    monkeypatch.setenv("SIM_HOME", str(home))


@pytest.fixture
def home_dated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> datetime.date:
    """SIM_HOME whose stream.yaml has another start date."""
    sim_home(tmp_path, monkeypatch, {r'^start_date: "[0-9-]+"$': 'start_date: "2027-06-14"'})
    return datetime.date(2027, 6, 14)


def test_the_transport_rates_come_from_stream_yaml(tmp_path, monkeypatch) -> None:
    """Rates of 0.3 in a copied stream.yaml: the run carries the defects they draw."""
    rate = r"(^  {name}:\n    value: )0$"
    sim_home(
        tmp_path,
        monkeypatch,
        {rate.format(name=n): r"\g<1>0.3" for n in ("gap_rate", "duplicate_rate", "reorder_rate")},
    )
    code, out = run(["--vehicles", str(VANS), "--max-messages", "12"], {})
    assert code == 0
    sim = FleetSimulation(42, load_fleet(size=VANS))
    serialiser = Serialiser(catalogue(), StepClock(YAML_DATE))
    rates = TransportRates(gap=0.3, duplicate=0.3, reorder=0.3)
    want = [p.data for p in islice(stream.payloads(sim, 1, rates, serialiser), 12)]
    assert lines(out) == want
    assert want != expected(count=12)


def test_the_default_start_date_is_stream_yaml_s(home_dated) -> None:
    code, out = run(BASE_ARGS, {})
    assert code == 0
    assert lines(out) == expected(start=home_dated)
    assert window_dates(lines(out)) == {"2027-06-14"}


def test_the_environment_start_date_moves_every_stamp(home_dated) -> None:
    code, out = run(BASE_ARGS, {"SIM_START_DATE": "2028-02-29"})
    assert code == 0
    assert lines(out) == expected(start=datetime.date(2028, 2, 29))


def test_the_argument_start_date_wins(home_dated) -> None:
    code, out = run(BASE_ARGS + ["--start-date", "2025-12-31"], {"SIM_START_DATE": "2028-02-29"})
    assert code == 0
    assert lines(out) == expected(start=datetime.date(2025, 12, 31))


def test_the_start_date_changes_only_the_date(home_dated) -> None:
    """Same bytes once the date part of window_start is set aside."""
    a = run(BASE_ARGS, {"SIM_START_DATE": "2028-02-29"})[1]
    b = run(BASE_ARGS, {"SIM_START_DATE": "2025-12-31"})[1]
    assert a != b
    assert a.replace(b'"window_start":"2028-02-29', b'"window_start":"2025-12-31') == b


# --- mode and stop -----------------------------------------------------------------------------


class FakePacer(stream.Pacer):
    def __init__(self) -> None:
        self.steps: list[int] = []
        self.slept: list[float] = []
        super().__init__(self.slept.append, lambda: 0.0)

    def wait_until(self, step: int) -> None:
        self.steps.append(step)
        super().wait_until(step)


class Flushes(io.BytesIO):
    """Standard output that counts its flushes."""

    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def test_paced_mode_waits_and_flushes_each_step_s_messages() -> None:
    pacer, out = FakePacer(), Flushes()
    assert main(BASE_ARGS + ["--mode", "paced"], {}, out, Factory(), pacer) == 0
    assert lines(out.getvalue()) == expected()
    assert len(pacer.steps) >= 2
    assert pacer.slept  # the fake clock never moves: every later step waits
    assert out.flushes == len(pacer.steps) + 1  # one per step's batch, one at the end


def test_batch_mode_never_waits_and_flushes_only_at_the_end() -> None:
    pacer, out = FakePacer(), Flushes()
    assert main(BASE_ARGS + ["--mode", "batch"], {}, out, Factory(), pacer) == 0
    assert lines(out.getvalue()) == expected()
    assert pacer.steps == []
    assert out.flushes == 1


def test_paced_mode_puts_each_step_s_messages_in_the_file_before_the_next_wait(
    tmp_path: Path,
) -> None:
    """At each wait, the lines on disk are those published so far: they grow at every step."""
    path = tmp_path / "out.jsonl"
    on_disk: list[int] = []

    class Watching(FakePacer):
        def wait_until(self, step: int) -> None:
            on_disk.append(path.read_bytes().count(b"\n"))
            super().wait_until(step)

    argv = BASE_ARGS + ["--mode", "paced", "--target", "file", "--out", str(path)]
    assert main(argv, {}, None, Factory(), Watching()) == 0
    assert lines(path.read_bytes()) == expected()
    assert on_disk[0] == 0
    assert len(on_disk) >= 3
    assert all(b > a for a, b in zip(on_disk, on_disk[1:], strict=False))


class Bounded(io.BytesIO):
    """Standard output that fails the test at once on a line past `limit`, so a run that does
    not stop fails instead of running a whole day."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, data) -> int:
        written = super().write(data)
        assert self.getvalue().count(b"\n") <= self.limit, "published past --max-messages"
        return written


@pytest.mark.parametrize("count", [1, 3, 12])
def test_max_messages_stops_the_run(count: int) -> None:
    out = Bounded(count)
    code = main(["--vehicles", str(VANS), "--max-messages", str(count)], {}, out, Factory())
    assert code == 0
    assert lines(out.getvalue()) == expected(count=count)
