"""FR-02: the seed tree gives independent, reproducible streams per vehicle and purpose."""

import hashlib
import os
import subprocess
import sys

import pytest

from simulator.rng import Stream, fleet_generator, vehicle_generator, vehicle_key


def test_vehicle_key_is_a_blake2b_digest() -> None:
    digest = hashlib.blake2b(b"VAN-042", digest_size=8).digest()
    assert vehicle_key("VAN-042") == int.from_bytes(digest, "big")


def test_vehicle_key_does_not_depend_on_the_python_hash_seed() -> None:
    """Python's hash() is salted per process; the key must be the same under any hash seed."""
    code = "from simulator.rng import vehicle_key; print(vehicle_key('VAN-042'))"
    keys = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert keys == {str(vehicle_key("VAN-042"))}


def test_same_inputs_give_the_same_stream() -> None:
    a = vehicle_generator(42, "VAN-001", Stream.PROFILE).random(5)
    b = vehicle_generator(42, "VAN-001", Stream.PROFILE).random(5)
    assert a.tolist() == b.tolist()


@pytest.mark.parametrize(
    ("seed", "vehicle", "stream"),
    [
        (43, "VAN-001", Stream.PROFILE),
        (42, "VAN-002", Stream.PROFILE),
        (42, "VAN-001", Stream.SCHEDULE),
    ],
)
def test_any_changed_input_gives_another_stream(seed: int, vehicle: str, stream: Stream) -> None:
    reference = vehicle_generator(42, "VAN-001", Stream.PROFILE).random(5).tolist()
    assert vehicle_generator(seed, vehicle, stream).random(5).tolist() != reference


def test_fleet_stream_differs_from_vehicle_streams() -> None:
    fleet = fleet_generator(42, Stream.AMBIENT).random(5).tolist()
    assert fleet != vehicle_generator(42, "VAN-001", Stream.AMBIENT).random(5).tolist()
    assert fleet == fleet_generator(42, Stream.AMBIENT).random(5).tolist()


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        vehicle_generator(-1, "VAN-001", Stream.PROFILE)


def test_stream_numbering_contract() -> None:
    """Renumbering a stream would silently change every run's data."""
    assert {s.name: s.value for s in Stream} == {
        "PARAMETERS": 1,
        "SCHEDULE": 2,
        "PROFILE": 3,
        "SENSOR_NOISE": 4,
        "TRANSPORT": 5,
        "SCENARIO": 6,
        "AMBIENT": 7,
    }
