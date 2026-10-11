"""FR-02: the seed tree gives independent, reproducible streams per vehicle and purpose."""

import hashlib
import os
import subprocess
import sys

import numpy as np
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


def test_unindexed_streams_keep_their_committed_entropy() -> None:
    """Adding indexed streams must not change the streams already drawn (seeding contract)."""
    expected = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([1, vehicle_key("VAN-001"), 1, 42]))
    )
    actual = vehicle_generator(42, "VAN-001", Stream.PARAMETERS)
    assert actual.random(3).tolist() == expected.random(3).tolist()
    fleet_expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0, 7, 42])))
    assert (
        fleet_generator(42, Stream.AMBIENT).random(3).tolist() == fleet_expected.random(3).tolist()
    )


def test_seed_sequence_pads_short_entropy_with_zeros() -> None:
    """The NumPy behaviour that makes separate namespaces for indexed streams necessary."""
    state = lambda entropy: np.random.SeedSequence(entropy).generate_state(4).tolist()  # noqa: E731
    assert state([1, 2, 3]) == state([1, 2, 3, 0])


@pytest.mark.parametrize("root", [0, 1, 7, 42])
def test_indexed_streams_never_reproduce_unindexed_ones(root: int) -> None:
    """Fleet stream [0, 7, r] is padded to [0, 7, r, 0]; an index must not land on it."""
    unindexed = fleet_generator(root, Stream.AMBIENT).random(3).tolist()
    for index in (0, 1, 7, 42):
        assert fleet_generator(index, Stream.AMBIENT, root).random(3).tolist() != unindexed
        assert fleet_generator(root, Stream.AMBIENT, index).random(3).tolist() != unindexed
    vehicle = vehicle_generator(root, "VAN-001", Stream.PROFILE).random(3).tolist()
    assert vehicle_generator(root, "VAN-001", Stream.PROFILE, 0).random(3).tolist() != vehicle


def test_indices_select_independent_streams() -> None:
    draws = {
        indices: vehicle_generator(42, "VAN-001", Stream.PROFILE, *indices).random(3).tolist()
        for indices in [(0,), (1,), (0, 0), (0, 1), (1, 0)]
    }
    assert len({tuple(value) for value in draws.values()}) == len(draws)
    again = vehicle_generator(42, "VAN-001", Stream.PROFILE, 0, 1).random(3).tolist()
    assert again == draws[(0, 1)]


def test_negative_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="indices must be non-negative"):
        vehicle_generator(42, "VAN-001", Stream.PROFILE, -1)
