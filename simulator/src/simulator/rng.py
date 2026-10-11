"""Seed tree (FR-02): one root seed, independent random streams per vehicle and per purpose.

Each stream is a PCG64 generator seeded with a SeedSequence built from a list of non-negative
integers, varying identifiers first and the root seed last, as the NumPy parallel random number
guide recommends. A vehicle is identified by a BLAKE2b digest of its ID (never Python `hash()`,
which is salted per process), not by its index in the fleet, so adding a vehicle does not change
any other vehicle's stream.

Streams can carry indices (the day, the trip): the road of trip k of day d depends only on the root
seed, the vehicle, d and k, so the number of draws one trip consumes never shifts another trip.
Indexed streams use their own namespaces and state the number of indices: SeedSequence pads
entropy shorter than its pool with zeros ([1, 2, 3] and [1, 2, 3, 0] give the same stream), so
appending indices to the unindexed form could reproduce an existing stream.
"""

import hashlib
from enum import IntEnum

import numpy as np

FLEET_NAMESPACE = 0
VEHICLE_NAMESPACE = 1
VEHICLE_INDEXED_NAMESPACE = 2
FLEET_INDEXED_NAMESPACE = 3


class Stream(IntEnum):
    """Purpose of a random stream. Values are part of the seeding contract: never renumber."""

    PARAMETERS = 1
    SCHEDULE = 2
    PROFILE = 3
    SENSOR_NOISE = 4
    TRANSPORT = 5
    SCENARIO = 6
    AMBIENT = 7


def vehicle_key(vehicle_id: str) -> int:
    """Stable 64-bit key of a vehicle ID (BLAKE2b, 8-byte digest)."""
    digest = hashlib.blake2b(vehicle_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _check_seed(root_seed: int) -> None:
    if root_seed < 0:
        raise ValueError("the root seed must be a non-negative integer")


def _check_indices(indices: tuple[int, ...]) -> None:
    if any(index < 0 for index in indices):
        raise ValueError("stream indices must be non-negative integers")


def _generator(entropy: list[int]) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def vehicle_generator(
    root_seed: int, vehicle_id: str, stream: Stream, *indices: int
) -> np.random.Generator:
    """Random stream of one vehicle for one purpose, optionally per day and trip."""
    _check_seed(root_seed)
    _check_indices(indices)
    key = vehicle_key(vehicle_id)
    if not indices:
        return _generator([VEHICLE_NAMESPACE, key, int(stream), root_seed])
    return _generator(
        [VEHICLE_INDEXED_NAMESPACE, key, int(stream), len(indices), *indices, root_seed]
    )


def fleet_generator(root_seed: int, stream: Stream, *indices: int) -> np.random.Generator:
    """Random stream shared by the whole fleet (e.g. the ambient temperature of a day)."""
    _check_seed(root_seed)
    _check_indices(indices)
    if not indices:
        return _generator([FLEET_NAMESPACE, int(stream), root_seed])
    return _generator([FLEET_INDEXED_NAMESPACE, int(stream), len(indices), *indices, root_seed])
