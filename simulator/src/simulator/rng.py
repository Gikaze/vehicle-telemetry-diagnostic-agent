"""Seed tree (FR-02): one root seed, independent random streams per vehicle and per purpose.

Each stream is a PCG64 generator seeded with a SeedSequence built from a list of non-negative
integers, varying identifiers first and the root seed last, as the NumPy parallel random number
guide recommends. A vehicle is identified by a BLAKE2b digest of its ID (never Python `hash()`,
which is salted per process), not by its index in the fleet, so adding a vehicle does not change
any other vehicle's stream.
"""

import hashlib
from enum import IntEnum

import numpy as np

FLEET_NAMESPACE = 0
VEHICLE_NAMESPACE = 1


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


def vehicle_generator(root_seed: int, vehicle_id: str, stream: Stream) -> np.random.Generator:
    """Random stream of one vehicle for one purpose."""
    _check_seed(root_seed)
    entropy = [VEHICLE_NAMESPACE, vehicle_key(vehicle_id), int(stream), root_seed]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def fleet_generator(root_seed: int, stream: Stream) -> np.random.Generator:
    """Random stream shared by the whole fleet (e.g. the ambient temperature of the day)."""
    _check_seed(root_seed)
    entropy = [FLEET_NAMESPACE, int(stream), root_seed]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
