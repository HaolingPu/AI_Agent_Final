"""One random stream per purpose, so adding, removing or reordering methods or sizes never changes the other numbers."""

import zlib

import numpy as np


def purpose_generator(*, seed: int, repeat: int, purpose: str) -> np.random.Generator:
    """A generator seeded by (seed, repeat, purpose name); the same purpose always draws the same numbers."""
    return np.random.default_rng([seed, repeat, zlib.crc32(purpose.encode())])
