"""Task selection methods: each returns the task ids of one subset."""

import numpy as np
import pandas as pd


def select_random(
    task_ids: pd.Index, *, subset_size: int, random_generator: np.random.Generator
) -> list[str]:
    """Floor baseline: a uniform random subset that ignores every agent's results."""
    return random_generator.choice(
        task_ids.to_numpy(), size=subset_size, replace=False
    ).tolist()
