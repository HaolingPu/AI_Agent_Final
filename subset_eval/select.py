"""Task selection methods: each returns the task ids of one subset."""

from typing import Literal

import numpy as np
import pandas as pd

SelectionMethod = Literal["random", "stratified", "pass_rate_band"]

# note (Haoling Pu): Ndzomga (arXiv 2603.23749) keeps tasks that 30-70 % of past agents solved.
PASS_RATE_BAND = (0.3, 0.7)


def select_random(
    task_ids: pd.Index, *, subset_size: int, random_generator: np.random.Generator
) -> list[str]:
    """Floor baseline: a uniform random subset that ignores every agent's results."""
    return random_generator.choice(
        task_ids.to_numpy(), size=subset_size, replace=False
    ).tolist()


def select_stratified(
    selection_outcomes: pd.DataFrame,
    *,
    subset_size: int,
    random_generator: np.random.Generator,
) -> list[str]:
    """Sort tasks by the selection agents' pass rate, cut them into subset_size near-equal groups, draw one per group.

    The subset keeps the full benchmark's mix of easy, medium and hard tasks, which a random subset only does on average.
    """
    pass_rate = selection_outcomes.mean(axis=0)
    if subset_size > len(pass_rate):
        raise ValueError(
            f"subset_size {subset_size} exceeds the {len(pass_rate)} tasks"
        )
    else:
        pass
    # note (Haoling Pu): many tasks share a pass rate (0 or 1 especially), so a random draw orders ties before cutting the groups.
    task_ids_by_difficulty = (
        pd.DataFrame(
            {
                "pass_rate": pass_rate,
                "tie_break_draw": random_generator.random(len(pass_rate)),
            }
        )
        .sort_values(["pass_rate", "tie_break_draw"])
        .index.to_numpy()
    )
    return [
        random_generator.choice(difficulty_group)
        for difficulty_group in np.array_split(task_ids_by_difficulty, subset_size)
    ]


def select_pass_rate_band(
    selection_outcomes: pd.DataFrame,
    *,
    subset_size: int,
    random_generator: np.random.Generator,
) -> list[str]:
    """Keep tasks the selection agents solved 30-70 % of the time, in random order.

    When the band holds fewer than subset_size tasks, the rest are the tasks closest to the band.
    """
    pass_rate = selection_outcomes.mean(axis=0)
    if subset_size > len(pass_rate):
        raise ValueError(
            f"subset_size {subset_size} exceeds the {len(pass_rate)} tasks"
        )
    else:
        pass
    band_low, band_high = PASS_RATE_BAND
    distance_to_band = np.maximum(band_low - pass_rate, pass_rate - band_high).clip(
        lower=0.0
    )
    # note (Haoling Pu): a random draw breaks ties, so tasks inside the band (distance 0) come out in random order.
    ranked_tasks = pd.DataFrame(
        {
            "distance_to_band": distance_to_band,
            "tie_break_draw": random_generator.random(len(pass_rate)),
        }
    ).sort_values(["distance_to_band", "tie_break_draw"])
    return ranked_tasks.index[:subset_size].tolist()
