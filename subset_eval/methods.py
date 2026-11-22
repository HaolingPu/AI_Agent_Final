"""Every selection method behind one interface: fit the selection agents once, pick a weighted subset, estimate the test agents' solve rates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from subset_eval.irt import IrtParameters, fit_irt
from subset_eval.select import (
    select_irt_max_information,
    select_pass_rate_band,
    select_random,
    select_stratified,
)
from subset_eval.tiny_benchmarks import (
    TinyBenchmarksFit,
    fit_tiny_benchmarks,
    gp_irt_solve_rate,
    select_anchor_points,
)

SelectionMethod = Literal[
    "random",
    "stratified",
    "pass_rate_band",
    "irt_max_information",
    "anchor_correctness",
    "anchor_irt",
]
SELECTION_METHODS: tuple[SelectionMethod, ...] = (
    "random",
    "stratified",
    "pass_rate_band",
    "irt_max_information",
    "anchor_correctness",
    "anchor_irt",
)
# note (Haoling Pu): tinyBenchmarks reports these three strategies with and without gp-IRT; they also get a "<method>_gp_irt" estimate.
GP_IRT_METHODS: tuple[SelectionMethod, ...] = (
    "random",
    "anchor_correctness",
    "anchor_irt",
)


@dataclass(kw_only=True)
class SelectionGroupFit:
    """What the methods learn from the selection agents, fit once per split and shared by every method and subset size."""

    selection_outcomes: pd.DataFrame
    irt_parameters: IrtParameters
    tiny_benchmarks_fit: TinyBenchmarksFit


def fit_selection_group(
    selection_outcomes: pd.DataFrame, *, random_generator: np.random.Generator
) -> SelectionGroupFit:
    return SelectionGroupFit(
        selection_outcomes=selection_outcomes,
        irt_parameters=fit_irt(selection_outcomes),
        tiny_benchmarks_fit=fit_tiny_benchmarks(
            selection_outcomes, random_generator=random_generator
        ),
    )


def select_task_weights(
    selection_method: SelectionMethod,
    selection_group_fit: SelectionGroupFit,
    *,
    subset_size: int,
    random_generator: np.random.Generator,
) -> pd.Series:
    """Task id -> weight, summing to 1; all but the anchor methods weigh their tasks equally."""
    selection_outcomes = selection_group_fit.selection_outcomes
    if selection_method == "random":
        task_weights = pd.Series(
            1.0 / subset_size,
            index=select_random(
                selection_outcomes.columns,
                subset_size=subset_size,
                random_generator=random_generator,
            ),
        )
    elif selection_method == "stratified":
        task_weights = pd.Series(
            1.0 / subset_size,
            index=select_stratified(
                selection_outcomes,
                subset_size=subset_size,
                random_generator=random_generator,
            ),
        )
    elif selection_method == "pass_rate_band":
        task_weights = pd.Series(
            1.0 / subset_size,
            index=select_pass_rate_band(
                selection_outcomes,
                subset_size=subset_size,
                random_generator=random_generator,
            ),
        )
    elif selection_method == "irt_max_information":
        task_weights = pd.Series(
            1.0 / subset_size,
            index=select_irt_max_information(
                selection_group_fit.irt_parameters, subset_size=subset_size
            ),
        )
    elif selection_method == "anchor_correctness":
        task_weights = select_anchor_points(
            selection_outcomes.T,
            subset_size=subset_size,
            random_generator=random_generator,
        )
    elif selection_method == "anchor_irt":
        tiny_benchmarks_irt = selection_group_fit.tiny_benchmarks_fit.irt
        # note (Haoling Pu): an IRT anchor embeds each task as its discrimination vector plus its difficulty, unscaled, as in the authors' code.
        task_weights = select_anchor_points(
            tiny_benchmarks_irt.discrimination.assign(
                difficulty=tiny_benchmarks_irt.difficulty
            ),
            subset_size=subset_size,
            random_generator=random_generator,
        )
    else:
        raise ValueError(f"unknown selection method {selection_method}")
    return task_weights


def estimate_solve_rates(
    selection_method: SelectionMethod,
    task_weights: pd.Series,
    test_outcomes: pd.DataFrame,
    selection_group_fit: SelectionGroupFit,
    *,
    subset_size: int,
) -> dict[str, pd.Series]:
    """Estimated full-benchmark solve rate of each test agent, keyed by method name: the weighted subset score, plus gp-IRT for GP_IRT_METHODS."""
    estimated_solve_rate_by_method = {
        selection_method: test_outcomes[task_weights.index] @ task_weights
    }
    if selection_method in GP_IRT_METHODS:
        estimated_solve_rate_by_method[f"{selection_method}_gp_irt"] = (
            gp_irt_solve_rate(
                test_outcomes,
                task_weights,
                selection_group_fit.tiny_benchmarks_fit,
                subset_size=subset_size,
                anchor_points=selection_method != "random",
            )
        )
    else:
        pass
    return estimated_solve_rate_by_method
