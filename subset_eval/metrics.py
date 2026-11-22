"""How well a subset ranks the test agents, and how much the full benchmark wobbles on its own."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

NOISE_CEILING_PERCENTILES = (2.5, 97.5)


@dataclass(kw_only=True)
class SubsetScore:
    kendall_tau: float
    top_k_overlap: float


@dataclass(kw_only=True)
class NoiseCeiling:
    """Central 95 % range of Kendall tau between bootstrap rankings and the full ranking."""

    kendall_tau_low: float
    kendall_tau_high: float


def rank_agreement(full_solve_rate: np.ndarray, subset_solve_rate: np.ndarray) -> float:
    """Kendall tau-b, which handles the ties a small subset produces."""
    kendall_tau = kendalltau(full_solve_rate, subset_solve_rate).statistic
    # note (Haoling Pu): a subset that gives every agent the same score carries no ranking information, so it counts as 0 instead of NaN.
    if np.isnan(kendall_tau):
        return 0.0
    else:
        return float(kendall_tau)


def score_subset(
    test_outcomes: pd.DataFrame, estimated_solve_rate: pd.Series, *, top_k: int
) -> SubsetScore:
    """Compare the test agents' ranking by a subset's estimate with their ranking on all tasks.

    The estimate is the plain subset mean for most methods, or a weighted / gp-IRT estimate for tinyBenchmarks.
    """
    if top_k > len(test_outcomes):
        raise ValueError(f"top_k {top_k} exceeds the {len(test_outcomes)} test agents")
    else:
        pass
    full_solve_rate = test_outcomes.mean(axis=1)
    subset_solve_rate = estimated_solve_rate.loc[test_outcomes.index]
    # note (Haoling Pu): keep="all" keeps every agent tied at the k-th score, so the overlap does not depend on row order.
    top_full_agent_ids = set(full_solve_rate.nlargest(top_k, keep="all").index)
    top_subset_agent_ids = set(subset_solve_rate.nlargest(top_k, keep="all").index)
    top_k_overlap = len(top_full_agent_ids & top_subset_agent_ids) / max(
        len(top_full_agent_ids), len(top_subset_agent_ids)
    )
    return SubsetScore(
        kendall_tau=rank_agreement(
            full_solve_rate.to_numpy(), subset_solve_rate.to_numpy()
        ),
        top_k_overlap=top_k_overlap,
    )


def noise_ceiling(
    test_outcomes: pd.DataFrame,
    *,
    num_bootstrap: int,
    random_generator: np.random.Generator,
) -> NoiseCeiling:
    """Resample all tasks with replacement and re-rank: a subset is good enough inside this range."""
    outcome_values = test_outcomes.to_numpy()
    full_solve_rate = outcome_values.mean(axis=1)
    num_tasks = outcome_values.shape[1]
    resampled_task_indices = random_generator.integers(
        0, num_tasks, size=(num_bootstrap, num_tasks)
    )
    kendall_taus = [
        rank_agreement(full_solve_rate, outcome_values[:, task_indices].mean(axis=1))
        for task_indices in resampled_task_indices
    ]
    kendall_tau_low, kendall_tau_high = np.percentile(
        kendall_taus, NOISE_CEILING_PERCENTILES
    )
    return NoiseCeiling(
        kendall_tau_low=float(kendall_tau_low), kendall_tau_high=float(kendall_tau_high)
    )
