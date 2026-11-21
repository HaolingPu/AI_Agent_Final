"""Contracts of the shared framework: data validation, splits and agreement metrics."""

import numpy as np
import pandas as pd
import pytest

from subset_eval.data import BenchmarkData, outcome_matrix
from subset_eval.metrics import noise_ceiling, score_subset
from subset_eval.split import split_by_date
from subset_eval.toy import make_toy_benchmark


@pytest.fixture
def toy_benchmark() -> BenchmarkData:
    return make_toy_benchmark(
        num_agents=30,
        num_tasks=40,
        num_trials=3,
        num_months=12,
        ability_gain_per_month=0.1,
        seed=0,
    )


def test_outcome_matrix_averages_trials(toy_benchmark: BenchmarkData) -> None:
    matrix = outcome_matrix(toy_benchmark)
    expected = toy_benchmark.runs.groupby(["agent_id", "task_id"])["resolved"].mean()
    assert matrix.shape == (30, 40)
    assert matrix.stack().sort_index().equals(expected.sort_index())


def test_outcome_matrix_rejects_a_pair_without_runs(
    toy_benchmark: BenchmarkData,
) -> None:
    """A loader that drops unresolved tasks instead of writing 0 must fail loudly."""
    runs = toy_benchmark.runs
    is_first_pair = (runs["agent_id"] == runs["agent_id"].iloc[0]) & (
        runs["task_id"] == runs["task_id"].iloc[0]
    )
    toy_benchmark.runs = runs[~is_first_pair]
    with pytest.raises(ValueError, match="no run"):
        outcome_matrix(toy_benchmark)


@pytest.mark.parametrize(
    "broken_table, message",
    [
        ("unknown_task", "missing from tasks"),
        ("duplicate_trial", "duplicate"),
        ("missing_date", "without a submit_date"),
    ],
)
def test_benchmark_rejects_broken_loader_output(
    toy_benchmark: BenchmarkData, broken_table: str, message: str
) -> None:
    """Loader mistakes that would silently change the matrix or the split."""
    agents = toy_benchmark.agents.copy()
    runs = toy_benchmark.runs.copy()
    if broken_table == "unknown_task":
        runs.loc[0, "task_id"] = "not-a-task"
    elif broken_table == "duplicate_trial":
        runs = pd.concat([runs, runs.iloc[[0]]])
    else:
        agents.loc[0, "submit_date"] = pd.NaT
    with pytest.raises(ValueError, match=message):
        BenchmarkData(name="toy", agents=agents, tasks=toy_benchmark.tasks, runs=runs)


def test_split_by_date_puts_the_cutoff_day_in_the_test_group(
    toy_benchmark: BenchmarkData,
) -> None:
    cutoff_date = pd.Timestamp("2024-07-01")
    submit_date = toy_benchmark.agents.set_index("agent_id")["submit_date"]
    assert (submit_date == cutoff_date).any()
    agent_split = split_by_date(toy_benchmark.agents, cutoff_date=cutoff_date)
    assert set(agent_split.selection_agent_ids).isdisjoint(agent_split.test_agent_ids)
    assert len(agent_split.selection_agent_ids) + len(agent_split.test_agent_ids) == 30
    assert (submit_date[agent_split.selection_agent_ids] < cutoff_date).all()
    assert (submit_date[agent_split.test_agent_ids] >= cutoff_date).all()


def test_score_subset_on_a_known_ranking() -> None:
    """Full ranking a > b > c > d; the one subset task puts d first and ties the rest."""
    outcomes = pd.DataFrame(
        {
            "easy_1": [1, 1, 1, 1],
            "easy_2": [1, 1, 1, 0],
            "mid_1": [1, 1, 1, 0],
            "mid_2": [1, 1, 0, 0],
            "hard_1": [1, 0, 0, 0],
            "hard_2": [1, 0, 0, 0],
            "reversed": [0, 0, 0, 1],
        },
        index=["a", "b", "c", "d"],
    )
    subset_score = score_subset(outcomes, ["reversed"], top_k=1)
    # 3 discordant pairs (d against each), 3 pairs tied in the subset: tau-b = -3 / sqrt(6 * 3).
    assert subset_score.kendall_tau == pytest.approx(-3 / np.sqrt(18))
    assert subset_score.top_k_overlap == 0.0


def test_score_subset_does_not_depend_on_row_order(
    toy_benchmark: BenchmarkData,
) -> None:
    """Ties at the k-th score once favoured whichever agent came first."""
    outcomes = outcome_matrix(toy_benchmark)
    subset_task_ids = outcomes.columns[:5].tolist()
    forward = score_subset(outcomes, subset_task_ids, top_k=5)
    backward = score_subset(outcomes.iloc[::-1], subset_task_ids, top_k=5)
    assert forward == backward


def test_noise_ceiling_is_one_when_all_tasks_agree() -> None:
    """Identical tasks give the same ranking under any resample."""
    outcomes = pd.DataFrame(np.tile([[1], [1], [0]], (1, 6)), index=["a", "b", "c"])
    ceiling = noise_ceiling(
        outcomes, num_bootstrap=50, random_generator=np.random.default_rng(0)
    )
    assert (ceiling.kendall_tau_low, ceiling.kendall_tau_high) == pytest.approx((1, 1))
