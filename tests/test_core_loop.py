"""Contracts of the shared framework: data validation, splits and agreement metrics."""

import numpy as np
import pandas as pd
import pytest

from subset_eval.data import BenchmarkData, outcome_matrix
from subset_eval.irt import IrtParameters, fit_irt
from subset_eval.metrics import noise_ceiling, score_subset
from subset_eval.select import (
    select_irt_information,
    select_pass_rate_band,
    select_stratified,
)
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


def test_pass_rate_band_takes_in_band_tasks_first_then_the_nearest() -> None:
    """Pass rates over 10 selection agents: in band 0.4 and 0.6; outside 0.2, 0.0 and 1.0."""
    pass_rate_by_task = {
        "band_a": 0.4,
        "band_b": 0.6,
        "near": 0.2,
        "never": 0.0,
        "always": 1.0,
    }
    selection_outcomes = pd.DataFrame(
        {
            task_id: [1] * round(10 * rate) + [0] * (10 - round(10 * rate))
            for task_id, rate in pass_rate_by_task.items()
        }
    )
    random_generator = np.random.default_rng(0)
    two_tasks = select_pass_rate_band(
        selection_outcomes, subset_size=2, random_generator=random_generator
    )
    three_tasks = select_pass_rate_band(
        selection_outcomes, subset_size=3, random_generator=random_generator
    )
    assert set(two_tasks) == {"band_a", "band_b"}
    assert set(three_tasks) == {"band_a", "band_b", "near"}


def test_pass_rate_band_samples_the_whole_band_when_it_exceeds_k() -> None:
    """Truncating the band by column order once picked the same in-band tasks every time."""
    selection_outcomes = pd.DataFrame(
        {f"band_{i}": [1] * 5 + [0] * 5 for i in range(6)}
        | {"never": [0] * 10, "always": [1] * 10}
    )
    picked_task_ids = set()
    for seed in range(20):
        picked_task_ids |= set(
            select_pass_rate_band(
                selection_outcomes,
                subset_size=2,
                random_generator=np.random.default_rng(seed),
            )
        )
    assert picked_task_ids == {f"band_{i}" for i in range(6)}


def test_stratified_takes_one_task_from_each_difficulty_group() -> None:
    """Pass rates 0.0, 0.1, ..., 0.9; with k = 2 the subset is always one easier-half and one harder-half task."""
    selection_outcomes = pd.DataFrame(
        {f"task_{i}": [1] * i + [0] * (10 - i) for i in range(10)}
    )
    harder_half = {f"task_{i}" for i in range(5)}
    for seed in range(20):
        subset_task_ids = select_stratified(
            selection_outcomes,
            subset_size=2,
            random_generator=np.random.default_rng(seed),
        )
        assert len(set(subset_task_ids)) == 2
        assert len(set(subset_task_ids) & harder_half) == 1


def test_stratified_shuffles_tasks_with_the_same_pass_rate() -> None:
    """Pass rates a 0, b 0.5, c 0.5, d 1: without a random tie-break the groups are always {a, b} / {c, d}."""
    selection_outcomes = pd.DataFrame(
        {"a": [0, 0], "b": [1, 0], "c": [1, 0], "d": [1, 1]}
    )
    subsets = {
        frozenset(
            select_stratified(
                selection_outcomes,
                subset_size=2,
                random_generator=np.random.default_rng(seed),
            )
        )
        for seed in range(50)
    }
    assert {frozenset({"a", "b"}), frozenset({"c", "d"})} <= subsets


def test_fit_irt_recovers_the_parameters_it_was_drawn_from() -> None:
    """A sign error in the gradient would flip or flatten the recovered abilities and difficulties."""
    random_generator = np.random.default_rng(0)
    ability = random_generator.normal(0.0, 1.0, size=200)
    difficulty = random_generator.normal(0.0, 1.0, size=60)
    discrimination = random_generator.lognormal(0.0, 0.3, size=60)
    solve_probability = 1.0 / (
        1.0 + np.exp(-discrimination * (ability[:, None] - difficulty[None, :]))
    )
    outcomes = pd.DataFrame(
        (random_generator.random(solve_probability.shape) < solve_probability).astype(
            int
        )
    )
    irt_parameters = fit_irt(outcomes)
    assert np.corrcoef(irt_parameters.ability, ability)[0, 1] > 0.9
    assert np.corrcoef(irt_parameters.difficulty, difficulty)[0, 1] > 0.9
    assert np.corrcoef(irt_parameters.discrimination, discrimination)[0, 1] > 0.3


def test_irt_information_prefers_discriminating_tasks_near_the_agents() -> None:
    """Information a^2 P (1 - P) peaks for steep tasks whose difficulty matches the agents."""
    irt_parameters = IrtParameters(
        ability=pd.Series([-0.5, 0.0, 0.5]),
        difficulty=pd.Series(
            {"steep_middle": 0.0, "flat_middle": 0.0, "steep_far": 4.0}
        ),
        discrimination=pd.Series(
            {"steep_middle": 3.0, "flat_middle": 0.3, "steep_far": 3.0}
        ),
    )
    assert select_irt_information(irt_parameters, subset_size=1) == ["steep_middle"]
