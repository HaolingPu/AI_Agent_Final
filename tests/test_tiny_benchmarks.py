"""Contracts of the tinyBenchmarks reproduction: anchor points, multidimensional IRT and gp-IRT."""

import numpy as np
import pandas as pd
import pytest

from subset_eval.tiny_benchmarks import (
    IRT_DIMENSIONS,
    MultidimensionalIrtParameters,
    TinyBenchmarksFit,
    fit_ability,
    fit_multidimensional_irt,
    fit_tiny_benchmarks,
    gp_irt_solve_rate,
    select_anchor_points,
)
from subset_eval.toy import make_toy_benchmark


def draw_two_dimensional_outcomes(
    random_generator: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    ability = random_generator.normal(0.0, 1.0, size=(150, 2))
    discrimination = random_generator.normal(0.0, 1.5, size=(60, 2))
    difficulty = random_generator.normal(0.0, 1.0, size=60)
    solve_probability = 1.0 / (1.0 + np.exp(-(ability @ discrimination.T - difficulty)))
    outcomes = pd.DataFrame(
        (random_generator.random(solve_probability.shape) < solve_probability).astype(
            int
        ),
        columns=[f"task_{i}" for i in range(60)],
    )
    return outcomes, solve_probability


def test_anchor_points_take_the_task_nearest_each_centroid_weighted_by_cluster_share() -> (
    None
):
    """Clusters {0..6} and {100, 101, 102}: anchors sit at 3 and 101 with weights 0.7 and 0.3."""
    positions = [0, 1, 2, 3, 4, 5, 6, 100, 101, 102]
    task_embedding = pd.DataFrame(
        {"position": positions}, index=[f"task_{p}" for p in positions]
    )
    task_weights = select_anchor_points(
        task_embedding, subset_size=2, random_generator=np.random.default_rng(0)
    )
    assert task_weights.to_dict() == pytest.approx({"task_3": 0.7, "task_101": 0.3})


def test_multidimensional_irt_recovers_the_solve_probabilities() -> None:
    """A sign error in either gradient would scramble the predicted probabilities."""
    random_generator = np.random.default_rng(0)
    outcomes, solve_probability = draw_two_dimensional_outcomes(random_generator)
    irt = fit_multidimensional_irt(
        outcomes, dimension=2, random_generator=random_generator
    )
    predicted = irt.solve_probability(fit_ability(outcomes, irt)).to_numpy()
    assert np.corrcoef(predicted.ravel(), solve_probability.ravel())[0, 1] > 0.9


def test_fit_ability_orders_agents_and_survives_perfect_scores() -> None:
    """Without a prior, all-solved and all-failed agents run off to the logit clip instead of crashing."""
    irt = MultidimensionalIrtParameters(
        discrimination=pd.DataFrame({0: [1.0, 1.0, 1.0]}, index=["a", "b", "c"]),
        difficulty=pd.Series({"a": -1.0, "b": 0.0, "c": 1.0}),
    )
    observed_outcomes = pd.DataFrame(
        [[1, 1, 1], [1, 1, 0], [0, 0, 0]],
        index=["all", "some", "none"],
        columns=["a", "b", "c"],
    )
    ability = fit_ability(observed_outcomes, irt)[0]
    assert ability["all"] > ability["some"] > ability["none"]


def test_gp_irt_weight_follows_lambda_with_a_quarter_variance_for_anchors() -> None:
    """With b^2 = sigma^2 / (4 n), lambda is 1/2 for anchor points and 1/5 for a random subset."""
    random_generator = np.random.default_rng(1)
    outcomes, _ = draw_two_dimensional_outcomes(random_generator)
    irt = fit_multidimensional_irt(
        outcomes, dimension=2, random_generator=random_generator
    )
    task_weights = pd.Series([0.5, 0.3, 0.2], index=["task_0", "task_7", "task_9"])
    score_variance = 0.2
    balanced_bias = np.sqrt(score_variance / (4 * len(task_weights)))

    def _estimate(validation_error: float, anchor_points: bool) -> pd.Series:
        return gp_irt_solve_rate(
            outcomes,
            task_weights,
            TinyBenchmarksFit(
                irt=irt,
                validation_error=validation_error,
                score_variance=score_variance,
            ),
            anchor_points=anchor_points,
        )

    weighted_score = outcomes[task_weights.index] @ task_weights
    p_irt_score = _estimate(0.0, True)
    assert _estimate(1e6, True).to_numpy() == pytest.approx(weighted_score.to_numpy())
    assert _estimate(balanced_bias, True).to_numpy() == pytest.approx(
        (0.5 * weighted_score + 0.5 * p_irt_score).to_numpy()
    )
    assert _estimate(balanced_bias, False).to_numpy() == pytest.approx(
        (0.2 * weighted_score + 0.8 * p_irt_score).to_numpy()
    )


def test_fit_tiny_benchmarks_picks_a_candidate_dimension() -> None:
    benchmark = make_toy_benchmark(
        num_agents=60,
        num_tasks=80,
        num_trials=1,
        num_months=12,
        ability_gain_per_month=0.1,
        seed=0,
    )
    outcomes = benchmark.runs.pivot(
        index="agent_id", columns="task_id", values="resolved"
    )
    tiny_benchmarks_fit = fit_tiny_benchmarks(
        outcomes, random_generator=np.random.default_rng(0)
    )
    assert tiny_benchmarks_fit.irt.discrimination.index.equals(outcomes.columns)
    assert tiny_benchmarks_fit.irt.discrimination.shape[1] in IRT_DIMENSIONS
    assert 0.0 < tiny_benchmarks_fit.validation_error < 0.5
