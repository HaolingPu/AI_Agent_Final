"""Contracts of the tinyBenchmarks reproduction: anchor points, multidimensional IRT and gp-IRT."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from scipy.special import logit

from subset_eval import tiny_benchmarks
from subset_eval.tiny_benchmarks import (
    MultidimensionalIrtParameters,
    TinyBenchmarksFit,
    fit_ability,
    fit_multidimensional_irt,
    fit_tiny_benchmarks,
    gp_irt_solve_rate,
    select_anchor_points,
)


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
        columns=[f"task_{task_number}" for task_number in range(60)],
    )
    return outcomes, solve_probability


def test_anchor_points_take_the_task_nearest_each_centroid_weighted_by_cluster_share() -> (
    None
):
    """Clusters {0..6} and {100, 101, 102}: anchors sit at 3 and 101 with weights 0.7 and 0.3."""
    positions = [0, 1, 2, 3, 4, 5, 6, 100, 101, 102]
    task_embedding = pd.DataFrame(
        {"position": positions}, index=[f"task_{position}" for position in positions]
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
    """Without a prior, all-solved and all-failed agents get a large but finite ability instead of crashing."""
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


@pytest.mark.parametrize(
    ("status", "is_kept"), [(2, True), (1, False)], ids=["precision_loss", "maxiter"]
)
def test_fit_ability_keeps_a_precision_loss_stop_and_raises_on_other_failures(
    monkeypatch: pytest.MonkeyPatch, status: int, is_kept: bool
) -> None:
    """A drifting ability on real SWE-bench data stopped BFGS with precision loss (status 2), which crashed the C2 run."""

    def _stopped_minimize(
        *arguments: object, **keyword_arguments: object
    ) -> OptimizeResult:
        return OptimizeResult(
            x=np.array([25.0]), success=False, status=status, message="stopped"
        )

    monkeypatch.setattr(tiny_benchmarks, "minimize", _stopped_minimize)
    irt = MultidimensionalIrtParameters(
        discrimination=pd.DataFrame({0: [1.0]}, index=["a"]),
        difficulty=pd.Series({"a": 0.0}),
    )
    observed_outcomes = pd.DataFrame([[1]], index=["agent"], columns=["a"])
    if is_kept:
        assert fit_ability(observed_outcomes, irt).loc["agent", 0] == 25.0
    else:
        with pytest.raises(RuntimeError, match="did not converge"):
            fit_ability(observed_outcomes, irt)


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
            subset_size=len(task_weights),
            anchor_points=anchor_points,
        )

    weighted_score = outcomes[task_weights.index] @ task_weights
    unseen_task_ids = outcomes.columns.difference(task_weights.index)
    p_irt_score = 3 / 60 * outcomes[task_weights.index].mean(axis=1) + 57 / 60 * (
        irt.solve_probability(fit_ability(outcomes[task_weights.index], irt))[
            unseen_task_ids
        ].mean(axis=1)
    )
    assert _estimate(0.0, True).to_numpy() == pytest.approx(p_irt_score.to_numpy())
    assert _estimate(1e6, True).to_numpy() == pytest.approx(weighted_score.to_numpy())
    assert _estimate(balanced_bias, True).to_numpy() == pytest.approx(
        (0.5 * weighted_score + 0.5 * p_irt_score).to_numpy()
    )
    assert _estimate(balanced_bias, False).to_numpy() == pytest.approx(
        (0.2 * weighted_score + 0.8 * p_irt_score).to_numpy()
    )


def test_fit_tiny_benchmarks_finds_two_dimensions_in_two_dimensional_data() -> None:
    outcomes, _ = draw_two_dimensional_outcomes(np.random.default_rng(0))
    tiny_benchmarks_fit = fit_tiny_benchmarks(
        outcomes, random_generator=np.random.default_rng(0)
    )
    assert tiny_benchmarks_fit.irt.discrimination.index.equals(outcomes.columns)
    assert tiny_benchmarks_fit.irt.discrimination.shape[1] == 2
    assert 0.0 < tiny_benchmarks_fit.validation_error < 0.5


def test_fit_tiny_benchmarks_holds_out_unique_agents_and_keeps_the_smallest_near_best_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake IRT fits predict 0.504 / 0.502 / 0.500 / 0.503 for dimensions 2 / 5 / 10 / 15 against an actual 0.5.

    Dimension 5 is the first within 0.0025 of the best error, so it wins with b = 0.002; the strict minimum would pick 10.
    Agents agent_0 and agent_5 repeat (as in the date-split bootstrap) and must be held out, never trained on.
    """
    predicted_solve_rate_by_dimension = {2: 0.504, 5: 0.502, 10: 0.500, 15: 0.503}
    agent_ids = [f"agent_{agent_number}" for agent_number in range(10)]
    # note (Haoling Pu): every agent solves one of the two odd-position tasks, so the actual held-out solve rate is 0.5.
    selection_outcomes = pd.DataFrame(
        [
            [agent_number % 2, 1, 1 - agent_number % 2, 0]
            for agent_number in list(range(10)) + [0, 5]
        ],
        index=agent_ids + ["agent_0", "agent_5"],
        columns=["task_0", "task_1", "task_2", "task_3"],
    )
    training_agent_ids = []
    held_out_calls = []
    real_fit_ability = tiny_benchmarks.fit_ability

    def _fake_fit_multidimensional_irt(
        outcomes: pd.DataFrame,
        *,
        dimension: int,
        random_generator: np.random.Generator,
    ) -> MultidimensionalIrtParameters:
        training_agent_ids.append(outcomes.index.tolist())
        return MultidimensionalIrtParameters(
            discrimination=pd.DataFrame(
                np.zeros((outcomes.shape[1], dimension)), index=outcomes.columns
            ),
            difficulty=pd.Series(
                -logit(predicted_solve_rate_by_dimension[dimension]),
                index=outcomes.columns,
            ),
        )

    def _recording_fit_ability(
        observed_outcomes: pd.DataFrame, irt: MultidimensionalIrtParameters
    ) -> pd.DataFrame:
        held_out_calls.append(
            (observed_outcomes.index.tolist(), observed_outcomes.columns.tolist())
        )
        return real_fit_ability(observed_outcomes, irt)

    monkeypatch.setattr(
        tiny_benchmarks, "fit_multidimensional_irt", _fake_fit_multidimensional_irt
    )
    monkeypatch.setattr(tiny_benchmarks, "fit_ability", _recording_fit_ability)
    tiny_benchmarks_fit = fit_tiny_benchmarks(
        selection_outcomes, random_generator=np.random.default_rng(0)
    )
    held_out_agent_ids = ["agent_0", "agent_5"]
    assert (
        training_agent_ids[:4]
        == [[agent_id for agent_id in agent_ids if agent_id not in held_out_agent_ids]]
        * 4
    )
    assert training_agent_ids[4] == selection_outcomes.index.tolist()
    assert held_out_calls == [(held_out_agent_ids, ["task_0", "task_2"])] * 4
    assert tiny_benchmarks_fit.irt.discrimination.shape[1] == 5
    assert tiny_benchmarks_fit.validation_error == pytest.approx(0.002)
    assert tiny_benchmarks_fit.score_variance == pytest.approx(
        selection_outcomes.var(axis=1, ddof=0).mean()
    )
