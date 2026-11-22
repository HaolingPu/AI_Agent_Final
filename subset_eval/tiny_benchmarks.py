"""Reproduction of tinyBenchmarks (Polo et al., arXiv 2402.14992): anchor points, multidimensional IRT and the gp-IRT estimator."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# note (Haoling Pu): settings follow the authors' code (github.com/felipemaiapolo/efficbench) where it is more specific than the paper.
IRT_DIMENSIONS = (2, 5, 10, 15)
DIMENSION_ERROR_TOLERANCE = 0.0025
VALIDATION_AGENT_STRIDE = 5
NUM_KMEANS_RESTARTS = 5
LOGIT_CLIP = 30.0
LOG_EPSILON = 1e-10
# note (Haoling Pu): the authors fit hierarchical priors by variational inference (py-irt); a MAP fit with fixed Gaussian priors stands in for it.
ABILITY_PRIOR_SD = 1.0
DISCRIMINATION_PRIOR_SD = 1.0
DIFFICULTY_PRIOR_SD = 3.0
INITIAL_PARAMETER_SD = 0.1


@dataclass(kw_only=True)
class MultidimensionalIrtParameters:
    """P(solve) = sigmoid(ability . discrimination - difficulty), with ability and discrimination in R^d."""

    discrimination: pd.DataFrame
    difficulty: pd.Series

    def solve_probability(self, ability: pd.DataFrame) -> pd.DataFrame:
        logit = (
            ability.to_numpy() @ self.discrimination.to_numpy().T
            - self.difficulty.to_numpy()
        )
        return pd.DataFrame(
            expit(np.clip(logit, -LOGIT_CLIP, LOGIT_CLIP)),
            index=ability.index,
            columns=self.difficulty.index,
        )


@dataclass(kw_only=True)
class TinyBenchmarksFit:
    irt: MultidimensionalIrtParameters
    # note (Haoling Pu): b and sigma^2 of the gp-IRT weight lambda = b^2 / (sigma^2 / n + b^2).
    validation_error: float
    score_variance: float


def fit_multidimensional_irt(
    outcomes: pd.DataFrame, *, dimension: int, random_generator: np.random.Generator
) -> MultidimensionalIrtParameters:
    """Maximum a posteriori multidimensional 2PL fit; a solve rate between 0 and 1 is one fractional observation."""
    solve_rate = outcomes.to_numpy(dtype=float)
    num_agents, num_tasks = solve_rate.shape
    ability_size = num_agents * dimension
    discrimination_size = num_tasks * dimension

    def _negative_log_posterior(
        flat_parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        ability = flat_parameters[:ability_size].reshape(num_agents, dimension)
        discrimination = flat_parameters[
            ability_size : ability_size + discrimination_size
        ].reshape(num_tasks, dimension)
        difficulty = flat_parameters[ability_size + discrimination_size :]
        logit = ability @ discrimination.T - difficulty
        negative_log_likelihood = np.sum(np.logaddexp(0.0, logit) - solve_rate * logit)
        probability_residual = expit(logit) - solve_rate
        negative_log_prior = 0.5 * (
            np.sum(ability**2) / ABILITY_PRIOR_SD**2
            + np.sum(discrimination**2) / DISCRIMINATION_PRIOR_SD**2
            + np.sum(difficulty**2) / DIFFICULTY_PRIOR_SD**2
        )
        gradient = np.concatenate(
            [
                (
                    probability_residual @ discrimination
                    + ability / ABILITY_PRIOR_SD**2
                ).ravel(),
                (
                    probability_residual.T @ ability
                    + discrimination / DISCRIMINATION_PRIOR_SD**2
                ).ravel(),
                -probability_residual.sum(axis=0) + difficulty / DIFFICULTY_PRIOR_SD**2,
            ]
        )
        return negative_log_likelihood + negative_log_prior, gradient

    # note (Haoling Pu): all-zero ability and discrimination is a saddle point (zero gradient), so the fit starts from a small random draw.
    result = minimize(
        _negative_log_posterior,
        x0=random_generator.normal(
            0.0,
            INITIAL_PARAMETER_SD,
            size=ability_size + discrimination_size + num_tasks,
        ),
        jac=True,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(
            f"multidimensional IRT fit did not converge: {result.message}"
        )
    else:
        pass
    return MultidimensionalIrtParameters(
        discrimination=pd.DataFrame(
            result.x[ability_size : ability_size + discrimination_size].reshape(
                num_tasks, dimension
            ),
            index=outcomes.columns,
        ),
        difficulty=pd.Series(
            result.x[ability_size + discrimination_size :], index=outcomes.columns
        ),
    )


def fit_ability(
    observed_outcomes: pd.DataFrame, irt: MultidimensionalIrtParameters
) -> pd.DataFrame:
    """Maximum likelihood ability of each agent from its observed tasks, task parameters fixed.

    As in the authors' code: one BFGS fit per agent, no prior, start at 0, logits clipped to +-30.
    An agent that solves every observed task drifts up until the gradient vanishes, so its ability is large but finite.
    """
    discrimination = irt.discrimination.loc[observed_outcomes.columns].to_numpy()
    difficulty = irt.difficulty.loc[observed_outcomes.columns].to_numpy()
    abilities = []
    # note (Haoling Pu): one joint fit over all agents ran out of evaluations: perfect-score agents keep drifting and starve the shared quasi-Newton memory.
    for solve_rate in observed_outcomes.to_numpy(dtype=float):

        def _negative_log_likelihood(ability: np.ndarray) -> tuple[float, np.ndarray]:
            logit = discrimination @ ability - difficulty
            solve_probability = expit(np.clip(logit, -LOGIT_CLIP, LOGIT_CLIP))
            negative_log_likelihood = -np.sum(
                solve_rate * np.log(solve_probability + LOG_EPSILON)
                + (1.0 - solve_rate) * np.log(1.0 - solve_probability + LOG_EPSILON)
            )
            logit_gradient = np.where(
                np.abs(logit) < LOGIT_CLIP,
                -solve_probability
                * (1.0 - solve_probability)
                * (
                    solve_rate / (solve_probability + LOG_EPSILON)
                    - (1.0 - solve_rate) / (1.0 - solve_probability + LOG_EPSILON)
                ),
                0.0,
            )
            return negative_log_likelihood, logit_gradient @ discrimination

        result = minimize(
            _negative_log_likelihood,
            x0=np.zeros(discrimination.shape[1]),
            jac=True,
            method="BFGS",
        )
        if not result.success:
            raise RuntimeError(f"ability fit did not converge: {result.message}")
        else:
            pass
        abilities.append(result.x)
    return pd.DataFrame(abilities, index=observed_outcomes.index)


def fit_tiny_benchmarks(
    selection_outcomes: pd.DataFrame, *, random_generator: np.random.Generator
) -> TinyBenchmarksFit:
    """Choose the IRT dimension on held-out selection agents, then fit that dimension on all of them.

    As in the authors' code: every fifth agent is held out, its ability is fit on the even-numbered tasks, and the error is the mean absolute gap between the predicted and the actual solve rate on the odd-numbered tasks.
    The smallest dimension within 0.0025 of the best error wins; its error is the bias b of the gp-IRT weight.
    """
    # note (Haoling Pu): the date split bootstraps its selection agents; a repeated agent on both sides of the hold-out would leak.
    unique_outcomes = selection_outcomes[~selection_outcomes.index.duplicated()]
    is_validation = np.arange(len(unique_outcomes)) % VALIDATION_AGENT_STRIDE == 0
    validation_outcomes = unique_outcomes[is_validation]
    seen_task_ids = unique_outcomes.columns[::2]
    unseen_task_ids = unique_outcomes.columns[1::2]
    validation_errors = []
    for dimension in IRT_DIMENSIONS:
        irt = fit_multidimensional_irt(
            unique_outcomes[~is_validation],
            dimension=dimension,
            random_generator=random_generator,
        )
        ability = fit_ability(validation_outcomes[seen_task_ids], irt)
        predicted_solve_rate = irt.solve_probability(ability)[unseen_task_ids].mean(
            axis=1
        )
        actual_solve_rate = validation_outcomes[unseen_task_ids].mean(axis=1)
        validation_errors.append(
            (predicted_solve_rate - actual_solve_rate).abs().mean()
        )
    validation_errors = np.array(validation_errors)
    chosen_position = int(
        np.argmax(
            validation_errors - validation_errors.min() < DIMENSION_ERROR_TOLERANCE
        )
    )
    return TinyBenchmarksFit(
        irt=fit_multidimensional_irt(
            selection_outcomes,
            dimension=IRT_DIMENSIONS[chosen_position],
            random_generator=random_generator,
        ),
        validation_error=float(validation_errors[chosen_position]),
        # note (Haoling Pu): the authors' code takes each agent's variance across tasks, then averages over agents.
        score_variance=float(
            selection_outcomes.to_numpy(dtype=float).var(axis=1).mean()
        ),
    )


def select_anchor_points(
    task_embedding: pd.DataFrame,
    *,
    subset_size: int,
    random_generator: np.random.Generator,
) -> pd.Series:
    """K-means on the task embeddings; keep the task closest to each centroid, weighted by its cluster's share of all tasks.

    The embedding is a task's results over the selection agents (correctness anchors) or its IRT parameters (IRT anchors).
    Two centroids can share their closest task, so the subset can hold fewer than subset_size tasks.
    """
    embedding = task_embedding.to_numpy(dtype=float)
    # note (Haoling Pu): the authors run K-means with 5 seeds and keep the lowest inertia.
    kmeans = min(
        (
            KMeans(n_clusters=subset_size, random_state=int(seed), n_init="auto").fit(
                embedding
            )
            for seed in random_generator.integers(2**31, size=NUM_KMEANS_RESTARTS)
        ),
        key=lambda fitted_kmeans: fitted_kmeans.inertia_,
    )
    anchor_positions = pairwise_distances(kmeans.cluster_centers_, embedding).argmin(
        axis=1
    )
    cluster_share = np.bincount(kmeans.labels_, minlength=subset_size) / len(embedding)
    return (
        pd.Series(cluster_share, index=task_embedding.index[anchor_positions])
        .groupby(level=0)
        .sum()
    )


def gp_irt_solve_rate(
    test_outcomes: pd.DataFrame,
    task_weights: pd.Series,
    tiny_benchmarks_fit: TinyBenchmarksFit,
    *,
    subset_size: int,
    anchor_points: bool,
) -> pd.Series:
    """gp-IRT estimate of each test agent's full-benchmark solve rate (eq. 4.4).

    It mixes the weighted subset score with p-IRT (observed results on the subset, IRT predictions on the other tasks).
    lambda = b^2 / (sigma^2 / n + b^2) trusts the subset more when the IRT bias b is large; anchor points use sigma^2 / 4, as in the authors' code.
    n is the requested subset_size, as in the authors' code, even when two anchors merged into one task.
    """
    seen_task_ids = task_weights.index
    unseen_task_ids = test_outcomes.columns.difference(seen_task_ids)
    seen_share = len(seen_task_ids) / test_outcomes.shape[1]
    ability = fit_ability(test_outcomes[seen_task_ids], tiny_benchmarks_fit.irt)
    p_irt_solve_rate = seen_share * test_outcomes[seen_task_ids].mean(axis=1) + (
        1.0 - seen_share
    ) * tiny_benchmarks_fit.irt.solve_probability(ability)[unseen_task_ids].mean(axis=1)
    if anchor_points:
        weighted_score_variance = tiny_benchmarks_fit.score_variance / (4 * subset_size)
    else:
        weighted_score_variance = tiny_benchmarks_fit.score_variance / subset_size
    squared_bias = tiny_benchmarks_fit.validation_error**2
    subset_weight = squared_bias / (weighted_score_variance + squared_bias)
    return (
        subset_weight * (test_outcomes[seen_task_ids] @ task_weights)
        + (1.0 - subset_weight) * p_irt_solve_rate
    )
