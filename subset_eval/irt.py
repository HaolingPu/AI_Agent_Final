"""Two-parameter logistic (2PL) IRT: agent ability, task difficulty and task discrimination."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

# note (Haoling Pu): Gaussian priors fix the scale (ability centred at 0) and keep tasks that every agent solves, or fails, at a finite difficulty.
ABILITY_PRIOR_SD = 1.0
DIFFICULTY_PRIOR_SD = 3.0
LOG_DISCRIMINATION_PRIOR_SD = 0.5


@dataclass(kw_only=True)
class IrtParameters:
    ability: pd.Series
    difficulty: pd.Series
    discrimination: pd.Series

    def task_information(self, ability: np.ndarray) -> pd.Series:
        """Fisher information of each task, averaged over the given abilities: a^2 * P * (1 - P)."""
        solve_probability = expit(
            self.discrimination.to_numpy()
            * (ability[:, None] - self.difficulty.to_numpy()[None, :])
        )
        information = (
            self.discrimination.to_numpy() ** 2
            * solve_probability
            * (1.0 - solve_probability)
        )
        return pd.Series(information.mean(axis=0), index=self.difficulty.index)


def fit_irt(outcomes: pd.DataFrame) -> IrtParameters:
    """Maximum a posteriori 2PL fit; a solve rate between 0 and 1 (several trials) counts as a fractional success."""
    solve_rate = outcomes.to_numpy(dtype=float)
    num_agents, num_tasks = solve_rate.shape

    def _negative_log_posterior(
        flat_parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        ability = flat_parameters[:num_agents]
        difficulty = flat_parameters[num_agents : num_agents + num_tasks]
        log_discrimination = flat_parameters[num_agents + num_tasks :]
        discrimination = np.exp(log_discrimination)
        logit = discrimination[None, :] * (ability[:, None] - difficulty[None, :])
        negative_log_likelihood = np.sum(np.logaddexp(0.0, logit) - solve_rate * logit)
        residual = expit(logit) - solve_rate
        negative_log_prior = 0.5 * (
            np.sum(ability**2) / ABILITY_PRIOR_SD**2
            + np.sum(difficulty**2) / DIFFICULTY_PRIOR_SD**2
            + np.sum(log_discrimination**2) / LOG_DISCRIMINATION_PRIOR_SD**2
        )
        gradient = np.concatenate(
            [
                (residual * discrimination[None, :]).sum(axis=1)
                + ability / ABILITY_PRIOR_SD**2,
                -(residual * discrimination[None, :]).sum(axis=0)
                + difficulty / DIFFICULTY_PRIOR_SD**2,
                (residual * logit).sum(axis=0)
                + log_discrimination / LOG_DISCRIMINATION_PRIOR_SD**2,
            ]
        )
        return negative_log_likelihood + negative_log_prior, gradient

    result = minimize(
        _negative_log_posterior,
        x0=np.zeros(num_agents + 2 * num_tasks),
        jac=True,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"IRT fit did not converge: {result.message}")
    else:
        pass
    return IrtParameters(
        ability=pd.Series(result.x[:num_agents], index=outcomes.index),
        difficulty=pd.Series(
            result.x[num_agents : num_agents + num_tasks], index=outcomes.columns
        ),
        discrimination=pd.Series(
            np.exp(result.x[num_agents + num_tasks :]), index=outcomes.columns
        ),
    )
