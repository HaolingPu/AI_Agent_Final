"""Synthetic benchmark in the unified format, so the framework runs before real data is loaded."""

import numpy as np
import pandas as pd

from subset_eval.data import BenchmarkData

TOY_START_DATE = pd.Timestamp("2024-01-01")


# note (Haoling Pu): the defaults are the toy benchmark every experiment runs on, so C1, C2 and C3 always see the same data.
def make_toy_benchmark(
    *,
    num_agents: int = 120,
    num_tasks: int = 300,
    num_trials: int = 1,
    num_months: int = 24,
    ability_gain_per_month: float = 0.1,
    seed: int,
) -> BenchmarkData:
    """Draw outcomes from a 2PL IRT model where newer agents are stronger.

    P(agent i solves task j) = sigmoid(discrimination_j * (ability_i - difficulty_j)),
    and ability_i grows by ability_gain_per_month with the agent's submit month.
    The data come from the same model that IRT selection fits, so IRT gains on toy data are an upper bound, not evidence.
    """
    random_generator = np.random.default_rng(seed)
    submit_month = np.sort(random_generator.integers(0, num_months, size=num_agents))
    ability = ability_gain_per_month * submit_month + random_generator.normal(
        0.0, 1.0, size=num_agents
    )
    difficulty = random_generator.normal(0.0, 1.5, size=num_tasks)
    discrimination = random_generator.lognormal(0.0, 0.4, size=num_tasks)
    solve_probability = 1.0 / (
        1.0 + np.exp(-discrimination * (ability[:, None] - difficulty[None, :]))
    )

    agent_ids = [f"toy-agent-{i:03d}" for i in range(num_agents)]
    task_ids = [f"toy-task-{j:03d}" for j in range(num_tasks)]
    agents = pd.DataFrame(
        {
            "agent_id": agent_ids,
            "scaffold": "toy-scaffold",
            "model": [f"toy-model-{i:03d}" for i in range(num_agents)],
            "model_org": "toy-org",
            "open_weight": False,
            "submit_date": [
                TOY_START_DATE + pd.DateOffset(months=int(month))
                for month in submit_month
            ],
            "model_release_date": pd.NaT,
        }
    )
    tasks = pd.DataFrame({"task_id": task_ids, "category": "toy"})

    resolved = (
        random_generator.random((num_trials, num_agents, num_tasks))
        < solve_probability[None, :, :]
    )
    trial_index, agent_index, task_index = np.indices(resolved.shape)
    runs = pd.DataFrame(
        {
            "agent_id": np.asarray(agent_ids)[agent_index.ravel()],
            "task_id": np.asarray(task_ids)[task_index.ravel()],
            "trial": trial_index.ravel() + 1,
            "resolved": resolved.ravel().astype(int),
            "cost_usd": np.nan,
            "total_tokens": np.nan,
        }
    )
    return BenchmarkData(name="toy", agents=agents, tasks=tasks, runs=runs)
