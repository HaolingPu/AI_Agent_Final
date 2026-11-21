"""Unified benchmark format shared by every loader: agents, tasks and runs tables."""

from dataclasses import dataclass

import pandas as pd

AGENT_COLUMNS = (
    "agent_id",
    "scaffold",
    "model",
    "model_org",
    "open_weight",
    "submit_date",
    "model_release_date",
)
TASK_COLUMNS = ("task_id", "category")
RUN_COLUMNS = ("agent_id", "task_id", "trial", "resolved", "cost_usd", "total_tokens")


@dataclass(kw_only=True)
class BenchmarkData:
    """One benchmark in long format. A loader's only job is to return this.

    agents: one row per submission; submit_date is a pandas Timestamp.
    tasks: one row per task.
    runs: one row per (agent, task, trial); resolved is 0 or 1, cost_usd and total_tokens may be NaN.
    """

    name: str
    agents: pd.DataFrame
    tasks: pd.DataFrame
    runs: pd.DataFrame

    def __post_init__(self) -> None:
        for table_name, table, columns in (
            ("agents", self.agents, AGENT_COLUMNS),
            ("tasks", self.tasks, TASK_COLUMNS),
            ("runs", self.runs, RUN_COLUMNS),
        ):
            missing_columns = set(columns) - set(table.columns)
            if missing_columns:
                raise ValueError(
                    f"{self.name}.{table_name} misses columns {sorted(missing_columns)}"
                )
            else:
                pass
        if not self.agents["agent_id"].is_unique or not self.tasks["task_id"].is_unique:
            raise ValueError(f"{self.name}: agent_id and task_id must be unique")
        elif not self.runs["agent_id"].isin(self.agents["agent_id"]).all():
            raise ValueError(
                f"{self.name}.runs has agent_id values missing from agents"
            )
        elif not self.runs["task_id"].isin(self.tasks["task_id"]).all():
            raise ValueError(f"{self.name}.runs has task_id values missing from tasks")
        elif not self.runs["resolved"].isin((0, 1)).all():
            raise ValueError(f"{self.name}.runs.resolved must be 0 or 1")
        elif self.runs.duplicated(["agent_id", "task_id", "trial"]).any():
            raise ValueError(
                f"{self.name}.runs has duplicate (agent, task, trial) rows"
            )
        elif not pd.api.types.is_datetime64_any_dtype(self.agents["submit_date"]):
            raise ValueError(f"{self.name}.agents.submit_date must be datetime64")
        elif self.agents["submit_date"].isna().any():
            raise ValueError(f"{self.name}.agents has agents without a submit_date")
        else:
            pass


def outcome_matrix(benchmark: BenchmarkData) -> pd.DataFrame:
    """Agents x tasks solve rate, averaged over trials (0/1 when there is one trial)."""
    matrix = benchmark.runs.pivot_table(
        index="agent_id", columns="task_id", values="resolved", aggfunc="mean"
    )
    # note (Haoling Pu): every agent must have every task; a loader writes unresolved tasks as 0, never leaves them out.
    if matrix.isna().any().any() or matrix.shape != (
        len(benchmark.agents),
        len(benchmark.tasks),
    ):
        raise ValueError(f"{benchmark.name}: some (agent, task) pairs have no run")
    else:
        pass
    return matrix.loc[benchmark.agents["agent_id"], benchmark.tasks["task_id"]]
