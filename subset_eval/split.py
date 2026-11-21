"""Split agents into a selection group (whose results pick tasks) and a test group (ranked on them)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SplitKind = Literal["random", "date"]


@dataclass(kw_only=True)
class AgentSplit:
    selection_agent_ids: list[str]
    test_agent_ids: list[str]

    def __post_init__(self) -> None:
        if not self.selection_agent_ids or len(self.test_agent_ids) < 2:
            raise ValueError(
                "a split needs at least one selection agent and two test agents"
            )
        else:
            pass


def split_random(
    agents: pd.DataFrame,
    *,
    selection_fraction: float,
    random_generator: np.random.Generator,
) -> AgentSplit:
    """Prior work's setting: selection and test agents come from the same pool."""
    shuffled_agent_ids = random_generator.permutation(
        agents["agent_id"].to_numpy()
    ).tolist()
    num_selection_agents = round(selection_fraction * len(shuffled_agent_ids))
    return AgentSplit(
        selection_agent_ids=shuffled_agent_ids[:num_selection_agents],
        test_agent_ids=shuffled_agent_ids[num_selection_agents:],
    )


def split_by_date(agents: pd.DataFrame, *, cutoff_date: pd.Timestamp) -> AgentSplit:
    """Agents submitted before cutoff_date select; agents on or after it are tested."""
    is_before_cutoff = agents["submit_date"] < cutoff_date
    return AgentSplit(
        selection_agent_ids=agents.loc[is_before_cutoff, "agent_id"].tolist(),
        test_agent_ids=agents.loc[~is_before_cutoff, "agent_id"].tolist(),
    )
