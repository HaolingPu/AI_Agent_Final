"""Terminal-Bench 4.1 loader: the official Harbor Hub leaderboard (67 rows, 2026-02 to 2026-11), 66 tasks x 5 trials per row.

Download the data first with scripts/fetch_terminal_bench_4_1.py (needs a free Harbor Hub account).
Unlike 2.0, the leaderboard is still open, and its top agent solves about 71 %, so it is the unsaturated contrast for C3.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from subset_eval.data import BenchmarkData

logger = logging.getLogger(__name__)

DATA_DIR = Path("terminal_bench_4_1")
NUM_TASKS = 66
NUM_TRIALS_PER_TASK = 5
SCAFFOLD_ALIASES = {"mini-SWE-agent": "mini-swe-agent"}


def load_terminal_bench_4_1(
    raw_data_dir: Path, *, drop_regrade_errors: bool
) -> BenchmarkData:
    """Every leaderboard row, scored on the trials linked to it.

    A trial with no reward (an agent error, a timeout, a failed regrade, ...) counts as unresolved, as for Terminal-Bench 2.0 and the leaderboard.
    drop_regrade_errors (robustness check): a failed regrade is left out instead, a task's score is the mean over its graded trials,
    and the tasks where some row has no graded trial left are dropped (13 of 66 as fetched 2026-11-27), so every row is kept.
    """
    data_dir = raw_data_dir / DATA_DIR
    if not (data_dir / "rows.json").exists():
        raise FileNotFoundError(
            f"{data_dir / 'rows.json'} not found; run scripts/fetch_terminal_bench_4_1.py"
        )
    else:
        pass

    agent_rows = []
    run_tables = []
    for row in json.loads((data_dir / "rows.json").read_text()):
        metadata = row["metadata"]
        # note (Haoling Pu): several rows run one model at different reasoning efforts, so the effort is part of the id.
        agent_id = f"{metadata['agent_display']['label']}__{metadata['model_display']['label']}__{metadata['reasoning_effort']}"
        row_trial_file = json.loads(
            (data_dir / "trials" / f"{row['id']}.json").read_text()
        )
        trials = pd.DataFrame(row_trial_file["trials"])
        if len(trials) != row_trial_file["num_linked_trials"]:
            raise ValueError(
                f"{agent_id}: {len(trials)} of {row_trial_file['num_linked_trials']} linked trials were downloaded"
            )
        else:
            pass
        trials["task_id"] = trials["Task"].str.removeprefix("terminal-bench/")
        trials["resolved"] = (trials["Reward"] == "1.000").astype(int)
        # note (Haoling Pu): most rows were re-verified in 2026-11 (agent "nop"), so a trial's start time is when it was graded, not when the agent ran; it only orders the trials of a task.
        trials = trials.sort_values(["task_id", "Started", "ID"])
        trials["trial"] = trials.groupby("task_id").cumcount() + 1
        trials["agent_id"] = agent_id
        # note (Haoling Pu): the linked trials' costs disagree with the leaderboard's total for about 20 rows (5 rows have none), and nothing downstream reads cost yet.
        trials["cost_usd"] = pd.to_numeric(
            trials["Cost"].replace("", None).str.removeprefix("$"), errors="raise"
        )
        trials["total_tokens"] = float("nan")
        run_tables.append(trials)
        num_successes = int(trials["resolved"].sum())
        scaffold = metadata["agent_display"]["label"]
        agent_rows.append(
            {
                "agent_id": agent_id,
                # note (Haoling Pu): two rows spell the scaffold "mini-SWE-agent", the other ten "mini-swe-agent".
                "scaffold": SCAFFOLD_ALIASES.get(scaffold, scaffold),
                "model": metadata["model_display"]["label"],
                "model_org": metadata["model_org"]["label"],
                # note (Haoling Pu): the leaderboard does not say whether a model is open-weight.
                "open_weight": None,
                # note (Haoling Pu): the leaderboard's date is the model's release date; there is no submission date (2.0 is dated by its first trial instead). Gemini 3.1 Pro and GPT-5.4 Mini were added in 2026-11 but sit early by release date.
                "submit_date": pd.Timestamp(metadata["date"]),
                "model_release_date": pd.Timestamp(metadata["date"]),
                "reasoning_effort": metadata["reasoning_effort"],
                # note (Haoling Pu): the per-row stats describe all linked trials, also in the graded-only variant.
                "missing_reward_share": (trials["Reward"] == "").mean(),
                # note (Haoling Pu): a failed regrade is a grading failure on the Hub, not the agent's, but it still counts as unresolved (up to 76 of 330 trials in a row).
                "regrade_error_share": (trials["Error"] == "RegradeError").mean(),
                # note (Haoling Pu): the Hub does not compute a row's score from its trials; the submitter's own count differs from the linked trials for 19 of 67 rows (by 1-9 of 330).
                "leaderboard_successes_minus_linked": row["metrics"]["successes"]
                - num_successes,
            }
        )
    agents = pd.DataFrame(agent_rows)
    runs = pd.concat(run_tables, ignore_index=True)

    trials_per_task = runs.groupby(["agent_id", "task_id"]).size()
    task_ids = sorted(runs["task_id"].unique())
    if len(task_ids) != NUM_TASKS:
        raise ValueError(
            f"terminal_bench_4_1: expected {NUM_TASKS} task ids, found {len(task_ids)}"
        )
    elif (
        len(trials_per_task) != len(agents) * NUM_TASKS
        or (trials_per_task != NUM_TRIALS_PER_TASK).any()
    ):
        raise ValueError(
            f"terminal_bench_4_1: every row should have {NUM_TRIALS_PER_TASK} trials on each of the {NUM_TASKS} tasks"
        )
    else:
        pass
    if drop_regrade_errors:
        runs = runs[runs["Error"] != "RegradeError"]
        graded_trials = (
            runs.groupby(["agent_id", "task_id"]).size().unstack(fill_value=0)
        )
        task_ids = graded_trials.columns[(graded_trials > 0).all()].tolist()
        runs = runs[runs["task_id"].isin(task_ids)]
        kept_graded_trials = graded_trials[task_ids]
        # note (Haoling Pu): a cell with fewer graded trials is a noisier mean, yet IRT weighs every cell alike.
        logger.info(
            f"terminal_bench_4_1: dropped the failed regrades; {len(task_ids)} of {NUM_TASKS} tasks have a graded trial for every row; "
            f"{int((kept_graded_trials < NUM_TRIALS_PER_TASK).sum().sum())} kept (row, task) cells rest on fewer than {NUM_TRIALS_PER_TASK} trials "
            f"(fewest {int(kept_graded_trials.min().min())})"
        )
    else:
        pass
    mismatched_agents = agents[agents["leaderboard_successes_minus_linked"] != 0]
    logger.info(
        f"terminal_bench_4_1: {len(agents)} rows; the leaderboard's success count differs from the linked trials for "
        + ", ".join(
            f"{agent_id} ({difference:+d})"
            for agent_id, difference in zip(
                mismatched_agents["agent_id"],
                mismatched_agents["leaderboard_successes_minus_linked"],
            )
        )
    )
    return BenchmarkData(
        name="terminal_bench_4_1_graded_only"
        if drop_regrade_errors
        else "terminal_bench_4_1",
        agents=agents,
        # note (Haoling Pu): the trial listing carries no task category.
        tasks=pd.DataFrame({"task_id": task_ids, "category": "terminal-bench"}),
        runs=runs[
            ["agent_id", "task_id", "trial", "resolved", "cost_usd", "total_tokens"]
        ],
    )
