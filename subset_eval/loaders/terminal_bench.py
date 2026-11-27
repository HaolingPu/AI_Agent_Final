"""Terminal-Bench 2.0 loader: one submission folder of the Hugging Face dataset harborframework/terminal-bench-2-leaderboard per agent, about 5 trials per task.

The dataset is over 100 GB of logs; the loader reads only metadata.yaml and each trial's result.json (about 170 MB).
Hugging Face's git server does not serve single files, so fetch them over HTTPS from the file list:

    from huggingface_hub import HfApi, hf_hub_download
    paths = [
        f.path
        for f in HfApi().list_repo_tree(
            "harborframework/terminal-bench-2-leaderboard",
            repo_type="dataset",
            recursive=True,
            path_in_repo="submissions/terminal-bench/2.0",
        )
        if f.path.endswith(("/metadata.yaml", "/metadata.yml", "/result.json"))
    ]
    for path in paths:
        hf_hub_download(
            "harborframework/terminal-bench-2-leaderboard",
            path,
            repo_type="dataset",
            local_dir="data/raw/terminal_bench_2_leaderboard",
        )

Keep the download slow (a few files at a time): Hugging Face answers HTTP 429 to fast anonymous downloads.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from subset_eval.data import BenchmarkData

logger = logging.getLogger(__name__)

SUBMISSIONS_DIR = Path("terminal_bench_2_leaderboard/submissions/terminal-bench/2.0")
NUM_TASKS = 89
# note (Haoling Pu): Terminal-Bench 2.1 changed tasks, so its trials are not comparable with 2.0 and are left out.
TERMINAL_BENCH_2_1_SOURCE = "terminal-bench/terminal-bench-2-1"
# note (Haoling Pu): five submissions spell this task with a dash; the others use the official dot.
TASK_ID_ALIASES = {"install-windows-3-11": "install-windows-3.11"}
# note (Haoling Pu): Crux's only job also holds earlier runs of other models (Sonnet 4, Opus 4.5, ...) from 2025-11 to 2026-02; its Opus 4.6 trials alone cover all 89 tasks and are the submission.
TRIAL_MODEL_BY_SUBMISSION = {"Crux__Claude-Opus-4.6": "claude-opus-4-6"}


def load_terminal_bench(raw_data_dir: Path) -> BenchmarkData:
    """Every submission with at least one 2.0 trial on each of the 89 tasks.

    A trial whose verifier never ran (an infrastructure error, a setup timeout, ...) has no reward and counts as unresolved, as on the public leaderboard.
    """
    submissions_dir = raw_data_dir / SUBMISSIONS_DIR
    if not submissions_dir.is_dir():
        raise FileNotFoundError(
            f"{submissions_dir} not found; see the download steps in {__file__}"
        )
    else:
        pass

    agent_rows = []
    run_tables = []
    left_out = []
    for submission_dir in sorted(
        path for path in submissions_dir.iterdir() if path.is_dir()
    ):
        # note (Haoling Pu): two submissions name the file metadata.yml.
        metadata_path = submission_dir / "metadata.yaml"
        if metadata_path.exists():
            metadata = yaml.safe_load(metadata_path.read_text())
        else:
            metadata = yaml.safe_load((submission_dir / "metadata.yml").read_text())
        trial_rows = []
        # note (Haoling Pu): <submission>/<job>/<trial>/result.json; the job folder's own result.json only summarizes it.
        for result_path in sorted(submission_dir.glob("*/*/result.json")):
            trial_result = json.loads(result_path.read_text())
            # note (Haoling Pu): task names come as "terminal-bench/<task>" or plain "<task>".
            task_name = trial_result["task_name"].rsplit("/", 1)[-1]
            verifier_result = trial_result["verifier_result"]
            agent_result = trial_result["agent_result"]
            if verifier_result is None:
                reward = 0.0
            else:
                reward = verifier_result["rewards"]["reward"]
            if agent_result is None:
                cost_usd = None
                total_tokens = None
            else:
                cost_usd = agent_result["cost_usd"]
                input_tokens = agent_result["n_input_tokens"]
                output_tokens = agent_result["n_output_tokens"]
                if input_tokens is None or output_tokens is None:
                    total_tokens = None
                else:
                    total_tokens = input_tokens + output_tokens
            trial_rows.append(
                {
                    "agent_id": submission_dir.name,
                    "source": trial_result["source"],
                    "trial_model": trial_result["config"]["agent"]["model_name"],
                    "task_id": TASK_ID_ALIASES.get(task_name, task_name),
                    "started_at": pd.to_datetime(trial_result["started_at"], utc=True),
                    "resolved": int(reward == 1.0),
                    "has_reward": verifier_result is not None,
                    "cost_usd": cost_usd,
                    "total_tokens": total_tokens,
                }
            )
        all_trials = pd.DataFrame(trial_rows)
        is_kept = all_trials["source"] != TERMINAL_BENCH_2_1_SOURCE
        if submission_dir.name in TRIAL_MODEL_BY_SUBMISSION:
            is_kept &= (
                all_trials["trial_model"]
                == TRIAL_MODEL_BY_SUBMISSION[submission_dir.name]
            )
        else:
            pass
        trials = all_trials[is_kept]
        if trials.empty:
            left_out.append(f"{submission_dir.name} (no Terminal-Bench 2.0 trials)")
        elif trials["task_id"].nunique() < NUM_TASKS:
            left_out.append(
                f"{submission_dir.name} (trials on {trials['task_id'].nunique()} of {NUM_TASKS} tasks)"
            )
        else:
            # note (Haoling Pu): trials are numbered per task in start order; some submissions ran more than 5 (two full runs, or retries).
            trials = trials.sort_values(["task_id", "started_at"])
            trials["trial"] = trials.groupby("task_id").cumcount() + 1
            run_tables.append(trials)
            models = metadata["models"]
            agent_rows.append(
                {
                    "agent_id": submission_dir.name,
                    "scaffold": metadata["agent_display_name"],
                    "model": " + ".join(
                        model["model_display_name"] for model in models
                    ),
                    "model_org": " + ".join(
                        sorted({model["model_org_display_name"] for model in models})
                    ),
                    # note (Haoling Pu): metadata.yaml does not say whether a model is open-weight.
                    "open_weight": None,
                    # note (Haoling Pu): there is no submission date, so an agent is dated by the day its first trial started.
                    "submit_date": trials["started_at"]
                    .min()
                    .tz_localize(None)
                    .normalize(),
                    "model_release_date": pd.NaT,
                    "missing_reward_share": 1.0 - trials["has_reward"].mean(),
                }
            )
    agents = pd.DataFrame(agent_rows)
    runs = pd.concat(run_tables, ignore_index=True)
    logger.info(
        f"terminal_bench: kept {len(agents)} submissions; left out {len(left_out)}: "
        + "; ".join(left_out)
    )
    heavy_error_agents = agents[agents["missing_reward_share"] > 0.1]
    logger.info(
        "terminal_bench: submissions where over 10 % of trials have no reward (counted as unresolved): "
        + ", ".join(
            f"{agent_id} {share:.0%}"
            for agent_id, share in zip(
                heavy_error_agents["agent_id"],
                heavy_error_agents["missing_reward_share"],
            )
        )
    )

    task_ids = sorted(runs["task_id"].unique())
    if len(task_ids) != NUM_TASKS:
        raise ValueError(
            f"terminal_bench: expected {NUM_TASKS} task ids, found {len(task_ids)}"
        )
    else:
        pass
    return BenchmarkData(
        name="terminal_bench",
        agents=agents,
        # note (Haoling Pu): result.json carries no task category.
        tasks=pd.DataFrame({"task_id": task_ids, "category": "terminal-bench"}),
        runs=runs[
            ["agent_id", "task_id", "trial", "resolved", "cost_usd", "total_tokens"]
        ].astype({"cost_usd": float, "total_tokens": float}),
    )
