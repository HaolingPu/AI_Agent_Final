"""SWE-bench Verified loader: one submission folder of github.com/SWE-bench/experiments per agent, one trial per task.

Download only the files the loader reads (about 5 MB instead of the full repo's logs):

    git clone --filter=blob:none --no-checkout --depth 1 https://github.com/SWE-bench/experiments.git data/raw/swe_bench_experiments
    cd data/raw/swe_bench_experiments
    git sparse-checkout set --no-cone '/evaluation/verified/*/metadata.yaml' '/evaluation/verified/*/metadata.yml' \
        '/evaluation/verified/*/results/results.json' '/evaluation/verified/*/per_instance_details.json'
    git checkout
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from subset_eval.data import BenchmarkData

logger = logging.getLogger(__name__)

SUBMISSIONS_DIR = Path("swe_bench_experiments/evaluation/verified")
NUM_TASKS = 500
# note (Wilson Zheng): two mini-SWE-agent submissions flag far fewer tasks as resolved than their metadata reports (10.2 % vs 52.8 %, 0 % vs 69.6 %); a gap this large means broken per-task data, so they are left out.
MAX_REPORTED_SOLVE_RATE_GAP = 1.0


@dataclass(kw_only=True)
class SubmissionResults:
    resolved_task_ids: set[str]
    # note (Wilson Zheng): only per_instance_details.json (mini-SWE-agent) names every task and its cost; results.json lists the resolved ones.
    detailed_task_ids: set[str]
    cost_by_task_id: dict[str, float]


def read_submission_results(submission_dir: Path) -> SubmissionResults | None:
    """Per-task results of one submission, or None when it has no per-task file or its file misses tasks."""
    results_path = submission_dir / "results" / "results.json"
    details_path = submission_dir / "per_instance_details.json"
    if details_path.exists():
        details = json.loads(details_path.read_text())
        cost_by_task_id = {
            task_id: float(detail["cost"])
            for task_id, detail in details.items()
            if detail.get("cost") is not None
        }
    else:
        details = {}
        cost_by_task_id = {}
    # note (Wilson Zheng): one submission has both files and they agree on every task, but its details miss 59 tasks, so results.json wins.
    if results_path.exists():
        resolved_task_ids = set(json.loads(results_path.read_text())["resolved"])
    elif len(details) == NUM_TASKS:
        resolved_task_ids = {
            task_id for task_id, detail in details.items() if detail["resolved"]
        }
    else:
        return None
    return SubmissionResults(
        resolved_task_ids=resolved_task_ids,
        detailed_task_ids=set(details),
        cost_by_task_id=cost_by_task_id,
    )


def read_metadata(submission_dir: Path) -> dict:
    """metadata.yaml, or metadata.yml for the few submissions that use that name."""
    yaml_path = submission_dir / "metadata.yaml"
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text())
    else:
        return yaml.safe_load((submission_dir / "metadata.yml").read_text())


def agent_row(submission_dir: Path, metadata: dict) -> dict:
    tags = metadata["tags"]
    model = tags.get("model")
    # note (Wilson Zheng): an agent that calls several models lists them all; one without a model list falls back to its display name unless that is "Undisclosed".
    if isinstance(model, list):
        model_name = " + ".join(str(name) for name in model)
    elif isinstance(model, str):
        model_name = model
    elif tags["model_display"] != "Undisclosed":
        model_name = tags["model_display"]
    else:
        model_name = None
    model_release_date = metadata["info"].get("model_release_date")
    return {
        "agent_id": submission_dir.name,
        "scaffold": tags["agent"],
        "model": model_name,
        "model_org": tags.get("model_org"),
        "open_weight": bool(tags["os_model"]),
        # note (Wilson Zheng): the folder name starts with the submission date, YYYYMMDD.
        "submit_date": pd.to_datetime(submission_dir.name[:8], format="%Y%m%d"),
        "model_release_date": (
            pd.NaT
            if model_release_date is None
            else pd.to_datetime(str(model_release_date), format="%Y%m%d")
        ),
    }


def load_swe_bench_verified(raw_data_dir: Path) -> BenchmarkData:
    """Every submission with complete per-task results; unresolved tasks are rows with resolved = 0."""
    submissions_dir = raw_data_dir / SUBMISSIONS_DIR
    if not submissions_dir.is_dir():
        raise FileNotFoundError(
            f"{submissions_dir} not found; see the download steps in {__file__}"
        )
    else:
        pass
    submission_dirs = sorted(
        path for path in submissions_dir.iterdir() if path.is_dir()
    )
    agent_rows = []
    results_by_agent_id = {}
    left_out = []
    for submission_dir in submission_dirs:
        submission_results = read_submission_results(submission_dir)
        metadata = read_metadata(submission_dir)
        reported_solve_rate = metadata["info"].get("resolved")
        if submission_results is None:
            left_out.append(f"{submission_dir.name} (no complete per-task results)")
        elif (
            reported_solve_rate is not None
            and abs(
                100 * len(submission_results.resolved_task_ids) / NUM_TASKS
                - float(reported_solve_rate)
            )
            > MAX_REPORTED_SOLVE_RATE_GAP
        ):
            left_out.append(
                f"{submission_dir.name} (per-task {100 * len(submission_results.resolved_task_ids) / NUM_TASKS:.1f} % vs reported {reported_solve_rate} %)"
            )
        else:
            agent_rows.append(agent_row(submission_dir, metadata))
            results_by_agent_id[submission_dir.name] = submission_results
    logger.info(
        f"swe_bench_verified: {len(agent_rows)} of {len(submission_dirs)} submissions kept; left out {len(left_out)}: "
        + "; ".join(left_out)
    )

    # note (Wilson Zheng): no file lists the 500 tasks on its own; the mini-SWE-agent details name all of them, and every resolved list must fall inside.
    task_ids = sorted(
        set().union(
            *(
                submission_results.detailed_task_ids
                | submission_results.resolved_task_ids
                for submission_results in results_by_agent_id.values()
            )
        )
    )
    if len(task_ids) != NUM_TASKS:
        raise ValueError(
            f"swe_bench_verified: expected {NUM_TASKS} task ids, found {len(task_ids)}"
        )
    else:
        pass
    tasks = pd.DataFrame(
        {
            "task_id": task_ids,
            # note (Wilson Zheng): an instance id is <owner>__<repo>-<issue number>, so the category is the repo.
            "category": [
                task_id.rsplit("-", 1)[0].replace("__", "/") for task_id in task_ids
            ],
        }
    )

    agent_ids = list(results_by_agent_id)
    runs = pd.DataFrame(
        {
            "agent_id": np.repeat(agent_ids, len(task_ids)),
            "task_id": np.tile(task_ids, len(agent_ids)),
            "trial": 1,
        }
    )
    runs["resolved"] = [
        int(task_id in results_by_agent_id[agent_id].resolved_task_ids)
        for agent_id, task_id in zip(runs["agent_id"], runs["task_id"])
    ]
    runs["cost_usd"] = [
        results_by_agent_id[agent_id].cost_by_task_id.get(task_id, np.nan)
        for agent_id, task_id in zip(runs["agent_id"], runs["task_id"])
    ]
    runs["total_tokens"] = np.nan
    return BenchmarkData(
        name="swe_bench_verified",
        agents=pd.DataFrame(agent_rows),
        tasks=tasks,
        runs=runs,
    )
