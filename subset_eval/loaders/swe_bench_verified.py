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
import re
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
# note (Haoling Pu): lower-cased model names that name a dated model without its date or in another style; checked by hand on the 2026-11 data.
MODEL_ALIASES = {
    "claude 3.7 sonnet": "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude opus 4.5": "claude-opus-4-5-20251101",
    "claude-4-5-opus": "claude-opus-4-5-20251101",
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "gpt-5-2": "gpt-5.2-2025-12-11",
    "gemini 2.0 flash (v20241212-experimental)": "gemini-2.0-flash-exp",
    "o4-mini": "o4-mini-20250416",
}


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


def agent_row(submission_dir: Path, metadata: dict[str, dict]) -> dict:
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
        # note (Haoling Pu): a few submissions name the file metadata.yml.
        metadata_path = submission_dir / "metadata.yaml"
        if metadata_path.exists():
            metadata = yaml.safe_load(metadata_path.read_text())
        else:
            metadata = yaml.safe_load((submission_dir / "metadata.yml").read_text())
        reported_solve_rate = metadata["info"].get("resolved")
        if submission_results is None:
            left_out.append(f"{submission_dir.name} (no complete per-task results)")
            continue
        else:
            per_task_solve_rate = (
                100 * len(submission_results.resolved_task_ids) / NUM_TASKS
            )
        # note (Haoling Pu): only 48 of 182 submissions report a solve rate, so only those can be checked against it.
        if (
            reported_solve_rate is not None
            and abs(per_task_solve_rate - float(reported_solve_rate))
            > MAX_REPORTED_SOLVE_RATE_GAP
        ):
            left_out.append(
                f"{submission_dir.name} (per-task {per_task_solve_rate:.1f} % vs reported {reported_solve_rate} %)"
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


def load_swe_bench_verified_mini_swe_agent(raw_data_dir: Path) -> BenchmarkData:
    """X3: only the mini-SWE-agent runs (one scaffold, many models), dated by the model's release instead of the submission.

    mini-SWE-agent re-runs older models in batches (11 submissions on 2026-02-17 alone), so its submit date says little about when a model appeared.
    """
    swe_bench = load_swe_bench_verified(raw_data_dir)
    is_mini_swe_agent = swe_bench.agents["scaffold"] == "mini-SWE-agent"
    agents = swe_bench.agents[is_mini_swe_agent].reset_index(drop=True)
    # note (Wilson Zheng): every mini-SWE-agent submission reports its model's release date; BenchmarkData rejects a missing one.
    agents["submit_date"] = agents["model_release_date"]
    return BenchmarkData(
        name="swe_bench_verified_mini_swe_agent",
        agents=agents,
        tasks=swe_bench.tasks,
        runs=swe_bench.runs[swe_bench.runs["agent_id"].isin(agents["agent_id"])],
    )


def load_swe_bench_verified_one_per_model(raw_data_dir: Path) -> BenchmarkData:
    """One submission per model: the best-scoring one (the earliest on a tie); a submission with an undisclosed model is kept as its own model.

    Several scaffolds run the same model (claude-3-5-sonnet-20241022 has 14 submissions), so a random split puts near-twins in both groups.
    Model names are matched after dropping a Hugging Face or provider prefix, lower-casing, and sorting the word parts (not the version numbers),
    so claude-4-sonnet-20250514 and claude-sonnet-4-20250514 are one model but gpt-4.5 and gpt-5.4 would not be;
    MODEL_ALIASES joins the remaining spellings of one model, and a "+" combination is matched as a set.
    """

    def _model_key(model: str) -> str:
        part_keys = set()
        for part in model.split("+"):
            name = part.strip().lower().rsplit("/", 1)[-1]
            tokens = [
                token
                for token in re.split(r"[-_ .()]+", MODEL_ALIASES.get(name, name))
                if token
            ]
            words = sorted(token for token in tokens if not token.isdigit())
            numbers = [token for token in tokens if token.isdigit()]
            part_keys.add(f"{'-'.join(words)}|{'-'.join(numbers)}")
        return " + ".join(sorted(part_keys))

    swe_bench = load_swe_bench_verified(raw_data_dir)
    agents = swe_bench.agents.copy()
    solve_rate = swe_bench.runs.groupby("agent_id")["resolved"].mean()
    agents["solve_rate"] = agents["agent_id"].map(solve_rate)
    # note (Haoling Pu): pandas stores a missing model as NaN; each such submission is its own model.
    agents["model_key"] = [
        agent_id if pd.isna(model) else _model_key(model)
        for agent_id, model in zip(agents["agent_id"], agents["model"])
    ]
    kept_agents = (
        agents.sort_values(["solve_rate", "submit_date"], ascending=[False, True])
        .drop_duplicates("model_key")
        .sort_values("agent_id")
        .drop(columns=["solve_rate", "model_key"])
        .reset_index(drop=True)
    )
    logger.info(
        f"swe_bench_verified_one_per_model: {len(kept_agents)} of {len(agents)} submissions, one per model"
    )
    return BenchmarkData(
        name="swe_bench_verified_one_per_model",
        agents=kept_agents,
        tasks=swe_bench.tasks,
        runs=swe_bench.runs[swe_bench.runs["agent_id"].isin(kept_agents["agent_id"])],
    )
