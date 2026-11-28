"""METR time-horizon loader: every run METR scored for its time-horizon reports, about 5-8 trials per (model, task).

Tasks come from HCAST, RE-Bench and SWAA, from seconds to many hours of human time. One row per model (METR's alias): METR runs each model in its own scaffold.
Download from github.com/METR/eval-analysis-public (about 40 MB):

    B=https://raw.githubusercontent.com/METR/eval-analysis-public/main
    curl -fL $B/reports/time-horizon-1-0/data/raw/runs.jsonl -o data/raw/metr/time-horizon-1-0/runs.jsonl
    curl -fL $B/reports/time-horizon-1-1/data/raw/runs.jsonl -o data/raw/metr/time-horizon-1-1/runs.jsonl
    curl -fL $B/data/external/release_dates.yaml -o data/raw/metr/release_dates.yaml
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from subset_eval.data import BenchmarkData

logger = logging.getLogger(__name__)

METR_ROOT = Path("metr")
MetrReport = Literal["time-horizon-1-0", "time-horizon-1-1"]
# note (Haoling Pu): a model that ran fewer than this share of the tasks is left out, so the tasks every kept model ran stay many. Missing tasks are mostly long ones, so each report's share is set to drop no recent model:
# 1.0 drops davinci-002 and GPT-2 (157 tasks left, mostly long HCAST / RE-Bench tasks lost); 1.1 drops GPT-4 0314, o1-preview and Claude 3.5 Sonnet (Old) (226 tasks left).
MIN_TASK_COVERAGE_BY_REPORT = {"time-horizon-1-0": 0.95, "time-horizon-1-1": 0.985}


def load_metr(raw_data_dir: Path, *, report: MetrReport) -> BenchmarkData:
    """Every model with at least its report's share of the tasks, on the tasks all of them ran; human baselines are left out.

    Every run counts with its score_binarized, as in METR's own fits: runs stopped by a server error or by the model's token / time budget
    (usageLimits, set per model by METR: about 2M tokens for older models, 8M-32M for recent ones) are failures unless the score says otherwise.
    METR weighs tasks by family when it fits time horizons; here every task counts once, like the other benchmarks,
    so the 66 SWAA tasks (seconds long, solved by almost every model) are 30-43 % of the tasks and add near-ties.
    """
    runs_path = raw_data_dir / METR_ROOT / report / "runs.jsonl"
    release_dates_path = raw_data_dir / METR_ROOT / "release_dates.yaml"
    if not runs_path.exists() or not release_dates_path.exists():
        raise FileNotFoundError(
            f"{runs_path} or {release_dates_path} not found; see the download steps in {__file__}"
        )
    else:
        pass
    metr_runs = pd.read_json(runs_path, lines=True)
    release_date_by_alias = yaml.safe_load(release_dates_path.read_text())["date"]
    metr_runs = metr_runs[metr_runs["alias"] != "human"]
    min_task_coverage = MIN_TASK_COVERAGE_BY_REPORT[report]

    num_tasks_by_alias = metr_runs.groupby("alias")["task_id"].nunique()
    num_tasks = metr_runs["task_id"].nunique()
    kept_aliases = num_tasks_by_alias.index[
        num_tasks_by_alias >= min_task_coverage * num_tasks
    ]
    metr_runs = metr_runs[metr_runs["alias"].isin(kept_aliases)]
    num_aliases_by_task = metr_runs.groupby("task_id")["alias"].nunique()
    task_ids = sorted(
        num_aliases_by_task.index[num_aliases_by_task == len(kept_aliases)]
    )
    metr_runs = metr_runs[metr_runs["task_id"].isin(task_ids)]
    logger.info(
        f"metr {report}: {len(kept_aliases)} of {len(num_tasks_by_alias)} models ran at least {min_task_coverage:.1%} of the {num_tasks} tasks; "
        f"left out {sorted(set(num_tasks_by_alias.index) - set(kept_aliases))}; {len(task_ids)} tasks were run by every kept model"
    )

    # note (Haoling Pu): run ids (sorted as text) only label the trials of a (model, task) cell; many runs have no start time.
    metr_runs = metr_runs.sort_values(["alias", "task_id", "run_id"])
    runs = pd.DataFrame(
        {
            "agent_id": metr_runs["alias"],
            "task_id": metr_runs["task_id"],
            "trial": metr_runs.groupby(["alias", "task_id"]).cumcount() + 1,
            "resolved": metr_runs["score_binarized"].astype(int),
            # note (Haoling Pu): a cost of 0 means METR has no cost for the run: every 1.1 run, and in 1.0 GPT-5.1-Codex-Max, DeepSeek-V3-0324, part of o3 / o4-mini / GPT-5 and some SWAA runs.
            "cost_usd": metr_runs["generation_cost"].where(
                metr_runs["generation_cost"] > 0
            ),
            "total_tokens": metr_runs["tokens_count"],
        }
    )
    agent_rows = []
    for alias, alias_runs in metr_runs.groupby("alias"):
        # note (Haoling Pu): METR has no submission date; a model is dated by its release, as on Terminal-Bench 4.1.
        release_date = pd.Timestamp(release_date_by_alias[alias])
        agent_rows.append(
            {
                "agent_id": alias,
                # note (Haoling Pu): SWAA runs use a plain one-shot solver, and 1.1 prefixes the task-setup step; the agent is the last part of the non-SWAA runs' scaffold.
                "scaffold": alias_runs.loc[
                    alias_runs["task_source"] != "SWAA", "scaffold"
                ]
                .str.split(",")
                .str[-1]
                .mode()
                .iloc[0],
                "model": alias_runs["model"].iloc[0],
                # note (Haoling Pu): the runs do not name the model's lab or whether its weights are open.
                "model_org": None,
                "open_weight": None,
                "submit_date": release_date,
                "model_release_date": release_date,
            }
        )
    tasks = (
        metr_runs.drop_duplicates("task_id")[["task_id", "task_family"]]
        .rename(columns={"task_family": "category"})
        .sort_values("task_id")
    )
    return BenchmarkData(
        name=f"metr_{report.removeprefix('time-horizon-').replace('-', '_')}",
        agents=pd.DataFrame(agent_rows),
        tasks=tasks.reset_index(drop=True),
        runs=runs.reset_index(drop=True),
    )
