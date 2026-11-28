"""Multi-SWE-bench loader: 3 scaffolds x 12 models, one run each on 1,632 issues in 7 languages (Python left out: it is SWE-bench Verified).

The 36 agents are the paper's own grid (MagentLess, MopenHands, MSWE-agent with each model), the only submissions on every language.
Every submission is dated 2025-03 or 2025-04, so an agent is dated by its model's release (2024-09 to 2025-04), as on METR and Terminal-Bench 4.1.
Pooling the languages weights each by its issue count (go 26 %, javascript 22 %); the leaderboard ranks within a language only.
Caveats: 68 % of issues are solved by no grid agent (the best solves 14 %), and many unresolved outcomes are empty or error patches
(up to 61 % of an agent's issues), counted as 0 as on the leaderboard, so part of the ranking reflects scaffold failures.
Download each submission's results/results.json (about 6 MB):

    git clone --filter=blob:none --no-checkout --depth 1 https://github.com/multi-swe-bench/experiments.git data/raw/multi_swe_bench
    cd data/raw/multi_swe_bench && git sparse-checkout set --no-cone '/evaluation/*/verified/*/results/results.json' && git checkout
"""

import json
from pathlib import Path

import pandas as pd

from subset_eval.data import BenchmarkData

SUBMISSIONS_ROOT = Path("multi_swe_bench/evaluation")
LANGUAGES = ("c", "c++", "go", "java", "javascript", "rust", "typescript")
SCAFFOLDS = ("MagentLess", "MopenHands", "MSWE-agent")
NUM_TASKS = 1632
# note (Haoling Pu): from each lab's release announcement (o1 and Claude 3.5 Sonnet (Oct) as in METR's release_dates.yaml).
MODEL_RELEASE_DATES = {
    "Qwen2.5-72B-Instruct": "2024-09-19",
    "Claude-3.5-Sonnet(Oct)": "2024-10-22",
    "GPT-4o-1120": "2024-11-20",
    "OpenAI-o1": "2024-12-05",
    "DeepSeek-V3": "2024-12-26",
    "DeepSeek-R1": "2025-01-20",
    "Doubao-1.5-pro": "2025-01-22",
    "OpenAI-o3-mini-high": "2025-01-31",
    "Claude-3.7-Sonnet": "2025-02-24",
    "Gemini-2.5-Pro": "2025-03-25",
    "Llama-4-Maverick": "2025-04-05",
    "Doubao-1.5-thinking": "2025-04-17",
}


def load_multi_swe_bench(raw_data_dir: Path) -> BenchmarkData:
    """The 36 grid agents on every non-Python issue; an issue a submission's lists do not name counts as unresolved, as on the leaderboard (every real grid file names all its issues)."""
    submissions_root = raw_data_dir / SUBMISSIONS_ROOT
    if not submissions_root.is_dir():
        raise FileNotFoundError(
            f"{submissions_root} not found; see the download steps in {__file__}"
        )
    else:
        pass
    agent_rows = []
    resolved_task_ids_by_agent_id: dict[str, set[str]] = {}
    task_ids: set[str] = set()
    for scaffold in SCAFFOLDS:
        for model, release_date in MODEL_RELEASE_DATES.items():
            agent_id = f"{scaffold}_{model}"
            resolved_task_ids_by_agent_id[agent_id] = set()
            for language in LANGUAGES:
                # note (Haoling Pu): each grid agent has exactly one folder per language, dated 20250329 or 20250426.
                (submission_dir,) = (submissions_root / language / "verified").glob(
                    f"2025????_{agent_id}"
                )
                results = json.loads(
                    (submission_dir / "results" / "results.json").read_text()
                )
                # note (Haoling Pu): results.json also holds counts (ints); every list names issues. The language prefix makes the task's category readable.
                task_ids |= {
                    f"{language}/{task_id}"
                    for task_ids_in_list in results.values()
                    if isinstance(task_ids_in_list, list)
                    for task_id in task_ids_in_list
                }
                resolved_task_ids_by_agent_id[agent_id] |= {
                    f"{language}/{task_id}" for task_id in results["resolved"]
                }
            agent_rows.append(
                {
                    "agent_id": agent_id,
                    "scaffold": scaffold,
                    "model": model,
                    "model_org": None,
                    "open_weight": None,
                    "submit_date": pd.Timestamp(release_date),
                    "model_release_date": pd.Timestamp(release_date),
                }
            )
    # note (Haoling Pu): each language's issue union equals its upstream index.json all_ids; a short count means a truncated results.json.
    if len(task_ids) != NUM_TASKS:
        raise ValueError(f"expected {NUM_TASKS} issues, found {len(task_ids)}")
    else:
        pass
    sorted_task_ids = sorted(task_ids)
    runs = pd.DataFrame(
        [
            {
                "agent_id": agent_id,
                "task_id": task_id,
                "trial": 1,
                "resolved": int(task_id in resolved_task_ids),
                "cost_usd": float("nan"),
                "total_tokens": float("nan"),
            }
            for agent_id, resolved_task_ids in resolved_task_ids_by_agent_id.items()
            for task_id in sorted_task_ids
        ]
    )
    tasks = pd.DataFrame(
        {
            "task_id": sorted_task_ids,
            "category": [task_id.split("/")[0] for task_id in sorted_task_ids],
        }
    )
    return BenchmarkData(
        name="multi_swe_bench",
        agents=pd.DataFrame(agent_rows),
        tasks=tasks,
        runs=runs,
    )
