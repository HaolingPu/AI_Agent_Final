import json
from pathlib import Path

import pandas as pd
import pytest

from subset_eval.data import outcome_matrix
from subset_eval.loaders import load_benchmark
from subset_eval.loaders.metr import METR_ROOT, load_metr

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def write_runs(raw_data_dir: Path, metr_runs: list[dict]) -> None:
    """A fake time-horizon-1-1 runs.jsonl and its release dates."""
    report_dir = raw_data_dir / METR_ROOT / "time-horizon-1-1"
    report_dir.mkdir(parents=True)
    (report_dir / "runs.jsonl").write_text(
        "\n".join(json.dumps(metr_run) for metr_run in metr_runs)
    )
    (raw_data_dir / METR_ROOT / "release_dates.yaml").write_text(
        "date:\n  Model A: 2024-01-01\n  Model B: 2025-01-01\n  Model C: 2025-06-01\n  Model D: 2025-07-01\n"
    )


def metr_run(
    alias: str,
    task_id: str,
    run_id: str,
    *,
    score: int,
    fatal_error_from: str | None = None,
    generation_cost: float = 1.0,
) -> dict:
    return {
        "alias": alias,
        "model": alias.lower(),
        "task_id": task_id,
        "task_family": task_id.split("/")[0],
        "run_id": run_id,
        "score_binarized": score,
        "fatal_error_from": fatal_error_from,
        "generation_cost": generation_cost,
        "tokens_count": 100.0,
        "scaffold": "setup,react",
        "task_source": "HCAST",
    }


def test_loader_rules_on_fake_runs(tmp_path: Path) -> None:
    task_ids = [f"family/task-{index}" for index in range(20)]
    # note (Haoling Pu): Model B misses the last task (19 of 20 = 95 % < 98.5 %), so it is left out; Model A's other runs keep all 20 tasks.
    metr_runs = [
        metr_run(alias, task_id, f"{alias}-{task_id}", score=1)
        for alias in ("Model A", "Model C")
        for task_id in task_ids
    ]
    metr_runs += [
        metr_run("Model B", task_id, f"B-{task_id}", score=1)
        for task_id in task_ids[:19]
    ]
    # note (Haoling Pu): Model D ran 18 of 20 tasks, so it is left out too.
    metr_runs += [
        metr_run("Model D", task_id, f"D-{task_id}", score=1)
        for task_id in task_ids[:18]
    ]
    # note (Haoling Pu): every run keeps its own score, errors included, as in METR's fits; a cost of 0 is unknown.
    metr_runs += [
        metr_run(
            "Model A",
            task_ids[0],
            "z-server",
            score=1,
            fatal_error_from="serverOrTask",
        ),
        metr_run(
            "Model A",
            task_ids[0],
            "z-limit",
            score=0,
            fatal_error_from="usageLimits",
            generation_cost=0.0,
        ),
        metr_run(
            "Model A",
            task_ids[1],
            "z-limit",
            score=1,
            fatal_error_from="usageLimits",
        ),
        metr_run("human", task_ids[0], "human-0", score=1),
    ]
    write_runs(tmp_path, metr_runs)

    benchmark = load_metr(tmp_path, report="time-horizon-1-1")

    assert benchmark.name == "metr_1_1"
    assert benchmark.agents["agent_id"].tolist() == ["Model A", "Model C"]
    assert benchmark.agents["submit_date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2025-06-01"),
    ]
    assert len(benchmark.tasks) == 20
    first_cell = benchmark.runs[
        (benchmark.runs["agent_id"] == "Model A")
        & (benchmark.runs["task_id"] == task_ids[0])
    ]
    assert first_cell["resolved"].tolist() == [1, 0, 1]
    assert first_cell["cost_usd"].isna().tolist() == [False, True, False]
    assert outcome_matrix(benchmark).loc["Model A", task_ids[0]] == pytest.approx(2 / 3)
    assert outcome_matrix(benchmark).loc["Model A", task_ids[1]] == 1.0
    assert benchmark.agents["scaffold"].tolist() == ["react", "react"]


@pytest.mark.skipif(
    not all(
        (RAW_DATA_DIR / METR_ROOT / file_name).exists()
        for file_name in (
            "release_dates.yaml",
            "time-horizon-1-0/runs.jsonl",
            "time-horizon-1-1/runs.jsonl",
        )
    ),
    reason="METR runs not downloaded to data/raw/",
)
@pytest.mark.parametrize(
    ("benchmark_name", "shape", "left_out_alias"),
    [
        ("metr_1_0", (31, 157), "GPT-2"),
        ("metr_1_1", (17, 226), "Claude 3.5 Sonnet (Old) (Inspect)"),
    ],
)
def test_real_reports_keep_models_with_almost_every_task(
    benchmark_name: str, shape: tuple[int, int], left_out_alias: str
) -> None:
    benchmark = load_benchmark(benchmark_name, raw_data_dir=RAW_DATA_DIR, seed=0)
    assert outcome_matrix(benchmark).shape == shape
    assert left_out_alias not in set(benchmark.agents["agent_id"])
    assert "human" not in set(benchmark.agents["agent_id"])
