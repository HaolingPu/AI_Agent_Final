import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subset_eval.data import BenchmarkData, outcome_matrix
from subset_eval.loaders import load_benchmark, terminal_bench_4_1
from subset_eval.loaders.terminal_bench_4_1 import DATA_DIR, load_terminal_bench_4_1

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HAS_RAW_DATA = (RAW_DATA_DIR / DATA_DIR / "rows.json").is_file()
ERROR_BY_REWARD = {"": "AgentTimeoutError", "regrade": "RegradeError"}


def write_rows(
    raw_data_dir: Path,
    trials_by_row: dict[str, list[tuple[str, str, str, str]]],
    *,
    leaderboard_successes: dict[str, int],
) -> None:
    """trials_by_row: row id -> (task, reward ("" when none, "regrade" for a failed regrade), started, cost) per linked trial."""
    data_dir = raw_data_dir / DATA_DIR
    (data_dir / "trials").mkdir(parents=True)
    rows = []
    for row_id, trials in trials_by_row.items():
        rows.append(
            {
                "id": row_id,
                "metadata": {
                    "agent_display": {"url": "https://a.example", "label": "Agent"},
                    "model_display": {
                        "url": "https://m.example",
                        "label": f"Model {row_id}",
                    },
                    "model_org": {"url": "https://o.example", "label": "Org"},
                    "agent_org": {"url": "https://o.example", "label": "Org"},
                    "date": "2026-09-22",
                    "display_date": "Sep 22, 2026",
                    "reasoning_effort": "max",
                },
                "metrics": {"successes": leaderboard_successes[row_id]},
            }
        )
        (data_dir / "trials" / f"{row_id}.json").write_text(
            json.dumps(
                {
                    "row_id": row_id,
                    "num_linked_trials": len(trials),
                    "trials": [
                        {
                            "ID": f"{row_id}-{index}",
                            "Task": f"terminal-bench/{task}",
                            "Reward": "" if reward == "regrade" else reward,
                            "Error": ERROR_BY_REWARD.get(reward, ""),
                            "Started": started,
                            "Cost": cost,
                        }
                        for index, (task, reward, started, cost) in enumerate(trials)
                    ],
                }
            )
        )
    (data_dir / "rows.json").write_text(json.dumps(rows))


def test_loader_rules_on_fake_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(terminal_bench_4_1, "NUM_TASKS", 2)
    monkeypatch.setattr(terminal_bench_4_1, "NUM_TRIALS_PER_TASK", 2)
    write_rows(
        tmp_path,
        {
            "r1": [
                ("task-a", "1.000", "2026-11-03 23:50", "$3.97"),
                # note (Haoling Pu): the earlier start is trial 1.
                ("task-a", "0.000", "2026-11-03 23:49", "$1.00"),
                # note (Haoling Pu): no reward counts as unresolved, and a missing cost stays NaN.
                ("task-b", "", "2026-11-03 23:49", ""),
                ("task-b", "1.000", "2026-11-03 23:51", "$120.50"),
            ],
        },
        leaderboard_successes={"r1": 3},
    )

    benchmark = load_terminal_bench_4_1(
        tmp_path, drop_regrade_errors=False, highest_effort_only=False
    )

    assert benchmark.agents["agent_id"].tolist() == ["Agent__Model r1__max"]
    agent = benchmark.agents.iloc[0]
    assert agent["submit_date"] == pd.Timestamp("2026-09-22")
    assert agent["missing_reward_share"] == pytest.approx(0.25)
    assert agent["leaderboard_successes_minus_linked"] == 1
    runs = benchmark.runs.set_index(["task_id", "trial"])
    assert runs.loc[("task-a", 1), "resolved"] == 0
    assert runs.loc[("task-a", 2), "resolved"] == 1
    assert runs.loc[("task-b", 1), "resolved"] == 0
    assert np.isnan(runs.loc[("task-b", 1), "cost_usd"])
    assert runs.loc[("task-b", 2), "cost_usd"] == pytest.approx(120.5)


@pytest.mark.parametrize("drop_regrade_errors", [False, True])
def test_loader_rejects_a_row_missing_a_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drop_regrade_errors: bool
) -> None:
    monkeypatch.setattr(terminal_bench_4_1, "NUM_TASKS", 2)
    monkeypatch.setattr(terminal_bench_4_1, "NUM_TRIALS_PER_TASK", 2)
    write_rows(
        tmp_path,
        {
            "r1": [
                ("task-a", "1.000", "2026-11-03 23:50", "$1"),
                ("task-a", "1.000", "2026-11-03 23:51", "$1"),
                ("task-b", "1.000", "2026-11-03 23:52", "$1"),
            ],
        },
        leaderboard_successes={"r1": 3},
    )

    with pytest.raises(ValueError, match="trials on each"):
        load_terminal_bench_4_1(
            tmp_path, drop_regrade_errors=drop_regrade_errors, highest_effort_only=False
        )


def test_graded_only_drops_failed_regrades_and_tasks_left_without_a_graded_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(terminal_bench_4_1, "NUM_TASKS", 2)
    monkeypatch.setattr(terminal_bench_4_1, "NUM_TRIALS_PER_TASK", 2)
    write_rows(
        tmp_path,
        {
            "r1": [
                ("task-a", "1.000", "2026-11-03 23:50", "$1"),
                ("task-a", "regrade", "2026-11-03 23:51", "$1"),
                ("task-b", "0.000", "2026-11-03 23:52", "$1"),
                ("task-b", "1.000", "2026-11-03 23:53", "$1"),
            ],
            "r2": [
                ("task-a", "1.000", "2026-11-03 23:50", "$1"),
                ("task-a", "1.000", "2026-11-03 23:51", "$1"),
                ("task-b", "regrade", "2026-11-03 23:52", "$1"),
                ("task-b", "regrade", "2026-11-03 23:53", "$1"),
            ],
        },
        leaderboard_successes={"r1": 2, "r2": 2},
    )

    as_zero = outcome_matrix(
        load_terminal_bench_4_1(
            tmp_path, drop_regrade_errors=False, highest_effort_only=False
        )
    )
    graded_only = outcome_matrix(
        load_terminal_bench_4_1(
            tmp_path, drop_regrade_errors=True, highest_effort_only=False
        )
    )

    assert as_zero.loc["Agent__Model r1__max", "task-a"] == 0.5
    assert as_zero.columns.tolist() == ["task-a", "task-b"]
    assert graded_only.loc["Agent__Model r1__max", "task-a"] == 1.0
    assert graded_only.columns.tolist() == ["task-a"]


def test_highest_effort_merges_scaffold_spellings_and_keeps_the_top_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(terminal_bench_4_1, "NUM_TASKS", 1)
    monkeypatch.setattr(terminal_bench_4_1, "NUM_TRIALS_PER_TASK", 1)
    write_rows(
        tmp_path,
        {
            "r1": [("task-a", "1.000", "2026-11-03 23:50", "$1")],
            "r2": [("task-a", "0.000", "2026-11-03 23:50", "$1")],
        },
        leaderboard_successes={"r1": 1, "r2": 0},
    )
    # note (Haoling Pu): one model under both spellings of mini-SWE-agent, at two efforts.
    rows_path = tmp_path / DATA_DIR / "rows.json"
    rows = json.loads(rows_path.read_text())
    for row, scaffold, effort in zip(
        rows, ("mini-SWE-agent", "mini-swe-agent"), ("high", "max")
    ):
        row["metadata"]["agent_display"]["label"] = scaffold
        row["metadata"]["model_display"]["label"] = "Model M"
        row["metadata"]["reasoning_effort"] = effort
    rows_path.write_text(json.dumps(rows))

    agents = load_terminal_bench_4_1(
        tmp_path, drop_regrade_errors=False, highest_effort_only=True
    ).agents

    assert agents["agent_id"].tolist() == ["mini-swe-agent__Model M__max"]


@pytest.fixture(scope="module")
def terminal_bench_4_1_data() -> BenchmarkData:
    return load_benchmark("terminal_bench_4_1", raw_data_dir=RAW_DATA_DIR, seed=0)


@pytest.mark.skipif(
    not HAS_RAW_DATA, reason="Terminal-Bench 4.1 not downloaded to data/raw/"
)
def test_real_data_shape_and_scores(terminal_bench_4_1_data: BenchmarkData) -> None:
    outcomes = outcome_matrix(terminal_bench_4_1_data)
    assert outcomes.shape == (67, 66)
    # note (Haoling Pu): the top row solves 235 of 330 trials (71.2 %), as on the leaderboard.
    assert outcomes.mean(axis=1).max() == pytest.approx(235 / 330)
    agents = terminal_bench_4_1_data.agents
    assert agents["leaderboard_successes_minus_linked"].abs().max() <= 9
    assert agents["submit_date"].min() >= pd.Timestamp("2026-02-01")


@pytest.mark.skipif(
    not HAS_RAW_DATA, reason="Terminal-Bench 4.1 not downloaded to data/raw/"
)
def test_real_data_graded_only_keeps_every_row() -> None:
    outcomes = outcome_matrix(
        load_benchmark(
            "terminal_bench_4_1_graded_only", raw_data_dir=RAW_DATA_DIR, seed=0
        )
    )
    # note (Haoling Pu): 31 (row, task) pairs have all 5 trials failed at regrade, on 13 tasks.
    assert outcomes.shape == (67, 53)


@pytest.mark.skipif(
    not HAS_RAW_DATA, reason="Terminal-Bench 4.1 not downloaded to data/raw/"
)
def test_real_data_highest_effort_keeps_one_row_per_scaffold_and_model() -> None:
    agents = load_benchmark(
        "terminal_bench_4_1_highest_effort", raw_data_dir=RAW_DATA_DIR, seed=0
    ).agents
    assert len(agents) == 54
    assert not agents.duplicated(["scaffold", "model"]).any()
    # note (Haoling Pu): four models were run at several efforts; GLM-5.3-Flash at "none" and "max".
    effort_by_model = agents.set_index("model")["reasoning_effort"]
    for model in ("Fable 5.1", "GPT-6 Astra", "Opus 5", "GLM-5.3-Flash"):
        assert effort_by_model[model] == "max"
