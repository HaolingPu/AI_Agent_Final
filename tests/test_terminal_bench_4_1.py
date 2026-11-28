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


def write_rows(
    raw_data_dir: Path,
    trials_by_row: dict[str, list[tuple[str, str, str, str]]],
    *,
    leaderboard_successes: dict[str, int],
) -> None:
    """trials_by_row: row id -> (task, reward or "" when none, started, cost) per linked trial."""
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
                            "Reward": reward,
                            "Error": "" if reward else "AgentTimeoutError",
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

    benchmark = load_terminal_bench_4_1(tmp_path)

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


def test_loader_rejects_a_row_missing_a_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        load_terminal_bench_4_1(tmp_path)


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
