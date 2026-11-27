import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from subset_eval.data import BenchmarkData, outcome_matrix
from subset_eval.loaders import load_benchmark, terminal_bench
from subset_eval.loaders.terminal_bench import SUBMISSIONS_DIR, load_terminal_bench

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HAS_RAW_DATA = (RAW_DATA_DIR / SUBMISSIONS_DIR).is_dir()


def write_submission(
    raw_data_dir: Path,
    name: str,
    trials: list[tuple[str, str, float | None, str]],
    *,
    metadata_file_name: str = "metadata.yaml",
    trial_model: str = "model-a",
) -> None:
    """trials: (task name, source, reward or None when the verifier never ran, started_at)."""
    submission_dir = raw_data_dir / SUBMISSIONS_DIR / name
    submission_dir.mkdir(parents=True)
    (submission_dir / metadata_file_name).write_text(
        "agent_display_name: Agent\n"
        "models:\n"
        "  - model_display_name: Model A\n"
        "    model_org_display_name: Org A\n"
    )
    # note (Haoling Pu): a job-level summary must not be read as a trial.
    (submission_dir / "job").mkdir()
    (submission_dir / "job" / "result.json").write_text(json.dumps({"stats": {}}))
    for index, (task_name, source, reward, started_at) in enumerate(trials):
        trial_dir = submission_dir / "job" / f"trial_{index}"
        trial_dir.mkdir()
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_name,
                    "source": source,
                    "started_at": started_at,
                    "config": {"agent": {"model_name": trial_model}},
                    "verifier_result": (
                        None if reward is None else {"rewards": {"reward": reward}}
                    ),
                    # note (Haoling Pu): a trial whose verifier never ran has no agent_result either.
                    "agent_result": (
                        None
                        if reward is None
                        else {
                            "cost_usd": 0.5,
                            "n_input_tokens": 10,
                            "n_output_tokens": 2,
                        }
                    ),
                }
            )
        )


def test_loader_rules_on_fake_submissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(terminal_bench, "NUM_TASKS", 2)
    write_submission(
        tmp_path,
        "complete",
        [
            ("terminal-bench/task-a", "terminal-bench", 1.0, "2026-02-03T05:00:00Z"),
            ("task-a", "terminal-bench", 0.0, "2026-02-01T23:00:00Z"),
            # note (Haoling Pu): no reward = the verifier never ran; it counts as unresolved.
            ("install-windows-3-11", "terminal-bench", None, "2026-02-02T00:00:00Z"),
            # note (Haoling Pu): a 2.1 trial is dropped even inside a 2.0 submission.
            (
                "task-a",
                "terminal-bench/terminal-bench-2-1",
                1.0,
                "2026-01-01T00:00:00Z",
            ),
        ],
    )
    write_submission(
        tmp_path,
        "missing_a_task",
        [("task-a", "terminal-bench", 1.0, "2026-02-01T00:00:00Z")],
    )
    write_submission(
        tmp_path,
        "metadata_yml",
        [
            ("task-a", "terminal-bench", 1.0, "2026-03-01T00:00:00Z"),
            ("install-windows-3.11", "terminal-bench", 1.0, "2026-03-01T00:00:00Z"),
        ],
        metadata_file_name="metadata.yml",
    )
    # note (Haoling Pu): like Crux, every trial here ran another model than the submission's, so none is kept.
    monkeypatch.setattr(
        terminal_bench, "TRIAL_MODEL_BY_SUBMISSION", {"other_model": "model-b"}
    )
    write_submission(
        tmp_path,
        "other_model",
        [
            ("task-a", "terminal-bench", 1.0, "2026-03-01T00:00:00Z"),
            ("install-windows-3.11", "terminal-bench", 1.0, "2026-03-01T00:00:00Z"),
        ],
    )
    write_submission(
        tmp_path,
        "only_2_1",
        [
            (
                "task-a",
                "terminal-bench/terminal-bench-2-1",
                1.0,
                "2026-05-01T00:00:00Z",
            ),
            (
                "install-windows-3.11",
                "terminal-bench/terminal-bench-2-1",
                1.0,
                "2026-05-01T00:00:00Z",
            ),
        ],
    )

    benchmark = load_terminal_bench(tmp_path)

    assert benchmark.agents["agent_id"].tolist() == ["complete", "metadata_yml"]
    assert benchmark.tasks["task_id"].tolist() == ["install-windows-3.11", "task-a"]
    runs = benchmark.runs[benchmark.runs["agent_id"] == "complete"].set_index(
        ["task_id", "trial"]
    )
    # note (Haoling Pu): trials are numbered by start time, so the earlier failure is trial 1.
    assert runs.loc[("task-a", 1), "resolved"] == 0
    assert runs.loc[("task-a", 2), "resolved"] == 1
    assert runs.loc[("install-windows-3.11", 1), "resolved"] == 0
    assert runs.loc[("task-a", 1), "total_tokens"] == 12
    assert np.isnan(runs.loc[("install-windows-3.11", 1), "total_tokens"])
    agent = benchmark.agents.iloc[0]
    assert agent["submit_date"] == pd.Timestamp("2026-02-01")
    assert agent["missing_reward_share"] == pytest.approx(1 / 3)
    assert agent["model"] == "Model A"


@pytest.fixture(scope="module")
def terminal_bench_data() -> BenchmarkData:
    return load_benchmark("terminal_bench", raw_data_dir=RAW_DATA_DIR, seed=0)


@pytest.mark.skipif(
    not HAS_RAW_DATA, reason="Terminal-Bench not downloaded to data/raw/"
)
def test_real_data_shape_and_dates(terminal_bench_data: BenchmarkData) -> None:
    outcomes = outcome_matrix(terminal_bench_data)
    assert outcomes.shape[1] == 89
    assert outcomes.shape[0] >= 60
    assert "0error-Ledger__Claude-Opus-4.7" not in outcomes.index
    submit_date = terminal_bench_data.agents["submit_date"]
    assert submit_date.min() >= pd.Timestamp("2025-11-01")
    assert submit_date.max() <= pd.Timestamp("2026-06-01")
