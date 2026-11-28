import json
from pathlib import Path

import pytest

from subset_eval.loaders import swe_bench_verified
from subset_eval.loaders.swe_bench_verified import (
    SUBMISSIONS_DIR,
    load_swe_bench_verified,
)

TASK_IDS = ["django__django-1", "django__django-2", "sympy__sympy-3", "sympy__sympy-4"]


def write_submission(
    raw_data_dir: Path,
    name: str,
    *,
    reported_solve_rate: float | None,
    resolved_task_ids: list[str] | None = None,
    resolved_by_detailed_task_id: dict[str, bool] | None = None,
    metadata_file_name: str = "metadata.yaml",
) -> None:
    """A submission folder with results.json (resolved ids) and/or per_instance_details.json (every task it ran)."""
    submission_dir = raw_data_dir / SUBMISSIONS_DIR / name
    submission_dir.mkdir(parents=True)
    (submission_dir / metadata_file_name).write_text(
        # note (Haoling Pu): most real submissions report no solve rate at all.
        (
            "info: {}\n"
            if reported_solve_rate is None
            else f"info:\n  resolved: {reported_solve_rate}\n"
        )
        + "tags:\n"
        "  agent: Agent\n"
        "  model: model-a\n"
        "  model_display: Model A\n"
        "  model_org: Org\n"
        "  os_model: false\n"
    )
    if resolved_task_ids is None:
        pass
    else:
        (submission_dir / "results").mkdir()
        (submission_dir / "results" / "results.json").write_text(
            json.dumps({"resolved": resolved_task_ids})
        )
    if resolved_by_detailed_task_id is None:
        pass
    else:
        (submission_dir / "per_instance_details.json").write_text(
            json.dumps(
                {
                    task_id: {"resolved": resolved, "cost": 0.5}
                    for task_id, resolved in resolved_by_detailed_task_id.items()
                }
            )
        )


def test_loader_rules_on_fake_submissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(swe_bench_verified, "NUM_TASKS", len(TASK_IDS))
    # note (Haoling Pu): names every task, so the task set is known; 2 of 4 resolved = 50 %.
    write_submission(
        tmp_path,
        "20250101_details_only",
        reported_solve_rate=50.0,
        resolved_by_detailed_task_id=dict(zip(TASK_IDS, [True, True, False, False])),
    )
    # note (Haoling Pu): its details miss a task, so results.json wins; the other tasks are unresolved.
    write_submission(
        tmp_path,
        "20250201_both_files",
        reported_solve_rate=25.0,
        resolved_task_ids=["sympy__sympy-3"],
        resolved_by_detailed_task_id={"django__django-1": False},
        metadata_file_name="metadata.yml",
    )
    # note (Haoling Pu): per-task results say 25 %, its metadata says 75 %, so it is left out.
    write_submission(
        tmp_path,
        "20250301_contradicts_its_score",
        reported_solve_rate=75.0,
        resolved_task_ids=["django__django-1"],
    )
    # note (Haoling Pu): no reported solve rate, so the score check is skipped and it is kept.
    write_submission(
        tmp_path,
        "20250215_no_reported_rate",
        reported_solve_rate=None,
        resolved_task_ids=["django__django-2"],
    )
    # note (Haoling Pu): incomplete details and no results.json, so it is left out.
    write_submission(
        tmp_path,
        "20250401_incomplete_details",
        reported_solve_rate=0.0,
        resolved_by_detailed_task_id={"django__django-1": False},
    )

    benchmark = load_swe_bench_verified(tmp_path)

    assert benchmark.agents["agent_id"].tolist() == [
        "20250101_details_only",
        "20250201_both_files",
        "20250215_no_reported_rate",
    ]
    assert not benchmark.agents["open_weight"].any()
    resolved = benchmark.runs.set_index(["agent_id", "task_id"])["resolved"]
    assert resolved["20250201_both_files"].to_dict() == {
        "django__django-1": 0,
        "django__django-2": 0,
        "sympy__sympy-3": 1,
        "sympy__sympy-4": 0,
    }
    assert (
        benchmark.tasks.set_index("task_id")["category"]["sympy__sympy-3"]
        == "sympy/sympy"
    )


def test_a_resolved_id_outside_the_task_set_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(swe_bench_verified, "NUM_TASKS", len(TASK_IDS))
    write_submission(
        tmp_path,
        "20250101_details_only",
        reported_solve_rate=0.0,
        resolved_by_detailed_task_id=dict.fromkeys(TASK_IDS, False),
    )
    write_submission(
        tmp_path,
        "20250201_unknown_task",
        reported_solve_rate=25.0,
        resolved_task_ids=["astropy__astropy-9"],
    )

    with pytest.raises(ValueError, match="expected 4 task ids, found 5"):
        load_swe_bench_verified(tmp_path)
