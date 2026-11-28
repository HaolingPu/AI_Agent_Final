import json
from pathlib import Path

import pytest

from subset_eval.data import outcome_matrix
from subset_eval.loaders import load_benchmark
from subset_eval.loaders import multi_swe_bench

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_unlisted_issues_count_as_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(multi_swe_bench, "LANGUAGES", ("go", "rust"))
    monkeypatch.setattr(multi_swe_bench, "SCAFFOLDS", ("MSWE-agent",))
    monkeypatch.setattr(multi_swe_bench, "NUM_TASKS", 3)
    monkeypatch.setattr(
        multi_swe_bench,
        "MODEL_RELEASE_DATES",
        {"Model-A": "2024-10-01", "Model-B": "2025-01-01"},
    )
    # note (Haoling Pu): Model-B names only its resolved go issue, like a short results.json; the other go issue still exists through Model-A.
    results_by_agent_language = {
        ("Model-A", "go"): {"resolved": ["o__r-1"], "unresolved_ids": ["o__r-2"]},
        ("Model-A", "rust"): {"resolved": [], "unresolved_ids": ["o__r-1"]},
        ("Model-B", "go"): {"resolved": ["o__r-2"], "total_instances": 2},
        ("Model-B", "rust"): {"resolved": ["o__r-1"]},
    }
    for (model, language), results in results_by_agent_language.items():
        results_dir = (
            tmp_path
            / multi_swe_bench.SUBMISSIONS_ROOT
            / language
            / "verified"
            / f"20250329_MSWE-agent_{model}"
            / "results"
        )
        results_dir.mkdir(parents=True)
        (results_dir / "results.json").write_text(json.dumps(results))

    benchmark = multi_swe_bench.load_multi_swe_bench(tmp_path)

    assert outcome_matrix(benchmark).to_dict(orient="index") == {
        "MSWE-agent_Model-A": {"go/o__r-1": 1.0, "go/o__r-2": 0.0, "rust/o__r-1": 0.0},
        "MSWE-agent_Model-B": {"go/o__r-1": 0.0, "go/o__r-2": 1.0, "rust/o__r-1": 1.0},
    }
    assert benchmark.tasks.set_index("task_id")["category"]["rust/o__r-1"] == "rust"


@pytest.mark.skipif(
    not (RAW_DATA_DIR / multi_swe_bench.SUBMISSIONS_ROOT).is_dir(),
    reason="Multi-SWE-bench results not downloaded to data/raw/",
)
def test_real_grid_matches_each_submissions_resolved_count() -> None:
    benchmark = load_benchmark("multi_swe_bench", raw_data_dir=RAW_DATA_DIR, seed=0)
    outcomes = outcome_matrix(benchmark)
    assert outcomes.shape == (36, 1632)
    # note (Haoling Pu): issue counts per language from the upstream index.json all_ids.
    assert benchmark.tasks["category"].value_counts().to_dict() == {
        "go": 428,
        "javascript": 356,
        "rust": 239,
        "typescript": 224,
        "c++": 129,
        "c": 128,
        "java": 128,
    }
    # note (Haoling Pu): every (agent, language) solve count equals that submission's own resolved_instances.
    submissions_root = RAW_DATA_DIR / multi_swe_bench.SUBMISSIONS_ROOT
    for agent_id in outcomes.index:
        for language in multi_swe_bench.LANGUAGES:
            (submission_dir,) = (submissions_root / language / "verified").glob(
                f"2025????_{agent_id}"
            )
            results = json.loads(
                (submission_dir / "results" / "results.json").read_text()
            )
            language_issues = outcomes.columns[
                outcomes.columns.str.startswith(f"{language}/")
            ]
            assert (
                outcomes.loc[agent_id, language_issues].sum()
                == results["resolved_instances"]
            )
