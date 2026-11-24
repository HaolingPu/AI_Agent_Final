from pathlib import Path

import pytest

from subset_eval.data import BenchmarkData, outcome_matrix
from subset_eval.loaders import load_benchmark
from subset_eval.loaders.swe_bench_verified import SUBMISSIONS_DIR

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# note (Wilson Zheng): the raw files are gitignored, so these tests run only after the download in swe_bench_verified.py.
pytestmark = pytest.mark.skipif(
    not (RAW_DATA_DIR / SUBMISSIONS_DIR).is_dir(),
    reason="SWE-bench experiments not downloaded to data/raw/",
)


@pytest.fixture(scope="module")
def swe_bench() -> BenchmarkData:
    return load_benchmark("swe_bench_verified", raw_data_dir=RAW_DATA_DIR, seed=0)


def test_keeps_every_submission_with_complete_and_consistent_results(
    swe_bench: BenchmarkData,
) -> None:
    # note (Wilson Zheng): 182 submissions, minus 7 without per-task results and 2 whose per-task results contradict their reported score.
    assert outcome_matrix(swe_bench).shape == (173, 500)
    assert (
        not swe_bench.agents["agent_id"]
        .isin(
            [
                "20250720_mini-v0.0.0-claude-3-7-sonnet-20250219",
                "20260226_mini-v2.0.0_gemini-3-pro-high",
                "20260219_mini-v2.0.0_gpt-5-2-codex",
            ]
        )
        .any()
    )


def test_solve_rates_match_the_public_leaderboard(swe_bench: BenchmarkData) -> None:
    solve_rate = outcome_matrix(swe_bench).mean(axis=1)
    assert solve_rate["20231010_rag_claude2"] == pytest.approx(0.044)
    assert solve_rate["20241022_tools_claude-3-5-sonnet-updated"] == pytest.approx(0.49)
    # note (Wilson Zheng): this submission has both result files; its per_instance_details.json misses 59 tasks, so results.json must win.
    assert solve_rate["20260901_mini-v2.4.2_gemini-3-5-flash"] == pytest.approx(0.718)


def test_tasks_are_grouped_by_repo(swe_bench: BenchmarkData) -> None:
    category_counts = swe_bench.tasks["category"].value_counts()
    assert category_counts["django/django"] == 231
    assert category_counts["sympy/sympy"] == 75
    assert category_counts.sum() == 500


def test_agent_metadata(swe_bench: BenchmarkData) -> None:
    agents = swe_bench.agents.set_index("agent_id")
    mini_swe_agent = agents.loc["20250726_mini-v1.0.0_claude-sonnet-4-20250514"]
    assert mini_swe_agent["scaffold"] == "mini-SWE-agent"
    assert mini_swe_agent["model_org"] == "Anthropic"
    assert str(mini_swe_agent["submit_date"].date()) == "2025-07-26"
    assert agents["model_release_date"].notna().sum() == 39
    assert agents["submit_date"].min().year == 2023


def test_cost_is_known_only_for_mini_swe_agent_runs(swe_bench: BenchmarkData) -> None:
    has_cost = swe_bench.runs.groupby("agent_id")["cost_usd"].apply(
        lambda cost: cost.notna().any()
    )
    scaffold = swe_bench.agents.set_index("agent_id")["scaffold"]
    assert (scaffold[has_cost[has_cost].index] == "mini-SWE-agent").all()
    assert swe_bench.runs["total_tokens"].isna().all()


def test_mini_swe_agent_variant_dates_agents_by_model_release() -> None:
    mini_swe_agent = load_benchmark(
        "swe_bench_verified_mini_swe_agent", raw_data_dir=RAW_DATA_DIR, seed=0
    )
    assert outcome_matrix(mini_swe_agent).shape == (39, 500)
    assert (mini_swe_agent.agents["scaffold"] == "mini-SWE-agent").all()
    # note (Wilson Zheng): Claude Sonnet 4.5 was re-run on 2026-02-17 but released on 2025-09-29.
    agents = mini_swe_agent.agents.set_index("agent_id")
    assert str(
        agents.loc["20260217_mini-v2.0.0_claude-4-5-sonnet-high", "submit_date"].date()
    ) == "2025-09-29"
