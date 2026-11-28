"""One entry point for every benchmark: each loader turns raw files into BenchmarkData."""

from pathlib import Path
from typing import Literal

from subset_eval.data import BenchmarkData
from subset_eval.loaders.metr import load_metr
from subset_eval.loaders.swe_bench_verified import (
    load_swe_bench,
    load_swe_bench_verified_mini_swe_agent,
    load_swe_bench_verified_one_per_model,
)
from subset_eval.loaders.terminal_bench import load_terminal_bench
from subset_eval.loaders.terminal_bench_4_1 import load_terminal_bench_4_1
from subset_eval.toy import make_toy_benchmark

# note (Haoling Pu): a new loader adds its name here and one elif below; the experiments then accept it as --benchmark.
BenchmarkName = Literal[
    "toy",
    "terminal_bench",
    "terminal_bench_4_1",
    "terminal_bench_4_1_graded_only",
    "terminal_bench_4_1_highest_effort",
    "swe_bench_verified",
    "swe_bench_lite",
    "swe_bench_verified_mini_swe_agent",
    "swe_bench_verified_one_per_model",
    "metr_1_0",
    "metr_1_1",
]


def load_benchmark(
    benchmark_name: BenchmarkName, *, raw_data_dir: Path, seed: int
) -> BenchmarkData:
    """raw_data_dir holds the downloaded raw files (data/raw/ by default, gitignored); the toy benchmark ignores it."""
    if benchmark_name == "toy":
        return make_toy_benchmark(seed=seed)
    elif benchmark_name == "terminal_bench":
        return load_terminal_bench(raw_data_dir)
    elif benchmark_name == "terminal_bench_4_1":
        return load_terminal_bench_4_1(
            raw_data_dir, drop_regrade_errors=False, highest_effort_only=False
        )
    elif benchmark_name == "terminal_bench_4_1_graded_only":
        return load_terminal_bench_4_1(
            raw_data_dir, drop_regrade_errors=True, highest_effort_only=False
        )
    elif benchmark_name == "terminal_bench_4_1_highest_effort":
        return load_terminal_bench_4_1(
            raw_data_dir, drop_regrade_errors=False, highest_effort_only=True
        )
    elif benchmark_name == "swe_bench_verified":
        return load_swe_bench(raw_data_dir, split="verified")
    elif benchmark_name == "swe_bench_lite":
        return load_swe_bench(raw_data_dir, split="lite")
    elif benchmark_name == "swe_bench_verified_mini_swe_agent":
        return load_swe_bench_verified_mini_swe_agent(raw_data_dir)
    elif benchmark_name == "swe_bench_verified_one_per_model":
        return load_swe_bench_verified_one_per_model(raw_data_dir)
    elif benchmark_name == "metr_1_0":
        return load_metr(raw_data_dir, report="time-horizon-1-0")
    elif benchmark_name == "metr_1_1":
        return load_metr(raw_data_dir, report="time-horizon-1-1")
    else:
        raise ValueError(f"unknown benchmark {benchmark_name}")
