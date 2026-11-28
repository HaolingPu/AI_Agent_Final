"""One entry point for every benchmark: each loader turns raw files into BenchmarkData."""

from pathlib import Path
from typing import Literal

from subset_eval.data import BenchmarkData
from subset_eval.loaders.swe_bench_verified import (
    load_swe_bench_verified,
    load_swe_bench_verified_mini_swe_agent,
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
    "swe_bench_verified",
    "swe_bench_verified_mini_swe_agent",
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
        return load_terminal_bench_4_1(raw_data_dir, drop_regrade_errors=False)
    elif benchmark_name == "terminal_bench_4_1_graded_only":
        return load_terminal_bench_4_1(raw_data_dir, drop_regrade_errors=True)
    elif benchmark_name == "swe_bench_verified":
        return load_swe_bench_verified(raw_data_dir)
    elif benchmark_name == "swe_bench_verified_mini_swe_agent":
        return load_swe_bench_verified_mini_swe_agent(raw_data_dir)
    else:
        raise ValueError(f"unknown benchmark {benchmark_name}")
