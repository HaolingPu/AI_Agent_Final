# Do Small Benchmark Subsets Still Rank Newer Agents?

A small subset of benchmark tasks can rank agents almost like the full benchmark, but prior work tests this on the same agents it selects from. We select subsets from older or weaker agents' results and check whether they still rank newer and stronger agents, on SWE-bench Verified and Terminal-Bench 2.0.

- [ROADMAP.md](ROADMAP.md): plan, core claims, data, roles and timeline.
- [BACKGROUND.md](BACKGROUND.md): prior work and concepts (IRT, Kendall tau, noise ceiling, decay curve).

## Layout

```
subset_eval/          shared framework (import from here)
  data.py             BenchmarkData: the unified agents / tasks / runs tables; outcome_matrix()
  toy.py              synthetic benchmark for development
  split.py            agent splits: random, by date
  select.py           task selection methods
  metrics.py          Kendall tau, top-k overlap, bootstrap noise ceiling
experiments/          entry points; each writes results/*.csv (gitignored)
tests/                pytest
```

## Setup

```bash
uv sync
uv run pre-commit install
```

## Run

```bash
uv run pytest -q
uv run python experiments/run_toy.py
```

## Adding a benchmark

Write `subset_eval/loaders/<benchmark>.py` with one function that returns `BenchmarkData` (see the column lists in [data.py](subset_eval/data.py)). Write every (agent, task, trial) as a row, with unresolved tasks as `resolved = 0`; `outcome_matrix` refuses missing pairs.

11-768 AI Agents, Fall 2026, Carnegie Mellon University.
