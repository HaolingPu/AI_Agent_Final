# Do Small Benchmark Subsets Still Rank Newer Agents?

A small subset of benchmark tasks can rank agents almost like the full benchmark, but prior work tests this on the same agents it selects from. We select subsets from older or weaker agents' results and check whether they still rank newer and stronger agents, on SWE-bench Verified and Terminal-Bench 2.0.

- [ONBOARDING.md](ONBOARDING.md): **start here** if you are writing a loader or running experiments.
- [ROADMAP.md](ROADMAP.md): plan, core claims, data, roles and timeline.
- [BACKGROUND.md](BACKGROUND.md): prior work and concepts (IRT, Kendall tau, noise ceiling, decay curve).

## Layout

```
subset_eval/          shared framework (import from here)
  data.py             BenchmarkData: the unified agents / tasks / runs tables; outcome_matrix()
  toy.py              synthetic benchmark for development
  split.py            agent splits: random, by date, weak-to-strong
  select.py           task selection: random, stratified, pass-rate band, IRT max-information
  tiny_benchmarks.py  tinyBenchmarks reproduction: anchor points, multidimensional IRT, gp-IRT
  methods.py          every method behind one interface (used by all experiments)
  seeding.py          one random stream per purpose name
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
uv run python experiments/run_core_loop.py    # core loop: every method on random / date / weak-to-strong splits (C1, X5)
uv run python experiments/run_decay.py        # C2: frozen subsets scored on later windows, shelf life
uv run python experiments/run_saturation.py   # C3: can the full benchmark still separate its top 10?
uv run python experiments/reproduce_tiny_benchmarks.py   # sanity check: our methods on tinyBenchmarks' own MMLU data
```

Each script takes `--benchmark` (default `toy`, synthetic data); a real benchmark becomes available once its loader is registered in `subset_eval/loaders/__init__.py`. Results go to `results/<benchmark>_<experiment>.csv`.

## Adding a benchmark

Write `subset_eval/loaders/<benchmark>.py` with one function that returns `BenchmarkData` (see the column lists in [data.py](subset_eval/data.py)). Write every (agent, task, trial) as a row, with unresolved tasks as `resolved = 0`; `outcome_matrix` refuses missing pairs.

11-768 AI Agents, Fall 2026, Carnegie Mellon University.
