# Getting Started

For Wilson and Wendy, and for your coding agent. [ROADMAP.md](ROADMAP.md) has the full plan, and [BACKGROUND.md](BACKGROUND.md) explains the concepts.

## The project in three lines

1. We pick a small subset of benchmark tasks, using older or weaker agents' results.
2. We check whether that subset still ranks newer or stronger agents the way the full benchmark does (Kendall tau).
3. The framework and the experiments already run on synthetic data. **What is missing is the real data: your loaders.**

## Setup

```bash
git clone https://github.com/HaolingPu/AI_Agent_Final.git
cd AI_Agent_Final
uv sync
uv run pre-commit install
uv run pytest -q                              # all tests should pass
uv run python experiments/run_core_loop.py    # synthetic data, about 30 s
```

If the repo sits in an iCloud-synced folder (Desktop or Documents on macOS), keep the virtual environment outside it, or imports break:

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/ai_agent_final
```

## Your part

|  | Wilson | Wendy |
|---|---|---|
| Benchmark | SWE-bench Verified | Terminal-Bench 2.0 |
| Loader file | `subset_eval/loaders/swe_bench_verified.py` | `subset_eval/loaders/terminal_bench.py` |
| Core claim | C2, shelf life: `experiments/run_decay.py` | C3, saturation: `experiments/run_saturation.py` (both benchmarks) |
| Extension | X3, model vs scaffold (mini-SWE-agent runs only) | X4, run-to-run noise (5 trials); X5, weak-to-strong (already a split in `run_core_loop.py`) |

## Step 1: write the loader

Write one function that reads the raw files and returns `BenchmarkData`. For example, `load_swe_bench_verified(raw_data_dir: Path) -> BenchmarkData`.

`BenchmarkData` (in [subset_eval/data.py](subset_eval/data.py)) holds three tables:

| Table | Columns |
|---|---|
| `agents` | `agent_id`, `scaffold`, `model`, `model_org`, `open_weight`, `submit_date`, `model_release_date` |
| `tasks` | `task_id`, `category` |
| `runs` | `agent_id`, `task_id`, `trial`, `resolved` (0 or 1), `cost_usd`, `total_tokens` |

Rules:
- Write one `runs` row per (agent, task, trial). **A failed task is a row with `resolved = 0`; never drop it.**
- Every agent needs a result on every task. If a submission has no per-task results, leave that agent out and write down how many you left out.
- `submit_date` is a pandas datetime: the day the result was submitted. It is required.
- Ids are unique, and a task has the same `task_id` for every agent.
- Unknown optional values (cost, tokens, release date) can be empty.
- Put the downloaded files in `data/raw/`. That folder is gitignored, so never commit raw data.

`BenchmarkData` checks most of these rules as soon as you build it, and raises an error that names the problem.

When the loader works, register it in [subset_eval/loaders/__init__.py](subset_eval/loaders/__init__.py): add its name to `BenchmarkName` and one `elif` to `load_benchmark`. Then add a short test in `tests/`. For example, check the number of agents and tasks, and check that one agent's solve rate matches the public leaderboard.

Where the data is. We found these on a first look, so check them against the files.
- **SWE-bench Verified**:
  - Source: repo `github.com/SWE-bench/experiments`, folders `evaluation/verified/<YYYYMMDD>_<agent>/`.
  - Per-task results are in `results/results.json` (the list of resolved tasks), or in `per_instance_details.json` for mini-SWE-agent submissions (this file also has per-task cost).
  - `metadata.yaml` gives the agent and model. The date is the folder name's prefix.
  - About 175 of the 182 submissions have per-task results. There is one trial per task.
- **Terminal-Bench 2.0**:
  - Source: Hugging Face dataset `harborframework/terminal-bench-2-leaderboard`, one folder per `agent__model`.
  - Each trial has a `result.json` with `task_name`, `verifier_result.rewards.reward`, `cost_usd`, tokens and `started_at`.
  - There are about 5 trials per task.
  - The date comes from the job folder name, or from `started_at` when the name has none.

## Step 2: run the experiments

```bash
uv run python experiments/run_core_loop.py --benchmark swe_bench_verified
uv run python experiments/run_decay.py --benchmark swe_bench_verified --cutoff-date 2025-06-30
uv run python experiments/run_saturation.py --benchmark terminal_bench
```

Each script prints a summary and writes `results/<benchmark>_<experiment>.csv`. How to read them (details in ROADMAP section 5):
- **Core loop**: compare each method with `random` within the same split; never compare raw tau across splits.
- **Decay (C2)**: a negative gap means the frozen subset ranks worse than a freshly chosen one, so it has gone stale.
- **Saturation (C3)**: the benchmark is saturated only if the separable pairs drop for the last-6-months pool too, not just for the growing leaderboard.

## Working rules

- **Branches**: work on your own branch (for example `wilson/swe-loader`) and open a pull request into `main`. Pull before you start.
- **Commits**: make one commit per finished change and push it right away.
- **Style**: match the existing code.
  - Use full-word names.
  - No `Any`, `getattr` or `hasattr`.
  - Every `if` has an `else`.
  - Sign comments `# note (Your Name): ...`.
  - pre-commit formats the code when you commit.
- **Secrets**: never commit API keys, tokens or `.env` files.
- **Shared files**: ask Haoling before changing `data.py`, `methods.py`, `metrics.py`, or the logic of the experiment scripts.
