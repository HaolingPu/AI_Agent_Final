# Roadmap

11-768 final project. Deadline: 2026-12-03. New to the terms (IRT, Kendall tau, noise ceiling, ...)? Read [BACKGROUND.md](BACKGROUND.md) first.

## 1. Question

Running an agent on a full benchmark is expensive. Prior work shows that a small subset of tasks (e.g. ~40 of 500) ranks agents almost like the full benchmark. But it selects the subset and tests it on the **same population of agents** (random split).

In practice a subset is selected once, from the results of **existing** agents, and then used on agents that come **later** and are **stronger**.

> Does a subset selected from older agents' results still rank newer agents correctly, and for how long?
> And when does the full benchmark itself stop telling the best agents apart?

## 2. Core claims (must do)

| ID | Claim | How we test it |
|---|---|---|
| **C1** | Random splits **overstate** how well small subsets work | Same methods and sizes, random split vs date split: compare Kendall tau |
| **C2** | A frozen subset expires after about N months, so it should be re-selected every N months | Decay curve: tau per month after the cutoff, against the noise ceiling |
| **C3** | A saturated benchmark (SWE-bench Verified) stops separating the top agents after some date; an unsaturated one (Terminal-Bench) does not yet | Per month: score gaps among the top 10 agents vs bootstrap noise, on both benchmarks |

Story: prior evaluations are too optimistic (C1) -> subsets expire, here is how fast (C2) -> the full benchmark expires too (C3).

## 3. Extensions (in priority order)

| ID | Idea |
|---|---|
| X1 | **Frontier-aware selection** (our own method): keep tasks slightly harder than the current best agent. Does the subset last longer? |
| X2 | Task life cycle: when does each task go from "unsolved" to "separates agents" to "solved by everyone"? (mostly a by-product of C3) |
| X3 | Model vs scaffold: redo C2 on mini-SWE-agent only (one scaffold, many models) |
| X4 | Run-to-run noise: use Terminal-Bench's 5 trials per task; add to the noise ceiling |
| X5 | Weak-to-strong split: select with the weakest 2/3 of agents, test on the strongest 1/3 |
| X6 | Adaptive testing + staleness alarm |

## 4. Data

| | SWE-bench Verified | Terminal-Bench 2.0 |
|---|---|---|
| Source | [SWE-bench/experiments](https://github.com/SWE-bench/experiments) `evaluation/verified/<date>_<name>/` | [harborframework/terminal-bench-2-leaderboard](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard) `submissions/terminal-bench/2.0/<agent>__<model>/` |
| Agents x tasks | 175 usable of 182 x 500 | 76 x ~89, 5 trials per task |
| Per-task 0/1 | `results/results.json` (135) or `per_instance_details.json` (40, also per-task cost); 7 have neither | `<job>/<trial>/result.json` -> `verifier_result.rewards.reward` |
| Agent date | folder name (2023-10 to 2026-09, ~3 years) | job folder name, else `started_at` (2025-11 to 2026-05, ~6 months) |
| Scaffold / model | `metadata.yaml`: `agent`, `model`, `model_org`, `os_model` | `metadata.yaml`: `agent_display_name`, `models[]` |
| Model release date | only 48 of 182 (`model_release_date`); fill the rest | not given |
| Cost / tokens | 40 mini-SWE-agent runs | every trial (`cost_usd`, tokens) |

- The long time axis (C2) uses **SWE-bench**; Terminal-Bench covers only ~6 months, so it serves C1, C3 (as the unsaturated contrast), X4 and X5.
- Model release dates: fill from the Artificial Analysis free API (`release_date`; attribution required) or by hand.
- Raw data is not committed (`data/raw/` is in `.gitignore`).

### Unified format (both benchmarks)

Three tables, long format:

| Table | Columns |
|---|---|
| `agents` | `agent_id`, `scaffold`, `model`, `model_org`, `open_weight`, `submit_date` (datetime, required), `model_release_date` |
| `tasks` | `task_id`, `category` (repo for SWE-bench) |
| `runs` | `agent_id`, `task_id`, `trial`, `resolved` (0/1), `cost_usd`, `total_tokens` (NaN when unknown) |

A loader returns `BenchmarkData(name=..., agents=..., tasks=..., runs=...)` from [subset_eval/data.py](subset_eval/data.py), which checks these columns. Every (agent, task, trial) is one row, unresolved tasks included as `resolved = 0`. The agents x tasks matrix is `outcome_matrix()`: the mean over trials.

## 5. Core loop

1. **Split** agents into a selection group and a test group (random / by date / weak-to-strong).
2. **Select** k tasks from the selection group's results: random; stratified (sort by pass rate, cut into k near-equal groups, one random task per group); 30-70 % pass-rate band (Ndzomga); IRT information (fit a 2PL model on the selection agents, keep the k tasks with the highest Fisher information over their abilities); tinyBenchmarks anchor points (K-means on each task's results or IRT parameters, keep the task nearest each centroid, weight it by its cluster's share of tasks).
3. **Rank** the test group by its subset score: the plain mean, the anchor-weighted mean, or the tinyBenchmarks gp-IRT estimate (for random and both anchor methods).
4. **Score**: Kendall tau vs. the test group's full-benchmark ranking; compare with the noise ceiling.

**Compare within a split.** Newer agents are closer to each other, so every subset (and the full benchmark) ranks them less reliably; on toy data even random selection drops in the date split, and so does the noise ceiling. A method's result is therefore read relative to random selection and to the noise ceiling **on the same split**, never as a raw tau across splits.

**Reading the other experiments.** Weak-to-strong defines its groups by the very score being ranked, so its test group is compressed: read it only relative to random, and note that it cannot be done in practice. C2 compares a frozen subset with a refreshed one on the same agents, because raw tau per window mostly tracks which agents fall in that window. C3 reports a fixed 6-month pool next to the growing leaderboard, because the top 10 of a larger pool is packed tighter by itself.

Subset sizes: SWE-bench 25 / 50 / 100 / 200; Terminal-Bench 10 / 20 / 40.

Date cutoffs (SWE-bench): 2025-06-30 main; also 2024-12, 2025-03, 2025-09.

## 6. Code layout

| Part | Input -> output |
|---|---|
| `subset_eval/data.py` | `BenchmarkData` (the three tables, validated) -> `outcome_matrix` |
| `subset_eval/loaders/` | raw files -> `BenchmarkData` (Wilson, Wendy) |
| `split` | tables + split kind -> selection group, test group |
| `subset_eval/select.py` | selection group + method + k -> k task ids |
| `subset_eval/tiny_benchmarks.py` | tinyBenchmarks reproduction: multidimensional IRT, anchor points, gp-IRT estimate |
| `subset_eval/metrics.py` | subset estimate + test group -> tau, top-5 agreement; bootstrap noise ceiling |
| `subset_eval/methods.py` | one interface over every method: fit the selection group once, pick weighted subsets, estimate test scores |
| `experiments/run_core_loop.py` | every method x split (random / date / weak-to-strong) x size -> `results/<benchmark>_core_loop.csv` (C1, X5) |
| `experiments/run_decay.py` | subsets frozen at a cutoff vs. refreshed before each later window, scored on the same window's agents; shelf life = first of two windows in a row where frozen ranks worse -> `results/<benchmark>_decay.csv` (C2) |
| `experiments/run_saturation.py` | per month, for all agents so far and for the last 6 months only: top-10 pairs the full benchmark separates beyond bootstrap noise, top-10 noise ceiling -> `results/<benchmark>_saturation.csv` (C3) |

## 7. Roles

Everyone owns data or framework **and** one core claim end to end (experiments, figure, report section).

| Person | Data / code | Core claim | Extensions |
|---|---|---|---|
| Haoling | Shared framework (split / select / evaluate / ceiling), figure style, report structure | **C1** on both benchmarks | X1 |
| Wilson | SWE-bench Verified loader, fill model release dates | **C2** decay curves | X3 |
| Wendy | Terminal-Bench 2.0 loader, confirm dates | **C3** on both benchmarks | X4, X5 |

## 8. Timeline

| Dates | Work |
|---|---|
| 11/21-11/23 | Agree on the unified format and function interfaces. Wilson and Wendy build loaders. |
| 11/24-11/27 | Each person runs their core claim (C1 / C2 / C3) with the shared framework (11/26 Thanksgiving) |
| 11/28-11/30 | Finish core claims; extensions if time allows |
| 12/01-12/03 | Figures and report |

## 9. Deliverables

1. Table: benchmark x split (random / date), cells = Kendall tau, with the noise ceiling (C1).
2. Decay curve per selection method, with the noise-ceiling band (C2).
3. Top-10 score gaps vs noise over time, SWE-bench Verified vs Terminal-Bench (C3).
4. Final report.

## 10. Open checks

- [ ] Terminal-Bench: dates for the ~18 submissions whose job folders are not named by date
- [ ] Terminal-Bench: 89 vs 90 tasks; drop submissions with missing trials or 2.1 jobs
- [ ] SWE-bench: what to do with the 7 submissions without per-task results (drop)
- [ ] What exactly is due on 12/03 (report, presentation, code)?
