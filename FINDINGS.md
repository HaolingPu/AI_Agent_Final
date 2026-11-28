# Findings

Status on 2026-11-28. Every number comes from `results/` (gitignored) and can be regenerated with the commands at the end. Definitions are in [BACKGROUND.md](BACKGROUND.md); the plan is in [ROADMAP.md](ROADMAP.md).

## 1. Benchmarks

| Benchmark (`--benchmark`) | Agents | Tasks | Trials per task | Dated by | Span | Top / median solve rate |
|---|---|---|---|---|---|---|
| `swe_bench_verified` | 173 | 500 | 1 | submission folder | 2023-10 to 2026-09 | 0.79 / 0.57 |
| `swe_bench_verified_one_per_model` | 110 | 500 | 1 | submission folder | 2023-10 to 2026-09 | 0.79 / 0.58 |
| `swe_bench_lite` | 84 | 300 | 1 | submission folder | 2023-10 to 2025-09 | 0.60 / 0.34 |
| `terminal_bench` (2.0) | 74 | 89 | 4.9 | first trial | 2025-11 to 2026-05 | 0.90 / 0.67 |
| `terminal_bench_4_1` | 67 | 66 | 5 | model release | 2026-02 to 2026-11 | 0.71 / 0.39 |
| `terminal_bench_4_1_highest_effort` | 54 | 66 | 5 | model release | 2026-02 to 2026-11 | 0.71 / 0.30 |
| `terminal_bench_4_1_graded_only` | 67 | 53 | 5 | model release | 2026-02 to 2026-11 | 0.69 / 0.39 |
| `metr_1_0` (time horizon 1.0) | 31 | 157 | 7.4 | model release | 2022-03 to 2025-11 | 0.85 / 0.72 |
| `metr_1_1` (time horizon 1.1) | 17 | 226 | 4.9 | model release | 2023-11 to 2026-02 | 0.84 / 0.71 |
| `multi_swe_bench` (7 languages, no Python) | 36 | 1,632 | 1 | model release | 2024-09 to 2025-04 | 0.14 / 0.05 |

- Multi-SWE-bench is a grid: 3 scaffolds (MagentLess, MopenHands, MSWE-agent) × 12 models, all run by the benchmark's authors.
- **Not used: tau2-bench.** Its public per-task `results.json` for about 14 of the 25 banking_knowledge submissions predates the v1.0.1 regrade, so they disagree with the leaderboard by up to 8.8 points. The regraded trajectory files are 90-230 MB each, and the airline / retail / telecom domains have only 8 submissions.
- **Not used: HAL.** Its traces are 113 GB of encrypted archives.

## 2. C1: random splits overstate how well subsets rank new agents

Plain random subsets of k tasks, the same subsets on both splits (paired by repeat), Kendall tau against the full benchmark. Brackets: 95 % bootstrap interval over 20 repeats.

| Benchmark | k | Random split | Date split | Date − random | Noise ceiling (2.5 %), date − random |
|---|---|---|---|---|---|
| SWE-bench Verified | 25 | 0.796 | 0.689 | −0.107 [−0.137, −0.080] | −0.05 |
| SWE-bench Verified | 100 | 0.904 | 0.833 | −0.071 [−0.081, −0.061] | −0.05 |
| SWE-bench Lite | 25 | 0.732 | 0.675 | −0.058 [−0.098, −0.014] | −0.03 |
| Terminal-Bench 2.0 | 10 | 0.763 | 0.790 | +0.027 [−0.009, +0.060] | +0.02 |
| Terminal-Bench 4.1 (highest effort) | 10 | 0.840 | 0.803 | −0.037 [−0.063, −0.013] | −0.05 |
| METR 1.0 | 10 | 0.765 | 0.603 | −0.162 [−0.238, −0.090] | −0.08 |
| METR 1.1 | 20 | 0.859 | 0.339 | −0.520 [−0.640, −0.388] | −0.41 |
| Multi-SWE-bench | 50 | 0.525 | 0.511 | −0.015 [−0.132, +0.104] | +0.02 |

- On 5 of 7 benchmarks the date split is lower. The two exceptions span only 6-7 months: Terminal-Bench 2.0 (submissions) and Multi-SWE-bench (model releases).
- **What this does and does not show.** Plain random subsets never look at the selection agents. So this gap is not staleness. It shows that **the newest agents are harder to rank than a random 30 %**, because they sit closer together. The noise ceiling falls too, but less than tau on SWE-bench and METR 1.0. A random-split evaluation therefore promises a tau that new agents do not get.
- METR 1.1 has only 5 test agents (10 pairs), so its tau is coarse.

## 3. X1 and X5: which selection method survives a shift

Each method minus plain random, on the same split and repeat (paired). k = smallest size, then a middle size.

| Benchmark | Split | k | Random tau | X1 irt_frontier − random | irt_max_information − random |
|---|---|---|---|---|---|
| SWE-bench Verified | date | 25 | 0.689 | **+0.058** [+0.030, +0.087] | +0.032 [+0.011, +0.055] |
| SWE-bench Verified | weak→strong | 25 | 0.398 | **+0.144** [+0.091, +0.195] | −0.038 [−0.096, +0.021] |
| SWE-bench Verified | weak→strong | 100 | 0.614 | **+0.131** [+0.104, +0.162] | −0.080 [−0.110, −0.047] |
| SWE-bench Lite | weak→strong | 25 | 0.471 | **+0.161** [+0.110, +0.213] | +0.105 [+0.030, +0.179] |
| Terminal-Bench 2.0 | weak→strong | 20 | 0.441 | **+0.224** [+0.164, +0.285] | −0.224 [−0.295, −0.156] |
| Terminal-Bench 4.1 (highest effort) | weak→strong | 40 | 0.726 | **+0.080** [+0.037, +0.123] | −0.079 [−0.117, −0.040] |
| METR 1.0 | date | 10 | 0.603 | **+0.222** [+0.139, +0.305] | +0.111 [+0.021, +0.206] |
| METR 1.0 | weak→strong | 20 | 0.427 | **+0.341** [+0.220, +0.460] | −0.157 [−0.282, −0.053] |
| METR 1.1 | date | 20 | 0.339 | **+0.608** [+0.459, +0.764] | +0.008 [−0.199, +0.214] |
| Multi-SWE-bench | date | 50 | 0.511 | **+0.124** [+0.026, +0.227] | +0.011 [−0.111, +0.137] |
| Multi-SWE-bench | weak→strong | 50 | 0.380 | −0.201 [−0.336, −0.059] | −0.197 [−0.336, −0.053] |

- **X1 (irt_frontier)** keeps the k tasks with the most Fisher information for abilities from the best selection agent up to one prior SD above it.
  - Under a weak→strong shift it beats random on every benchmark except two: Terminal-Bench 4.1 at k = 10 and 20 (a tie), and Multi-SWE-bench, where it loses (−0.15 to −0.26).
  - On the date split it wins or ties everywhere except SWE-bench Lite.
  - **The Multi-SWE-bench counterexample.** The weak selection agents solve 3.5 % of all tasks; 86 % of a random 50 are solved by none of them. A task no selection agent solves has no usable difficulty estimate, so both IRT methods can only pick tasks the weak agents did solve. For the test agents those are easy: they solve 44 % of X1's 50 tasks (56 % of irt_max_information's) but only 8 % of all tasks. The ranking on easy tasks differs from the ranking on the whole benchmark, and irt_max_information loses just as much (−0.197 at k = 50). One repeat at k = 50; not tested further.
  - Its cost: on a random split, where the test agents look like the selection agents, it loses (SWE-bench Verified −0.137 at k = 25). It spends its tasks above the agents being ranked.
- **irt_max_information**, the best method on random splits (+0.045 on SWE-bench Verified at k = 25), drops below random under weak→strong on most benchmarks. It aims at the selection agents' abilities, which the new agents have passed. On one weak→strong repeat, SWE-bench Verified's test agents solve 97 % of its 50 tasks (random 50: 76 %), and METR 1.0's solve 99 % of its 20 (random 20: 84 %).
- The tinyBenchmarks anchor methods and gp-IRT never beat random reliably. Our reproduction matches the paper on its own MMLU data, so this is not a bug.
- Robustness (k = 10 / 25, weak→strong, X1 − random): Terminal-Bench 4.1 all rows +0.129, graded-only +0.190; SWE-bench one-per-model +0.128.
- Intervals hold for these agents. The date and weak→strong test groups are fixed, so repeats vary only the selection-agent bootstrap. There is no correction for the many comparisons.

## 4. C2: frozen subsets go stale slowly on SWE-bench

A subset frozen at the cutoff vs one re-selected before each later window, scored on the same window's agents (20 repeats).

- **SWE-bench Verified, cutoff 2024-12-31, 3-month windows.** The gap averaged over methods is −0.023, −0.011, −0.037 and −0.054 at 3, 6, 9 and 12 months. Its 95 % band is fully below 0 at 9 months (high −0.003) and 12 months (high −0.012).
- **Same test window, older subset** (`figures/c2/swe_bench_verified_decay_by_age.png`). The 2025-12 window falls from −0.015 (subset 2 months old) to −0.024, −0.046 and −0.054 (11 months old).
- **No single (method, size) reaches a shelf life.** No method is behind beyond its band in two windows in a row, on any of the four cutoffs.
- **The frozen X1 subset does not decay:** gap −0.005, +0.046, −0.004 and −0.002 at 3-12 months. irt_max_information's gap: −0.055, −0.012, −0.019 and −0.057.
- One-per-model SWE-bench: −0.079 at 12 months.
- Lite (6-month windows) and METR 1.0 have too few agents per window for a clear result (bands cross 0).

## 5. Why subsets go stale: tasks get used up

The share of a frozen subset's tasks that a later window's agents solve at least 90 % of the time:

| Benchmark | Subset (k) | At the cutoff | Last window |
|---|---|---|---|
| SWE-bench Verified (cutoff 2024-12) | every task | 0.28 | 0.49 (12 months) |
| | irt_max_information (25) | 0.33 | 0.74 |
| | X1 irt_frontier (25) | 0.00 | 0.24 |
| METR 1.0 (cutoff 2024-12) | every task | 0.66 | 0.72 (6 months) |
| | irt_max_information (10) | 0.74 | 0.95 |
| | X1 irt_frontier (10) | 0.08 | 0.30 |

The subset that is most informative at the cutoff is the one that the next models solve first.

## 6. C3: the full benchmark stops telling the top 10 apart

The number of top-10 pairs (of 45) whose score gap is beyond task-resampling noise, on the leaderboard so far:

| Benchmark | First month | Last month |
|---|---|---|
| SWE-bench Verified | 39 (2024-06) | 2 (2026-09) |
| SWE-bench Verified, one per model | 40 (2024-07) | 5 (2026-09) |
| SWE-bench Lite | 35 (2024-05) | 25 (2025-09) |
| Terminal-Bench 2.0 | 20 (2026-01) | 7 (2026-05) |
| Terminal-Bench 4.1, highest effort | 32 (2026-07) | 6 (2026-11) |
| METR 1.0 | 32 (2024-09) | 19 (2025-11) |

- A pool of only the last 6 months separates more pairs (SWE-bench Verified 24 in 2026-05; Terminal-Bench 4.1 18 in 2026-11).
- So part of the drop comes from a growing pool packing its top 10 tighter. The rest is saturation: SWE-bench Verified still ends far lower than its 2024 level.

## 7. Cost: cut trials, not tasks

Share of a full evaluation's dollars vs Kendall tau, date split (`figures/cost/cost_vs_tau_date.png`). A task's cost is the median share of a costed agent's spend.

| Benchmark | Every task, 1 trial: cost / tau | Best subset, all trials, same or lower cost | Best subset at any cost |
|---|---|---|---|
| Terminal-Bench 2.0 | 0.19 / **0.913** | irt_max_information k=10: 0.13 / 0.864 | irt_max_information k=40: 0.57 / 0.932 |
| Terminal-Bench 4.1 (highest effort) | 0.20 / **0.898** | irt_max_information k=10: 0.19 / 0.835 | X1 k=40: 0.60 / 0.935 |
| METR 1.0 | 0.15 / **0.880** | anchor_irt k=20: 0.15 / 0.782 | X1 k=20: 0.44 / 0.878 |

- When a benchmark runs several trials per task, one trial of every task beats every subset at the same cost, by 0.05-0.10 tau. A subset needs about 3× the money to catch up.
- The frontier tasks X1 picks are long: they cost 1.3-1.8× an average task on SWE-bench and Terminal-Bench, and 3-4× on METR.
- On one-trial benchmarks (SWE-bench), a subset is the only way to save.

## 8. Caveats

- **Few test agents.** METR 1.1 has 5 test agents and METR 1.0 about 10. On METR 1.1 the date and weak→strong test groups are the same 5 agents.
- **Dates.** Terminal-Bench 4.1, METR and Multi-SWE-bench are dated by model release, not by submission. Multi-SWE-bench dates are hand-entered from the labs' announcements.
- **Weak→strong** splits by the very score being ranked, so it cannot be done in practice. Read it only relative to random.
- **Costs** come from few agents: mini-SWE-agent only on SWE-bench, and 23 of 31 models on METR 1.0. METR 1.1 and Multi-SWE-bench have no costs.

## 9. Reproduce

```bash
export UV_PROJECT_ENVIRONMENT=~/.venvs/ai_agent_final
uv run python experiments/run_core_loop.py --benchmark <name> --subset-sizes 10 20 40      # SWE-bench and Multi-SWE-bench: 25 50 100 200
uv run python experiments/summarize_core_loop.py --benchmarks swe_bench_verified swe_bench_lite terminal_bench terminal_bench_4_1_highest_effort metr_1_0 metr_1_1 multi_swe_bench
uv run python experiments/run_decay.py --benchmark swe_bench_verified --cutoff-date 2024-12-31 --window-months 3 --subset-sizes 25 50 100 200 --output-dir results/c2/2024-12-31
uv run python experiments/run_saturation.py --benchmark <name>
uv run python experiments/run_cost.py --benchmark terminal_bench_4_1_highest_effort --trial-counts 1 2 3
uv run python experiments/run_task_lifecycle.py --benchmark swe_bench_verified --cutoff-date 2024-12-31 --subset-sizes 25 100
```
