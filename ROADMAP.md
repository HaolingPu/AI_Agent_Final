# Roadmap

11-768 final project. Deadline: 2026-12-03.

## 1. Question

Running an agent on a full benchmark is expensive. Prior work (tinyBenchmarks, Agent Psychometrics,
PTA-IRT) shows that a small subset of tasks (e.g. ~40 of 500) ranks agents almost like the full set.

But they choose the subset and test it on the **same population of agents** (random split).
In practice, a subset is chosen once from the results of **existing** agents and then used on
agents that are **newer** and **stronger**.

**Does a subset chosen from older / weaker agents' results still rank newer / stronger agents
correctly, and for how long?**

## 2. Benchmarks

Both have public per-agent, per-task pass/fail results, so the core study costs $0.

| Benchmark | Agents x tasks | Dates | Source | Note |
|---|---|---|---|---|
| SWE-bench Verified | 182 x 500 | yes (folder name) | [SWE-bench/experiments](https://github.com/SWE-bench/experiments), `evaluation/verified/*/results/results.json` | near saturated |
| Terminal-Bench 2.0 | ~75 x 89, 5 trials per task | to check (trial timestamps in `result.json`) | [harborframework/terminal-bench-2-leaderboard](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard) (full repo 40 GB; `result.json` only ~350 MB) | not saturated; 5 trials give run-to-run noise |

Artificial Analysis Intelligence Index: external reference only (free API gives the headline index,
release date and cost; no per-task results). Requires attribution.

## 3. Data format (shared by both benchmarks)

```
            task_1 task_2 ... task_N
agent_A       1      0          1      date 2024-08, full score 0.33
agent_B       1      1          0      date 2025-03, full score 0.62
```

- `outcomes`: agents x tasks, 0/1 (Terminal-Bench: mean over the 5 trials, plus the raw trials)
- `dates`: one submission date per agent
- `full_score`: mean over all tasks (agent strength)

## 4. Core loop

1. **Split** agents into a *selection group* and a *test group*.
2. **Select** k tasks using only the selection group's results.
3. **Rank** the test group by their score on the k tasks.
4. **Score**: compare with the full-benchmark ranking of the test group.

Repeat over:

| Axis | Values |
|---|---|
| Split | random (prior work) / by date / weak-to-strong |
| Selection method | random, stratified by difficulty, 30-70 % pass-rate filter (Ndzomga), IRT (tinyBenchmarks) |
| Subset size k | SWE-bench 25 / 50 / 100 / 200; Terminal-Bench 10 / 20 / 40 |

Splits:
- **Random**: random half of agents selects, the other half is tested (repeat with many seeds).
- **By date**: agents before the cutoff select, agents after it are tested. SWE-bench cutoff 2025-06-30
  (~125 / ~57); also 2024-12, 2025-03, 2025-09. Terminal-Bench cutoff after dates are confirmed.
- **Weak-to-strong**: the weakest ~2/3 by full score select, the strongest ~1/3 are tested.

## 5. Metrics

- **Kendall tau** between the subset ranking and the full ranking of the test group (1 = identical).
- **Top-k agreement**: are the top 5 agents the same?
- **Score error** (MAE) where a method predicts the full score (IRT).
- **Noise ceiling**: resample the full task set with replacement (bootstrap) and re-rank the test group;
  the spread of tau shows how much the full benchmark itself wobbles. A subset is good enough when its
  tau falls inside this range.
- **Floor**: a random subset of the same size.

## 6. Experiments

| ID | Experiment | Question |
|---|---|---|
| E1 | Random split, all methods and sizes | Do we reproduce "small subsets are enough"? (sanity check of our code) |
| E2 | Date split | How much does tau drop on newer agents? **(main result)** |
| E3 | Weak-to-strong split | Does a subset chosen from weak agents rank strong agents? |
| E4 | Date split, tau per month after the cutoff | How long does a frozen subset last? (decay curve) |
| E5 | Terminal-Bench only, use the 5 trials | How much does the ranking move when the same agents are rerun? Add to the noise ceiling |
| E6 (optional) | Adaptive testing + staleness alarm | Only if time allows |

E1-E3 are one loop with different settings: 2 benchmarks x 3 splits x 4 methods x 4 sizes.

## 7. Deliverables

1. Table: benchmark x split, cells = Kendall tau (with noise ceiling).
2. Decay curve: tau vs. months after the cutoff, with the noise-ceiling band.
3. tau vs. subset size, per method.
4. Final report.

## 8. Code layout (planned)

| Part | Input -> output |
|---|---|
| `data/` loaders | benchmark name -> outcomes, dates, full_score |
| `split` | data + split kind -> selection group, test group |
| `select` | selection group + method + k -> k task ids |
| `evaluate` | k tasks + test group -> tau, top-k agreement, MAE |
| `ceiling` | test group -> bootstrap tau range |
| `experiments/` | runs the grid, writes `results/*.csv`, draws `figures/` |

Raw data is not committed (`data/raw/` is in `.gitignore`); each loader downloads it.

## 9. Roles

| Person | Owns |
|---|---|
| Haoling | Framework (split / select / evaluate / ceiling), experiment grid, figures, report structure |
| [Name B] | SWE-bench Verified: loader, dates, run the grid, E4 decay curve |
| [Name C] | Terminal-Bench 2.0: loader, confirm dates, run the grid, E5 run-to-run noise |

## 10. Timeline

| Dates | Work |
|---|---|
| 11/21-11/23 | Agree on the data format and function interfaces. B and C build their loaders. C checks Terminal-Bench dates. |
| 11/24-11/27 | E1-E3 on both benchmarks (11/26 Thanksgiving) |
| 11/28-11/30 | E4 decay curve, E5 noise, cross-benchmark comparison; E6 if time allows |
| 12/01-12/03 | Figures and report |

## 11. Open checks

- [ ] Terminal-Bench: can we get one date per submission from `result.json` timestamps?
- [ ] Terminal-Bench: 89 vs 90 tasks; which submissions to drop (missing trials, 2.1 jobs)
- [ ] SWE-bench: re-check for new submissions before the final run
- [ ] What exactly is due on 12/03 (report, presentation, code)?

## Related work

- tinyBenchmarks (arXiv 2402.14992): IRT picks ~100 items per LLM benchmark.
- Agent Psychometrics (arXiv 2604.00594): IRT + adaptive testing on SWE-bench Verified, random split.
- Ndzomga (arXiv 2603.23749): keeps tasks with 30-70 % pass rate; date split on Terminal-Bench only.
- PACE (arXiv 2607.02032, Neubig et al.): cheap non-agentic proxy predicts agentic scores; leave-one-out over 14 models.
- Observational Scaling Laws (arXiv 2405.10938): predicts strong models from weaker public models.
