# Background

Concepts and prior work behind the [roadmap](ROADMAP.md).

## Agents and benchmarks

- **Agent = scaffold + model.** The scaffold is the shell (how the agent reads code, calls tools, runs commands); the model is the brain. The same model can score very differently in different scaffolds.
- Common scaffolds: **mini-SWE-agent** (SWE-bench team, bash only; used to test almost every new model), SWE-agent, OpenHands, **Terminus 2** (Terminal-Bench's own), Claude Code, Codex CLI, Gemini CLI. mini-SWE-agent and Terminus 2 pair one fixed scaffold with many models, which separates model progress from scaffold progress.
- **SWE-bench Verified**: 500 real GitHub issues from Python repos; the agent must write a patch that makes the hidden tests pass. Top agents now solve ~80 % or more, so it is close to **saturated**: it barely separates the best agents.
- **Terminal-Bench 2.0**: ~89 hard tasks done in a Linux terminal (build, debug, data, security, ...). Not saturated. Each task is run 5 times per agent.
- Newer SWE-bench variants exist (SWE-bench Pro, SWE-rebench) but do not publish dated per-task results.

## Measuring agreement

### Kendall tau

How similar two rankings are. Take every pair of agents and check whether both rankings order them the same way: tau = (agreeing pairs - disagreeing pairs) / all pairs. 1 = identical, 0 = unrelated, -1 = reversed.

Example: full ranking A > B > C, subset ranking A > C > B. Pairs A-B and A-C agree, B-C disagrees: tau = (2 - 1) / 3 = 0.33.

### Noise ceiling (bootstrap)

A subset's tau is never exactly 1, so how high is high enough? The full benchmark is itself only one sample of possible tasks. Resample the 500 tasks **with replacement** (some appear twice, some not at all), re-rank the agents, and repeat ~1000 times. The range of tau against the original ranking (say 0.88-0.95) is how much the full benchmark wobbles on its own.

**A subset is good enough when its tau falls inside this range.** This only resamples existing results: no agent runs, $0.

## Selecting tasks

### Item Response Theory (IRT)

From educational testing (GRE, TOEFL). Each agent gets an **ability** theta, each task a **difficulty** b and a **discrimination** a. The chance that agent i solves task j is

    P = sigmoid(a_j * (theta_i - b_j))

- Ability far above difficulty -> likely solved.
- **High discrimination**: a slightly stronger agent is much more likely to solve it, so the task separates agents well. **Low discrimination**: strong and weak agents do about equally, so the task says little.

All three are estimated from the existing 0/1 results.

### Prior methods

| Paper | Selection | Tasks kept | Test split |
|---|---|---|---|
| tinyBenchmarks (arXiv 2402.14992) | IRT: pick informative items, then estimate the full score | ~100 per LLM benchmark | random |
| Agent Psychometrics (arXiv 2604.00594) | IRT + **adaptive testing**: each agent gets the next most informative task given its answers so far | ~40 of 500 on SWE-bench Verified | random |
| PTA-IRT (2026) | IRT + agent traces | 10 % | random |
| Ndzomga (arXiv 2603.23749) | keep tasks that 30-70 % of past agents solved | 30-56 % | by date, Terminal-Bench only |
| PACE (arXiv 2607.02032, Neubig et al.) | cheap non-agentic items + regression predict agentic scores | <1 % of cost | leave-one-out over 14 models |

Related: Observational Scaling Laws (arXiv 2405.10938) predicts strong models from weaker public models.

**The gap.** IRT difficulty and discrimination are estimated from **older** agents. When newer agents get stronger, yesterday's "hard, informative" tasks can become easy for everyone, and tasks dropped as "impossible" can become the ones that separate the best agents. Random splits never see this, because the test agents come from the same period as the selection agents.

## Splits

- **Random split**: shuffle agents, half select, half are tested. What prior work does.
- **Date split**: agents before a cutoff date select, agents after it are tested. What happens in practice.
- **Weak-to-strong split**: weaker agents select, stronger agents are tested.

## Decay curve (our main figure)

1. Select a subset from agents before the cutoff (e.g. 2025-06-30) and **freeze** it.
2. Group later agents by submission month.
3. Compute tau for each month and plot it against the noise-ceiling band.

```
tau
1.0 |
    |  ===================   <- noise-ceiling band
0.9 |  *--*--*
    |           *
0.8 |              *--*      <- leaves the band: time to re-select
    +--+--+--+--+--+--+-->  months after the cutoff
       1  2  3  4  5  6
```

The month where the curve leaves the band is the subset's **shelf life**. One curve per selection method shows which lasts longest. Repeating with several cutoffs checks that the result is not a coincidence.

## Task life cycle and benchmark saturation

Each task tends to move through three stages:

    nobody solves it  ->  some agents do (separates agents)  ->  everyone solves it (no longer useful)

A whole benchmark expires when its top agents' score gaps become smaller than the bootstrap noise: it can no longer tell them apart and needs new tasks.
