"""Cost: how much of a full evaluation's dollars a subset needs, next to how well it ranks; and fewer trials per task as the other way to save (X4).

Two ways to spend a fraction of the budget on each test agent:
    a task subset: k selected tasks, every trial (the core loop's setting);
    fewer trials: every task, or a subset, run with only t of its trials (Terminal-Bench runs 5 per task).
Both estimates are scored against the test agents' full ranking (every task, every trial).

A task's cost share is the median, over agents with a known cost on every task, of that task's share of the agent's own cost of one trial of every task;
shares rather than dollars, so expensive models do not set the profile. A subset of hard, long tasks then costs more than k / N of the budget.
SWE-bench has costs only for mini-SWE-agent runs, so its profile comes from one scaffold.
gp-IRT is reported only with every trial: its noise term is fit on all-trial means, so it would overweigh a one-trial subset score.
Splits: random and date only; weak-to-strong cannot be done in practice, and cost is a practical question.
"""

import argparse
import logging
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd

from subset_eval.data import outcome_matrix
from subset_eval.loaders import BenchmarkName, load_benchmark
from subset_eval.methods import (
    SELECTION_METHODS,
    estimate_solve_rates,
    fit_selection_group,
    select_task_weights,
)
from subset_eval.metrics import score_subset
from subset_eval.seeding import purpose_generator
from subset_eval.split import AgentSplit, split_by_date, split_random

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument(
        "--trial-counts",
        type=int,
        nargs="+",
        default=[1],
        help="trials per task to keep besides all of them; a task with fewer trials keeps all it has",
    )
    parser.add_argument("--selection-fraction", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--benchmark",
        choices=get_args(BenchmarkName),
        default="terminal_bench_4_1_highest_effort",
    )
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    cli_arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    benchmark = load_benchmark(
        cli_arguments.benchmark,
        raw_data_dir=cli_arguments.raw_data_dir,
        seed=cli_arguments.seed,
    )
    outcomes = outcome_matrix(benchmark)
    runs = benchmark.runs
    agent_task_cost_usd = runs.pivot_table(
        index="agent_id", columns="task_id", values="cost_usd", aggfunc="mean"
    ).reindex(columns=outcomes.columns)
    costed_agent_task_cost_usd = agent_task_cost_usd.dropna()
    if costed_agent_task_cost_usd.empty:
        raise ValueError(f"{benchmark.name}: no agent has a known cost on every task")
    elif max(cli_arguments.subset_sizes) >= outcomes.shape[1]:
        raise ValueError(
            f"subset sizes must be below the {outcomes.shape[1]} tasks, got {cli_arguments.subset_sizes}"
        )
    else:
        task_cost_share = costed_agent_task_cost_usd.div(
            costed_agent_task_cost_usd.sum(axis=1), axis=0
        ).median(axis=0)
        task_cost_share = task_cost_share / task_cost_share.sum()
    num_trials_by_cell = runs.groupby(["agent_id", "task_id"]).size()
    trials_per_task = float(num_trials_by_cell.mean())
    cutoff_date = benchmark.agents["submit_date"].quantile(
        cli_arguments.selection_fraction
    )
    logger.info(
        f"{benchmark.name}: {outcomes.shape[0]} agents x {outcomes.shape[1]} tasks, {trials_per_task:.2f} trials per task on average, "
        f"cost profile from {len(costed_agent_task_cost_usd)} agents with a cost on every task, date cutoff {cutoff_date:%Y-%m}"
    )

    def _purpose_generator(repeat: int, purpose: str) -> np.random.Generator:
        return purpose_generator(
            seed=cli_arguments.seed, repeat=repeat, purpose=purpose
        )

    rows = []
    for repeat in range(cli_arguments.num_repeats):
        for split_kind in ("random", "date"):
            if split_kind == "random":
                agent_split: AgentSplit = split_random(
                    benchmark.agents,
                    selection_fraction=cli_arguments.selection_fraction,
                    random_generator=_purpose_generator(repeat, "random_split"),
                )
                selection_agent_ids = agent_split.selection_agent_ids
            else:
                agent_split = split_by_date(benchmark.agents, cutoff_date=cutoff_date)
                # note (Haoling Pu): as in the core loop, the fixed date split's selection agents are bootstrapped to give a spread across repeats.
                selection_agent_ids = (
                    _purpose_generator(repeat, "date_selection_bootstrap")
                    .choice(
                        agent_split.selection_agent_ids,
                        size=len(agent_split.selection_agent_ids),
                        replace=True,
                    )
                    .tolist()
                )
            test_outcomes = outcomes.loc[agent_split.test_agent_ids]
            selection_group_fit = fit_selection_group(
                outcomes.loc[selection_agent_ids],
                random_generator=_purpose_generator(
                    repeat, f"tiny_benchmarks_fit/{split_kind}"
                ),
            )
            # note (Haoling Pu): each test agent's trials of a task are shuffled once per repeat; keeping the first t is a random t-trial run.
            test_runs = runs[runs["agent_id"].isin(agent_split.test_agent_ids)].copy()
            test_runs["trial_order"] = _purpose_generator(
                repeat, f"trial_order/{split_kind}"
            ).random(len(test_runs))
            test_runs["trial_rank"] = test_runs.groupby(["agent_id", "task_id"])[
                "trial_order"
            ].rank(method="first")
            test_num_trials_by_cell = num_trials_by_cell.loc[agent_split.test_agent_ids]
            # note (Haoling Pu): subsets are selected once per split, so every trial count scores the same subsets (a paired comparison).
            task_weights_by_method_and_size = {
                (selection_method, subset_size): select_task_weights(
                    selection_method,
                    selection_group_fit,
                    subset_size=subset_size,
                    random_generator=_purpose_generator(
                        repeat,
                        f"selection/{split_kind}/{selection_method}/{subset_size}",
                    ),
                )
                for selection_method in SELECTION_METHODS
                for subset_size in cli_arguments.subset_sizes
            }
            observed_outcomes_by_trial_count = {"all": test_outcomes}
            for trial_count in cli_arguments.trial_counts:
                observed_outcomes_by_trial_count[str(trial_count)] = (
                    test_runs[test_runs["trial_rank"] <= trial_count]
                    .pivot_table(
                        index="agent_id",
                        columns="task_id",
                        values="resolved",
                        aggfunc="mean",
                    )
                    .loc[test_outcomes.index, test_outcomes.columns]
                )
            for (
                trial_count,
                observed_outcomes,
            ) in observed_outcomes_by_trial_count.items():
                if trial_count == "all":
                    trial_share = 1.0
                else:
                    trial_share = (
                        test_num_trials_by_cell.clip(upper=int(trial_count)).sum()
                        / test_num_trials_by_cell.sum()
                    )
                every_task_score = score_subset(
                    test_outcomes,
                    observed_outcomes.mean(axis=1),
                    top_k=cli_arguments.top_k,
                )
                rows.append(
                    {
                        "split": split_kind,
                        "repeat": repeat,
                        "method": "every_task",
                        "subset_size": outcomes.shape[1],
                        "trial_count": trial_count,
                        "task_share": 1.0,
                        "cost_share": trial_share,
                        "kendall_tau": every_task_score.kendall_tau,
                        "top_k_overlap": every_task_score.top_k_overlap,
                    }
                )
                for (
                    selection_method,
                    subset_size,
                ), task_weights in task_weights_by_method_and_size.items():
                    subset_cost_share = task_cost_share[task_weights.index].sum()
                    estimated_solve_rate_by_method = estimate_solve_rates(
                        selection_method,
                        task_weights,
                        observed_outcomes,
                        selection_group_fit,
                        subset_size=subset_size,
                    )
                    if trial_count == "all":
                        pass
                    else:
                        estimated_solve_rate_by_method = {
                            selection_method: estimated_solve_rate_by_method[
                                selection_method
                            ]
                        }
                    for (
                        method_name,
                        estimated_solve_rate,
                    ) in estimated_solve_rate_by_method.items():
                        subset_score = score_subset(
                            test_outcomes,
                            estimated_solve_rate,
                            top_k=cli_arguments.top_k,
                        )
                        rows.append(
                            {
                                "split": split_kind,
                                "repeat": repeat,
                                "method": method_name,
                                "subset_size": subset_size,
                                "trial_count": trial_count,
                                "task_share": len(task_weights) / outcomes.shape[1],
                                "cost_share": subset_cost_share * trial_share,
                                "kendall_tau": subset_score.kendall_tau,
                                "top_k_overlap": subset_score.top_k_overlap,
                            }
                        )
        logger.info(f"repeat {repeat} done")

    results = pd.DataFrame(rows)
    output_path = cli_arguments.output_dir / f"{cli_arguments.benchmark}_cost.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"wrote {len(results)} rows to {output_path}")
    summary = results.groupby(
        ["split", "trial_count", "subset_size", "method"], sort=True
    )[["task_share", "cost_share", "kendall_tau"]].mean()
    for split_kind in ("random", "date"):
        print(f"\n{split_kind} split: Kendall tau (rows: trials per task, subset size)")
        print(
            summary.loc[split_kind]["kendall_tau"]
            .unstack("method")
            .round(3)
            .to_string()
        )
        every_trial = summary.loc[(split_kind, "all")]
        print(
            f"\n{split_kind} split, every trial: dollars spent / tasks kept (1 = an average task)"
        )
        print(
            (every_trial["cost_share"] / every_trial["task_share"])
            .unstack("method")
            .round(2)
            .to_string()
        )


if __name__ == "__main__":
    main()
