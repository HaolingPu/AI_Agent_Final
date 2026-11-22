"""Run the core loop on toy data: split agents, select tasks, score the subset against the noise ceiling."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from subset_eval.data import outcome_matrix
from subset_eval.metrics import noise_ceiling, score_subset
from subset_eval.select import (
    PASS_RATE_BAND,
    SelectionMethod,
    select_pass_rate_band,
    select_random,
)
from subset_eval.split import AgentSplit, SplitKind, split_by_date, split_random
from subset_eval.toy import make_toy_benchmark

logger = logging.getLogger(__name__)

SPLIT_KINDS: tuple[SplitKind, ...] = ("random", "date")
SELECTION_METHODS: tuple[SelectionMethod, ...] = ("random", "pass_rate_band")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--num-bootstrap", type=int, default=200)
    parser.add_argument(
        "--subset-sizes", type=int, nargs="+", default=[10, 25, 50, 100]
    )
    parser.add_argument("--selection-fraction", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("results/toy_core_loop.csv")
    )
    cli_arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    benchmark = make_toy_benchmark(
        num_agents=120,
        num_tasks=300,
        num_trials=1,
        num_months=24,
        ability_gain_per_month=0.1,
        seed=cli_arguments.seed,
    )
    outcomes = outcome_matrix(benchmark)
    # note (Haoling Pu): the date cutoff keeps the same selection fraction as the random split, so only the split kind differs.
    cutoff_date = benchmark.agents["submit_date"].quantile(
        cli_arguments.selection_fraction
    )
    logger.info(
        f"{benchmark.name}: {outcomes.shape[0]} agents x {outcomes.shape[1]} tasks, date cutoff {cutoff_date:%Y-%m}"
    )

    rows = []
    for repeat in range(cli_arguments.num_repeats):
        # note (Haoling Pu): one generator per repeat, and the same random subsets for both splits, so the two splits are a paired comparison.
        # note (Haoling Pu): one stream per purpose, so adding a method does not shift the splits or the ceilings.
        split_generator, ceiling_generator, selection_generator = np.random.default_rng(
            [cli_arguments.seed, repeat]
        ).spawn(3)
        random_subset_task_ids_by_size = {
            subset_size: select_random(
                outcomes.columns,
                subset_size=subset_size,
                random_generator=selection_generator,
            )
            for subset_size in cli_arguments.subset_sizes
        }
        for split_kind in SPLIT_KINDS:
            if split_kind == "random":
                agent_split: AgentSplit = split_random(
                    benchmark.agents,
                    selection_fraction=cli_arguments.selection_fraction,
                    random_generator=split_generator,
                )
            else:
                agent_split = split_by_date(benchmark.agents, cutoff_date=cutoff_date)
            test_outcomes = outcomes.loc[agent_split.test_agent_ids]
            ceiling = noise_ceiling(
                test_outcomes,
                num_bootstrap=cli_arguments.num_bootstrap,
                random_generator=ceiling_generator,
            )
            selection_outcomes = outcomes.loc[agent_split.selection_agent_ids]
            # note (Haoling Pu): when k exceeds this count, the band subset is filled with near-band tasks and is no longer Ndzomga's method.
            num_tasks_in_band = int(
                selection_outcomes.mean(axis=0).between(*PASS_RATE_BAND).sum()
            )
            for selection_method in SELECTION_METHODS:
                for subset_size in cli_arguments.subset_sizes:
                    if selection_method == "random":
                        subset_task_ids = random_subset_task_ids_by_size[subset_size]
                    elif selection_method == "pass_rate_band":
                        subset_task_ids = select_pass_rate_band(
                            selection_outcomes,
                            subset_size=subset_size,
                            random_generator=selection_generator,
                        )
                    else:
                        raise ValueError(f"unknown selection method {selection_method}")
                    subset_score = score_subset(
                        test_outcomes, subset_task_ids, top_k=cli_arguments.top_k
                    )
                    rows.append(
                        {
                            "split": split_kind,
                            "repeat": repeat,
                            "method": selection_method,
                            "subset_size": subset_size,
                            "num_test_agents": len(agent_split.test_agent_ids),
                            "num_tasks_in_band": num_tasks_in_band,
                            "kendall_tau": subset_score.kendall_tau,
                            "top_k_overlap": subset_score.top_k_overlap,
                            "ceiling_low": ceiling.kendall_tau_low,
                            "ceiling_high": ceiling.kendall_tau_high,
                        }
                    )

    results = pd.DataFrame(rows)
    cli_arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cli_arguments.output, index=False)
    logger.info(f"wrote {len(results)} rows to {cli_arguments.output}")
    # note (Haoling Pu): methods are compared within one split; newer agents are closer together, so raw tau across splits is not comparable.
    kendall_tau_by_method = results.pivot_table(
        index=["split", "subset_size"], columns="method", values="kendall_tau"
    )
    kendall_tau_by_method["band_minus_random"] = (
        kendall_tau_by_method["pass_rate_band"] - kendall_tau_by_method["random"]
    )
    ceiling_by_split = results.groupby(["split", "subset_size"])[
        ["ceiling_low", "ceiling_high", "num_tasks_in_band"]
    ].mean()
    kendall_tau_by_method = kendall_tau_by_method.join(ceiling_by_split)
    print(kendall_tau_by_method.round(3).to_string())


if __name__ == "__main__":
    main()
