"""C1 core loop: split agents, select tasks with every method, score each subset against the noise ceiling."""

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
from subset_eval.metrics import noise_ceiling, score_subset
from subset_eval.seeding import purpose_generator
from subset_eval.select import PASS_RATE_BAND
from subset_eval.split import (
    AgentSplit,
    SplitKind,
    split_by_date,
    split_random,
    split_weak_to_strong,
)

logger = logging.getLogger(__name__)

SPLIT_KINDS: tuple[SplitKind, ...] = ("random", "date", "weak_to_strong")


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
    parser.add_argument("--benchmark", choices=get_args(BenchmarkName), default="toy")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    cli_arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    benchmark = load_benchmark(
        cli_arguments.benchmark,
        raw_data_dir=cli_arguments.raw_data_dir,
        seed=cli_arguments.seed,
    )
    output_path = cli_arguments.output_dir / f"{cli_arguments.benchmark}_core_loop.csv"
    outcomes = outcome_matrix(benchmark)
    # note (Haoling Pu): p-IRT predicts the tasks outside the subset, so a subset of every task leaves nothing to predict.
    if max(cli_arguments.subset_sizes) >= outcomes.shape[1]:
        raise ValueError(
            f"subset sizes must be below the {outcomes.shape[1]} tasks, got {cli_arguments.subset_sizes}"
        )
    else:
        pass
    # note (Haoling Pu): the date cutoff keeps the same selection fraction as the random split, so only the split kind differs.
    cutoff_date = benchmark.agents["submit_date"].quantile(
        cli_arguments.selection_fraction
    )
    logger.info(
        f"{benchmark.name}: {outcomes.shape[0]} agents x {outcomes.shape[1]} tasks, date cutoff {cutoff_date:%Y-%m}"
    )

    def _purpose_generator(repeat: int, purpose: str) -> np.random.Generator:
        return purpose_generator(
            seed=cli_arguments.seed, repeat=repeat, purpose=purpose
        )

    rows = []
    for repeat in range(cli_arguments.num_repeats):
        for split_kind in SPLIT_KINDS:
            if split_kind == "random":
                agent_split: AgentSplit = split_random(
                    benchmark.agents,
                    selection_fraction=cli_arguments.selection_fraction,
                    random_generator=_purpose_generator(repeat, "random_split"),
                )
            elif split_kind == "date":
                agent_split = split_by_date(benchmark.agents, cutoff_date=cutoff_date)
            elif split_kind == "weak_to_strong":
                # note (Haoling Pu): the groups come from the very score being ranked, so the test group's spread is compressed; read this split only relative to random, never as raw tau.
                agent_split = split_weak_to_strong(
                    outcomes.mean(axis=1),
                    selection_fraction=cli_arguments.selection_fraction,
                )
            else:
                raise ValueError(f"unknown split kind {split_kind}")
            # note (Haoling Pu): the date and weak-to-strong splits are the same every repeat, so their selection agents are bootstrapped to give each method a real spread.
            if split_kind == "random":
                selection_agent_ids = agent_split.selection_agent_ids
            else:
                selection_agent_ids = (
                    _purpose_generator(repeat, f"{split_kind}_selection_bootstrap")
                    .choice(
                        agent_split.selection_agent_ids,
                        size=len(agent_split.selection_agent_ids),
                        replace=True,
                    )
                    .tolist()
                )
            test_outcomes = outcomes.loc[agent_split.test_agent_ids]
            ceiling = noise_ceiling(
                test_outcomes,
                num_bootstrap=cli_arguments.num_bootstrap,
                random_generator=_purpose_generator(repeat, f"ceiling/{split_kind}"),
            )
            selection_group_fit = fit_selection_group(
                outcomes.loc[selection_agent_ids],
                random_generator=_purpose_generator(
                    repeat, f"tiny_benchmarks_fit/{split_kind}"
                ),
            )
            # note (Haoling Pu): when k exceeds this count, the band subset is filled with near-band tasks and is no longer Ndzomga's method.
            num_tasks_in_band = int(
                selection_group_fit.selection_outcomes.mean(axis=0)
                .between(*PASS_RATE_BAND)
                .sum()
            )
            for selection_method in SELECTION_METHODS:
                for subset_size in cli_arguments.subset_sizes:
                    # note (Haoling Pu): random subsets ignore the split's seed, so every split draws the same ones and the splits are a paired comparison.
                    if selection_method == "random":
                        selection_purpose = f"random_subsets/{subset_size}"
                    else:
                        selection_purpose = (
                            f"selection/{split_kind}/{selection_method}/{subset_size}"
                        )
                    task_weights = select_task_weights(
                        selection_method,
                        selection_group_fit,
                        subset_size=subset_size,
                        random_generator=_purpose_generator(repeat, selection_purpose),
                    )
                    for method_name, estimated_solve_rate in estimate_solve_rates(
                        selection_method,
                        task_weights,
                        test_outcomes,
                        selection_group_fit,
                        subset_size=subset_size,
                    ).items():
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
                                "num_subset_tasks": len(task_weights),
                                "num_test_agents": len(agent_split.test_agent_ids),
                                "num_tasks_in_band": num_tasks_in_band,
                                "irt_dimension": selection_group_fit.tiny_benchmarks_fit.irt.discrimination.shape[
                                    1
                                ],
                                "kendall_tau": subset_score.kendall_tau,
                                "top_k_overlap": subset_score.top_k_overlap,
                                "ceiling_low": ceiling.kendall_tau_low,
                                "ceiling_high": ceiling.kendall_tau_high,
                            }
                        )
        logger.info(f"repeat {repeat} done")

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"wrote {len(results)} rows to {output_path}")
    # note (Haoling Pu): methods are compared within one split; newer agents are closer together, so raw tau across splits is not comparable.
    kendall_tau_by_method = results.pivot_table(
        index=["split", "subset_size"], columns="method", values="kendall_tau"
    )
    ceiling_by_split = results.groupby(["split", "subset_size"])[
        ["ceiling_low", "ceiling_high", "num_tasks_in_band", "irt_dimension"]
    ].mean()
    print(kendall_tau_by_method.join(ceiling_by_split).round(3).to_string())
    print("\nKendall tau minus random:")
    print(
        kendall_tau_by_method.drop(columns="random")
        .sub(kendall_tau_by_method["random"], axis=0)
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
