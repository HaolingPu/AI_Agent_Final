"""Run the core loop on toy data: split agents, select tasks, score the subset against the noise ceiling."""

import argparse
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from subset_eval.data import outcome_matrix
from subset_eval.irt import fit_irt
from subset_eval.metrics import noise_ceiling, score_subset
from subset_eval.select import (
    PASS_RATE_BAND,
    SelectionMethod,
    select_irt_max_information,
    select_pass_rate_band,
    select_random,
    select_stratified,
)
from subset_eval.split import (
    AgentSplit,
    SplitKind,
    split_by_date,
    split_random,
    split_weak_to_strong,
)
from subset_eval.tiny_benchmarks import (
    fit_tiny_benchmarks,
    gp_irt_solve_rate,
    select_anchor_points,
)
from subset_eval.toy import make_toy_benchmark

logger = logging.getLogger(__name__)

SPLIT_KINDS: tuple[SplitKind, ...] = ("random", "date", "weak_to_strong")
SELECTION_METHODS: tuple[SelectionMethod, ...] = (
    "random",
    "stratified",
    "pass_rate_band",
    "irt_max_information",
    "anchor_correctness",
    "anchor_irt",
)
# note (Haoling Pu): tinyBenchmarks reports these three strategies with and without gp-IRT; they also get a "<method>_gp_irt" row.
GP_IRT_METHODS: tuple[SelectionMethod, ...] = (
    "random",
    "anchor_correctness",
    "anchor_irt",
)


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
        # note (Haoling Pu): each purpose seeds from its own name, so adding, removing or reordering methods or sizes never changes the other numbers.
        return np.random.default_rng(
            [cli_arguments.seed, repeat, zlib.crc32(purpose.encode())]
        )

    rows = []
    for repeat in range(cli_arguments.num_repeats):
        # note (Haoling Pu): the same random subsets serve both splits, so the two splits are a paired comparison.
        random_subset_task_ids_by_size = {
            subset_size: select_random(
                outcomes.columns,
                subset_size=subset_size,
                random_generator=_purpose_generator(
                    repeat, f"random_subsets/{subset_size}"
                ),
            )
            for subset_size in cli_arguments.subset_sizes
        }
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
                agent_split = split_weak_to_strong(
                    outcomes.mean(axis=1),
                    selection_fraction=cli_arguments.selection_fraction,
                )
            else:
                raise ValueError(f"unknown split kind {split_kind}")
            if split_kind == "random":
                selection_agent_ids = agent_split.selection_agent_ids
            else:
                # note (Haoling Pu): the date and weak-to-strong splits are the same every repeat, so their selection agents are bootstrapped to give each method a real spread.
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
            selection_outcomes = outcomes.loc[selection_agent_ids]
            # note (Haoling Pu): when k exceeds this count, the band subset is filled with near-band tasks and is no longer Ndzomga's method.
            num_tasks_in_band = int(
                selection_outcomes.mean(axis=0).between(*PASS_RATE_BAND).sum()
            )
            selection_irt_parameters = fit_irt(selection_outcomes)
            tiny_benchmarks_fit = fit_tiny_benchmarks(
                selection_outcomes,
                random_generator=_purpose_generator(
                    repeat, f"tiny_benchmarks_fit/{split_kind}"
                ),
            )
            # note (Haoling Pu): an IRT anchor embeds each task as its discrimination vector plus its difficulty, unscaled, as in the authors' code.
            irt_task_embedding = tiny_benchmarks_fit.irt.discrimination.assign(
                difficulty=tiny_benchmarks_fit.irt.difficulty
            )
            for selection_method in SELECTION_METHODS:
                for subset_size in cli_arguments.subset_sizes:
                    selection_purpose = (
                        f"selection/{split_kind}/{selection_method}/{subset_size}"
                    )
                    # note (Haoling Pu): every method yields task weights summing to 1; all but the anchor methods weigh their tasks equally.
                    if selection_method == "random":
                        task_weights = pd.Series(
                            1.0 / subset_size,
                            index=random_subset_task_ids_by_size[subset_size],
                        )
                    elif selection_method == "stratified":
                        task_weights = pd.Series(
                            1.0 / subset_size,
                            index=select_stratified(
                                selection_outcomes,
                                subset_size=subset_size,
                                random_generator=_purpose_generator(
                                    repeat, selection_purpose
                                ),
                            ),
                        )
                    elif selection_method == "pass_rate_band":
                        task_weights = pd.Series(
                            1.0 / subset_size,
                            index=select_pass_rate_band(
                                selection_outcomes,
                                subset_size=subset_size,
                                random_generator=_purpose_generator(
                                    repeat, selection_purpose
                                ),
                            ),
                        )
                    elif selection_method == "irt_max_information":
                        task_weights = pd.Series(
                            1.0 / subset_size,
                            index=select_irt_max_information(
                                selection_irt_parameters, subset_size=subset_size
                            ),
                        )
                    elif selection_method == "anchor_correctness":
                        task_weights = select_anchor_points(
                            selection_outcomes.T,
                            subset_size=subset_size,
                            random_generator=_purpose_generator(
                                repeat, selection_purpose
                            ),
                        )
                    elif selection_method == "anchor_irt":
                        task_weights = select_anchor_points(
                            irt_task_embedding,
                            subset_size=subset_size,
                            random_generator=_purpose_generator(
                                repeat, selection_purpose
                            ),
                        )
                    else:
                        raise ValueError(f"unknown selection method {selection_method}")
                    estimated_solve_rate_by_method = {
                        selection_method: test_outcomes[task_weights.index]
                        @ task_weights
                    }
                    if selection_method in GP_IRT_METHODS:
                        estimated_solve_rate_by_method[f"{selection_method}_gp_irt"] = (
                            gp_irt_solve_rate(
                                test_outcomes,
                                task_weights,
                                tiny_benchmarks_fit,
                                subset_size=subset_size,
                                anchor_points=selection_method != "random",
                            )
                        )
                    else:
                        pass
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
                                "num_subset_tasks": len(task_weights),
                                "num_test_agents": len(agent_split.test_agent_ids),
                                "num_tasks_in_band": num_tasks_in_band,
                                "irt_dimension": tiny_benchmarks_fit.irt.discrimination.shape[
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
    cli_arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cli_arguments.output, index=False)
    logger.info(f"wrote {len(results)} rows to {cli_arguments.output}")
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
