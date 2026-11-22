"""C2 decay curve: freeze each method's subset at a cutoff date, then score it on every later window of agents against that window's noise ceiling."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from subset_eval.data import outcome_matrix
from subset_eval.methods import (
    SELECTION_METHODS,
    estimate_solve_rates,
    fit_selection_group,
    select_task_weights,
)
from subset_eval.metrics import noise_ceiling, score_subset
from subset_eval.seeding import purpose_generator
from subset_eval.toy import make_toy_benchmark

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--num-bootstrap", type=int, default=200)
    parser.add_argument(
        "--subset-sizes", type=int, nargs="+", default=[10, 25, 50, 100]
    )
    parser.add_argument("--cutoff-date", type=pd.Timestamp, default="2025-01-01")
    parser.add_argument("--window-months", type=int, default=3)
    parser.add_argument("--min-test-agents", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/toy_decay.csv"))
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
    submit_date = benchmark.agents.set_index("agent_id")["submit_date"]
    cutoff_date = cli_arguments.cutoff_date
    selection_agent_ids = submit_date.index[submit_date < cutoff_date].tolist()
    later_submit_date = submit_date[submit_date >= cutoff_date]
    # note (Haoling Pu): a window is window_months calendar months; window 0 starts in the cutoff's month.
    months_after_cutoff = (later_submit_date.dt.year - cutoff_date.year) * 12 + (
        later_submit_date.dt.month - cutoff_date.month
    )
    window_start_month = (
        months_after_cutoff // cli_arguments.window_months
    ) * cli_arguments.window_months
    test_agent_ids_by_window = {}
    for start_month, window_agent_ids in window_start_month.groupby(
        window_start_month
    ).groups.items():
        # note (Haoling Pu): tau over a handful of agents is mostly noise, so sparse windows are left out rather than plotted.
        if len(window_agent_ids) < cli_arguments.min_test_agents:
            logger.info(
                f"skip window at month {start_month}: {len(window_agent_ids)} agents < {cli_arguments.min_test_agents}"
            )
        else:
            test_agent_ids_by_window[int(start_month)] = window_agent_ids.tolist()
    logger.info(
        f"{benchmark.name}: {len(selection_agent_ids)} selection agents before {cutoff_date:%Y-%m-%d}; "
        f"test agents per window {({start: len(ids) for start, ids in test_agent_ids_by_window.items()})}"
    )

    def _purpose_generator(repeat: int, purpose: str) -> np.random.Generator:
        return purpose_generator(
            seed=cli_arguments.seed, repeat=repeat, purpose=purpose
        )

    rows = []
    for repeat in range(cli_arguments.num_repeats):
        # note (Haoling Pu): the selection agents are fixed by the cutoff, so they are bootstrapped to give each method a real spread across repeats.
        bootstrap_selection_agent_ids = (
            _purpose_generator(repeat, "decay_selection_bootstrap")
            .choice(selection_agent_ids, size=len(selection_agent_ids), replace=True)
            .tolist()
        )
        selection_group_fit = fit_selection_group(
            outcomes.loc[bootstrap_selection_agent_ids],
            random_generator=_purpose_generator(repeat, "decay_tiny_benchmarks_fit"),
        )
        ceiling_by_window = {
            start_month: noise_ceiling(
                outcomes.loc[test_agent_ids],
                num_bootstrap=cli_arguments.num_bootstrap,
                random_generator=_purpose_generator(
                    repeat, f"decay_ceiling/{start_month}"
                ),
            )
            for start_month, test_agent_ids in test_agent_ids_by_window.items()
        }
        for selection_method in SELECTION_METHODS:
            for subset_size in cli_arguments.subset_sizes:
                # note (Haoling Pu): the subset is chosen once and then frozen for every later window.
                task_weights = select_task_weights(
                    selection_method,
                    selection_group_fit,
                    subset_size=subset_size,
                    random_generator=_purpose_generator(
                        repeat, f"decay_selection/{selection_method}/{subset_size}"
                    ),
                )
                for start_month, test_agent_ids in test_agent_ids_by_window.items():
                    test_outcomes = outcomes.loc[test_agent_ids]
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
                                "repeat": repeat,
                                "method": method_name,
                                "subset_size": subset_size,
                                "window_start_month": start_month,
                                "num_test_agents": len(test_agent_ids),
                                "kendall_tau": subset_score.kendall_tau,
                                "top_k_overlap": subset_score.top_k_overlap,
                                "ceiling_low": ceiling_by_window[
                                    start_month
                                ].kendall_tau_low,
                                "ceiling_high": ceiling_by_window[
                                    start_month
                                ].kendall_tau_high,
                            }
                        )
        logger.info(f"repeat {repeat} done")

    results = pd.DataFrame(rows)
    cli_arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cli_arguments.output, index=False)
    logger.info(f"wrote {len(results)} rows to {cli_arguments.output}")

    mean_tau = results.pivot_table(
        index=["subset_size", "method"],
        columns="window_start_month",
        values="kendall_tau",
    )
    ceiling_low = results.groupby("window_start_month")["ceiling_low"].mean()
    print("Mean Kendall tau per window (columns: months after the cutoff):")
    print(mean_tau.round(3).to_string())
    print("\nNoise ceiling, low end:")
    print(ceiling_low.round(3).to_frame().T.to_string())
    # note (Haoling Pu): shelf life = start month of the first window whose mean tau falls below the mean ceiling; NaN = never fell below in the observed windows.
    is_below_ceiling = mean_tau.lt(ceiling_low, axis=1)
    shelf_life_months = is_below_ceiling.idxmax(axis=1).where(
        is_below_ceiling.any(axis=1)
    )
    print("\nShelf life (months after the cutoff; NaN = still inside the band):")
    print(shelf_life_months.unstack("subset_size").to_string())


if __name__ == "__main__":
    main()
