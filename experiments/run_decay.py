"""C2 decay: freeze each method's subset at a cutoff date, then compare it on every later window of agents with a subset refreshed just before that window.

The refreshed subset is chosen the same way from every agent submitted before the window starts, and both are scored on the same test agents.
So their tau gap measures staleness alone, not which agents happen to fall in a window.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from subset_eval.data import outcome_matrix
from subset_eval.decay import shelf_life_months, window_start_month
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

    benchmark = make_toy_benchmark(seed=cli_arguments.seed)
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
    start_month_by_agent = window_start_month(
        submit_date,
        cutoff_date=cutoff_date,
        window_months=cli_arguments.window_months,
    )
    test_agent_ids_by_window = {}
    for start_month in range(
        0, int(start_month_by_agent.max()) + 1, cli_arguments.window_months
    ):
        window_agent_ids = start_month_by_agent.index[
            start_month_by_agent == start_month
        ].tolist()
        # note (Haoling Pu): tau over a handful of agents is mostly noise, so sparse windows are left out.
        if len(window_agent_ids) < cli_arguments.min_test_agents:
            logger.info(
                f"skip window at month {start_month}: {len(window_agent_ids)} agents < {cli_arguments.min_test_agents}"
            )
        else:
            test_agent_ids_by_window[start_month] = window_agent_ids
    # note (Haoling Pu): a window's subset is refreshed from every agent before its first day; window 0 starts at the cutoff, so its refreshed subset is the frozen one.
    selection_date_by_window = {
        start_month: (
            cutoff_date
            if start_month == 0
            else (cutoff_date.to_period("M") + start_month).start_time
        )
        for start_month in test_agent_ids_by_window
    }
    selection_dates = sorted({cutoff_date} | set(selection_date_by_window.values()))
    logger.info(
        f"{benchmark.name}: cutoff {cutoff_date:%Y-%m-%d}, test agents per window "
        f"{({start: len(ids) for start, ids in test_agent_ids_by_window.items()})}"
    )

    def _purpose_generator(repeat: int, purpose: str) -> np.random.Generator:
        return purpose_generator(
            seed=cli_arguments.seed, repeat=repeat, purpose=purpose
        )

    rows = []
    for repeat in range(cli_arguments.num_repeats):
        selection_group_fit_by_date = {}
        for selection_date in selection_dates:
            selection_agent_ids = submit_date.index[
                submit_date < selection_date
            ].tolist()
            # note (Haoling Pu): the selection agents are fixed by the date, so they are bootstrapped to give each method a real spread across repeats.
            bootstrap_selection_agent_ids = (
                _purpose_generator(
                    repeat, f"decay_selection_bootstrap/{selection_date:%Y-%m-%d}"
                )
                .choice(
                    selection_agent_ids, size=len(selection_agent_ids), replace=True
                )
                .tolist()
            )
            selection_group_fit_by_date[selection_date] = fit_selection_group(
                outcomes.loc[bootstrap_selection_agent_ids],
                random_generator=_purpose_generator(
                    repeat, f"decay_fit/{selection_date:%Y-%m-%d}"
                ),
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
                # note (Haoling Pu): every selection date reuses the same random stream, so frozen and refreshed subsets differ only through the agents' data; random subsets come out identical.
                task_weights_by_date = {
                    selection_date: select_task_weights(
                        selection_method,
                        selection_group_fit,
                        subset_size=subset_size,
                        random_generator=_purpose_generator(
                            repeat, f"decay_selection/{selection_method}/{subset_size}"
                        ),
                    )
                    for selection_date, selection_group_fit in selection_group_fit_by_date.items()
                }
                for start_month, test_agent_ids in test_agent_ids_by_window.items():
                    test_outcomes = outcomes.loc[test_agent_ids]
                    selection_date = selection_date_by_window[start_month]
                    frozen_estimates = estimate_solve_rates(
                        selection_method,
                        task_weights_by_date[cutoff_date],
                        test_outcomes,
                        selection_group_fit_by_date[cutoff_date],
                        subset_size=subset_size,
                    )
                    refreshed_estimates = estimate_solve_rates(
                        selection_method,
                        task_weights_by_date[selection_date],
                        test_outcomes,
                        selection_group_fit_by_date[selection_date],
                        subset_size=subset_size,
                    )
                    for method_name, frozen_solve_rate in frozen_estimates.items():
                        frozen_score = score_subset(
                            test_outcomes, frozen_solve_rate, top_k=cli_arguments.top_k
                        )
                        refreshed_score = score_subset(
                            test_outcomes,
                            refreshed_estimates[method_name],
                            top_k=cli_arguments.top_k,
                        )
                        rows.append(
                            {
                                "repeat": repeat,
                                "method": method_name,
                                "subset_size": subset_size,
                                "window_start_month": start_month,
                                "num_test_agents": len(test_agent_ids),
                                "frozen_kendall_tau": frozen_score.kendall_tau,
                                "refreshed_kendall_tau": refreshed_score.kendall_tau,
                                "kendall_tau_gap": frozen_score.kendall_tau
                                - refreshed_score.kendall_tau,
                                "frozen_top_k_overlap": frozen_score.top_k_overlap,
                                "refreshed_top_k_overlap": refreshed_score.top_k_overlap,
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

    def _by_window(value_column: str) -> pd.DataFrame:
        return results.pivot_table(
            index=["subset_size", "method"],
            columns="window_start_month",
            values=value_column,
        ).sort_index(axis=1)

    mean_gap = _by_window("kendall_tau_gap")
    print(
        "Frozen subset, mean Kendall tau per window (columns: months after the cutoff):"
    )
    print(_by_window("frozen_kendall_tau").round(3).to_string())
    print("\nNoise ceiling of each window:")
    print(
        results.groupby("window_start_month")[["ceiling_low", "ceiling_high"]]
        .mean()
        .T.round(3)
        .to_string()
    )
    print(
        "\nMean tau gap, frozen minus refreshed (negative = the frozen subset went stale):"
    )
    print(mean_gap.round(3).to_string())
    print(
        "\nShelf life (months after the cutoff; NaN = never behind two windows in a row):"
    )
    print(shelf_life_months(mean_gap).unstack("subset_size").to_string())


if __name__ == "__main__":
    main()
