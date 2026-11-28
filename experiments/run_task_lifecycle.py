"""Why a frozen subset goes stale (C2 mechanism, X2): the share of its tasks that each later window's agents have used up.

A task stops telling a window's agents apart when their mean solve rate on it is at least 0.9 ("used up") or 0 ("out of reach").
With one trial per task (SWE-bench) that is the share of the window's agents that solve it; with 5 trials (Terminal-Bench) it is stricter,
so a method's shares are read only against the every-task row of the same window and benchmark (windows differ in size too).
Per later window, this reports both shares for each method's subset frozen at the cutoff.
Subsets are selected exactly as in run_decay.py (same random streams), so repeat r freezes the same subset as decay's repeat r.
"""

import argparse
import logging
from pathlib import Path
from typing import get_args

import pandas as pd

from subset_eval.data import outcome_matrix
from subset_eval.decay import window_start_month
from subset_eval.loaders import BenchmarkName, load_benchmark
from subset_eval.methods import (
    SELECTION_METHODS,
    fit_selection_group,
    select_task_weights,
)
from subset_eval.seeding import purpose_generator

logger = logging.getLogger(__name__)

# note (Haoling Pu): "used up" = a window's mean solve rate on the task is at least this; "out of reach" = it is 0.
USED_UP_SOLVE_SHARE = 0.9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[25, 100])
    parser.add_argument("--cutoff-date", type=pd.Timestamp, default="2024-12-31")
    parser.add_argument("--window-months", type=int, default=3)
    parser.add_argument("--min-test-agents", type=int, default=8)
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
    outcomes = outcome_matrix(benchmark)
    submit_date = benchmark.agents.set_index("agent_id")["submit_date"]
    start_month_by_agent = window_start_month(
        submit_date,
        cutoff_date=cli_arguments.cutoff_date,
        window_months=cli_arguments.window_months,
    )
    window_agent_ids = {}
    for start_month, agent_ids in start_month_by_agent.groupby(
        start_month_by_agent
    ).groups.items():
        if len(agent_ids) < cli_arguments.min_test_agents:
            logger.info(
                f"skip window at month {start_month}: {len(agent_ids)} agents < {cli_arguments.min_test_agents}"
            )
        else:
            window_agent_ids[start_month] = agent_ids.tolist()
    # note (Haoling Pu): one row per task, one column per window: the window's agents' mean solve rate on the task.
    window_solve_share = pd.DataFrame(
        {
            start_month: outcomes.loc[agent_ids].mean(axis=0)
            for start_month, agent_ids in window_agent_ids.items()
        }
    )
    is_used_up = window_solve_share >= USED_UP_SOLVE_SHARE
    is_out_of_reach = window_solve_share == 0
    selection_agent_ids = submit_date.index[
        submit_date < cli_arguments.cutoff_date
    ].tolist()
    logger.info(
        f"{benchmark.name}: {len(selection_agent_ids)} selection agents before {cli_arguments.cutoff_date:%Y-%m-%d}; "
        f"agents per window { ({start: len(ids) for start, ids in window_agent_ids.items()}) }"
    )

    rows = [
        {
            "repeat": -1,
            "method": "every_task",
            "subset_size": outcomes.shape[1],
            "window_start_month": start_month,
            "num_window_agents": len(window_agent_ids[start_month]),
            "used_up_share": is_used_up[start_month].mean(),
            "out_of_reach_share": is_out_of_reach[start_month].mean(),
        }
        for start_month in window_solve_share.columns
    ]
    for repeat in range(cli_arguments.num_repeats):
        bootstrap_selection_agent_ids = (
            purpose_generator(
                seed=cli_arguments.seed,
                repeat=repeat,
                purpose=f"decay_selection_bootstrap/{cli_arguments.cutoff_date:%Y-%m-%d}",
            )
            .choice(selection_agent_ids, size=len(selection_agent_ids), replace=True)
            .tolist()
        )
        selection_group_fit = fit_selection_group(
            outcomes.loc[bootstrap_selection_agent_ids],
            random_generator=purpose_generator(
                seed=cli_arguments.seed,
                repeat=repeat,
                purpose=f"decay_fit/{cli_arguments.cutoff_date:%Y-%m-%d}",
            ),
        )
        for selection_method in SELECTION_METHODS:
            for subset_size in cli_arguments.subset_sizes:
                task_weights = select_task_weights(
                    selection_method,
                    selection_group_fit,
                    subset_size=subset_size,
                    random_generator=purpose_generator(
                        seed=cli_arguments.seed,
                        repeat=repeat,
                        purpose=f"decay_selection/{selection_method}/{subset_size}",
                    ),
                )
                # note (Haoling Pu): anchor subsets weigh their tasks, so each share is weighted the same way.
                for start_month in window_solve_share.columns:
                    rows.append(
                        {
                            "repeat": repeat,
                            "method": selection_method,
                            "subset_size": subset_size,
                            "window_start_month": start_month,
                            "num_window_agents": len(window_agent_ids[start_month]),
                            "used_up_share": float(
                                is_used_up.loc[task_weights.index, start_month]
                                @ task_weights
                            ),
                            "out_of_reach_share": float(
                                is_out_of_reach.loc[task_weights.index, start_month]
                                @ task_weights
                            ),
                        }
                    )
        logger.info(f"repeat {repeat} done")

    results = pd.DataFrame(rows)
    output_path = (
        cli_arguments.output_dir / f"{cli_arguments.benchmark}_task_lifecycle.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"wrote {len(results)} rows to {output_path}")
    for share_column in ("used_up_share", "out_of_reach_share"):
        print(
            f"\n{share_column} (columns: months after the cutoff; mean over repeats):"
        )
        print(
            results.pivot_table(
                index=["subset_size", "method"],
                columns="window_start_month",
                values=share_column,
                aggfunc="mean",
            )
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
