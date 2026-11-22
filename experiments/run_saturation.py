"""C3 saturation: month by month, can the full benchmark still tell its top-k agents apart beyond task-resampling noise?

Two leaderboards per month: every agent so far, and only the agents of the last pool-months months.
The top-k of a growing pool is packed tighter by itself, so saturation counts only if the fixed-window pool shows it too.
The bootstrap resamples tasks, not runs: "separable" means separable given these runs.
"""

import argparse
import logging
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd

from subset_eval.data import outcome_matrix
from subset_eval.loaders import BenchmarkName, load_benchmark
from subset_eval.metrics import noise_ceiling, separable_pairs
from subset_eval.seeding import purpose_generator

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pool-months", type=int, default=6)
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
    output_path = cli_arguments.output_dir / f"{cli_arguments.benchmark}_saturation.csv"
    outcomes = outcome_matrix(benchmark)
    submit_date = benchmark.agents.set_index("agent_id")["submit_date"]
    num_tasks = outcomes.shape[1]
    # note (Haoling Pu): multinomial counts are the task counts of a resample with replacement; one set serves every agent and month, so all gaps are paired.
    task_counts = purpose_generator(
        seed=cli_arguments.seed, repeat=0, purpose="saturation_bootstrap"
    ).multinomial(
        num_tasks,
        np.full(num_tasks, 1.0 / num_tasks),
        size=cli_arguments.num_bootstrap,
    )
    bootstrap_solve_rate = pd.DataFrame(
        outcomes.to_numpy(dtype=float) @ task_counts.T / num_tasks,
        index=outcomes.index,
    )
    full_solve_rate = outcomes.mean(axis=1)
    pool_names = ("all_so_far", f"last_{cli_arguments.pool_months}_months")

    rows = []
    for month in pd.period_range(
        submit_date.min().to_period("M"), submit_date.max().to_period("M"), freq="M"
    ):
        for pool_name in pool_names:
            if pool_name == "all_so_far":
                is_in_pool = submit_date <= month.end_time
            else:
                is_in_pool = (submit_date <= month.end_time) & (
                    submit_date >= (month - cli_arguments.pool_months + 1).start_time
                )
            pool_agent_ids = submit_date.index[is_in_pool]
            if len(pool_agent_ids) < cli_arguments.top_k:
                logger.info(
                    f"skip {month} {pool_name}: {len(pool_agent_ids)} agents < top-k {cli_arguments.top_k}"
                )
            else:
                # note (Haoling Pu): exactly k agents; ties at the k-th place go to the lower agent id, so the top-k never depends on row order.
                top_agent_ids = (
                    full_solve_rate[pool_agent_ids]
                    .rename_axis("agent_id")
                    .reset_index(name="solve_rate")
                    .sort_values(["solve_rate", "agent_id"], ascending=[False, True])[
                        "agent_id"
                    ]
                    .head(cli_arguments.top_k)
                    .tolist()
                )
                top_bootstrap_solve_rate = bootstrap_solve_rate.loc[
                    top_agent_ids
                ].to_numpy()
                is_separable = separable_pairs(top_bootstrap_solve_rate)
                ceiling = noise_ceiling(
                    outcomes.loc[top_agent_ids],
                    num_bootstrap=cli_arguments.num_bootstrap,
                    random_generator=purpose_generator(
                        seed=cli_arguments.seed,
                        repeat=0,
                        purpose=f"saturation_ceiling/{pool_name}/{month}",
                    ),
                )
                rows.append(
                    {
                        "month": str(month),
                        "pool": pool_name,
                        "num_pool_agents": len(pool_agent_ids),
                        "top_solve_rate": full_solve_rate[top_agent_ids[0]],
                        "kth_solve_rate": full_solve_rate[top_agent_ids[-1]],
                        "median_adjacent_gap": float(
                            np.median(
                                -np.diff(full_solve_rate[top_agent_ids].to_numpy())
                            )
                        ),
                        "median_adjacent_gap_standard_deviation": float(
                            np.median(
                                np.diff(top_bootstrap_solve_rate, axis=0).std(axis=1)
                            )
                        ),
                        # note (Haoling Pu): a per-pair 95 % test, not adjusted for the number of pairs; truly tied agents add about 1 in 40 false "separable" pairs.
                        "num_separable_pairs": int(is_separable.sum()),
                        "num_pairs": len(is_separable),
                        "ceiling_low": ceiling.kendall_tau_low,
                        "ceiling_high": ceiling.kendall_tau_high,
                    }
                )

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"wrote {len(results)} rows to {output_path}")
    print(
        results.pivot(
            index="month",
            columns="pool",
            values=["num_pool_agents", "num_separable_pairs", "ceiling_low"],
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
