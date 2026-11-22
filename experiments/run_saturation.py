"""C3 saturation: month by month, can the full benchmark still tell its top-k agents apart beyond task-resampling noise?"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from subset_eval.data import outcome_matrix
from subset_eval.metrics import noise_ceiling
from subset_eval.seeding import purpose_generator
from subset_eval.toy import make_toy_benchmark

logger = logging.getLogger(__name__)

SEPARATION_PERCENTILE = 2.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("results/toy_saturation.csv")
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
    submit_date = benchmark.agents.set_index("agent_id")["submit_date"]
    num_tasks = outcomes.shape[1]
    # note (Haoling Pu): one set of task resamples serves every agent and month, so the bootstrap gaps between two agents are paired.
    task_counts = purpose_generator(
        seed=cli_arguments.seed, repeat=0, purpose="saturation_bootstrap"
    ).multinomial(
        num_tasks, np.full(num_tasks, 1.0 / num_tasks), size=cli_arguments.num_bootstrap
    )
    bootstrap_solve_rate = pd.DataFrame(
        outcomes.to_numpy(dtype=float) @ task_counts.T / num_tasks,
        index=outcomes.index,
    )
    full_solve_rate = outcomes.mean(axis=1)

    rows = []
    for month in pd.period_range(
        submit_date.min().to_period("M"), submit_date.max().to_period("M"), freq="M"
    ):
        # note (Haoling Pu): the leaderboard at a month's end holds every agent submitted so far.
        leaderboard_agent_ids = submit_date.index[submit_date <= month.end_time]
        if len(leaderboard_agent_ids) < cli_arguments.top_k:
            logger.info(
                f"skip {month}: {len(leaderboard_agent_ids)} agents < top-k {cli_arguments.top_k}"
            )
        else:
            # note (Haoling Pu): ties at the k-th place go to the lower agent id, so the top-k never depends on row order.
            top_agent_ids = (
                full_solve_rate[leaderboard_agent_ids]
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
            adjacent_gap = -np.diff(full_solve_rate[top_agent_ids].to_numpy())
            bootstrap_adjacent_gap = -np.diff(top_bootstrap_solve_rate, axis=0)
            # note (Haoling Pu): a higher-ranked agent is separable from a lower one when the lower end of their 95 % bootstrap gap interval stays above 0.
            higher_rank, lower_rank = np.triu_indices(cli_arguments.top_k, k=1)
            is_separable = (
                np.percentile(
                    top_bootstrap_solve_rate[higher_rank]
                    - top_bootstrap_solve_rate[lower_rank],
                    SEPARATION_PERCENTILE,
                    axis=1,
                )
                > 0
            )
            is_adjacent = lower_rank == higher_rank + 1
            ceiling = noise_ceiling(
                outcomes.loc[top_agent_ids],
                num_bootstrap=cli_arguments.num_bootstrap,
                random_generator=purpose_generator(
                    seed=cli_arguments.seed,
                    repeat=0,
                    purpose=f"saturation_ceiling/{month}",
                ),
            )
            rows.append(
                {
                    "month": str(month),
                    "num_leaderboard_agents": len(leaderboard_agent_ids),
                    "top_solve_rate": full_solve_rate[top_agent_ids[0]],
                    "kth_solve_rate": full_solve_rate[top_agent_ids[-1]],
                    "median_adjacent_gap": float(np.median(adjacent_gap)),
                    "median_adjacent_gap_sd": float(
                        np.median(bootstrap_adjacent_gap.std(axis=1))
                    ),
                    "num_separable_adjacent_pairs": int(
                        is_separable[is_adjacent].sum()
                    ),
                    "num_separable_pairs": int(is_separable.sum()),
                    "num_pairs": len(is_separable),
                    "ceiling_low": ceiling.kendall_tau_low,
                    "ceiling_high": ceiling.kendall_tau_high,
                }
            )

    results = pd.DataFrame(rows)
    cli_arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(cli_arguments.output, index=False)
    logger.info(f"wrote {len(results)} rows to {cli_arguments.output}")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
