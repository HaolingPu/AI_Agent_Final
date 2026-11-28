"""Cross-benchmark tables from run_core_loop.py outputs, as markdown, for the report.

C1: the same plain random subsets scored on a random split vs a date split (paired by repeat: same subsets, different test agents).
Plain random never reads the selection agents, so this gap is how much harder the newest agents are to rank than a random 30 %
(they sit closer together), not staleness; the noise ceiling's own drop is shown next to it.
X1 / X5: each method minus random on the same split and repeat (paired: same selection and test agents), per subset size.
Intervals: 95 % bootstrap intervals of the mean over repeats. On the date and weak-to-strong splits the test agents are fixed,
so repeats vary only the selection-agent bootstrap and the methods' seeds: the intervals hold for these agents, not new ones,
and are not corrected for the many comparisons.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from subset_eval.seeding import purpose_generator

NUM_BOOTSTRAP = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="names whose results/<name>_core_loop.csv exist",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["irt_frontier", "irt_max_information", "stratified", "anchor_irt"],
        help="methods compared with random in the X1 / X5 table",
    )
    parser.add_argument("--seed", type=int, default=0)
    cli_arguments = parser.parse_args()

    def _interval(per_repeat_values: pd.Series, purpose: str) -> str:
        # note (Haoling Pu): one random stream per cell, so adding a benchmark or method leaves the other intervals unchanged.
        bootstrap_means = (
            purpose_generator(seed=cli_arguments.seed, repeat=0, purpose=purpose)
            .choice(
                per_repeat_values.to_numpy(),
                size=(NUM_BOOTSTRAP, len(per_repeat_values)),
                replace=True,
            )
            .mean(axis=1)
        )
        low, high = np.quantile(bootstrap_means, [0.025, 0.975])
        return f"{per_repeat_values.mean():+.3f} [{low:+.3f}, {high:+.3f}]"

    def _markdown(rows: list[dict[str, str]]) -> str:
        # note (Haoling Pu): pandas' to_markdown needs the tabulate package; a pipe table is three lines of code.
        header = list(rows[0])
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += [
            "| " + " | ".join(row[column] for column in header) + " |" for row in rows
        ]
        return "\n".join(lines)

    c1_rows = []
    method_rows = []
    for benchmark_name in cli_arguments.benchmarks:
        results = pd.read_csv(
            cli_arguments.results_dir / f"{benchmark_name}_core_loop.csv"
        )
        # note (Haoling Pu): one tau per (split, size, repeat, method); the gp-IRT estimates are their own method names.
        kendall_tau_table = results.pivot_table(
            index=["split", "subset_size", "repeat"],
            columns="method",
            values="kendall_tau",
        )
        ceiling_low_by_split = results.groupby("split")["ceiling_low"].mean()
        num_test_agents_by_split = results.groupby("split")["num_test_agents"].mean()
        for subset_size in sorted(results["subset_size"].unique()):
            random_split_tau = kendall_tau_table.loc[("random", subset_size), "random"]
            date_split_tau = kendall_tau_table.loc[("date", subset_size), "random"]
            c1_rows.append(
                {
                    "benchmark": benchmark_name,
                    "k": str(subset_size),
                    "random split": f"{random_split_tau.mean():.3f}",
                    "date split": f"{date_split_tau.mean():.3f}",
                    "date - random": _interval(
                        date_split_tau - random_split_tau,
                        f"{benchmark_name}/c1/{subset_size}",
                    ),
                    "ceiling low, date - random": f"{ceiling_low_by_split['date'] - ceiling_low_by_split['random']:+.2f}",
                    "test agents (random / date)": f"{num_test_agents_by_split['random']:.0f} / {num_test_agents_by_split['date']:.0f}",
                }
            )
            for split_kind in ("random", "date", "weak_to_strong"):
                split_tau = kendall_tau_table.loc[(split_kind, subset_size)]
                method_row = {
                    "benchmark": benchmark_name,
                    "split": split_kind,
                    "k": str(subset_size),
                    "test agents": f"{num_test_agents_by_split[split_kind]:.0f}",
                    "random": f"{split_tau['random'].mean():.3f}",
                }
                for selection_method in cli_arguments.methods:
                    method_row[f"{selection_method} - random"] = _interval(
                        split_tau[selection_method] - split_tau["random"],
                        f"{benchmark_name}/{split_kind}/{subset_size}/{selection_method}",
                    )
                method_rows.append(method_row)

    print("## C1: random subsets, random split vs date split\n")
    print(_markdown(c1_rows))
    print("\n## X1 / X5: method minus random, same split and repeat\n")
    print(_markdown(method_rows))


if __name__ == "__main__":
    main()
