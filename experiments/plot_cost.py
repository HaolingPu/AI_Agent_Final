"""Cost figure: Kendall tau against the share of a full evaluation's dollars, one panel per benchmark.

Solid lines: a selected task subset run with every trial, one point per subset size.
Black dashed line: every task run with fewer trials, one point per trial count (the right end is every task, every trial: tau 1 at cost 1).
A point above and left of another ranks better for less money. Bars are the central 95 % of tau across repeats.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

# note (Haoling Pu): same colour and marker per method as plot_decay.py; only the methods the text discusses.
METHOD_STYLES = {
    "random": ("#2a78d6", "o"),
    "stratified": ("#eb6834", "s"),
    "irt_max_information": ("#eda100", "D"),
    "irt_frontier": ("#8a5cd6", "X"),
}
EVERY_TASK_COLOR = "#2b2a27"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e8e7e3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost-csv",
        type=Path,
        nargs="+",
        required=True,
        help="one run_cost.py output per benchmark",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True, help="one panel title per csv"
    )
    parser.add_argument("--split", choices=("random", "date"), default="date")
    parser.add_argument("--output", type=Path, required=True)
    cli_arguments = parser.parse_args()
    if len(cli_arguments.labels) != len(cli_arguments.cost_csv):
        raise ValueError("give one label per --cost-csv")
    else:
        pass

    plt.rcParams.update(
        {
            "font.size": 9,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.edgecolor": GRID_COLOR,
        }
    )
    figure, axes = plt.subplots(
        1,
        len(cli_arguments.cost_csv),
        figsize=(4.2 * len(cli_arguments.cost_csv), 3.6),
        sharey=True,
        squeeze=False,
    )
    num_repeats_by_panel = []
    for axis, cost_csv, label in zip(
        axes[0], cli_arguments.cost_csv, cli_arguments.labels
    ):
        results = pd.read_csv(cost_csv)
        results = results[results["split"] == cli_arguments.split]
        num_repeats_by_panel.append(results["repeat"].nunique())
        # note (Haoling Pu): per point, the mean cost share and tau over repeats, and tau's 2.5 / 97.5 percentiles.
        summary = (
            results.groupby(["method", "subset_size", "trial_count"])
            .agg(
                cost_share=("cost_share", "mean"),
                kendall_tau=("kendall_tau", "mean"),
                tau_low=("kendall_tau", lambda tau: tau.quantile(0.025)),
                tau_high=("kendall_tau", lambda tau: tau.quantile(0.975)),
            )
            .reset_index()
        )
        every_task = summary[summary["method"] == "every_task"].sort_values(
            "cost_share"
        )
        curves = [
            (
                every_task,
                EVERY_TASK_COLOR,
                "*",
                "--",
                "every task, fewer trials",
            )
        ]
        for selection_method, (color, marker) in METHOD_STYLES.items():
            curves.append(
                (
                    summary[
                        (summary["method"] == selection_method)
                        & (summary["trial_count"] == "all")
                    ].sort_values("cost_share"),
                    color,
                    marker,
                    "-",
                    selection_method.replace("_", " "),
                )
            )
        for curve, color, marker, line_style, curve_label in curves:
            axis.errorbar(
                curve["cost_share"],
                curve["kendall_tau"],
                yerr=[
                    curve["kendall_tau"] - curve["tau_low"],
                    curve["tau_high"] - curve["kendall_tau"],
                ],
                fmt=line_style,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.5,
                elinewidth=0.8,
                capsize=2,
                label=curve_label,
            )
        axis.set_xscale("log")
        # note (Haoling Pu): plain decimals at fixed shares; matplotlib's log labels overlap below 1.
        axis.set_xticks(
            [0.05, 0.1, 0.2, 0.5, 1.0], labels=["0.05", "0.1", "0.2", "0.5", "1"]
        )
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_xlabel("share of the full evaluation's dollars")
        axis.set_title(label, color=TEXT_COLOR)
        axis.grid(color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(f"Kendall tau vs full ranking ({cli_arguments.split} split)")
    if len(set(num_repeats_by_panel)) != 1:
        raise ValueError(
            f"the csvs have different repeat counts {num_repeats_by_panel}; rerun them alike"
        )
    else:
        pass
    # note (Haoling Pu): below the panels, as in plot_decay.py; the one-trial SWE-bench panel has no dashed curve to label.
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(legend_labels),
    )
    figure.text(
        0.0,
        -0.13,
        f"x: mean cost share over {num_repeats_by_panel[0]} repeats; y: mean tau, bars from its 2.5th to 97.5th percentile over repeats. "
        "A task's cost is the median share of a costed agent's spend (SWE-bench: mini-SWE-agent runs only), so k hard tasks can cost more than k / N.\n"
        "Fewer trials: cost counts trials. One-trial benchmarks have no fewer-trial curve. Subsets run with fewer trials sit in run_cost.py's table, not here.",
        fontsize=7,
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    figure.tight_layout()
    cli_arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(cli_arguments.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
