"""C2 figure: per subset size, the frozen subset's tau per window against the noise-ceiling band, and its gap to a refreshed subset."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from subset_eval.methods import SELECTION_METHODS

# note (Wilson Zheng): categorical slots in a fixed order, validated for colour-blind separation; each method also gets its own marker, since three colours sit below 3:1 contrast.
METHOD_STYLES = {
    "random": ("#2a78d6", "o"),
    "stratified": ("#eb6834", "s"),
    "pass_rate_band": ("#1baf7a", "^"),
    "irt_max_information": ("#eda100", "D"),
    "anchor_correctness": ("#e87ba4", "v"),
    "anchor_irt": ("#008300", "P"),
}
CEILING_COLOR = "#d6d5d0"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e8e7e3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decay_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    cli_arguments = parser.parse_args()

    results = pd.read_csv(cli_arguments.decay_csv)
    # note (Wilson Zheng): the gp-IRT variants track their base method closely; the figure keeps the six selection methods.
    results = results[results["method"].isin(SELECTION_METHODS)]
    mean_by_window = results.groupby(
        ["subset_size", "method", "window_start_month"], as_index=False
    )[["frozen_kendall_tau", "kendall_tau_gap"]].mean()
    ceiling = results.groupby("window_start_month")[
        ["ceiling_low", "ceiling_high"]
    ].mean()
    subset_sizes = sorted(results["subset_size"].unique())

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
        2,
        len(subset_sizes),
        figsize=(2.6 * len(subset_sizes), 5.2),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for column, subset_size in enumerate(subset_sizes):
        tau_axis, gap_axis = axes[0, column], axes[1, column]
        tau_axis.fill_between(
            ceiling.index,
            ceiling["ceiling_low"],
            ceiling["ceiling_high"],
            color=CEILING_COLOR,
            linewidth=0,
            label="noise ceiling (95 %)",
        )
        gap_axis.axhline(0.0, color=TEXT_COLOR, linewidth=1)
        for method, (color, marker) in METHOD_STYLES.items():
            method_means = mean_by_window[
                (mean_by_window["subset_size"] == subset_size)
                & (mean_by_window["method"] == method)
            ]
            for axis, value_column in (
                (tau_axis, "frozen_kendall_tau"),
                (gap_axis, "kendall_tau_gap"),
            ):
                axis.plot(
                    method_means["window_start_month"],
                    method_means[value_column],
                    color=color,
                    marker=marker,
                    markersize=5,
                    linewidth=1.5,
                    label=method,
                )
        tau_axis.set_title(f"{subset_size} tasks", color=TEXT_COLOR)
        gap_axis.set_xlabel("months after the cutoff")
        gap_axis.set_xticks(ceiling.index)
        for axis in (tau_axis, gap_axis):
            axis.grid(color=GRID_COLOR, linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Kendall tau, frozen subset")
    axes[1, 0].set_ylabel("tau gap, frozen − refreshed")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    if cli_arguments.title:
        figure.suptitle(cli_arguments.title, color=TEXT_COLOR)
    else:
        pass
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    cli_arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(cli_arguments.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
