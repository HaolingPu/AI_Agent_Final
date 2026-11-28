"""C3 figure: month by month, how many pairs of the top-k agents the full benchmark tells apart beyond task-resampling noise, one panel per benchmark.

Each month's point is the leaderboard at the end of that month.

Solid: every agent so far (the public leaderboard). Dashed: only the agents of the last few months, since a growing pool packs its top 10 tighter by itself.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# note (Haoling Pu): same text colours as the C2 figures; one colour per pool.
POOL_COLORS = {"all_so_far": "#2a78d6", "fixed_window": "#eb6834"}
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e8e7e3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saturation-csv",
        type=Path,
        nargs="+",
        required=True,
        help="one run_saturation.py output per benchmark",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True, help="one panel title per csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    cli_arguments = parser.parse_args()
    if len(cli_arguments.labels) != len(cli_arguments.saturation_csv):
        raise ValueError("give one label per --saturation-csv")
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
        len(cli_arguments.saturation_csv),
        figsize=(4.2 * len(cli_arguments.saturation_csv), 3.4),
        sharey=True,
        squeeze=False,
    )
    for axis, saturation_csv, label in zip(
        axes[0], cli_arguments.saturation_csv, cli_arguments.labels
    ):
        saturation = pd.read_csv(saturation_csv)
        saturation["month_start"] = pd.PeriodIndex(
            saturation["month"], freq="M"
        ).to_timestamp()
        num_pairs = int(saturation["num_pairs"].max())
        for pool_name, pool_rows in saturation.groupby("pool"):
            if pool_name == "all_so_far":
                pool_kind = "all_so_far"
                pool_label = "every agent so far"
                line_style = "-"
            else:
                pool_kind = "fixed_window"
                pool_label = f"{pool_name.replace('_', ' ')} only"
                line_style = "--"
            axis.plot(
                pool_rows["month_start"],
                pool_rows["num_separable_pairs"],
                line_style,
                color=POOL_COLORS[pool_kind],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=pool_label,
            )
        axis.set_ylim(0, num_pairs + 2)
        axis.set_title(label, color=TEXT_COLOR)
        axis.set_xlabel("leaderboard at month end")
        axis.grid(color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        # note (Haoling Pu): about six month ticks per panel, never between months.
        axis.xaxis.set_major_locator(
            mdates.MonthLocator(interval=max(1, saturation["month"].nunique() // 6))
        )
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", labelrotation=45)
        axis.legend(frameon=False, loc="upper right")
    # note (Haoling Pu): num_pairs = k (k - 1) / 2 for the top k agents.
    top_k = round((1 + (1 + 8 * num_pairs) ** 0.5) / 2)
    axes[0][0].set_ylabel(f"top-{top_k} pairs told apart (of {num_pairs})")
    figure.text(
        0.0,
        -0.02,
        "A pair is told apart when the 2.5th percentile of the better agent's bootstrap score gap (tasks resampled) is above 0; per pair, not adjusted for the number of pairs.",
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
