"""C2 control figure: the same test window, scored by frozen subsets of different ages, one line per test window.

Across cutoffs, a later window holds the same test agents and the same refreshed subset (same selection date, same seed), so only the frozen subset's age changes along a line.
A line that falls with age shows staleness that is not explained by which agents happen to land in a window.
Each point averages the gap over every method but plain random (0 by construction) and every subset size; bars are the central 95 % of that average across repeats.
Window 0 of a cutoff starts on the cutoff day, so its agents differ from the same calendar window seen from an earlier cutoff; window 0 rows (gap 0) are left out.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# note (Haoling Pu): same categorical slots and text colours as plot_decay_summary.py.
WINDOW_STYLES = (
    ("#2a78d6", "o"),
    ("#eb6834", "s"),
    ("#1baf7a", "^"),
    ("#eda100", "D"),
    ("#8a5cd6", "v"),
)
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e8e7e3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decay-csv",
        type=Path,
        nargs="+",
        required=True,
        help="one run_decay.py output per cutoff, written to a folder named after its cutoff date (results/c2/<cutoff>/)",
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=3,
        help="the runs' --window-months; windows line up across cutoffs only when the cutoffs are a multiple of it apart",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    cli_arguments = parser.parse_args()

    gap_tables = []
    for decay_csv in cli_arguments.decay_csv:
        cutoff_date = pd.Timestamp(decay_csv.parent.name)
        results = pd.read_csv(decay_csv)
        results = results[
            (results["method"] != "random") & (results["window_start_month"] > 0)
        ]
        per_repeat_gap = (
            results.groupby(["window_start_month", "repeat"])["kendall_tau_gap"]
            .mean()
            .reset_index()
        )
        per_repeat_gap["window_start"] = [
            (cutoff_date.to_period("M") + int(start_month)).start_time
            for start_month in per_repeat_gap["window_start_month"]
        ]
        # note (Haoling Pu): the frozen subset knows agents up to the cutoff day, so a quarter-end cutoff is about 2 months old at the month-3 window's first day, not 3.
        per_repeat_gap["subset_age_months"] = (
            (per_repeat_gap["window_start"] - cutoff_date).dt.days / 30.44
        ).round().astype(int)
        gap_tables.append(per_repeat_gap)
    gap_by_repeat = pd.concat(gap_tables, ignore_index=True)
    gap = (
        gap_by_repeat.groupby(["window_start", "subset_age_months"])["kendall_tau_gap"]
        .agg(
            mean="mean",
            low=lambda repeat_gap: repeat_gap.quantile(0.025),
            high=lambda repeat_gap: repeat_gap.quantile(0.975),
        )
        .reset_index()
    )
    # note (Haoling Pu): only windows seen at two or more ages can show a trend.
    num_ages_by_window = gap.groupby("window_start")["subset_age_months"].nunique()
    shared_windows = num_ages_by_window.index[num_ages_by_window >= 2]
    if len(shared_windows) > len(WINDOW_STYLES):
        raise ValueError(
            f"{len(shared_windows)} windows are seen at two ages or more; at most {len(WINDOW_STYLES)} fit one figure"
        )
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
    figure, axis = plt.subplots(figsize=(5.5, 3.6))
    axis.axhline(0.0, color=TEXT_COLOR, linewidth=1)
    offsets = [
        0.2 * (position - (len(shared_windows) - 1) / 2)
        for position in range(len(shared_windows))
    ]
    for window_start, (color, marker), offset in zip(
        shared_windows, WINDOW_STYLES, offsets
    ):
        window_gap = gap[gap["window_start"] == window_start]
        window_end = (
            window_start.to_period("M") + cli_arguments.window_months - 1
        ).to_timestamp()
        axis.errorbar(
            window_gap["subset_age_months"] + offset,
            window_gap["mean"],
            yerr=[
                window_gap["mean"] - window_gap["low"],
                window_gap["high"] - window_gap["mean"],
            ],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            elinewidth=1,
            capsize=2,
            label=f"{window_start:%Y-%m} to {window_end:%Y-%m}",
        )
    axis.set_xticks(sorted(gap["subset_age_months"].unique()))
    axis.set_xlabel(
        "age of the frozen subset at the window's start (months from the cutoff day)"
    )
    axis.set_ylabel("tau gap, frozen − refreshed")
    axis.grid(color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, title="test window", loc="lower left")
    if cli_arguments.title:
        axis.set_title(cli_arguments.title, color=TEXT_COLOR)
    else:
        pass
    figure.text(
        0.0,
        -0.02,
        f"Bars: central 95 % over {gap_by_repeat['repeat'].nunique()} repeats, which resample the selection agents only (test agents fixed). "
        "Along a line, only the frozen subset's age changes.",
        fontsize=7,
        color=TEXT_COLOR,
        ha="left",
        va="top",
    )
    figure.tight_layout()
    cli_arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(cli_arguments.output, dpi=200, bbox_inches="tight")
    print(
        gap[gap["window_start"].isin(shared_windows)]
        .pivot(index="window_start", columns="subset_age_months", values="mean")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
