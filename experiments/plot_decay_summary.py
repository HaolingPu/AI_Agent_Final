"""C2 headline figure: how far a frozen subset falls behind a refreshed one as months pass, one line per cutoff.

Each point averages the gap over the selection methods that use agent data (random is never stale by construction) and every subset size.
Error bars are the central 95 % of that average across repeats, which resample the selection agents only.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# note (Wilson Zheng): the first four categorical slots, in their fixed, colour-blind-validated order; one marker per cutoff as well.
CUTOFF_STYLES = (
    ("#2a78d6", "o"),
    ("#eb6834", "s"),
    ("#1baf7a", "^"),
    ("#eda100", "D"),
)
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e8e7e3"


def mean_gap_by_window(decay_csv: Path) -> pd.DataFrame:
    """Per window: the method- and size-averaged gap, and its 2.5 / 97.5 percentiles across repeats."""
    results = pd.read_csv(decay_csv)
    # note (Haoling Pu): every method but plain random, the gp-IRT ones included; run_decay's shelf life is per (method, size), so its band is wider than this averaged one.
    per_repeat_gap = (
        results[results["method"] != "random"]
        .groupby(["window_start_month", "repeat"])["kendall_tau_gap"]
        .mean()
        .unstack()
    )
    return pd.DataFrame(
        {
            "mean": per_repeat_gap.mean(axis=1),
            "low": per_repeat_gap.quantile(0.025, axis=1),
            "high": per_repeat_gap.quantile(0.975, axis=1),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decay-csv",
        type=Path,
        nargs="+",
        required=True,
        help="one run_decay.py output per cutoff",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True, help="one label per --decay-csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    cli_arguments = parser.parse_args()
    if len(cli_arguments.labels) != len(cli_arguments.decay_csv):
        raise ValueError("give one label per --decay-csv")
    elif len(cli_arguments.decay_csv) > len(CUTOFF_STYLES):
        raise ValueError(f"at most {len(CUTOFF_STYLES)} cutoffs fit one figure")
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
    # note (Wilson Zheng): a small horizontal offset per cutoff keeps the error bars from hiding each other.
    offsets = [
        0.15 * (position - (len(cli_arguments.decay_csv) - 1) / 2)
        for position in range(len(cli_arguments.decay_csv))
    ]
    window_months = set()
    for decay_csv, label, (color, marker), offset in zip(
        cli_arguments.decay_csv, cli_arguments.labels, CUTOFF_STYLES, offsets
    ):
        gap = mean_gap_by_window(decay_csv)
        window_months |= set(gap.index)
        axis.errorbar(
            gap.index + offset,
            gap["mean"],
            yerr=[gap["mean"] - gap["low"], gap["high"] - gap["mean"]],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            elinewidth=1,
            capsize=2,
            label=label,
        )
    axis.set_xticks(sorted(window_months))
    axis.set_xlabel("months after the cutoff")
    axis.set_ylabel("tau gap, frozen − refreshed")
    axis.grid(color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, title="cutoff", loc="lower left")
    if cli_arguments.title:
        axis.set_title(cli_arguments.title, color=TEXT_COLOR)
    else:
        pass
    num_repeats = pd.read_csv(cli_arguments.decay_csv[0])["repeat"].nunique()
    # note (Haoling Pu): the caption says what the bars are, since a reader sees the figure without this docstring.
    figure.text(
        0.0,
        -0.02,
        f"Bars: central 95 % over {num_repeats} repeats, which resample the selection agents only (test agents fixed). "
        "Month 0 is 0 by construction.",
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
