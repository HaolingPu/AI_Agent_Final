"""Time windows after a cutoff date, and when a frozen subset falls behind a refreshed one (C2)."""

import pandas as pd

# note (Haoling Pu): with 20 repeats this percentile falls between the two largest gaps; with one repeat it is that repeat's gap.
SHELF_LIFE_UPPER_QUANTILE = 0.975


def window_start_month(
    submit_date: pd.Series, *, cutoff_date: pd.Timestamp, window_months: int
) -> pd.Series:
    """Months from the cutoff's month to the start of each later agent's window; agents before the cutoff are dropped.

    Windows are window_months calendar months; window 0 runs from the cutoff day to the end of its last month, so it can be shorter.
    """
    later_submit_date = submit_date[submit_date >= cutoff_date]
    months_after_cutoff = (later_submit_date.dt.year - cutoff_date.year) * 12 + (
        later_submit_date.dt.month - cutoff_date.month
    )
    return (months_after_cutoff // window_months) * window_months


def shelf_life_months(decay_results: pd.DataFrame, *, window_months: int) -> pd.Series:
    """First window that starts two windows in a row where the frozen subset is clearly behind a refreshed one, per (subset size, method).

    decay_results: run_decay.py rows, one per (repeat, window, subset size, method), with kendall_tau_gap = tau(frozen) - tau(refreshed).
    A window is behind when the 97.5th percentile of the gap over repeats is below 0, i.e. the whole 95 % band is under zero; a mean below 0 alone is mostly noise.
    A window left out for too few test agents breaks a run, so two windows in a row are always adjacent in time.
    NaN means never behind in two windows in a row, including behind only in the last window, which has no next window to confirm it.
    """
    upper_gap_by_window = decay_results.pivot_table(
        index=["subset_size", "method"],
        columns="window_start_month",
        values="kendall_tau_gap",
        aggfunc=lambda gap: gap.quantile(SHELF_LIFE_UPPER_QUANTILE),
    )
    all_window_months = range(
        0, int(upper_gap_by_window.columns.max()) + 1, window_months
    )
    is_behind = upper_gap_by_window.reindex(columns=all_window_months).lt(0)
    starts_two_behind = is_behind & is_behind.shift(-1, axis=1, fill_value=False)
    return starts_two_behind.idxmax(axis=1).where(starts_two_behind.any(axis=1))
