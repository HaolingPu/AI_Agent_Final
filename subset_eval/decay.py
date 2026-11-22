"""Time windows after a cutoff date, and when a frozen subset falls behind a refreshed one (C2)."""

import pandas as pd


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


def shelf_life_months(mean_gap_by_window: pd.DataFrame) -> pd.Series:
    """First window that starts two windows in a row where the frozen subset ranks worse than a refreshed one.

    mean_gap_by_window: rows are (method, size), columns are window start months, values are mean tau(frozen) - tau(refreshed).
    NaN means the frozen subset never fell behind for two windows in a row.
    """
    is_behind = mean_gap_by_window.sort_index(axis=1).lt(0)
    starts_two_behind = is_behind & is_behind.shift(-1, axis=1, fill_value=False)
    return starts_two_behind.idxmax(axis=1).where(starts_two_behind.any(axis=1))
