"""Contracts of the pieces the experiments share: method dispatch, decay windows and shelf life, pair separability."""

import numpy as np
import pandas as pd
import pytest

from subset_eval.data import outcome_matrix
from subset_eval.decay import shelf_life_months, window_start_month
from subset_eval.methods import (
    GP_IRT_METHODS,
    SELECTION_METHODS,
    estimate_solve_rates,
    fit_selection_group,
    select_task_weights,
)
from subset_eval.metrics import separable_pairs
from subset_eval.toy import make_toy_benchmark


def test_every_method_returns_weights_and_gp_irt_only_where_the_paper_has_it() -> None:
    outcomes = outcome_matrix(
        make_toy_benchmark(num_agents=60, num_tasks=80, num_months=12, seed=0)
    )
    selection_group_fit = fit_selection_group(
        outcomes.iloc[:40], random_generator=np.random.default_rng(0)
    )
    test_outcomes = outcomes.iloc[40:]
    for selection_method in SELECTION_METHODS:
        task_weights = select_task_weights(
            selection_method,
            selection_group_fit,
            subset_size=10,
            random_generator=np.random.default_rng(0),
        )
        assert task_weights.sum() == pytest.approx(1.0)
        assert len(task_weights) <= 10
        assert task_weights.index.isin(outcomes.columns).all()
        estimates = estimate_solve_rates(
            selection_method,
            task_weights,
            test_outcomes,
            selection_group_fit,
            subset_size=10,
        )
        if selection_method in GP_IRT_METHODS:
            assert set(estimates) == {selection_method, f"{selection_method}_gp_irt"}
        else:
            assert set(estimates) == {selection_method}
    with pytest.raises(ValueError, match="unknown selection method"):
        select_task_weights(
            "not_a_method",
            selection_group_fit,
            subset_size=10,
            random_generator=np.random.default_rng(0),
        )


def test_window_start_month_with_a_mid_month_cutoff_and_a_year_rollover() -> None:
    """Cutoff 2025-01-15, 3-month windows: window 0 is 01-15 to 03-31, window 3 starts 04-01, 2026-01 is window 12."""
    submit_date = pd.Series(
        pd.to_datetime(
            ["2025-01-10", "2025-01-20", "2025-03-31", "2025-04-01", "2026-01-01"]
        ),
        index=["before", "first_day", "window_end", "next_window", "next_year"],
    )
    start_month = window_start_month(
        submit_date, cutoff_date=pd.Timestamp("2025-01-15"), window_months=3
    )
    assert start_month.to_dict() == {
        "first_day": 0,
        "window_end": 0,
        "next_window": 3,
        "next_year": 12,
    }


def decay_rows(gaps_by_window: dict[int, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subset_size": 10,
                "method": "irt_max_information",
                "repeat": repeat,
                "window_start_month": window,
                "kendall_tau_gap": gap,
            }
            for window, gaps in gaps_by_window.items()
            for repeat, gap in enumerate(gaps)
        ]
    )


def test_shelf_life_needs_the_whole_band_below_zero_two_windows_in_a_row() -> None:
    """Windows arrive out of order; a mean below 0 whose band crosses 0 does not count, nor does a single dip or a run across a skipped window."""
    clearly_behind = [-0.1, -0.1, -0.1]
    band_crosses_zero = [-0.2, -0.2, 0.05]
    cases = {
        "behind_from_3": ({6: clearly_behind, 0: [0.0] * 3, 3: clearly_behind}, 3),
        "mean_below_zero_only": (
            {0: [0.0] * 3, 3: band_crosses_zero, 6: band_crosses_zero},
            None,
        ),
        "single_dip": ({0: [0.0] * 3, 3: clearly_behind, 6: [0.1] * 3}, None),
        "window_3_skipped": ({0: clearly_behind, 6: clearly_behind}, None),
    }
    for name, (gaps_by_window, expected_month) in cases.items():
        shelf_life = shelf_life_months(
            decay_rows(gaps_by_window), window_months=3
        ).iloc[0]
        if expected_month is None:
            assert np.isnan(shelf_life), name
        else:
            assert shelf_life == expected_month, name


def test_separable_pairs_keeps_a_clear_leader_apart_and_ties_together() -> None:
    """A solves all 40 tasks; B and C give identical results on half of them."""
    random_generator = np.random.default_rng(0)
    outcomes = np.array([[1] * 40, [1] * 20 + [0] * 20, [1] * 20 + [0] * 20])
    task_counts = random_generator.multinomial(40, np.full(40, 1 / 40), size=500)
    bootstrap_solve_rate = outcomes @ task_counts.T / 40
    # note (Haoling Pu): pairs follow triu order: (A, B), (A, C), (B, C).
    assert separable_pairs(bootstrap_solve_rate).tolist() == [True, True, False]
    assert not separable_pairs(bootstrap_solve_rate[::-1]).any()
