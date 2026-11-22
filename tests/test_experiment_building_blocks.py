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


def test_shelf_life_needs_two_windows_behind_in_a_row_and_respects_month_order() -> (
    None
):
    """Columns arrive as 6, 0, 3; a single dip does not count."""
    mean_gap_by_window = pd.DataFrame(
        {
            6: [0.1, -0.1, -0.1, 0.1],
            0: [0.0, 0.0, -0.1, 0.0],
            3: [0.1, -0.1, -0.1, -0.1],
        },
        index=["never_behind", "behind_from_3", "behind_from_0", "single_dip"],
    )
    shelf_life = shelf_life_months(mean_gap_by_window)
    assert np.isnan(shelf_life["never_behind"])
    assert shelf_life["behind_from_3"] == 3
    assert shelf_life["behind_from_0"] == 0
    assert np.isnan(shelf_life["single_dip"])


def test_separable_pairs_keeps_a_clear_leader_apart_and_ties_together() -> None:
    """A solves all 40 tasks; B and C give identical results on half of them."""
    random_generator = np.random.default_rng(0)
    outcomes = np.array([[1] * 40, [1] * 20 + [0] * 20, [1] * 20 + [0] * 20])
    task_counts = random_generator.multinomial(40, np.full(40, 1 / 40), size=500)
    bootstrap_solve_rate = outcomes @ task_counts.T / 40
    # note (Haoling Pu): pairs follow triu order: (A, B), (A, C), (B, C).
    assert separable_pairs(bootstrap_solve_rate).tolist() == [True, True, False]
    assert not separable_pairs(bootstrap_solve_rate[::-1]).any()
