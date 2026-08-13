from __future__ import annotations

import math

import pandas as pd
import pytest

from news_sentiment_trading.inference import (
    automatic_hac_lags,
    daily_rank_ic,
    hac_alpha_beta,
    hac_mean,
    holm_adjust,
    moving_block_bootstrap_mean_ci,
)


def test_automatic_hac_lags_known_answers() -> None:
    assert automatic_hac_lags(0) == 0
    assert automatic_hac_lags(1) == 0
    assert automatic_hac_lags(25) == 2
    assert automatic_hac_lags(100) == 4
    assert automatic_hac_lags(348) == 5


def test_hac_mean_autocorrelated_known_answer() -> None:
    # Mean=1.8, gamma(0)=0.56, gamma(1)=-0.208. With one Bartlett
    # lag the long-run variance is 0.56 + 2*(1/2)*(-0.208) = 0.352.
    result = hac_mean(pd.Series([1.0, 2.0, 1.0, 3.0, 2.0]), lags=1)

    assert result.mean == pytest.approx(1.8)
    assert result.standard_error == pytest.approx(math.sqrt(0.352 / 5.0))
    assert result.t_statistic == pytest.approx(6.78400525299968)
    assert result.p_value_two_sided == pytest.approx(1.1688907563388551e-11)
    assert result.lags == 1
    assert result.observations == 5


def test_hac_mean_preserves_explicit_missing_date_in_lag_pairs() -> None:
    # Four finite observations have mean 3.75. The explicit missing third row
    # prevents 2.0 and 4.0 from becoming an artificial lag-one pair. The HAC
    # meat is 28.75 + 5.875 = 34.625, so Var(mean)=34.625/4**2.
    index = pd.date_range("2026-01-05", periods=5, freq="B")
    series = pd.Series([1.0, 2.0, float("nan"), 4.0, 8.0], index=index)

    result = hac_mean(series, lags=1)

    assert result.mean == pytest.approx(3.75)
    assert result.standard_error == pytest.approx(math.sqrt(34.625 / 16.0))
    assert result.observations == 4


def test_circular_block_bootstrap_seeded_known_answer() -> None:
    series = pd.Series([-0.2, 0.1, 0.7, -1.3, 0.2, 2.1, -0.4, 0.9])

    first = moving_block_bootstrap_mean_ci(series, samples=101, block_length=3, seed=77)
    repeated = moving_block_bootstrap_mean_ci(series, samples=101, block_length=3, seed=77)
    other_seed = moving_block_bootstrap_mean_ci(series, samples=101, block_length=3, seed=78)

    assert first == pytest.approx((-0.125, 0.65625))
    assert repeated == first
    assert other_seed == pytest.approx((-0.08125, 0.63125))
    assert other_seed != first


def test_circular_block_bootstrap_full_length_blocks_preserve_mean() -> None:
    series = pd.Series([1.0, 2.0, 4.0, 9.0])
    interval = moving_block_bootstrap_mean_ci(
        series, samples=100, block_length=len(series), seed=20260801
    )
    assert interval == pytest.approx((4.0, 4.0))


def test_circular_block_bootstrap_rejects_calendar_gap_instead_of_compressing_it() -> None:
    series = pd.Series([1.0, 2.0, float("nan"), 4.0, 8.0])
    with pytest.raises(ValueError, match="complete finite series"):
        moving_block_bootstrap_mean_ci(series, samples=100, block_length=2, seed=77)


def test_daily_rank_ic_ties_missingness_and_coverage_known_answers() -> None:
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    score = pd.DataFrame(
        [
            [1.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, float("nan")],
            [1.0, 1.0, 1.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        index=dates,
        columns=list("ABCD"),
    )
    forward = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, float("nan"), 3.0, 4.0],
        ],
        index=dates,
        columns=list("ABCD"),
    )

    rank_ic = daily_rank_ic(score, forward, minimum_assets=4)

    assert rank_ic.iloc[0] == pytest.approx(0.9486832980505139)
    assert rank_ic.iloc[1:].isna().all()
    assert rank_ic.name == "rank_ic"


def test_hac_alpha_beta_coefficients_and_covariance_known_answer() -> None:
    benchmark = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    strategy = pd.Series([-3.7, -1.6, 0.4, 2.7, 4.3, 6.8])

    result = hac_alpha_beta(strategy, benchmark, lags=1)

    assert result.intercept_daily == pytest.approx(0.4476190476190472)
    assert result.beta == pytest.approx(2.071428571428572)
    assert result.intercept_hac_se == pytest.approx(0.02400434279056851)
    assert result.beta_hac_se == pytest.approx(0.020622008868125447)
    assert result.lags == 1
    assert result.observations == 6


def test_hac_alpha_beta_missing_row_does_not_compress_lag_pairs() -> None:
    benchmark = pd.Series([-2.0, -1.0, float("nan"), 0.0, 1.0, 2.0, 3.0])
    strategy = pd.Series([-3.7, -1.6, float("nan"), 0.4, 2.7, 4.3, 6.8])

    with_gap = hac_alpha_beta(strategy, benchmark, lags=1)
    compressed = hac_alpha_beta(
        strategy.dropna().reset_index(drop=True), benchmark.dropna().reset_index(drop=True), lags=1
    )

    assert with_gap.intercept_daily == pytest.approx(compressed.intercept_daily)
    assert with_gap.beta == pytest.approx(compressed.beta)
    assert with_gap.intercept_hac_se != pytest.approx(compressed.intercept_hac_se)
    assert with_gap.beta_hac_se != pytest.approx(compressed.beta_hac_se)


def test_holm_adjustment_restores_original_order_and_running_maximum() -> None:
    adjusted = holm_adjust([0.04, 0.001, 0.03, 0.20])
    assert adjusted == pytest.approx([0.09, 0.004, 0.09, 0.20])
