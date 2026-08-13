"""Adversarial tests for leakage, isolation, missingness, and accounting.

Every fixture in this module is generated from synthetic bytes.  The tests are
deliberately phrased as invariants: changing data that was unavailable at a
decision boundary must not change that decision or any already-closed result.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from news_sentiment_trading.config import SignalParameters, load_config
from news_sentiment_trading.data import (
    PRICE_COLUMNS,
    SENTIMENT_COLUMNS,
    align_signal_to_return_end,
    aligned_forward_returns,
    load_asset_csv,
    load_panel,
)
from news_sentiment_trading.features import build_features, strictly_prior_rolling
from news_sentiment_trading.portfolio import evaluate_portfolio, market_neutral_weights
from news_sentiment_trading.signals import bounded_holding, event_signal
from news_sentiment_trading.synthetic import synthetic_panel, write_synthetic_csvs
from news_sentiment_trading.walk_forward import WalkForwardResult, run_walk_forward

_COMPONENT_COLUMNS = (
    "positivePartscr",
    "certaintyPartscr",
    "finupPartscr",
    "negativePartscr",
    "uncertaintyPartscr",
    "findownPartscr",
)


def _primary_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "primary.toml"


@pytest.fixture(scope="module")
def isolation_case() -> tuple[pd.DataFrame, WalkForwardResult]:
    """A faster two-fold panel that retains the locked inner design and grid."""

    tickers = ("AAPL", "AMZN", "DB", "DIS")
    panel = synthetic_panel(sessions=370, tickers=tickers, seed=20260899)
    config = load_config(_primary_config_path())
    config = replace(config, universe=replace(config.universe, primary=tickers))
    return panel, run_walk_forward(panel, config)


def _rerun_isolation_panel(panel: pd.DataFrame) -> WalkForwardResult:
    tickers = tuple(str(item) for item in panel.index.get_level_values("ticker").unique())
    config = load_config(_primary_config_path())
    config = replace(config, universe=replace(config.universe, primary=tickers))
    return run_walk_forward(panel, config)


def test_adversarial_future_proxy_only_rewards_the_leaked_feature() -> None:
    """An explicit future proxy wins; the production strict-prior transform does not."""

    rng = np.random.default_rng(7719)
    innovations = pd.DataFrame({"x": rng.normal(size=4_000)})
    future_return = innovations["x"].shift(-1).rename("future_return")

    prior_mean = strictly_prior_rolling(innovations, 50, "mean", minimum_fraction=1.0)["x"]
    prior_std = strictly_prior_rolling(innovations, 50, "std", minimum_fraction=1.0)["x"]
    production_score = innovations["x"].sub(prior_mean).div(prior_std)
    leaked_score = innovations["x"].shift(-1)

    assert leaked_score.corr(future_return) == pytest.approx(1.0)
    assert abs(float(production_score.corr(future_return))) < 0.05


def test_current_row_cannot_change_its_own_training_scalers_or_rvt_threshold(
    panel: pd.DataFrame,
) -> None:
    parameters = SignalParameters(20, 1.5, 0.75)
    original = build_features(panel, parameters)
    date = panel.index.get_level_values("date").unique()[180]

    mutated = panel.copy()
    mutated.loc[(date, "AAPL"), list(_COMPONENT_COLUMNS)] = [10_000, 10_000, 10_000, 1, 1, 1]
    mutated.loc[(date, "AAPL"), "RVT"] = 1_000_000
    changed = build_features(mutated, parameters)

    assert changed.prior_mean.loc[date, "AAPL"] == pytest.approx(
        original.prior_mean.loc[date, "AAPL"]
    )
    assert changed.prior_std.loc[date, "AAPL"] == pytest.approx(
        original.prior_std.loc[date, "AAPL"]
    )
    assert changed.prior_rvt_threshold.loc[date, "AAPL"] == pytest.approx(
        original.prior_rvt_threshold.loc[date, "AAPL"]
    )
    assert changed.score.loc[date, "AAPL"] != pytest.approx(original.score.loc[date, "AAPL"])


def test_future_suffix_is_invariant_for_all_earlier_feature_and_timing_artifacts(
    panel: pd.DataFrame,
) -> None:
    parameters = SignalParameters(50, 2.0, 0.75)
    dates = panel.index.get_level_values("date").unique()
    cutoff = pd.Timestamp(dates[260])
    original_features = build_features(panel, parameters)
    original_returns = aligned_forward_returns(panel)
    original_signal = event_signal(original_features, parameters.z_threshold)

    mutated = panel.copy()
    suffix = mutated.index.get_level_values("date") > cutoff
    mutated.loc[suffix, list(_COMPONENT_COLUMNS)] *= 100_000
    mutated.loc[suffix, "RVT"] *= 100_000
    mutated.loc[suffix, "Adjusted Open"] *= 3.0
    mutated.loc[suffix, "Adj Close"] *= 2.0
    changed_features = build_features(mutated, parameters)
    changed_returns = aligned_forward_returns(mutated)
    changed_signal = event_signal(changed_features, parameters.z_threshold)

    for field in (
        "raw",
        "score",
        "prior_mean",
        "prior_std",
        "prior_rvt_threshold",
        "active_news",
        "available",
        "imputed",
    ):
        pd.testing.assert_frame_equal(
            getattr(original_features, field).loc[:cutoff],
            getattr(changed_features, field).loc[:cutoff],
        )
    pd.testing.assert_frame_equal(original_signal.loc[:cutoff], changed_signal.loc[:cutoff])

    closed_returns = original_returns.returns.index <= cutoff
    pd.testing.assert_frame_equal(
        original_returns.returns.loc[closed_returns],
        changed_returns.returns.loc[closed_returns],
    )
    pd.testing.assert_frame_equal(
        align_signal_to_return_end(original_signal, original_returns).loc[closed_returns],
        align_signal_to_return_end(changed_signal, changed_returns).loc[closed_returns],
    )


def test_one_session_fill_never_backfills_or_creates_an_unbounded_stale_signal(
    panel: pd.DataFrame,
) -> None:
    dates = panel.index.get_level_values("date").unique()
    mutated = panel.copy()
    mutated.loc[(dates[0], "AAPL"), list(_COMPONENT_COLUMNS)] = np.nan
    mutated.loc[(dates[100], "AAPL"), list(_COMPONENT_COLUMNS)] = np.nan
    mutated.loc[(dates[101], "AAPL"), list(_COMPONENT_COLUMNS)] = np.nan
    mutated.loc[(dates[102], "AAPL"), list(_COMPONENT_COLUMNS)] = [99, 99, 99, 1, 1, 1]

    filled = build_features(
        mutated,
        SignalParameters(20, 1.5, 0.5, missing_policy="ffill_1"),
    )
    no_fill = build_features(
        mutated,
        SignalParameters(20, 1.5, 0.5, missing_policy="no_fill"),
    )
    filled_signal = bounded_holding(event_signal(filled, 1.5), holding_days=1)

    assert pd.isna(filled.raw.loc[dates[0], "AAPL"]), "a later row must never backfill row zero"
    assert filled.imputed.loc[dates[100], "AAPL"]
    assert not filled.imputed.loc[dates[101], "AAPL"]
    assert pd.isna(filled.raw.loc[dates[101], "AAPL"])
    assert pd.isna(no_fill.raw.loc[dates[100], "AAPL"])
    assert pd.isna(no_fill.raw.loc[dates[101], "AAPL"])
    assert filled_signal.loc[dates[101], "AAPL"] == 0


def test_embargoed_return_and_outer_test_mutation_cannot_change_fold_selection(
    isolation_case: tuple[pd.DataFrame, WalkForwardResult],
) -> None:
    panel, original = isolation_case
    fold = original.folds[0]
    first_execution = pd.Timestamp(fold.first_execution_date)
    mutated = panel.copy()
    unavailable = mutated.index.get_level_values("date") >= first_execution
    mutated.loc[unavailable, "Adjusted Open"] *= 5.0
    mutated.loc[unavailable, "Adj Close"] *= 7.0
    mutated.loc[unavailable, list(_COMPONENT_COLUMNS)] *= 10_000
    mutated.loc[unavailable, "RVT"] *= 10_000
    changed = _rerun_isolation_panel(mutated)

    assert not original.return_frame.returns.loc[first_execution].equals(
        changed.return_frame.returns.loc[first_execution]
    ), "the adversarial mutation must actually change the embargoed return"
    assert original.candidate_scores[0] == changed.candidate_scores[0]
    assert original.selections[0] == changed.selections[0]
    assert original.folds[0].selection_data_end < original.folds[0].first_execution_date


def test_closed_fold_is_immutable_to_every_later_suffix(
    isolation_case: tuple[pd.DataFrame, WalkForwardResult],
) -> None:
    panel, original = isolation_case
    fold = original.folds[0]
    fold_dates = original.parameters_by_date.index[
        original.parameters_by_date["fold_id"].eq(fold.fold_id)
    ]
    test_end = pd.Timestamp(fold.test_end)
    mutated = panel.copy()
    later = mutated.index.get_level_values("date") > test_end
    mutated.loc[later, "Adjusted Open"] *= 11.0
    mutated.loc[later, "Adj Close"] *= 13.0
    mutated.loc[later, list(_COMPONENT_COLUMNS)] *= 100_000
    mutated.loc[later, "RVT"] *= 100_000
    changed = _rerun_isolation_panel(mutated)

    assert original.candidate_scores[0] == changed.candidate_scores[0]
    assert original.selections[0] == changed.selections[0]
    pd.testing.assert_frame_equal(
        original.parameters_by_date.loc[fold_dates], changed.parameters_by_date.loc[fold_dates]
    )
    pd.testing.assert_frame_equal(original.signal.loc[fold_dates], changed.signal.loc[fold_dates])
    pd.testing.assert_frame_equal(original.score.loc[fold_dates], changed.score.loc[fold_dates])
    pd.testing.assert_frame_equal(
        original.asset_returns.loc[fold_dates], changed.asset_returns.loc[fold_dates]
    )
    pd.testing.assert_frame_equal(
        original.portfolio.weights.loc[fold_dates], changed.portfolio.weights.loc[fold_dates]
    )
    pd.testing.assert_frame_equal(
        original.portfolio.pretrade_weights.loc[fold_dates],
        changed.portfolio.pretrade_weights.loc[fold_dates],
    )
    portfolio_fields = (
        "gross_return",
        "net_return",
        "transaction_cost",
        "turnover",
        "long_exposure",
        "short_exposure",
        "gross_exposure",
        "net_exposure",
        "pretrade_financing_balance",
        "financing_balance",
        "unused_gross_capacity",
    )
    for portfolio_name in ("portfolio", "fixed_baseline_portfolio"):
        original_portfolio = getattr(original, portfolio_name)
        changed_portfolio = getattr(changed, portfolio_name)
        pd.testing.assert_frame_equal(
            original_portfolio.weights.loc[fold_dates],
            changed_portfolio.weights.loc[fold_dates],
        )
        pd.testing.assert_frame_equal(
            original_portfolio.pretrade_weights.loc[fold_dates],
            changed_portfolio.pretrade_weights.loc[fold_dates],
        )
        for field in portfolio_fields:
            pd.testing.assert_series_equal(
                getattr(original_portfolio, field).loc[fold_dates],
                getattr(changed_portfolio, field).loc[fold_dates],
            )
    for field in (
        "rebalanced_return",
        "static_return",
        "cost_aware_rebalanced_return",
        "cost_aware_rebalanced_turnover",
        "cost_aware_rebalanced_transaction_cost",
    ):
        pd.testing.assert_series_equal(
            getattr(original.benchmark, field).loc[fold_dates],
            getattr(changed.benchmark, field).loc[fold_dates],
        )
    pd.testing.assert_frame_equal(
        original.benchmark.rebalanced_weights.loc[fold_dates],
        changed.benchmark.rebalanced_weights.loc[fold_dates],
    )
    pd.testing.assert_frame_equal(
        original.benchmark.static_weights.loc[fold_dates],
        changed.benchmark.static_weights.loc[fold_dates],
    )
    pd.testing.assert_series_equal(
        original.return_frame.signal_dates.loc[fold_dates],
        changed.return_frame.signal_dates.loc[fold_dates],
    )
    pd.testing.assert_series_equal(
        original.return_frame.execution_dates.loc[fold_dates],
        changed.return_frame.execution_dates.loc[fold_dates],
    )


def test_later_fold_mutation_cannot_change_its_or_prior_parameter_choices(
    isolation_case: tuple[pd.DataFrame, WalkForwardResult],
) -> None:
    panel, original = isolation_case
    assert len(original.folds) >= 2
    second_fold = original.folds[1]
    first_execution = pd.Timestamp(second_fold.first_execution_date)
    mutated = panel.copy()
    unavailable = mutated.index.get_level_values("date") >= first_execution
    mutated.loc[unavailable, "Adjusted Open"] *= 3.0
    mutated.loc[unavailable, "Adj Close"] *= 4.0
    mutated.loc[unavailable, list(_COMPONENT_COLUMNS)] *= 50_000
    mutated.loc[unavailable, "RVT"] *= 50_000
    changed = _rerun_isolation_panel(mutated)
    assert original.selections[:2] == changed.selections[:2]
    assert original.candidate_scores[:2] == changed.candidate_scores[:2]


def test_market_neutral_asymmetric_books_use_the_smaller_side_and_one_sided_cash() -> None:
    signal = pd.DataFrame(
        [[1, 1, 1, -1, 0], [1, 1, 0, 0, 0]],
        columns=list("ABCDE"),
    )
    weights = market_neutral_weights(signal, per_asset_cap=0.20, gross_limit=1.0)

    assert weights.iloc[0, :3].tolist() == pytest.approx([0.2 / 3] * 3)
    assert weights.iloc[0, 3] == pytest.approx(-0.2)
    assert weights.iloc[0].sum() == pytest.approx(0.0)
    assert weights.iloc[0].abs().sum() == pytest.approx(0.4)
    assert weights.iloc[1].abs().sum() == pytest.approx(0.0)


def test_entry_and_same_day_final_liquidation_are_both_charged() -> None:
    date = pd.DatetimeIndex(["2026-01-02"])
    weights = pd.DataFrame({"A": [0.3], "B": [-0.2]}, index=date)
    returns = pd.DataFrame({"A": [0.10], "B": [-0.05]}, index=date)
    result = evaluate_portfolio(weights, returns, cost_bps=25, liquidate=True)

    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert result.transaction_cost.iloc[0] == pytest.approx(0.0025)
    assert result.gross_return.iloc[0] == pytest.approx(0.04)
    assert result.net_return.iloc[0] == pytest.approx(0.0375)
    assert result.trade_count == 4


@settings(max_examples=50, deadline=None)
@given(
    raw_weights=arrays(
        np.float64,
        (5, 4),
        elements=st.floats(-2, 2, allow_nan=False, allow_infinity=False),
    ),
    asset_returns=arrays(
        np.float64,
        (5, 4),
        elements=st.floats(-0.5, 0.5, allow_nan=False, allow_infinity=False),
    ),
    cost_bps=st.one_of(
        st.just(0.0),
        st.floats(0.001, 250, allow_nan=False, allow_infinity=False),
    ),
)
def test_portfolio_accounting_identity_holds_for_random_bounded_books(
    raw_weights: np.ndarray,
    asset_returns: np.ndarray,
    cost_bps: float,
) -> None:
    gross = np.abs(raw_weights).sum(axis=1, keepdims=True)
    weights_array = raw_weights / np.maximum(gross, 1.0)
    index = pd.bdate_range("2026-01-02", periods=5)
    columns = list("ABCD")
    weights = pd.DataFrame(weights_array, index=index, columns=columns)
    returns = pd.DataFrame(asset_returns, index=index, columns=columns)

    result = evaluate_portfolio(weights, returns, cost_bps=cost_bps, liquidate=True)
    previous = np.vstack([np.zeros((1, weights.shape[1])), weights_array[:-1]])
    expected_turnover = np.abs(weights_array - previous).sum(axis=1)
    expected_turnover[-1] += np.abs(weights_array[-1]).sum()
    expected_gross = (weights_array * asset_returns).sum(axis=1)
    expected_cost = expected_turnover * cost_bps / 10_000.0

    np.testing.assert_allclose(result.turnover.to_numpy(), expected_turnover, rtol=1e-12)
    np.testing.assert_allclose(result.gross_return.to_numpy(), expected_gross, rtol=1e-12)
    np.testing.assert_allclose(result.transaction_cost.to_numpy(), expected_cost, rtol=1e-12)
    np.testing.assert_allclose(
        result.net_return.to_numpy(), expected_gross - expected_cost, rtol=1e-12
    )


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("missing_column", "missing columns"),
        ("invalid_date", "invalid Date"),
        ("unsorted", "dates are not strictly increasing"),
        ("negative_volume", "negative volume"),
        ("nonnumeric_price", "missing/non-numeric price"),
        ("nonfinite_price", "finite|price|adjustment"),
        ("negative_rvt", "nonnegative|sentiment|RVT"),
        ("negative_component", "nonnegative|sentiment|component"),
    ],
)
def test_source_schema_fails_closed_on_invalid_values(
    tmp_path: Path,
    case: str,
    expected_message: str,
) -> None:
    frame = synthetic_panel(sessions=60, tickers=("AAPL",), seed=99).reset_index()
    frame = frame.rename(columns={"date": "Date"}).drop(columns=["ticker", "Adjusted Open"])
    if case == "missing_column":
        frame = frame.drop(columns=["High"])
    elif case == "invalid_date":
        frame["Date"] = frame["Date"].astype(object)
        frame.loc[4, "Date"] = "not-a-date"
    elif case == "unsorted":
        frame.loc[[4, 5], "Date"] = frame.loc[[5, 4], "Date"].to_numpy()
    elif case == "negative_volume":
        frame.loc[4, "Volume"] = -1
    elif case == "nonnumeric_price":
        frame["Close"] = frame["Close"].astype(object)
        frame.loc[4, "Close"] = "not-a-number"
    elif case == "nonfinite_price":
        frame.loc[4, "Open"] = np.inf
    elif case == "negative_rvt":
        frame.loc[4, "RVT"] = -0.01
    elif case == "negative_component":
        frame.loc[4, "positivePartscr"] = -0.01
    else:  # pragma: no cover - guarded by the parameter table
        raise AssertionError(case)
    path = tmp_path / f"{case}.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=expected_message):
        load_asset_csv(path, "AAPL")


def test_panel_rejects_a_single_asset_calendar_gap(tmp_path: Path) -> None:
    source = write_synthetic_csvs(tmp_path / "calendar", sessions=60)
    path = source / "DB.csv"
    frame = pd.read_csv(path).drop(index=11)
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="price calendar differs"):
        load_panel(source)


def test_price_and_sentiment_schema_contract_is_complete() -> None:
    """Guard the adversarial schema matrix when source columns evolve."""

    assert set(PRICE_COLUMNS) == {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    assert set(_COMPONENT_COLUMNS).issubset(SENTIMENT_COLUMNS)
