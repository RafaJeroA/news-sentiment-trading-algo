from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from news_sentiment_trading.metrics import performance_metrics
from news_sentiment_trading.portfolio import (
    benchmarks,
    directional_weights,
    drift_adjusted_turnover,
    evaluate_portfolio,
    long_only_weights,
    market_neutral_weights,
)


def test_long_only_caps_and_cash() -> None:
    signal = pd.DataFrame([[1, 0, 0], [1, 1, 1]], columns=list("ABC"))
    weights = long_only_weights(signal, per_asset_cap=0.20)
    assert weights.iloc[0].tolist() == [0.2, 0.0, 0.0]
    assert weights.iloc[1].sum() == pytest.approx(0.6)
    assert (weights >= 0).all().all()


def test_market_neutrality_and_one_sided_cash() -> None:
    signal = pd.DataFrame([[1, -1, 0], [1, 1, 0]], columns=list("ABC"))
    weights = market_neutral_weights(signal, per_asset_cap=0.20)
    assert weights.iloc[0].sum() == pytest.approx(0.0)
    assert weights.iloc[0].abs().sum() == pytest.approx(0.4)
    assert weights.iloc[1].abs().sum() == 0.0


def test_directional_weights_enforce_signal_domain_and_cap() -> None:
    signal = pd.DataFrame([[1, -1, 0], [1, 1, -1]], columns=list("ABC"))
    weights = directional_weights(signal, per_asset_cap=0.20)
    assert weights.iloc[0].tolist() == pytest.approx([0.2, -0.2, 0.0])
    assert weights.iloc[1].abs().max() == pytest.approx(0.2)
    assert weights.iloc[1].abs().sum() == pytest.approx(0.6)

    with pytest.raises(ValueError, match="per_asset_cap"):
        directional_weights(signal, per_asset_cap=0.0)
    with pytest.raises(ValueError, match="signal values"):
        directional_weights(pd.DataFrame([[2, 0, 0]]), per_asset_cap=0.20)


def test_target_weight_cost_entry_reverse_exit() -> None:
    index = pd.date_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.0, 1.0, -1.0]}, index=index)
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=index)
    result = evaluate_portfolio(weights, returns, cost_bps=100, liquidate=True)
    assert result.turnover.tolist() == pytest.approx([0.0, 1.0, 3.0])
    assert result.transaction_cost.sum() == pytest.approx(0.04)
    assert result.trade_count == 3


def test_neutral_cash_and_unused_capacity_are_distinct() -> None:
    index = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.2], "B": [-0.2]}, index=index)
    returns = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=index)
    result = evaluate_portfolio(weights, returns, cost_bps=0, liquidate=False)
    assert result.financing_balance.iloc[0] == pytest.approx(1.0)
    assert result.unused_gross_capacity.iloc[0] == pytest.approx(0.6)
    assert result.long_exposure.iloc[0] == pytest.approx(0.2)
    assert result.short_exposure.iloc[0] == pytest.approx(0.2)


def test_drift_turnover_uses_previous_row_return_not_current_row() -> None:
    index = pd.date_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=index)
    returns = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=index)
    base = evaluate_portfolio(
        weights,
        returns,
        cost_bps=0,
        liquidate=False,
        turnover_convention="drift_adjusted",
    )
    changed_returns = returns.copy()
    changed_returns.iloc[1] = [0.40, -0.20]
    changed = evaluate_portfolio(
        weights,
        changed_returns,
        cost_bps=0,
        liquidate=False,
        turnover_convention="drift_adjusted",
    )
    assert base.turnover.iloc[0] == pytest.approx(1.0)
    assert base.turnover.iloc[1] == pytest.approx(1.0 / 21.0)
    assert changed.turnover.iloc[1] == pytest.approx(base.turnover.iloc[1])
    assert changed.turnover.iloc[2] != pytest.approx(base.turnover.iloc[2])


def test_equal_returns_need_no_rebalance_only_when_fully_invested() -> None:
    index = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.10, 0.0], "B": [0.10, 0.0]}, index=index)
    fully_invested = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=index)
    with_cash = pd.DataFrame({"A": [0.2, 0.2], "B": [0.2, 0.2]}, index=index)
    full = drift_adjusted_turnover(fully_invested, returns, liquidate=False)
    partial = drift_adjusted_turnover(with_cash, returns, liquidate=False)
    assert full.turnover.tolist() == pytest.approx([1.0, 0.0])
    assert partial.turnover.tolist() == pytest.approx([0.4, 0.44 / 1.04 - 0.4])


def test_drift_entry_exit_reversal_cash_and_long_short_known_answers() -> None:
    dates = pd.date_range("2020-01-01", periods=2)
    returns = pd.DataFrame({"A": [0.10, 0.0], "B": [-0.10, 0.0]}, index=dates)
    exit_weights = pd.DataFrame({"A": [0.2, 0.0], "B": [-0.2, 0.0]}, index=dates)
    reverse_weights = pd.DataFrame({"A": [0.2, -0.2], "B": [-0.2, 0.2]}, index=dates)
    exited = drift_adjusted_turnover(exit_weights, returns, liquidate=False)
    reversed_book = drift_adjusted_turnover(reverse_weights, returns, liquidate=False)
    assert exited.turnover.iloc[0] == pytest.approx(0.4)
    assert exited.turnover.iloc[1] == pytest.approx(0.4 / 1.04)
    assert reversed_book.turnover.iloc[1] == pytest.approx(
        abs(-0.2 - 0.22 / 1.04) + abs(0.2 - (-0.18 / 1.04))
    )
    assert exited.pretrade_financing_balance.iloc[1] == pytest.approx(1.0 - 0.04 / 1.04)


def test_drift_liquidation_and_costs_reconcile_exactly() -> None:
    dates = pd.date_range("2020-01-01", periods=2)
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=dates)
    returns = pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.0]}, index=dates)
    result = evaluate_portfolio(
        weights,
        returns,
        cost_bps=100,
        liquidate=True,
        turnover_convention="drift_adjusted",
    )
    expected_second_rebalance = abs(0.5 - 0.55 / 1.04) + abs(0.5 - 0.5 / 1.04)
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert result.turnover.iloc[1] == pytest.approx(expected_second_rebalance + 1.0)
    assert result.final_liquidation_nav_equivalent_turnover == pytest.approx(1.0)
    pd.testing.assert_series_equal(
        result.transaction_cost,
        (result.turnover * 0.01).rename("transaction_cost"),
    )
    pd.testing.assert_series_equal(
        result.net_return,
        (result.gross_return - result.transaction_cost).rename("net_return"),
    )


def test_exit_only_cost_day_is_in_trading_hit_rate() -> None:
    index = pd.date_range("2020-01-01", periods=2)
    weights = pd.DataFrame({"A": [0.2, 0.0]}, index=index)
    returns = pd.DataFrame({"A": [0.0, 0.0]}, index=index)
    result = evaluate_portfolio(weights, returns, cost_bps=10, liquidate=False)
    metrics = performance_metrics(result.net_return, portfolio=result)
    assert metrics["exposure_active_days"] == 1
    assert metrics["trading_active_days"] == 2
    assert metrics["nonzero_net_return_days"] == 2
    assert metrics["hit_rate_trading_days"] == 0.0


def test_portfolio_return_reconciles() -> None:
    index = pd.date_range("2020-01-01", periods=2)
    weights = pd.DataFrame({"A": [0.5, 0.0], "B": [0.5, 1.0]}, index=index)
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, -0.10]}, index=index)
    result = evaluate_portfolio(weights, returns, cost_bps=10, liquidate=False)
    expected_gross = (weights * returns).sum(axis=1)
    pd.testing.assert_series_equal(result.gross_return, expected_gross.rename("gross_return"))
    pd.testing.assert_series_equal(
        result.net_return,
        (expected_gross - result.transaction_cost).rename("net_return"),
    )


def test_benchmarks_known_answer() -> None:
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.10]})
    result = benchmarks(returns)
    assert result.rebalanced_return.tolist() == pytest.approx([0.05, 0.05])
    assert result.static_return.iloc[0] == pytest.approx(0.05)
    assert result.static_weights.iloc[1, 0] == pytest.approx(1.1 / 2.1)


def test_cost_aware_rebalanced_benchmark_charges_actual_rebalancing() -> None:
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.10]})
    result = benchmarks(returns, rebalanced_cost_bps=10)
    expected_second_rebalance = abs(0.5 - 0.55 / 1.049) + abs(0.5 - 0.5 / 1.049)
    assert result.rebalanced_return.tolist() == pytest.approx([0.05, 0.05])
    assert result.cost_aware_rebalanced_turnover.tolist() == pytest.approx(
        [1.0, expected_second_rebalance + 1.05]
    )
    assert result.cost_aware_rebalanced_transaction_cost.tolist() == pytest.approx(
        [0.001, 0.001 * (expected_second_rebalance + 1.05)]
    )
    assert result.cost_aware_rebalanced_return.tolist() == pytest.approx(
        [
            0.05 - 0.001,
            0.05 - 0.001 * (expected_second_rebalance + 1.05),
        ]
    )


def test_benchmarks_reject_nonfinite_and_nonabsorbing_bankruptcy() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        benchmarks(pd.DataFrame({"A": [0.0, np.inf], "B": [0.0, 0.0]}))
    with pytest.raises(ValueError, match="after bankruptcy must be zero"):
        benchmarks(pd.DataFrame({"A": [-1.0, 0.25], "B": [-1.0, 0.25]}))
    with pytest.raises(ValueError, match="partially bankrupt"):
        benchmarks(pd.DataFrame({"A": [-1.0, 0.0], "B": [0.0, 0.10]}))


def test_portfolio_rejects_invalid_or_resurrected_asset_paths() -> None:
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    with pytest.raises(ValueError, match="below -100%"):
        evaluate_portfolio(
            weights,
            pd.DataFrame({"A": [-1.01, 0.0], "B": [0.0, 0.0]}),
            cost_bps=0,
        )
    with pytest.raises(ValueError, match="cannot resurrect"):
        evaluate_portfolio(
            weights,
            pd.DataFrame({"A": [-1.0, 0.0], "B": [0.0, 0.0]}),
            cost_bps=0,
        )
    bankrupt_short = pd.DataFrame({"A": [1.0]}, index=[0])
    with pytest.raises(ValueError, match="portfolio bankruptcy"):
        evaluate_portfolio(
            pd.DataFrame({"A": [-1.0]}, index=[0]),
            bankrupt_short,
            cost_bps=0,
            turnover_convention="drift_adjusted",
        )


def test_rebalanced_benchmark_bankruptcy_is_absorbing() -> None:
    result = benchmarks(pd.DataFrame({"A": [-1.0, 0.0], "B": [-1.0, 0.0]}))
    assert result.rebalanced_return.tolist() == pytest.approx([-1.0, 0.0])
    assert result.static_return.tolist() == pytest.approx([-1.0, 0.0])
    assert result.rebalanced_weights.iloc[1].sum() == 0.0
    assert result.static_weights.iloc[1].sum() == 0.0
