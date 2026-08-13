from __future__ import annotations

import math

import pandas as pd
import pytest

from news_sentiment_trading.inference import hac_mean, holm_adjust
from news_sentiment_trading.metrics import (
    drawdown_series,
    maximum_drawdown,
    maximum_drawdown_duration,
    performance_metrics,
    relative_wealth_metrics,
    spread_statistics,
    wealth_index,
)


def test_compounded_returns_not_arithmetic_sum() -> None:
    returns = pd.Series([0.10, -0.10])
    assert wealth_index(returns).iloc[-1] == pytest.approx(0.99)
    assert performance_metrics(returns)["total_return"] == pytest.approx(-0.01)


def test_active_spread_is_not_compounded_as_investable_wealth() -> None:
    strategy = pd.Series([0.10, -0.10])
    benchmark = pd.Series([0.05, 0.05])
    spread = strategy - benchmark
    statistics = spread_statistics(spread)
    relative = relative_wealth_metrics(strategy, benchmark)
    assert statistics["arithmetic_sum"] == pytest.approx(-0.10)
    assert "total_return" not in statistics
    assert relative["relative_total_return"] == pytest.approx(0.99 / 1.1025 - 1.0)


def test_drawdown_known_answer() -> None:
    returns = pd.Series([0.10, -0.20, 0.05, 0.20])
    drawdown = drawdown_series(returns)
    assert drawdown.iloc[1] == pytest.approx(-0.20)
    assert maximum_drawdown(returns) == pytest.approx(-0.20)
    assert maximum_drawdown_duration(returns) == 2


def test_flat_strategy_edge_cases() -> None:
    metrics = performance_metrics(pd.Series([0.0] * 20))
    assert metrics["total_return"] == 0.0
    assert metrics["maximum_drawdown"] == 0.0
    assert math.isnan(metrics["sharpe_ratio"])
    assert math.isnan(metrics["hit_rate_nonzero_return_days"])


def test_metrics_reject_nonabsorbing_post_bankruptcy_returns() -> None:
    with pytest.raises(ValueError, match="after bankruptcy must be zero"):
        performance_metrics(pd.Series([0.10, -1.0, 0.50]))

    metrics = performance_metrics(pd.Series([0.10, -1.0, 0.0]))
    assert metrics["total_return"] == -1.0
    assert metrics["hit_rate_nonzero_return_days"] == pytest.approx(0.5)


def test_compounding_rejects_nonfinite_wealth_overflow() -> None:
    with pytest.raises(ValueError, match="compounded wealth is non-finite"):
        wealth_index(pd.Series([1e308, 1e308]))


def test_hac_constant_series_has_zero_uncertainty() -> None:
    result = hac_mean(pd.Series([0.01] * 100), lags=5)
    assert result.mean == pytest.approx(0.01)
    assert result.standard_error == pytest.approx(0.0, abs=1e-15)


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm_adjust([0.01, 0.04])
    assert adjusted == pytest.approx([0.02, 0.04])
