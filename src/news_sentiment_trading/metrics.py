"""Conventional compounded performance and risk metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from news_sentiment_trading.portfolio import PortfolioResult


def validated_return_path(returns: pd.Series) -> pd.Series:
    """Validate simple returns and reject non-absorbing bankruptcy paths."""

    clean = returns.astype(float)
    values = clean.to_numpy(dtype=float)
    if clean.isna().any() or not np.isfinite(values).all():
        raise ValueError("returns contain missing or non-finite values")
    if (clean < -1.0).any():
        raise ValueError("a simple return below -100% is invalid")
    bankrupt = np.flatnonzero(values == -1.0)
    if len(bankrupt) and np.any(values[bankrupt[0] + 1 :] != 0.0):
        raise ValueError("returns after bankruptcy must be zero")
    return clean


def compounded_total_return(returns: pd.Series) -> float:
    """Return compounded total return after centralized path validation."""

    wealth = wealth_index(returns)
    return float(wealth.iloc[-1] - 1.0) if len(wealth) else 0.0


def wealth_index(returns: pd.Series) -> pd.Series:
    """Compound simple returns from an initial wealth of one."""

    clean = validated_return_path(returns)
    with np.errstate(over="ignore", invalid="ignore"):
        wealth = (1.0 + clean).cumprod()
    if not np.isfinite(wealth.to_numpy(dtype=float)).all():
        raise ValueError("compounded wealth is non-finite")
    wealth.name = "wealth"
    return wealth


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = wealth_index(returns)
    with_initial = pd.concat([pd.Series([1.0]), wealth.reset_index(drop=True)], ignore_index=True)
    peak = with_initial.cummax()
    drawdown = with_initial.div(peak).sub(1.0).iloc[1:]
    drawdown.index = returns.index
    drawdown.name = "drawdown"
    return drawdown


def maximum_drawdown(returns: pd.Series) -> float:
    drawdown = drawdown_series(returns)
    return float(drawdown.min()) if len(drawdown) else 0.0


def maximum_drawdown_duration(returns: pd.Series) -> int:
    drawdown = drawdown_series(returns)
    longest = 0
    current = 0
    for value in drawdown.to_numpy(dtype=float):
        if value < -1e-15:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _safe_std(series: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    return float(series.std(ddof=1))


def performance_metrics(
    returns: pd.Series,
    annualization: int = 252,
    risk_free_rate: float = 0.0,
    portfolio: PortfolioResult | None = None,
) -> dict[str, Any]:
    """Compute the preregistered performance metric set."""

    if annualization <= 0:
        raise ValueError("annualization must be positive")
    clean = validated_return_path(returns)
    wealth = wealth_index(clean)
    final_wealth = float(wealth.iloc[-1]) if len(wealth) else 1.0
    total_return = final_wealth - 1.0
    years = len(clean) / annualization
    cagr = (
        float(final_wealth ** (1.0 / years) - 1.0)
        if years > 0 and final_wealth > 0
        else (-1.0 if final_wealth == 0 else float("nan"))
    )
    annual_mean = float(clean.mean() * annualization) if len(clean) else float("nan")
    annual_volatility = _safe_std(clean) * math.sqrt(annualization)
    daily_rf = risk_free_rate / annualization
    excess = clean - daily_rf
    daily_std = _safe_std(excess)
    sharpe = (
        float(excess.mean() / daily_std * math.sqrt(annualization))
        if daily_std > 0 and math.isfinite(daily_std)
        else float("nan")
    )
    downside = excess.clip(upper=0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else float("nan")
    sortino = (
        float(excess.mean() / downside_std * math.sqrt(annualization))
        if downside_std > 0 and math.isfinite(downside_std)
        else float("nan")
    )
    nonzero = clean.ne(0.0)
    hit_rate = float(clean[nonzero].gt(0).mean()) if nonzero.any() else float("nan")
    output: dict[str, Any] = {
        "observations": int(len(clean)),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_mean_return": annual_mean,
        "annualized_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": maximum_drawdown(clean),
        "maximum_drawdown_duration": maximum_drawdown_duration(clean),
        "hit_rate_nonzero_return_days": hit_rate,
        "final_wealth": final_wealth,
    }
    if portfolio is not None:
        exposure_active = portfolio.gross_exposure.gt(0)
        trading_active = exposure_active | portfolio.turnover.gt(0)
        exposure_hit_rate = (
            float(clean.loc[exposure_active].gt(0).mean())
            if exposure_active.any()
            else float("nan")
        )
        trading_hit_rate = (
            float(clean.loc[trading_active].gt(0).mean()) if trading_active.any() else float("nan")
        )
        output.update(
            {
                "turnover_total": float(portfolio.turnover.sum()),
                "turnover_annualized": float(
                    portfolio.turnover.mean() * annualization if len(portfolio.turnover) else 0.0
                ),
                "long_exposure_mean": float(portfolio.long_exposure.mean()),
                "short_exposure_mean": float(portfolio.short_exposure.mean()),
                "gross_exposure_mean": float(portfolio.gross_exposure.mean()),
                "gross_exposure_max": float(portfolio.gross_exposure.max()),
                "net_exposure_mean": float(portfolio.net_exposure.mean()),
                "financing_balance_mean": float(portfolio.financing_balance.mean()),
                "unused_gross_capacity_mean": float(portfolio.unused_gross_capacity.mean()),
                "turnover_convention": portfolio.turnover_convention,
                "final_liquidation_turnover": portfolio.final_liquidation_turnover,
                "final_liquidation_nav_equivalent_turnover": (
                    portfolio.final_liquidation_nav_equivalent_turnover
                ),
                "exposure_active_days": int(exposure_active.sum()),
                "exposure_active_day_fraction": float(exposure_active.mean()),
                "hit_rate_exposure_days": exposure_hit_rate,
                "trading_active_days": int(trading_active.sum()),
                "trading_active_day_fraction": float(trading_active.mean()),
                "hit_rate_trading_days": trading_hit_rate,
                "nonzero_net_return_days": int(nonzero.sum()),
                "trade_count": int(portfolio.trade_count),
                "transaction_cost_total": float(portfolio.transaction_cost.sum()),
            }
        )
    return output


def spread_statistics(
    spread: pd.Series,
    annualization: int = 252,
) -> dict[str, Any]:
    """Describe an additive return spread without compounding it as wealth."""

    if annualization <= 0:
        raise ValueError("annualization must be positive")
    clean = spread.astype(float)
    if clean.isna().any() or not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("spread contains missing or non-finite values")
    standard_deviation = _safe_std(clean)
    annualized_volatility = standard_deviation * math.sqrt(annualization)
    sharpe = (
        float(clean.mean() / standard_deviation * math.sqrt(annualization))
        if standard_deviation > 0 and math.isfinite(standard_deviation)
        else float("nan")
    )
    return {
        "observations": int(len(clean)),
        "arithmetic_sum": float(clean.sum()),
        "mean_daily_spread": float(clean.mean()) if len(clean) else float("nan"),
        "annualized_mean_spread": (
            float(clean.mean() * annualization) if len(clean) else float("nan")
        ),
        "annualized_volatility": annualized_volatility,
        "sharpe_of_spread": sharpe,
        "positive_day_fraction": float(clean.gt(0).mean()) if len(clean) else float("nan"),
    }


def relative_wealth_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, Any]:
    """Compare compounded wealth levels without compounding an additive spread."""

    strategy, benchmark = strategy_returns.align(benchmark_returns, join="inner")
    if len(strategy) != len(strategy_returns) or len(benchmark) != len(benchmark_returns):
        raise ValueError("strategy and benchmark indexes must match exactly")
    strategy_wealth = wealth_index(strategy)
    benchmark_wealth = wealth_index(benchmark)
    strategy_final = float(strategy_wealth.iloc[-1]) if len(strategy_wealth) else 1.0
    benchmark_final = float(benchmark_wealth.iloc[-1]) if len(benchmark_wealth) else 1.0
    if benchmark_final <= 0:
        raise ValueError("benchmark final wealth must be positive")
    return {
        "observations": int(len(strategy)),
        "strategy_final_wealth": strategy_final,
        "benchmark_final_wealth": benchmark_final,
        "relative_total_return": strategy_final / benchmark_final - 1.0,
    }
