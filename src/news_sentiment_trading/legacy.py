"""Quarantined clean-room reproduction of the legacy homework calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from news_sentiment_trading.features import bull_bear_components


@dataclass(frozen=True)
class LegacyResult:
    ticker: str
    regime: str
    window: int
    z_threshold: float
    rvt_quantile: float
    strategy_arithmetic_return: float
    benchmark_arithmetic_return: float
    arithmetic_excess_return: float
    strategy_compounded_return: float
    benchmark_compounded_return: float
    one_way_trades: int
    observations: int


LEGACY_PARAMETERS: dict[str, tuple[int, float, float]] = {
    "FB": (50, 1.5, 0.50),
    "JPM": (20, 1.5, 0.50),
}


def _one_way_trades(position: pd.Series) -> int:
    clean = position.dropna()
    if clean.empty:
        return 0
    changes = clean.diff().abs().fillna(clean.abs()).sum()
    exit_size = abs(float(clean.iloc[-1]))
    return int(changes + exit_size)


def reproduce_legacy(
    panel: pd.DataFrame,
    ticker: str,
    long_short: bool,
    cost_rate: float = 0.01,
) -> LegacyResult:
    """Recompute the submitted arithmetic backtest without publishing its code."""

    if ticker not in LEGACY_PARAMETERS:
        raise ValueError(f"legacy reproduction is defined only for {sorted(LEGACY_PARAMETERS)}")
    asset = panel.xs(ticker, level="ticker").copy()
    components = bull_bear_components(panel).xs(ticker, level="ticker")
    denominator = components["BULL"] + components["BEAR"]
    ratio = (100.0 * components["BULL"] / denominator.where(denominator.ne(0))).ffill().bfill()
    legacy = pd.concat(
        [asset["Adj Close"].rename("price"), ratio.rename("ratio"), asset["RVT"]], axis=1
    ).dropna()
    window, z_threshold, rvt_quantile = LEGACY_PARAMETERS[ticker]
    mean = legacy["ratio"].rolling(window).mean()
    standard_deviation = legacy["ratio"].rolling(window).std(ddof=0).replace(0.0, np.nan)
    score = (legacy["ratio"] - mean) / standard_deviation
    active_news = legacy["RVT"].ge(legacy["RVT"].rolling(window).quantile(rvt_quantile))
    raw = pd.Series(np.nan, index=legacy.index, dtype=float)
    raw.loc[score.gt(z_threshold) & active_news] = 1.0
    raw.loc[score.lt(-z_threshold) & active_news] = -1.0 if long_short else 0.0
    position = raw.ffill().fillna(0.0).shift(1).rename("position")
    benchmark_return = legacy["price"].pct_change()
    strategy_return = benchmark_return * position
    joined = pd.concat(
        [strategy_return.rename("strategy"), benchmark_return.rename("benchmark"), position],
        axis=1,
    ).dropna()

    turnover = joined["position"].diff().abs().fillna(joined["position"].abs())
    turnover.iloc[-1] += abs(float(joined["position"].iloc[-1]))
    net_strategy = joined["strategy"] - turnover * cost_rate
    benchmark_cost = pd.Series(0.0, index=joined.index)
    benchmark_cost.iloc[0] = cost_rate
    benchmark_cost.iloc[-1] += cost_rate
    net_benchmark = joined["benchmark"] - benchmark_cost
    strategy_arithmetic = float(net_strategy.sum())
    benchmark_arithmetic = float(net_benchmark.sum())
    return LegacyResult(
        ticker=ticker,
        regime="long_short" if long_short else "long_only",
        window=window,
        z_threshold=z_threshold,
        rvt_quantile=rvt_quantile,
        strategy_arithmetic_return=strategy_arithmetic,
        benchmark_arithmetic_return=benchmark_arithmetic,
        arithmetic_excess_return=strategy_arithmetic - benchmark_arithmetic,
        strategy_compounded_return=float((1.0 + net_strategy).prod() - 1.0),
        benchmark_compounded_return=float((1.0 + net_benchmark).prod() - 1.0),
        one_way_trades=_one_way_trades(joined["position"]),
        observations=len(joined),
    )


def reproduce_fb_jpm(panel: pd.DataFrame) -> tuple[LegacyResult, ...]:
    return tuple(
        reproduce_legacy(panel, ticker, long_short)
        for ticker in ("FB", "JPM")
        for long_short in (False, True)
    )
