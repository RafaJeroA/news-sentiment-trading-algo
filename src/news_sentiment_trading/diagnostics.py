"""Predefined descriptive diagnostics that do not alter primary selection."""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from news_sentiment_trading.config import PrimaryConfig, SignalParameters
from news_sentiment_trading.data import (
    align_signal_to_return_end,
    aligned_forward_returns,
    asset_prices,
)
from news_sentiment_trading.features import build_features
from news_sentiment_trading.inference import daily_rank_ic, hac_mean
from news_sentiment_trading.metrics import (
    performance_metrics,
    relative_wealth_metrics,
    spread_statistics,
)
from news_sentiment_trading.portfolio import benchmarks, build_weights, evaluate_portfolio
from news_sentiment_trading.signals import bounded_holding, event_signal
from news_sentiment_trading.walk_forward import WalkForwardResult


def corrected_fb_jpm(panel: pd.DataFrame, config: PrimaryConfig) -> dict[str, dict[str, Any]]:
    """Re-evaluate the two legacy assets with corrected timing and accounting.

    This is a descriptive full-sample bridge to the homework, not confirmatory
    out-of-sample evidence. The parameter pairs are the exact legacy choices.
    """

    return_frame = aligned_forward_returns(panel, config.execution.convention)
    legacy_parameters = {
        "FB": SignalParameters(50, 1.5, 0.5),
        "JPM": SignalParameters(20, 1.5, 0.5),
    }
    output: dict[str, dict[str, Any]] = {}
    for ticker, parameters in legacy_parameters.items():
        features = build_features(panel, parameters)
        signal = bounded_holding(
            event_signal(features, parameters.z_threshold), parameters.holding_days
        )
        aligned = align_signal_to_return_end(signal, return_frame).loc[:, [ticker]]
        returns = return_frame.returns.loc[:, [ticker]]
        regimes: dict[str, Any] = {}
        for kind in ("long_only", "directional"):
            weights = build_weights(
                aligned,
                kind,
                1.0,
                config.portfolio.neutral_gross_limit,
            )
            cost_results: dict[str, Any] = {}
            for cost_bps in (config.research.primary_cost_bps, 100):
                portfolio = evaluate_portfolio(weights, returns, cost_bps, liquidate=True)
                cost_results[str(cost_bps)] = performance_metrics(
                    portfolio.net_return,
                    config.research.annualization,
                    config.research.risk_free_rate,
                    portfolio,
                )
            regimes[kind] = cost_results
        output[ticker] = {
            "scope": "descriptive full-sample corrected bridge; not OOS",
            "exposure_rule": "100% in the single active name; otherwise cash",
            "parameters": dataclasses.asdict(parameters),
            "buy_and_hold_benchmark": performance_metrics(
                benchmarks(returns).static_return,
                config.research.annualization,
                config.research.risk_free_rate,
            ),
            "regimes": regimes,
        }
    return output


def crisis_decomposition(
    result: WalkForwardResult,
    config: PrimaryConfig,
    crisis_start: str = "2020-02-19",
) -> dict[str, Any]:
    """Split the locked OOS sequence at the predefined COVID crisis boundary."""

    boundary = pd.Timestamp(crisis_start)
    segments = {
        "pre_crisis_and_crisis_excluded": result.portfolio.net_return.index < boundary,
        "crisis": result.portfolio.net_return.index >= boundary,
    }
    output: dict[str, Any] = {"crisis_start": crisis_start}
    for name, mask in segments.items():
        strategy = result.portfolio.net_return.loc[mask]
        benchmark = result.benchmark.rebalanced_return.loc[mask]
        output[name] = {
            "strategy": performance_metrics(
                strategy,
                config.research.annualization,
                config.research.risk_free_rate,
            ),
            "benchmark": performance_metrics(
                benchmark,
                config.research.annualization,
                config.research.risk_free_rate,
            ),
            "active_spread": spread_statistics(
                strategy - benchmark,
                config.research.annualization,
            ),
            "relative_wealth": relative_wealth_metrics(strategy, benchmark),
        }
    return output


def signal_coverage(result: WalkForwardResult) -> dict[str, Any]:
    """Return OOS score and event availability by asset and in aggregate."""

    rows: dict[str, Any] = {}
    for ticker in result.signal.columns:
        signal = result.signal[ticker]
        rows[str(ticker)] = {
            "observations": int(len(signal)),
            "score_available": int(result.score[ticker].notna().sum()),
            "positive_events": int(signal.gt(0).sum()),
            "negative_events": int(signal.lt(0).sum()),
            "active_events": int(signal.ne(0).sum()),
        }
    return {
        "by_asset": rows,
        "aggregate_active_asset_days": int(result.signal.ne(0).sum().sum()),
        "aggregate_long_asset_days": int(result.signal.gt(0).sum().sum()),
        "aggregate_short_asset_days": int(result.signal.lt(0).sum().sum()),
    }


def score_sorting(result: WalkForwardResult, groups: int = 5) -> list[dict[str, Any]]:
    """Pool date-wise score quantiles and their subsequent one-session returns."""

    if groups < 2:
        raise ValueError("groups must be at least two")
    observations: dict[int, list[float]] = {group: [] for group in range(1, groups + 1)}
    for date in result.score.index:
        joined = pd.concat(
            [
                result.score.loc[date].rename("score"),
                result.asset_returns.loc[date].rename("return"),
            ],
            axis=1,
        ).dropna()
        if len(joined) < groups or joined["score"].nunique() < groups:
            continue
        ranks = joined["score"].rank(method="first")
        buckets = pd.qcut(ranks, q=groups, labels=False) + 1
        for group in range(1, groups + 1):
            group_return = joined.loc[buckets.eq(group), "return"].astype(float).mean()
            observations[group].append(float(group_return))
    return [
        {
            "score_group": group,
            "date_observations": len(values),
            "mean_forward_return": float(np.mean(values)) if values else None,
        }
        for group, values in observations.items()
    ]


def signal_decay(
    panel: pd.DataFrame,
    result: WalkForwardResult,
    maximum_horizon: int = 3,
) -> list[dict[str, Any]]:
    """Measure rank IC for open-to-open horizons one through three."""

    if maximum_horizon < 1:
        raise ValueError("maximum_horizon must be positive")
    prices = asset_prices(panel, "Adjusted Open")
    date_positions = {pd.Timestamp(date): position for position, date in enumerate(prices.index)}
    rows: list[dict[str, Any]] = []
    for horizon in range(1, maximum_horizon + 1):
        forward = pd.DataFrame(
            np.nan,
            index=result.score.index,
            columns=result.score.columns,
            dtype=float,
        )
        for return_end in result.score.index:
            signal_date = pd.Timestamp(result.return_frame.signal_dates.loc[return_end])
            position = date_positions[signal_date]
            entry = position + 1
            exit_position = entry + horizon
            if exit_position < len(prices):
                forward.loc[return_end] = (
                    prices.iloc[exit_position].div(prices.iloc[entry]).sub(1.0)
                )
        rank_ic = daily_rank_ic(result.score, forward)
        valid = rank_ic.dropna()
        if valid.empty:
            rows.append(
                {
                    "horizon_sessions": horizon,
                    "observations": 0,
                    "mean_rank_ic": None,
                    "hac_p_value_two_sided": None,
                }
            )
        else:
            inference = hac_mean(rank_ic)
            rows.append(
                {
                    "horizon_sessions": horizon,
                    "observations": len(valid),
                    "mean_rank_ic": inference.mean,
                    "hac_p_value_two_sided": inference.p_value_two_sided,
                }
            )
    return rows


def parameter_stability(result: WalkForwardResult) -> dict[str, int]:
    """Count selected shared parameter combinations across outer folds."""

    labels = [
        (
            f"w={record.selected.window};z={record.selected.z_threshold};"
            f"q={record.selected.rvt_quantile}"
        )
        for record in result.selections
    ]
    return dict(sorted(Counter(labels).items()))
