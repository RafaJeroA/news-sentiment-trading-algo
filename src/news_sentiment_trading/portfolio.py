"""Auditable panel weights, exposure, turnover, costs, and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from news_sentiment_trading.config import PortfolioKind

TurnoverConvention = Literal["target_weight", "drift_adjusted"]


@dataclass(frozen=True)
class TurnoverResult:
    """Auditable pre-trade state and one-way risky-asset turnover."""

    convention: TurnoverConvention
    pretrade_weights: pd.DataFrame
    pretrade_financing_balance: pd.Series
    turnover: pd.Series
    final_liquidation_turnover: float
    final_liquidation_nav_equivalent_turnover: float
    trade_count: int


@dataclass(frozen=True)
class PortfolioResult:
    weights: pd.DataFrame
    pretrade_weights: pd.DataFrame
    pretrade_financing_balance: pd.Series
    gross_return: pd.Series
    net_return: pd.Series
    transaction_cost: pd.Series
    turnover: pd.Series
    long_exposure: pd.Series
    short_exposure: pd.Series
    gross_exposure: pd.Series
    net_exposure: pd.Series
    financing_balance: pd.Series
    unused_gross_capacity: pd.Series
    turnover_convention: TurnoverConvention
    final_liquidation_turnover: float
    final_liquidation_nav_equivalent_turnover: float
    trade_count: int


@dataclass(frozen=True)
class BenchmarkResult:
    rebalanced_return: pd.Series
    static_return: pd.Series
    cost_aware_rebalanced_return: pd.Series
    cost_aware_rebalanced_turnover: pd.Series
    cost_aware_rebalanced_transaction_cost: pd.Series
    cost_aware_rebalanced_cost_bps: float
    rebalanced_weights: pd.DataFrame
    static_weights: pd.DataFrame


def long_only_weights(signal: pd.DataFrame, per_asset_cap: float = 0.20) -> pd.DataFrame:
    """Equal active long weights with a cap and explicit residual cash."""

    if not 0 < per_asset_cap <= 1:
        raise ValueError("per_asset_cap must be in (0, 1]")
    active = signal.gt(0)
    counts = active.sum(axis=1)
    allocation = counts.map(
        lambda count: 0.0 if count == 0 else min(per_asset_cap, 1.0 / float(count))
    )
    return active.astype(float).mul(allocation, axis=0)


def market_neutral_weights(
    signal: pd.DataFrame,
    per_asset_cap: float = 0.20,
    gross_limit: float = 1.0,
) -> pd.DataFrame:
    """Build balanced books, or hold cash if either side is unavailable."""

    if not 0 < per_asset_cap <= 1:
        raise ValueError("per_asset_cap must be in (0, 1]")
    if not 0 < gross_limit <= 1:
        raise ValueError("gross_limit must be in (0, 1]")
    output = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date, row in signal.iterrows():
        long_names = row.index[row.gt(0)]
        short_names = row.index[row.lt(0)]
        if len(long_names) == 0 or len(short_names) == 0:
            continue
        book_gross = min(
            gross_limit / 2.0,
            per_asset_cap * len(long_names),
            per_asset_cap * len(short_names),
        )
        output.loc[date, long_names] = book_gross / len(long_names)
        output.loc[date, short_names] = -book_gross / len(short_names)
    return output


def directional_weights(signal: pd.DataFrame, per_asset_cap: float = 0.20) -> pd.DataFrame:
    """Secondary directional long-short weights with no neutrality claim."""

    if not 0 < per_asset_cap <= 1:
        raise ValueError("per_asset_cap must be in (0, 1]")
    if not signal.isin((-1, 0, 1)).all().all():
        raise ValueError("directional signal values must be in {-1, 0, 1}")
    active = signal.ne(0)
    counts = active.sum(axis=1)
    allocation = counts.map(
        lambda count: 0.0 if count == 0 else min(per_asset_cap, 1.0 / float(count))
    )
    return signal.astype(float).mul(allocation, axis=0)


def build_weights(
    signal: pd.DataFrame,
    kind: PortfolioKind,
    per_asset_cap: float = 0.20,
    neutral_gross_limit: float = 1.0,
) -> pd.DataFrame:
    if kind == "long_only":
        return long_only_weights(signal, per_asset_cap)
    if kind == "market_neutral":
        return market_neutral_weights(signal, per_asset_cap, neutral_gross_limit)
    if kind == "directional":
        return directional_weights(signal, per_asset_cap)
    raise ValueError(f"unsupported portfolio kind: {kind}")


def _aligned_portfolio_inputs(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    if not weights.index.is_unique or not weights.columns.is_unique:
        raise ValueError("portfolio weight labels must be unique")
    if not asset_returns.index.is_unique or not asset_returns.columns.is_unique:
        raise ValueError("asset return labels must be unique")
    aligned_returns = asset_returns.reindex(index=weights.index, columns=weights.columns)
    if aligned_returns.isna().any().any():
        raise ValueError("asset returns do not fully cover portfolio weights")
    return_values = aligned_returns.to_numpy(dtype=float)
    if not np.isfinite(return_values).all():
        raise ValueError("asset returns contain non-finite values")
    if (return_values < -1.0).any():
        raise ValueError("asset simple returns cannot be below -100%")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("weights contain non-finite values")
    for ticker in aligned_returns.columns:
        path = aligned_returns[ticker].to_numpy(dtype=float)
        bankrupt = np.flatnonzero(path == -1.0)
        if len(bankrupt) and np.any(path[bankrupt[0] + 1 :] != 0.0):
            raise ValueError(f"{ticker}: asset returns after bankruptcy must be zero")
        if len(bankrupt) and weights[ticker].iloc[bankrupt[0] + 1 :].abs().gt(1e-12).any():
            raise ValueError(f"{ticker}: portfolio targets cannot resurrect a bankrupt asset")
    return aligned_returns


def target_weight_turnover(weights: pd.DataFrame, *, liquidate: bool = True) -> TurnoverResult:
    """Reproduce the locked v1.0 target-to-target turnover convention."""

    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("weights contain non-finite values")
    pretrade = weights.shift(1).fillna(0.0)
    changes = weights.sub(pretrade)
    turnover = changes.abs().sum(axis=1).rename("turnover")
    trade_count = int(changes.ne(0.0).sum().sum())
    final_liquidation = 0.0
    if liquidate and len(weights):
        final_liquidation = float(weights.iloc[-1].abs().sum())
        turnover.iloc[-1] += final_liquidation
        trade_count += int(weights.iloc[-1].ne(0.0).sum())
    return TurnoverResult(
        convention="target_weight",
        pretrade_weights=pretrade,
        pretrade_financing_balance=(1.0 - pretrade.sum(axis=1)).rename(
            "pretrade_financing_balance"
        ),
        turnover=turnover,
        final_liquidation_turnover=final_liquidation,
        final_liquidation_nav_equivalent_turnover=final_liquidation,
        trade_count=trade_count,
    )


def _drifted_risky_weights(
    target: pd.Series, returns: pd.Series, transaction_cost: float
) -> tuple[pd.Series, float]:
    """Return risky weights after one interval, normalized by post-cost NAV."""

    risky_values = target.mul(1.0 + returns)
    ending_nav = (
        1.0
        - transaction_cost
        + float(np.dot(target.to_numpy(dtype=float), returns.to_numpy(dtype=float)))
    )
    if ending_nav <= 1e-12:
        if abs(ending_nav) <= 1e-12 and np.allclose(
            risky_values.to_numpy(dtype=float), 0.0, atol=1e-12, rtol=0.0
        ):
            return pd.Series(0.0, index=target.index, dtype=float), 0.0
        raise ValueError("cannot normalize drifted weights after portfolio bankruptcy")
    return risky_values.div(ending_nav), ending_nav


def drift_adjusted_turnover(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    cost_bps: float = 0.0,
    liquidate: bool = True,
) -> TurnoverResult:
    """Measure trading from return-drifted pre-trade risky weights.

    Row ``k`` targets are established before row ``k`` returns.  Consequently the
    pre-trade state for row ``k + 1`` is row ``k`` targets drifted by row ``k``
    returns. Transaction costs are funded from the residual financing balance,
    so prior rebalance cost is included in the next pre-trade denominator. The
    final liquidation cost is converted to start-of-row NAV units before it is
    added to row turnover, preserving ``cost = rate * turnover`` exactly.
    """

    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    aligned_returns = _aligned_portfolio_inputs(weights, asset_returns)
    pretrade = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    turnover = pd.Series(0.0, index=weights.index, name="turnover")
    cost_rate = cost_bps / 10_000.0
    trade_count = 0
    ending_weights = pd.Series(0.0, index=weights.columns, dtype=float)
    ending_nav = 1.0
    for position in range(len(weights)):
        changes = weights.iloc[position].sub(pretrade.iloc[position])
        rebalance_turnover = float(changes.abs().sum())
        turnover.iloc[position] = rebalance_turnover
        trade_count += int(changes.abs().gt(1e-12).sum())
        ending_weights, ending_nav = _drifted_risky_weights(
            weights.iloc[position],
            aligned_returns.iloc[position],
            cost_rate * rebalance_turnover,
        )
        if position + 1 < len(weights):
            pretrade.iloc[position + 1] = ending_weights
    final_liquidation = 0.0
    nav_equivalent_liquidation = 0.0
    if liquidate and len(weights):
        final_liquidation = float(ending_weights.abs().sum())
        nav_equivalent_liquidation = ending_nav * final_liquidation
        turnover.iloc[-1] += nav_equivalent_liquidation
        trade_count += int(ending_weights.abs().gt(1e-12).sum())
    return TurnoverResult(
        convention="drift_adjusted",
        pretrade_weights=pretrade,
        pretrade_financing_balance=(1.0 - pretrade.sum(axis=1)).rename(
            "pretrade_financing_balance"
        ),
        turnover=turnover,
        final_liquidation_turnover=final_liquidation,
        final_liquidation_nav_equivalent_turnover=nav_equivalent_liquidation,
        trade_count=trade_count,
    )


def evaluate_portfolio(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float,
    liquidate: bool = True,
    turnover_convention: TurnoverConvention = "target_weight",
) -> PortfolioResult:
    """Evaluate targets under an explicit turnover convention.

    ``target_weight`` is the preregistered v1.0 convention and remains the
    default. ``drift_adjusted`` is a post-v1.0 accounting robustness. Costs are
    one-way; initial entry and, when requested, final liquidation are included.
    """

    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    aligned_returns = _aligned_portfolio_inputs(weights, asset_returns)

    long_exposure = weights.clip(lower=0.0).sum(axis=1).rename("long_exposure")
    short_exposure = (-weights.clip(upper=0.0)).sum(axis=1).rename("short_exposure")
    gross_exposure = weights.abs().sum(axis=1).rename("gross_exposure")
    net_exposure = weights.sum(axis=1).rename("net_exposure")
    if (gross_exposure > 1.0 + 1e-10).any():
        raise ValueError("gross exposure exceeds 100%")

    if turnover_convention == "target_weight":
        turnover_result = target_weight_turnover(weights, liquidate=liquidate)
    elif turnover_convention == "drift_adjusted":
        turnover_result = drift_adjusted_turnover(
            weights,
            aligned_returns,
            cost_bps=cost_bps,
            liquidate=liquidate,
        )
    else:
        raise ValueError(f"unsupported turnover convention: {turnover_convention}")

    cost_rate = cost_bps / 10_000.0
    transaction_cost = (turnover_result.turnover * cost_rate).rename("transaction_cost")
    gross_return = (weights * aligned_returns).sum(axis=1).rename("gross_return")
    net_return = gross_return.sub(transaction_cost).rename("net_return")
    if (net_return < -1.0 - 1e-12).any():
        raise ValueError("transaction costs produce a portfolio return below -100%")
    financing_balance = (1.0 - net_exposure).rename("financing_balance")
    unused_gross_capacity = (1.0 - gross_exposure).rename("unused_gross_capacity")
    return PortfolioResult(
        weights=weights,
        pretrade_weights=turnover_result.pretrade_weights,
        pretrade_financing_balance=turnover_result.pretrade_financing_balance,
        gross_return=gross_return,
        net_return=net_return,
        transaction_cost=transaction_cost,
        turnover=turnover_result.turnover,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        financing_balance=financing_balance,
        unused_gross_capacity=unused_gross_capacity,
        turnover_convention=turnover_result.convention,
        final_liquidation_turnover=turnover_result.final_liquidation_turnover,
        final_liquidation_nav_equivalent_turnover=(
            turnover_result.final_liquidation_nav_equivalent_turnover
        ),
        trade_count=turnover_result.trade_count,
    )


def benchmarks(asset_returns: pd.DataFrame, *, rebalanced_cost_bps: float = 0.0) -> BenchmarkResult:
    """Return registered gross benchmarks and a cost-aware rebalance robustness."""

    if asset_returns.empty:
        raise ValueError("asset return panel is empty")
    if len(asset_returns.columns) == 0:
        raise ValueError("asset return panel has no assets")
    if rebalanced_cost_bps < 0:
        raise ValueError("rebalanced_cost_bps cannot be negative")
    if asset_returns.isna().any().any():
        raise ValueError("benchmark returns contain missing values")
    values = asset_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("benchmark returns contain non-finite values")
    if (values < -1.0).any():
        raise ValueError("benchmark simple returns cannot be below -100%")
    for ticker in asset_returns.columns:
        path = asset_returns[ticker].to_numpy(dtype=float)
        bankrupt = np.flatnonzero(path == -1.0)
        if len(bankrupt) and np.any(path[bankrupt[0] + 1 :] != 0.0):
            raise ValueError(f"{ticker}: benchmark returns after bankruptcy must be zero")
    for position in range(len(asset_returns) - 1):
        row = asset_returns.iloc[position]
        if row.eq(-1.0).any() and not row.eq(-1.0).all():
            raise ValueError(
                "cannot rebalance a partially bankrupt universe without a delisting rule"
            )

    tickers = list(asset_returns.columns)
    initial = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    rebalanced_weights = pd.DataFrame(index=asset_returns.index, columns=tickers, dtype=float)
    rebalanced_values: list[float] = []
    rebalanced_current = initial.copy()
    for date, returns_row in asset_returns.iterrows():
        rebalanced_weights.loc[date] = rebalanced_current
        portfolio_return = float((rebalanced_current * returns_row).sum())
        rebalanced_values.append(portfolio_return)
        if portfolio_return <= -1.0:
            rebalanced_current = pd.Series(0.0, index=tickers)
    rebalanced_return = pd.Series(
        rebalanced_values,
        index=asset_returns.index,
        dtype=float,
        name="rebalanced_equal_weight",
    )

    static_weights = pd.DataFrame(index=asset_returns.index, columns=tickers, dtype=float)
    static_values: list[float] = []
    current = initial.copy()
    for date, returns_row in asset_returns.iterrows():
        static_weights.loc[date] = current
        portfolio_return = float((current * returns_row).sum())
        static_values.append(portfolio_return)
        denominator = 1.0 + portfolio_return
        if denominator <= 0:
            current = pd.Series(0.0, index=tickers)
        else:
            current = current.mul(1.0 + returns_row).div(denominator)
    static_return = pd.Series(
        static_values,
        index=asset_returns.index,
        dtype=float,
        name="static_equal_weight",
    )
    cost_aware = evaluate_portfolio(
        rebalanced_weights,
        asset_returns,
        rebalanced_cost_bps,
        liquidate=True,
        turnover_convention="drift_adjusted",
    )
    return BenchmarkResult(
        rebalanced_return=rebalanced_return,
        static_return=static_return,
        cost_aware_rebalanced_return=cost_aware.net_return.rename(
            "cost_aware_rebalanced_equal_weight"
        ),
        cost_aware_rebalanced_turnover=cost_aware.turnover.rename("cost_aware_rebalanced_turnover"),
        cost_aware_rebalanced_transaction_cost=cost_aware.transaction_cost.rename(
            "cost_aware_rebalanced_transaction_cost"
        ),
        cost_aware_rebalanced_cost_bps=float(rebalanced_cost_bps),
        rebalanced_weights=rebalanced_weights,
        static_weights=static_weights,
    )
