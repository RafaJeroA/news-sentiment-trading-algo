"""Deterministic runners for the preregistered descriptive robustness checks.

Aggregate summaries support restricted local reports. In-memory audit records retain
the weights, returns, costs, benchmark, folds, and selections needed to write ignored,
row-level empirical artifacts for every variant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import pandas as pd

from news_sentiment_trading.config import (
    ExecutionConvention,
    FeatureKind,
    MissingPolicy,
    PortfolioKind,
    PrimaryConfig,
)
from news_sentiment_trading.data import aligned_forward_returns
from news_sentiment_trading.metrics import compounded_total_return
from news_sentiment_trading.portfolio import (
    BenchmarkResult,
    PortfolioResult,
    benchmarks,
    build_weights,
    evaluate_portfolio,
)
from news_sentiment_trading.walk_forward import (
    WalkForwardResult,
    make_outer_slices,
    run_walk_forward,
)


@dataclass(frozen=True)
class RobustnessAuditRecord:
    """Auditable daily state and selection provenance for one robustness row."""

    group: str
    variant: str
    portfolio_kind: str
    cost_bps: int
    portfolio: PortfolioResult
    asset_returns: pd.DataFrame
    benchmark: BenchmarkResult
    dates: pd.DatetimeIndex
    fold_manifest: list[dict[str, Any]] | None = None
    selection_manifest: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class RobustnessResult:
    """Aggregate tables plus ignored row-level audit records for every variant."""

    summaries: pd.DataFrame
    selections: pd.DataFrame
    audit_records: tuple[RobustnessAuditRecord, ...]


def _compounded_total(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return compounded_total_return(returns)


def _summary_row(
    *,
    group: str,
    variant: str,
    portfolio_kind: str,
    cost_bps: int,
    portfolio: PortfolioResult,
    benchmark: BenchmarkResult,
    annualization: int,
    dates: pd.DatetimeIndex | None = None,
    excluded_ticker: str = "",
) -> dict[str, Any]:
    index = portfolio.net_return.index if dates is None else dates
    net_return = portfolio.net_return.reindex(index)
    benchmark_return = benchmark.rebalanced_return.reindex(index)
    if net_return.isna().any() or benchmark_return.isna().any():
        raise ValueError("robustness summary dates are not fully covered")
    active_return = net_return - benchmark_return
    gross_exposure = portfolio.gross_exposure.reindex(index)
    net_exposure = portfolio.net_exposure.reindex(index)
    turnover = portfolio.turnover.reindex(index)
    transaction_cost = portfolio.transaction_cost.reindex(index)
    long_exposure = portfolio.long_exposure.reindex(index)
    short_exposure = portfolio.short_exposure.reindex(index)
    financing_balance = portfolio.financing_balance.reindex(index)
    unused_gross_capacity = portfolio.unused_gross_capacity.reindex(index)
    active = gross_exposure.gt(0.0)
    observations = len(index)
    return {
        "group": group,
        "variant": variant,
        "excluded_ticker": excluded_ticker,
        "portfolio_kind": portfolio_kind,
        "cost_bps": int(cost_bps),
        "cost_direction": "one_way",
        "turnover_convention": portfolio.turnover_convention,
        "cost_charged_on": (
            "absolute target-weight change"
            if portfolio.turnover_convention == "target_weight"
            else "absolute drift-adjusted risky-weight change"
        ),
        "cost_applies_to": f"{group}/{variant} strategy net return and active spread",
        "observations": observations,
        "strategy_total_return": _compounded_total(net_return),
        "benchmark_total_return": _compounded_total(benchmark_return),
        "cost_aware_rebalanced_benchmark_total_return": _compounded_total(
            benchmark.cost_aware_rebalanced_return.reindex(index)
        ),
        "cost_aware_benchmark_cost_bps": benchmark.cost_aware_rebalanced_cost_bps,
        "cost_aware_benchmark_turnover_convention": "drift_adjusted",
        "annualized_mean_net_return": (
            float(net_return.mean() * annualization) if observations else float("nan")
        ),
        "annualized_mean_active_return": (
            float(active_return.mean() * annualization) if observations else float("nan")
        ),
        "active_days": int(active.sum()),
        "active_day_fraction": float(active.mean()) if observations else float("nan"),
        "gross_exposure_mean": (float(gross_exposure.mean()) if observations else float("nan")),
        "net_exposure_mean": float(net_exposure.mean()) if observations else float("nan"),
        "long_exposure_mean": float(long_exposure.mean()) if observations else float("nan"),
        "short_exposure_mean": float(short_exposure.mean()) if observations else float("nan"),
        "financing_balance_mean": (
            float(financing_balance.mean()) if observations else float("nan")
        ),
        "unused_gross_capacity_mean": (
            float(unused_gross_capacity.mean()) if observations else float("nan")
        ),
        "turnover_total": float(turnover.sum()),
        "transaction_cost_total": float(transaction_cost.sum()),
    }


def _selection_rows(group: str, variant: str, result: WalkForwardResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selection in result.selections:
        parameters = asdict(selection.selected)
        rows.append(
            {
                "group": group,
                "variant": variant,
                "fold_id": selection.fold_id,
                **parameters,
                "used_fixed_fallback": selection.used_fixed_fallback,
                "candidates_tested": selection.candidates_tested,
                "candidates_valid": selection.candidates_valid,
                "selection_data_end": selection.selection_data_end,
            }
        )
    return rows


def _variant_config(
    config: PrimaryConfig,
    *,
    feature_kind: FeatureKind | None = None,
    missing_policy: MissingPolicy | None = None,
    holding_days: int | None = None,
    convention: ExecutionConvention | None = None,
    primary_universe: tuple[str, ...] | None = None,
) -> PrimaryConfig:
    execution = replace(
        config.execution,
        convention=config.execution.convention if convention is None else convention,
        missing_policy=(
            config.execution.missing_policy if missing_policy is None else missing_policy
        ),
        holding_days=config.execution.holding_days if holding_days is None else holding_days,
    )
    features = replace(
        config.features,
        primary_kind=(config.features.primary_kind if feature_kind is None else feature_kind),
    )
    universe = (
        config.universe
        if primary_universe is None
        else replace(config.universe, primary=primary_universe)
    )
    return replace(config, execution=execution, features=features, universe=universe)


def _panel_without(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if "ticker" not in panel.index.names:
        raise ValueError("panel index must include a ticker level")
    keep = panel.index.get_level_values("ticker") != ticker
    reduced = panel.loc[keep].copy()
    if reduced.empty:
        raise ValueError("leave-one-out variant cannot remove the entire universe")
    return reduced


def _validate_primary_result(
    panel: pd.DataFrame, config: PrimaryConfig, result: WalkForwardResult
) -> None:
    """Fail closed if a cached primary result does not match panel/config invariants."""

    if tuple(result.asset_returns.columns) != config.universe.primary:
        raise ValueError("primary_result universe does not match configuration")
    full_returns = aligned_forward_returns(panel, config.execution.convention).returns
    expected_returns = full_returns.iloc[config.walk_forward.outer_initial_sessions :]
    if not result.asset_returns.equals(expected_returns):
        raise ValueError("primary_result returns do not match this panel/configuration")
    expected_slices = make_outer_slices(
        pd.DatetimeIndex(full_returns.index),
        config.walk_forward.outer_initial_sessions,
        config.walk_forward.outer_test_sessions,
    )
    expected_bounds = [
        (str(pd.Timestamp(block[0]).date()), str(pd.Timestamp(block[-1]).date()))
        for block in expected_slices
    ]
    observed_bounds = [(fold.test_start, fold.test_end) for fold in result.folds]
    if observed_bounds != expected_bounds:
        raise ValueError("primary_result fold boundaries do not match configuration")
    grid = set(config.features.parameters(config.execution))
    if any(record.selected not in grid for record in result.selections):
        raise ValueError("primary_result contains a selection outside the locked grid")
    expected_weights = build_weights(
        result.signal,
        config.portfolio.primary,
        config.portfolio.long_only_weight_per_asset,
        config.portfolio.neutral_gross_limit,
    )
    if not result.portfolio.weights.equals(expected_weights):
        raise ValueError("primary_result weights do not match the configured portfolio")


def run_robustness(
    panel: pd.DataFrame,
    config: PrimaryConfig,
    *,
    primary_result: WalkForwardResult | None = None,
    crisis_start: str | pd.Timestamp = "2020-02-19",
) -> RobustnessResult:
    """Run every locked descriptive robustness family deterministically.

    Portfolio/cost checks reuse the primary selected signal.  Feature, missingness,
    holding, execution, and leave-one-out checks rerun the full nested selection.
    The optional ``primary_result`` avoids recomputing an already completed primary
    run; callers are responsible for passing the result from this exact panel/config.
    """

    if config.features.primary_kind != "ratio":
        raise ValueError("robustness runner expects the preregistered ratio primary")
    if config.execution.missing_policy != "no_fill":
        raise ValueError("robustness runner expects no-fill as the primary policy")
    if config.execution.holding_days != 1:
        raise ValueError("robustness runner expects one-session primary holding")
    if config.execution.convention != "next_adjusted_open":
        raise ValueError("robustness runner expects next-adjusted-open primary execution")

    base = run_walk_forward(panel, config) if primary_result is None else primary_result
    _validate_primary_result(panel, config, base)
    rows: list[dict[str, Any]] = []
    selection_rows = _selection_rows("parameter_policy", "selected", base)
    audit_records: list[RobustnessAuditRecord] = []

    def append_audit_record(
        group: str,
        variant: str,
        portfolio_kind: str,
        cost_bps: int,
        portfolio: PortfolioResult,
        benchmark: BenchmarkResult,
        asset_returns: pd.DataFrame,
        *,
        dates: pd.DatetimeIndex | None = None,
        walk_forward: WalkForwardResult | None = None,
    ) -> None:
        selected_dates = portfolio.net_return.index if dates is None else dates
        audit_records.append(
            RobustnessAuditRecord(
                group=group,
                variant=variant,
                portfolio_kind=portfolio_kind,
                cost_bps=int(cost_bps),
                portfolio=portfolio,
                asset_returns=asset_returns,
                benchmark=benchmark,
                dates=pd.DatetimeIndex(selected_dates),
                fold_manifest=(None if walk_forward is None else walk_forward.fold_manifest()),
                selection_manifest=(
                    None if walk_forward is None else walk_forward.selection_manifest()
                ),
            )
        )

    def append_walk_forward_summary(
        group: str,
        variant: str,
        result: WalkForwardResult,
        *,
        excluded_ticker: str = "",
    ) -> None:
        rows.append(
            _summary_row(
                group=group,
                variant=variant,
                portfolio_kind=config.portfolio.primary,
                cost_bps=config.research.primary_cost_bps,
                portfolio=result.portfolio,
                benchmark=result.benchmark,
                annualization=config.research.annualization,
                excluded_ticker=excluded_ticker,
            )
        )
        selection_rows.extend(_selection_rows(group, variant, result))
        append_audit_record(
            group,
            variant,
            config.portfolio.primary,
            config.research.primary_cost_bps,
            result.portfolio,
            result.benchmark,
            result.asset_returns,
            walk_forward=result,
        )

    rows.append(
        _summary_row(
            group="parameter_policy",
            variant="selected",
            portfolio_kind=config.portfolio.primary,
            cost_bps=config.research.primary_cost_bps,
            portfolio=base.portfolio,
            benchmark=base.benchmark,
            annualization=config.research.annualization,
        )
    )
    append_audit_record(
        "parameter_policy",
        "selected",
        config.portfolio.primary,
        config.research.primary_cost_bps,
        base.portfolio,
        base.benchmark,
        base.asset_returns,
        walk_forward=base,
    )

    drift_portfolio = evaluate_portfolio(
        base.portfolio.weights,
        base.asset_returns,
        config.research.primary_cost_bps,
        liquidate=True,
        turnover_convention="drift_adjusted",
    )
    for variant, portfolio in (
        ("registered_target_weight", base.portfolio),
        ("post_v1_drift_adjusted", drift_portfolio),
    ):
        rows.append(
            _summary_row(
                group="turnover_convention",
                variant=variant,
                portfolio_kind=config.portfolio.primary,
                cost_bps=config.research.primary_cost_bps,
                portfolio=portfolio,
                benchmark=base.benchmark,
                annualization=config.research.annualization,
            )
        )
        append_audit_record(
            "turnover_convention",
            variant,
            config.portfolio.primary,
            config.research.primary_cost_bps,
            portfolio,
            base.benchmark,
            base.asset_returns,
            walk_forward=base,
        )
    rows.append(
        _summary_row(
            group="parameter_policy",
            variant="fixed_no_tuning",
            portfolio_kind=config.portfolio.primary,
            cost_bps=config.research.primary_cost_bps,
            portfolio=base.fixed_baseline_portfolio,
            benchmark=base.benchmark,
            annualization=config.research.annualization,
        )
    )
    append_audit_record(
        "parameter_policy",
        "fixed_no_tuning",
        config.portfolio.primary,
        config.research.primary_cost_bps,
        base.fixed_baseline_portfolio,
        base.benchmark,
        base.asset_returns,
    )

    portfolio_kinds: tuple[PortfolioKind, ...] = (
        "long_only",
        "market_neutral",
        "directional",
    )
    for portfolio_kind in portfolio_kinds:
        weights = build_weights(
            base.signal,
            portfolio_kind,
            config.portfolio.long_only_weight_per_asset,
            config.portfolio.neutral_gross_limit,
        )
        for cost_bps in config.research.cost_scenarios_bps:
            portfolio = evaluate_portfolio(weights, base.asset_returns, cost_bps, liquidate=True)
            scenario_benchmark = benchmarks(
                base.asset_returns,
                rebalanced_cost_bps=cost_bps,
            )
            rows.append(
                _summary_row(
                    group="portfolio_cost",
                    variant=f"{portfolio_kind}_{cost_bps}bps",
                    portfolio_kind=portfolio_kind,
                    cost_bps=cost_bps,
                    portfolio=portfolio,
                    benchmark=scenario_benchmark,
                    annualization=config.research.annualization,
                )
            )
            append_audit_record(
                "portfolio_cost",
                f"{portfolio_kind}_{cost_bps}bps",
                portfolio_kind,
                cost_bps,
                portfolio,
                scenario_benchmark,
                base.asset_returns,
                walk_forward=base,
            )

    rows.append(
        _summary_row(
            group="missing_policy",
            variant="no_fill",
            portfolio_kind=config.portfolio.primary,
            cost_bps=config.research.primary_cost_bps,
            portfolio=base.portfolio,
            benchmark=base.benchmark,
            annualization=config.research.annualization,
        )
    )
    append_audit_record(
        "missing_policy",
        "no_fill",
        config.portfolio.primary,
        config.research.primary_cost_bps,
        base.portfolio,
        base.benchmark,
        base.asset_returns,
        walk_forward=base,
    )
    selection_rows.extend(_selection_rows("missing_policy", "no_fill", base))
    missing_result = run_walk_forward(panel, _variant_config(config, missing_policy="ffill_1"))
    append_walk_forward_summary("missing_policy", "ffill_1", missing_result)

    rows.append(
        _summary_row(
            group="feature",
            variant="ratio",
            portfolio_kind=config.portfolio.primary,
            cost_bps=config.research.primary_cost_bps,
            portfolio=base.portfolio,
            benchmark=base.benchmark,
            annualization=config.research.annualization,
        )
    )
    append_audit_record(
        "feature",
        "ratio",
        config.portfolio.primary,
        config.research.primary_cost_bps,
        base.portfolio,
        base.benchmark,
        base.asset_returns,
        walk_forward=base,
    )
    selection_rows.extend(_selection_rows("feature", "ratio", base))
    feature_result = run_walk_forward(panel, _variant_config(config, feature_kind="log_ratio"))
    append_walk_forward_summary("feature", "log_ratio", feature_result)

    rows.append(
        _summary_row(
            group="holding",
            variant="1_session",
            portfolio_kind=config.portfolio.primary,
            cost_bps=config.research.primary_cost_bps,
            portfolio=base.portfolio,
            benchmark=base.benchmark,
            annualization=config.research.annualization,
        )
    )
    append_audit_record(
        "holding",
        "1_session",
        config.portfolio.primary,
        config.research.primary_cost_bps,
        base.portfolio,
        base.benchmark,
        base.asset_returns,
        walk_forward=base,
    )
    selection_rows.extend(_selection_rows("holding", "1_session", base))
    holding_result = run_walk_forward(panel, _variant_config(config, holding_days=3))
    append_walk_forward_summary("holding", "3_sessions", holding_result)

    rows.append(
        _summary_row(
            group="execution",
            variant="next_adjusted_open",
            portfolio_kind=config.portfolio.primary,
            cost_bps=config.research.primary_cost_bps,
            portfolio=base.portfolio,
            benchmark=base.benchmark,
            annualization=config.research.annualization,
        )
    )
    append_audit_record(
        "execution",
        "next_adjusted_open",
        config.portfolio.primary,
        config.research.primary_cost_bps,
        base.portfolio,
        base.benchmark,
        base.asset_returns,
        walk_forward=base,
    )
    selection_rows.extend(_selection_rows("execution", "next_adjusted_open", base))
    close_result = run_walk_forward(
        panel, _variant_config(config, convention="lagged_adjusted_close")
    )
    append_walk_forward_summary("execution", "lagged_adjusted_close", close_result)

    for ticker in config.universe.primary:
        retained = tuple(item for item in config.universe.primary if item != ticker)
        reduced_panel = _panel_without(panel, ticker)
        leave_one_out = run_walk_forward(
            reduced_panel,
            _variant_config(config, primary_universe=retained),
        )
        append_walk_forward_summary(
            "leave_one_out",
            f"exclude_{ticker}",
            leave_one_out,
            excluded_ticker=ticker,
        )

    crisis_timestamp = pd.Timestamp(crisis_start)
    oos_dates = pd.DatetimeIndex(base.portfolio.net_return.index)
    periods = (
        ("full_oos", oos_dates),
        ("pre_crisis_component", oos_dates[oos_dates < crisis_timestamp]),
        ("crisis_component", oos_dates[oos_dates >= crisis_timestamp]),
    )
    for variant, dates in periods:
        rows.append(
            _summary_row(
                group="crisis_period",
                variant=variant,
                portfolio_kind=config.portfolio.primary,
                cost_bps=config.research.primary_cost_bps,
                portfolio=base.portfolio,
                benchmark=base.benchmark,
                annualization=config.research.annualization,
                dates=dates,
            )
        )
        append_audit_record(
            "crisis_period",
            variant,
            config.portfolio.primary,
            config.research.primary_cost_bps,
            base.portfolio,
            base.benchmark,
            base.asset_returns,
            dates=dates,
            walk_forward=base,
        )

    pre_crisis_dates = oos_dates[oos_dates < crisis_timestamp]
    if len(pre_crisis_dates):
        crisis_excluded = evaluate_portfolio(
            base.portfolio.weights.loc[pre_crisis_dates],
            base.asset_returns.loc[pre_crisis_dates],
            config.research.primary_cost_bps,
            liquidate=True,
        )
        rows.append(
            _summary_row(
                group="crisis_period",
                variant="crisis_excluded_liquidated",
                portfolio_kind=config.portfolio.primary,
                cost_bps=config.research.primary_cost_bps,
                portfolio=crisis_excluded,
                benchmark=base.benchmark,
                annualization=config.research.annualization,
            )
        )
        append_audit_record(
            "crisis_period",
            "crisis_excluded_liquidated",
            config.portfolio.primary,
            config.research.primary_cost_bps,
            crisis_excluded,
            base.benchmark,
            base.asset_returns.loc[pre_crisis_dates],
            dates=pre_crisis_dates,
            walk_forward=base,
        )

    summaries = pd.DataFrame(rows).reset_index(drop=True)
    selections = pd.DataFrame(selection_rows).reset_index(drop=True)
    selections["selection_cost_bps"] = config.walk_forward.selection_cost_bps
    selections["selection_cost_direction"] = "one_way"
    selections["selection_cost_charged_on"] = "absolute target-weight change"
    selections["selection_cost_applies_to"] = "training-only parameter-selection objective"
    if len(audit_records) != len(summaries):
        raise AssertionError("every robustness summary must have one audit record")
    return RobustnessResult(
        summaries=summaries,
        selections=selections,
        audit_records=tuple(audit_records),
    )
