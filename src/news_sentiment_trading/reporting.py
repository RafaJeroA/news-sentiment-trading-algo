"""Deterministic machine-readable artifacts and generated research reports."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from news_sentiment_trading.config import PortfolioKind, PrimaryConfig
from news_sentiment_trading.diagnostics import (
    corrected_fb_jpm,
    crisis_decomposition,
    parameter_stability,
    score_sorting,
    signal_coverage,
    signal_decay,
)
from news_sentiment_trading.inference import (
    daily_rank_ic,
    hac_alpha_beta,
    hac_mean,
    holm_adjust,
    moving_block_bootstrap_mean_ci,
)
from news_sentiment_trading.metrics import (
    performance_metrics,
    relative_wealth_metrics,
    spread_statistics,
    wealth_index,
)
from news_sentiment_trading.portfolio import (
    PortfolioResult,
    TurnoverConvention,
    build_weights,
    evaluate_portfolio,
)
from news_sentiment_trading.robustness import RobustnessResult
from news_sentiment_trading.walk_forward import WalkForwardResult

PRIMARY_ARTIFACT_FILES = frozenset(
    {
        "asset_returns.csv",
        "benchmark_returns.csv",
        "candidate_scores.json",
        "configuration.json",
        "corrected_fb_jpm.json",
        "cost_scenarios.json",
        "crisis_decomposition.json",
        "directional_daily.csv",
        "directional_pretrade_weights.csv",
        "directional_weights.csv",
        "fixed_baseline_daily.csv",
        "fixed_baseline_pretrade_weights.csv",
        "fixed_baseline_weights.csv",
        "fold_manifest.json",
        "generated_figure_manifest.json",
        "inference.json",
        "long_only_daily.csv",
        "long_only_pretrade_weights.csv",
        "long_only_weights.csv",
        "market_neutral_daily.csv",
        "market_neutral_pretrade_weights.csv",
        "market_neutral_weights.csv",
        "metrics.json",
        "parameter_stability.json",
        "parameters_by_date.csv",
        "per_asset_attribution.json",
        "per_fold_metrics.json",
        "portfolio_daily.csv",
        "portfolio_pretrade_weights.csv",
        "portfolio_variants.json",
        "portfolio_weights.csv",
        "rank_ic.csv",
        "run_metadata.json",
        "score_sorting.json",
        "scores.csv",
        "selected_parameters.json",
        "signal_coverage.json",
        "signal_decay.json",
        "signals.csv",
        "timing_by_return_end.csv",
        "wealth.png",
    }
)


@dataclass(frozen=True)
class ArtifactProvenance:
    """Content-addressed provenance for one artifact bundle.

    This record establishes file identity and research lineage only. It does not
    grant redistribution, disclosure, or publication permission.
    """

    artifact_manifest_sha256: str
    git_commit: str
    preregistration_commit: str
    configuration_hash: str
    environment_lock_hash: str
    source_gate_hash: str
    input_hashes: dict[str, str]


def load_artifact_provenance(path: str | Path) -> ArtifactProvenance:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {field.name for field in dataclasses.fields(ArtifactProvenance)}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("artifact provenance fields differ from the required schema")
    if not isinstance(raw["input_hashes"], dict):
        raise ValueError("artifact provenance input_hashes must be an object")
    return ArtifactProvenance(
        artifact_manifest_sha256=str(raw["artifact_manifest_sha256"]),
        git_commit=str(raw["git_commit"]),
        preregistration_commit=str(raw["preregistration_commit"]),
        configuration_hash=str(raw["configuration_hash"]),
        environment_lock_hash=str(raw["environment_lock_hash"]),
        source_gate_hash=str(raw["source_gate_hash"]),
        input_hashes={str(key): str(value) for key, value in raw["input_hashes"].items()},
    )


def write_artifact_provenance(artifact_dir: str | Path, destination: str | Path) -> Path:
    """Write a separate, write-once content address for independent review."""

    artifacts = Path(artifact_dir).resolve()
    output = Path(destination).resolve()
    if output.is_relative_to(artifacts):
        raise ValueError("artifact provenance must be stored outside the artifact bundle")
    if output.exists():
        raise FileExistsError(f"artifact provenance already exists: {output}")
    metadata = json.loads((artifacts / "run_metadata.json").read_text(encoding="utf-8"))
    payload = {
        "artifact_manifest_sha256": file_hash(artifacts / "ARTIFACT_MANIFEST.json"),
        "git_commit": metadata["git_commit"],
        "preregistration_commit": metadata["preregistration_commit"],
        "configuration_hash": metadata["configuration_hash"],
        "environment_lock_hash": metadata["environment_lock_hash"],
        "source_gate_hash": metadata["source_gate_hash"],
        "input_hashes": metadata["input_hashes"],
    }
    return write_json(output, payload)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def write_csv(
    path: str | Path,
    frame: pd.DataFrame | pd.Series,
    *,
    include_index: bool = True,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = frame.to_frame() if isinstance(frame, pd.Series) else frame
    materialized.to_csv(
        output,
        index=include_index,
        date_format="%Y-%m-%d",
        float_format="%.12g",
        lineterminator="\n",
    )
    return output


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one_way_cost_basis(
    rate_bps: int,
    applies_to: str,
    turnover_convention: TurnoverConvention = "target_weight",
) -> dict[str, Any]:
    charged_on = {
        "target_weight": "absolute target-weight change",
        "drift_adjusted": "absolute drift-adjusted risky-weight change",
    }[turnover_convention]
    return {
        "applies_to": applies_to,
        "charged_on": charged_on,
        "direction": "one_way",
        "rate_bps": int(rate_bps),
        "turnover_convention": turnover_convention,
    }


def _with_cost_basis_columns(
    frame: pd.DataFrame,
    basis: dict[str, Any],
    *,
    prefix: str,
) -> pd.DataFrame:
    output = frame.copy()
    output[f"{prefix}_rate_bps"] = int(basis["rate_bps"])
    output[f"{prefix}_direction"] = str(basis["direction"])
    output[f"{prefix}_charged_on"] = str(basis["charged_on"])
    output[f"{prefix}_applies_to"] = str(basis["applies_to"])
    output[f"{prefix}_turnover_convention"] = str(basis["turnover_convention"])
    return output


def _metrics_and_inference(
    result: WalkForwardResult, config: PrimaryConfig
) -> tuple[dict[str, Any], dict[str, Any], pd.Series]:
    strategy = result.portfolio.net_return
    benchmark = result.benchmark.rebalanced_return
    active_return = (strategy - benchmark).rename("active_return")
    rank_ic = daily_rank_ic(result.score, result.asset_returns)
    ic_hac = hac_mean(rank_ic)
    active_hac = hac_mean(active_return)
    adjusted = holm_adjust([ic_hac.p_value_two_sided, active_hac.p_value_two_sided])
    beta = hac_alpha_beta(strategy, benchmark)
    net_bootstrap = moving_block_bootstrap_mean_ci(
        strategy,
        samples=10_000,
        block_length=10,
        seed=config.research.seed,
    )
    active_bootstrap = moving_block_bootstrap_mean_ci(
        active_return,
        samples=10_000,
        block_length=10,
        seed=config.research.seed + 1,
    )
    hac_lags = sorted({0, ic_hac.lags, active_hac.lags, 5, 10})
    hac_sensitivity = {
        str(lag): {
            "rank_ic": hac_mean(rank_ic, lags=lag),
            "active_return": hac_mean(active_return, lags=lag),
        }
        for lag in hac_lags
        if lag < min(ic_hac.observations, active_hac.observations)
    }
    block_sensitivity = {
        str(block_length): {
            "samples": 2_000,
            "strategy_mean_ci": moving_block_bootstrap_mean_ci(
                strategy,
                samples=2_000,
                block_length=block_length,
                seed=config.research.seed + 100 + block_length,
            ),
            "active_mean_ci": moving_block_bootstrap_mean_ci(
                active_return,
                samples=2_000,
                block_length=block_length,
                seed=config.research.seed + 200 + block_length,
            ),
        }
        for block_length in (5, 10, 20)
    }
    strategy_cost_basis = _one_way_cost_basis(
        config.research.primary_cost_bps,
        "strategy net returns, active-return statistics, and relative wealth",
    )
    inference_cost_basis = _one_way_cost_basis(
        config.research.primary_cost_bps,
        (
            "strategy/active bootstrap inference, active-return HAC and Holm inference, "
            "benchmark beta/intercept, and the trading-hypothesis decision"
        ),
    )
    metrics = {
        "cost_basis": strategy_cost_basis,
        "strategy_net": performance_metrics(
            strategy,
            config.research.annualization,
            config.research.risk_free_rate,
            result.portfolio,
        ),
        "strategy_gross": performance_metrics(
            result.portfolio.gross_return,
            config.research.annualization,
            config.research.risk_free_rate,
        ),
        "fixed_baseline_net": performance_metrics(
            result.fixed_baseline_portfolio.net_return,
            config.research.annualization,
            config.research.risk_free_rate,
            result.fixed_baseline_portfolio,
        ),
        "rebalanced_equal_weight": performance_metrics(
            benchmark,
            config.research.annualization,
            config.research.risk_free_rate,
        ),
        "static_equal_weight": performance_metrics(
            result.benchmark.static_return,
            config.research.annualization,
            config.research.risk_free_rate,
        ),
        "cost_aware_rebalanced_equal_weight": {
            **performance_metrics(
                result.benchmark.cost_aware_rebalanced_return,
                config.research.annualization,
                config.research.risk_free_rate,
            ),
            "turnover_total": float(result.benchmark.cost_aware_rebalanced_turnover.sum()),
            "transaction_cost_total": float(
                result.benchmark.cost_aware_rebalanced_transaction_cost.sum()
            ),
            "cost_basis": _one_way_cost_basis(
                int(result.benchmark.cost_aware_rebalanced_cost_bps),
                "supplemental cost-aware rebalanced equal-weight benchmark",
                "drift_adjusted",
            ),
        },
        "active_spread": spread_statistics(
            active_return,
            config.research.annualization,
        ),
        "relative_wealth": relative_wealth_metrics(strategy, benchmark),
    }
    inference = {
        "cost_basis": inference_cost_basis,
        "rank_ic_hac": ic_hac,
        "active_return_hac": active_hac,
        "holm_adjusted_p_values": {
            "rank_ic": adjusted[0],
            "active_return": adjusted[1],
        },
        "strategy_mean_block_bootstrap_ci": net_bootstrap,
        "active_mean_block_bootstrap_ci": active_bootstrap,
        "benchmark_exposure": beta,
        "inference_specification": {
            "confidence_level": 0.95,
            "hac": {
                "method": "Newey-West with Bartlett weights",
                "automatic_lag_rule": "floor(4*(T/100)^(2/9))",
                "rank_ic_lags": ic_hac.lags,
                "active_return_lags": active_hac.lags,
            },
            "moving_block_bootstrap": {
                "method": "circular synchronized-date moving-block bootstrap",
                "samples": 10_000,
                "block_length": 10,
                "strategy_seed": config.research.seed,
                "active_return_seed": config.research.seed + 1,
            },
            "limitations": [
                "HAC and block choices are fixed approximations in a short dependent sample.",
                "The block bootstrap assumes local stationarity within the resampled series.",
                "Sensitivity results are descriptive and do not alter the preregistered test.",
            ],
        },
        "assumption_sensitivity": {
            "hac_by_lag": hac_sensitivity,
            "moving_block_by_length": block_sensitivity,
        },
        "multiple_testing_disclosure": {
            "parameter_configurations_per_outer_fold": len(
                config.features.parameters(config.execution)
            ),
            "confirmatory_p_values": 2,
            "adjustment": "Holm family-wise correction",
        },
    }
    return metrics, inference, rank_ic


def _per_fold_metrics(result: WalkForwardResult, config: PrimaryConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in result.folds:
        start = pd.Timestamp(fold.test_start)
        end = pd.Timestamp(fold.test_end)
        index = result.portfolio.net_return.loc[start:end].index
        strategy = result.portfolio.net_return.loc[index]
        strategy_gross = result.portfolio.gross_return.loc[index]
        benchmark = result.benchmark.rebalanced_return.loc[index]
        rows.append(
            {
                "fold_id": fold.fold_id,
                "start": fold.test_start,
                "end": fold.test_end,
                "cost_basis": _one_way_cost_basis(
                    config.research.primary_cost_bps,
                    "fold strategy net return and active spread",
                ),
                "strategy": performance_metrics(
                    strategy,
                    config.research.annualization,
                    config.research.risk_free_rate,
                ),
                "strategy_net": performance_metrics(
                    strategy,
                    config.research.annualization,
                    config.research.risk_free_rate,
                ),
                "strategy_gross": performance_metrics(
                    strategy_gross,
                    config.research.annualization,
                    config.research.risk_free_rate,
                ),
                "benchmark": performance_metrics(
                    benchmark,
                    config.research.annualization,
                    config.research.risk_free_rate,
                ),
                "mean_active_return_annualized": float(
                    (strategy - benchmark).mean() * config.research.annualization
                ),
                "cost_and_exposure": {
                    "transaction_cost_total": float(
                        result.portfolio.transaction_cost.loc[index].sum()
                    ),
                    "turnover_total": float(result.portfolio.turnover.loc[index].sum()),
                    "long_exposure_mean": float(result.portfolio.long_exposure.loc[index].mean()),
                    "short_exposure_mean": float(result.portfolio.short_exposure.loc[index].mean()),
                    "gross_exposure_mean": float(result.portfolio.gross_exposure.loc[index].mean()),
                    "net_exposure_mean": float(result.portfolio.net_exposure.loc[index].mean()),
                    "financing_balance_mean": float(
                        result.portfolio.financing_balance.loc[index].mean()
                    ),
                    "unused_gross_capacity_mean": float(
                        result.portfolio.unused_gross_capacity.loc[index].mean()
                    ),
                    "exposure_active_days": int(
                        result.portfolio.gross_exposure.loc[index].gt(0).sum()
                    ),
                },
            }
        )
    return rows


def _cost_scenarios(result: WalkForwardResult, config: PrimaryConfig) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for cost_bps in config.research.cost_scenarios_bps:
        portfolio = evaluate_portfolio(
            result.portfolio.weights,
            result.asset_returns,
            cost_bps,
            liquidate=True,
        )
        scenario = performance_metrics(
            portfolio.net_return,
            config.research.annualization,
            config.research.risk_free_rate,
            portfolio,
        )
        scenario["cost_basis"] = _one_way_cost_basis(
            cost_bps,
            "scenario strategy net return",
        )
        output[str(cost_bps)] = scenario
    return output


def _portfolio_variant_results(
    result: WalkForwardResult, config: PrimaryConfig
) -> dict[PortfolioKind, PortfolioResult]:
    output: dict[PortfolioKind, PortfolioResult] = {}
    kinds: tuple[PortfolioKind, ...] = (
        "long_only",
        "market_neutral",
        "directional",
    )
    for kind in kinds:
        weights = build_weights(
            result.signal,
            kind,
            config.portfolio.long_only_weight_per_asset,
            config.portfolio.neutral_gross_limit,
        )
        portfolio = evaluate_portfolio(
            weights,
            result.asset_returns,
            config.research.primary_cost_bps,
            liquidate=True,
        )
        output[kind] = portfolio
    return output


def _portfolio_variants(result: WalkForwardResult, config: PrimaryConfig) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind, portfolio in _portfolio_variant_results(result, config).items():
        metrics = performance_metrics(
            portfolio.net_return,
            config.research.annualization,
            config.research.risk_free_rate,
            portfolio,
        )
        metrics["cost_basis"] = _one_way_cost_basis(
            config.research.primary_cost_bps,
            f"{kind} strategy net return",
        )
        output[kind] = metrics
    return output


def _save_wealth_figure(result: WalkForwardResult, output: Path, config: PrimaryConfig) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5))
    wealth_index(result.portfolio.net_return).plot(ax=axis, label="Walk-forward strategy, net")
    wealth_index(result.fixed_baseline_portfolio.net_return).plot(
        ax=axis, label="Fixed baseline, net"
    )
    wealth_index(result.benchmark.rebalanced_return).plot(
        ax=axis, label="Rebalanced equal-weight benchmark"
    )
    axis.set_title(
        f"Outer-test compounded wealth ({config.research.primary_cost_bps} bps one-way cost)"
    )
    axis.set_ylabel("Wealth (initial = 1)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160, metadata={"Software": "news-sentiment-trading-algo"})
    plt.close(figure)
    return output


def write_primary_artifacts(
    result: WalkForwardResult,
    panel: pd.DataFrame,
    config: PrimaryConfig,
    output_dir: str | Path,
    input_hashes: dict[str, str],
    git_commit: str,
    lock_hash: str,
    preregistration_commit: str,
    source_gate_hash: str,
) -> dict[str, str]:
    """Write every primary artifact deterministically and return its SHA-256 manifest."""

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"artifact directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics, inference, rank_ic = _metrics_and_inference(result, config)
    selection_basis = _one_way_cost_basis(
        config.walk_forward.selection_cost_bps,
        "training-only parameter-selection objective",
    )
    evaluation_basis = _one_way_cost_basis(
        config.research.primary_cost_bps,
        "primary strategy net return and transaction cost",
    )
    selections = result.selection_manifest()
    for selection in selections:
        selection["cost_basis"] = selection_basis
    candidate_scores = []
    for fold in result.candidate_scores:
        candidate_fold = []
        for score in fold:
            candidate = dataclasses.asdict(score)
            candidate["cost_basis"] = selection_basis
            candidate_fold.append(candidate)
        candidate_scores.append(candidate_fold)
    corrected = corrected_fb_jpm(panel, config)
    for ticker_payload in corrected.values():
        for regime, scenarios in ticker_payload["regimes"].items():
            for rate, scenario in scenarios.items():
                scenario["cost_basis"] = _one_way_cost_basis(
                    int(rate),
                    f"{regime} corrected full-sample strategy net return",
                )
    crisis = crisis_decomposition(result, config)
    crisis["cost_basis"] = _one_way_cost_basis(
        config.research.primary_cost_bps,
        "crisis-segment strategy net returns and active spreads",
    )
    stability: dict[str, Any] = {
        "cost_basis": _one_way_cost_basis(
            config.walk_forward.selection_cost_bps,
            "training-only parameter-selection objective",
        ),
        "selection_counts": parameter_stability(result),
    }
    per_asset_contribution = (result.portfolio.weights * result.asset_returns).sum(axis=0)
    benchmark_returns = pd.concat(
        [
            result.benchmark.rebalanced_return,
            result.benchmark.static_return,
            result.benchmark.cost_aware_rebalanced_return,
            result.benchmark.cost_aware_rebalanced_turnover,
            result.benchmark.cost_aware_rebalanced_transaction_cost,
        ],
        axis=1,
    )
    portfolio_daily = pd.concat(
        [
            result.portfolio.gross_return,
            result.portfolio.net_return,
            result.portfolio.transaction_cost,
            result.portfolio.turnover,
            result.portfolio.long_exposure,
            result.portfolio.short_exposure,
            result.portfolio.gross_exposure,
            result.portfolio.net_exposure,
            result.portfolio.pretrade_financing_balance,
            result.portfolio.financing_balance,
            result.portfolio.unused_gross_capacity,
        ],
        axis=1,
    )
    fixed_daily = pd.concat(
        [
            result.fixed_baseline_portfolio.gross_return,
            result.fixed_baseline_portfolio.net_return,
            result.fixed_baseline_portfolio.transaction_cost,
            result.fixed_baseline_portfolio.turnover,
            result.fixed_baseline_portfolio.long_exposure,
            result.fixed_baseline_portfolio.short_exposure,
            result.fixed_baseline_portfolio.gross_exposure,
            result.fixed_baseline_portfolio.net_exposure,
            result.fixed_baseline_portfolio.pretrade_financing_balance,
            result.fixed_baseline_portfolio.financing_balance,
            result.fixed_baseline_portfolio.unused_gross_capacity,
        ],
        axis=1,
    )
    portfolio_daily = _with_cost_basis_columns(
        portfolio_daily,
        evaluation_basis,
        prefix="evaluation_cost",
    )
    fixed_daily = _with_cost_basis_columns(
        fixed_daily,
        _one_way_cost_basis(
            config.research.primary_cost_bps,
            "fixed-baseline strategy net return and transaction cost",
        ),
        prefix="evaluation_cost",
    )
    parameters_by_date = _with_cost_basis_columns(
        result.parameters_by_date,
        selection_basis,
        prefix="selection_cost",
    )
    write_json(output / "configuration.json", config.as_dict())
    write_json(
        output / "run_metadata.json",
        {
            "git_commit": git_commit,
            "preregistration_commit": preregistration_commit,
            "configuration_hash": config.digest(),
            "environment_lock_hash": lock_hash,
            "source_gate_hash": source_gate_hash,
            "input_hashes": input_hashes,
            "seed": config.research.seed,
            "evaluation_cost_basis": evaluation_basis,
            "selection_cost_basis": selection_basis,
            "registered_turnover_convention": result.portfolio.turnover_convention,
            "supplemental_cost_aware_benchmark_basis": _one_way_cost_basis(
                int(result.benchmark.cost_aware_rebalanced_cost_bps),
                "supplemental rebalanced equal-weight benchmark net return",
                "drift_adjusted",
            ),
        },
    )
    write_json(output / "fold_manifest.json", result.fold_manifest())
    write_json(output / "selected_parameters.json", selections)
    write_json(
        output / "candidate_scores.json",
        candidate_scores,
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "inference.json", inference)
    write_json(output / "per_fold_metrics.json", _per_fold_metrics(result, config))
    write_json(output / "cost_scenarios.json", _cost_scenarios(result, config))
    write_json(output / "portfolio_variants.json", _portfolio_variants(result, config))
    write_json(output / "corrected_fb_jpm.json", corrected)
    write_json(output / "crisis_decomposition.json", crisis)
    write_json(output / "signal_coverage.json", signal_coverage(result))
    write_json(output / "score_sorting.json", score_sorting(result))
    write_json(output / "signal_decay.json", signal_decay(panel, result))
    write_json(output / "parameter_stability.json", stability)
    contribution_values = {ticker: float(value) for ticker, value in per_asset_contribution.items()}
    write_json(
        output / "per_asset_attribution.json",
        {
            "metric": "arithmetic_sum_of_daily_gross_weight_times_return",
            "unit": "NAV simple-return contribution",
            "values": contribution_values,
            "reconciliation": {
                "contribution_sum": float(per_asset_contribution.sum()),
                "gross_return_arithmetic_sum": float(result.portfolio.gross_return.sum()),
                "gross_return_compounded_total": float(
                    performance_metrics(result.portfolio.gross_return)["total_return"]
                ),
            },
        },
    )
    write_csv(output / "parameters_by_date.csv", parameters_by_date)
    write_csv(output / "signals.csv", result.signal)
    write_csv(output / "scores.csv", result.score)
    write_csv(output / "asset_returns.csv", result.asset_returns)
    write_csv(output / "portfolio_weights.csv", result.portfolio.weights)
    write_csv(output / "portfolio_pretrade_weights.csv", result.portfolio.pretrade_weights)
    write_csv(output / "portfolio_daily.csv", portfolio_daily)
    write_csv(
        output / "fixed_baseline_weights.csv",
        result.fixed_baseline_portfolio.weights,
    )
    write_csv(
        output / "fixed_baseline_pretrade_weights.csv",
        result.fixed_baseline_portfolio.pretrade_weights,
    )
    write_csv(output / "fixed_baseline_daily.csv", fixed_daily)
    for kind, variant in _portfolio_variant_results(result, config).items():
        variant_daily = pd.concat(
            [
                variant.gross_return,
                variant.net_return,
                variant.transaction_cost,
                variant.turnover,
                variant.long_exposure,
                variant.short_exposure,
                variant.gross_exposure,
                variant.net_exposure,
                variant.pretrade_financing_balance,
                variant.financing_balance,
                variant.unused_gross_capacity,
            ],
            axis=1,
        )
        variant_daily = _with_cost_basis_columns(
            variant_daily,
            _one_way_cost_basis(
                config.research.primary_cost_bps,
                f"{kind} strategy net return and transaction cost",
            ),
            prefix="evaluation_cost",
        )
        write_csv(output / f"{kind}_weights.csv", variant.weights)
        write_csv(output / f"{kind}_pretrade_weights.csv", variant.pretrade_weights)
        write_csv(output / f"{kind}_daily.csv", variant_daily)
    write_csv(output / "benchmark_returns.csv", benchmark_returns)
    write_csv(output / "rank_ic.csv", rank_ic)
    timing = pd.concat(
        [result.return_frame.signal_dates, result.return_frame.execution_dates], axis=1
    ).loc[result.asset_returns.index]
    write_csv(output / "timing_by_return_end.csv", timing)
    figure = _save_wealth_figure(result, output / "wealth.png", config)
    write_json(
        output / "generated_figure_manifest.json",
        [
            {
                "cost_basis": _one_way_cost_basis(
                    config.research.primary_cost_bps,
                    "strategy net wealth path",
                ),
                "file": figure.name,
                "sha256": file_hash(figure),
                "title": (
                    "Outer-test compounded wealth "
                    f"({config.research.primary_cost_bps} bps one-way cost)"
                ),
            }
        ],
    )
    manifest = {
        path.name: file_hash(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"
    }
    write_json(output / "ARTIFACT_MANIFEST.json", manifest)
    manifest["ARTIFACT_MANIFEST.json"] = file_hash(output / "ARTIFACT_MANIFEST.json")
    return manifest


def validate_primary_artifacts(artifact_dir: str | Path, provenance: ArtifactProvenance) -> None:
    """Validate hashes and locked primary shape before generating claims."""

    artifacts = Path(artifact_dir)
    manifest_path = artifacts / "ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("artifact manifest must be a nonempty object")
    if set(manifest) != PRIMARY_ARTIFACT_FILES:
        raise ValueError("primary artifact manifest file set differs from the locked schema")
    actual_files = {
        path.name
        for path in artifacts.iterdir()
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"
    }
    if actual_files != PRIMARY_ARTIFACT_FILES:
        raise ValueError("primary artifact directory file set differs from the locked schema")
    if file_hash(manifest_path) != provenance.artifact_manifest_sha256:
        raise ValueError("artifact manifest does not match the external provenance")
    for name, expected_hash in manifest.items():
        path = artifacts / str(name)
        if not path.is_file() or file_hash(path) != expected_hash:
            raise ValueError(f"artifact hash mismatch: {name}")
    folds = json.loads((artifacts / "fold_manifest.json").read_text(encoding="utf-8"))
    if len(folds) != 6 or [item["test_observations"] for item in folds] != [58] * 6:
        raise ValueError("primary fold manifest is not six 58-observation folds")
    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    if metrics["strategy_net"]["observations"] != 348:
        raise ValueError("primary metrics do not contain 348 OOS observations")
    metadata = json.loads((artifacts / "run_metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "git_commit": provenance.git_commit,
        "preregistration_commit": provenance.preregistration_commit,
        "configuration_hash": provenance.configuration_hash,
        "environment_lock_hash": provenance.environment_lock_hash,
        "source_gate_hash": provenance.source_gate_hash,
        "input_hashes": provenance.input_hashes,
    }
    mismatched = [
        field for field, expected in expected_metadata.items() if metadata.get(field) != expected
    ]
    if mismatched:
        raise ValueError(f"primary artifact provenance mismatch: {mismatched}")


def write_robustness_artifacts(
    result: RobustnessResult,
    output_dir: str | Path,
    *,
    git_commit: str,
    preregistration_commit: str,
    configuration_hash: str,
    lock_hash: str,
    source_gate_hash: str,
    input_hashes: dict[str, str],
) -> dict[str, str]:
    """Write immutable robustness summaries, selections, provenance, and hashes."""

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"artifact directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection_rates = {int(value) for value in result.selections["selection_cost_bps"].unique()}
    selection_directions = set(result.selections["selection_cost_direction"].astype(str).unique())
    if len(selection_rates) != 1 or selection_directions != {"one_way"}:
        raise ValueError("robustness selections do not have one locked one-way cost basis")
    selection_basis = _one_way_cost_basis(
        selection_rates.pop(),
        "training-only parameter-selection objective",
    )
    write_csv(output / "robustness_summaries.csv", result.summaries, include_index=False)
    write_csv(output / "robustness_selections.csv", result.selections, include_index=False)
    variant_root = output / "variants"
    for record in result.audit_records:
        variant_dir = variant_root / f"{record.group}__{record.variant}"
        dates = record.dates
        weights = record.portfolio.weights.reindex(dates)
        asset_returns = record.asset_returns.reindex(index=dates, columns=weights.columns)
        if weights.isna().any().any() or asset_returns.isna().any().any():
            raise ValueError(f"incomplete robustness audit rows: {record.group}/{record.variant}")
        daily = pd.concat(
            [
                record.portfolio.gross_return.reindex(dates),
                record.portfolio.net_return.reindex(dates),
                record.portfolio.transaction_cost.reindex(dates),
                record.portfolio.turnover.reindex(dates),
                record.portfolio.long_exposure.reindex(dates),
                record.portfolio.short_exposure.reindex(dates),
                record.portfolio.gross_exposure.reindex(dates),
                record.portfolio.net_exposure.reindex(dates),
                record.portfolio.pretrade_financing_balance.reindex(dates),
                record.portfolio.financing_balance.reindex(dates),
                record.portfolio.unused_gross_capacity.reindex(dates),
            ],
            axis=1,
        )
        evaluation_basis = _one_way_cost_basis(
            record.cost_bps,
            f"{record.group}/{record.variant} strategy net return and transaction cost",
            record.portfolio.turnover_convention,
        )
        daily = _with_cost_basis_columns(
            daily,
            evaluation_basis,
            prefix="evaluation_cost",
        )
        benchmark = pd.concat(
            [
                record.benchmark.rebalanced_return.reindex(dates),
                record.benchmark.static_return.reindex(dates),
                record.benchmark.cost_aware_rebalanced_return.reindex(dates),
                record.benchmark.cost_aware_rebalanced_turnover.reindex(dates),
                record.benchmark.cost_aware_rebalanced_transaction_cost.reindex(dates),
            ],
            axis=1,
        )
        write_json(
            variant_dir / "variant_metadata.json",
            {
                "group": record.group,
                "variant": record.variant,
                "portfolio_kind": record.portfolio_kind,
                "cost_bps": record.cost_bps,
                "cost_direction": "one_way",
                "evaluation_cost_basis": evaluation_basis,
                "selection_cost_basis": selection_basis,
                "observations": len(dates),
            },
        )
        write_csv(variant_dir / "weights.csv", weights)
        write_csv(
            variant_dir / "pretrade_weights.csv",
            record.portfolio.pretrade_weights.reindex(dates),
        )
        write_csv(variant_dir / "asset_returns.csv", asset_returns)
        write_csv(variant_dir / "portfolio_daily.csv", daily)
        write_csv(variant_dir / "benchmark_returns.csv", benchmark)
        if record.fold_manifest is not None:
            write_json(variant_dir / "fold_manifest.json", record.fold_manifest)
        if record.selection_manifest is not None:
            selection_manifest = []
            for selection in record.selection_manifest:
                selection_record = dict(selection)
                selection_record["cost_basis"] = selection_basis
                selection_manifest.append(selection_record)
            write_json(variant_dir / "selected_parameters.json", selection_manifest)
    write_json(
        output / "run_metadata.json",
        {
            "git_commit": git_commit,
            "preregistration_commit": preregistration_commit,
            "configuration_hash": configuration_hash,
            "environment_lock_hash": lock_hash,
            "source_gate_hash": source_gate_hash,
            "input_hashes": input_hashes,
            "scope": "predefined descriptive robustness matrix",
            "evaluation_cost_convention": {
                "applies_to": "each variant strategy net return and transaction cost",
                "charged_on": "variant-specific explicit turnover convention",
                "direction": "one_way",
                "rate_and_convention_location": (
                    "robustness_summaries.csv and each variant_metadata.json"
                ),
            },
            "selection_cost_basis": selection_basis,
        },
    )
    manifest = {
        path.relative_to(output).as_posix(): file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"
    }
    write_json(output / "ARTIFACT_MANIFEST.json", manifest)
    manifest["ARTIFACT_MANIFEST.json"] = file_hash(output / "ARTIFACT_MANIFEST.json")
    return manifest


def validate_robustness_artifacts(
    artifact_dir: str | Path,
    provenance: ArtifactProvenance,
    primary_artifact_dir: str | Path | None = None,
    primary_provenance: ArtifactProvenance | None = None,
) -> None:
    """Validate a robustness bundle and, when supplied, its primary-run provenance."""

    artifacts = Path(artifact_dir)
    manifest = json.loads((artifacts / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_path = artifacts / "ARTIFACT_MANIFEST.json"
    if file_hash(manifest_path) != provenance.artifact_manifest_sha256:
        raise ValueError("robustness manifest does not match the external provenance")
    actual_files = {
        path.relative_to(artifacts).as_posix()
        for path in artifacts.rglob("*")
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"
    }
    if set(manifest) != actual_files:
        raise ValueError("robustness artifact file set differs from its manifest")
    for name, expected_hash in manifest.items():
        path = artifacts / str(name)
        if not path.is_file() or file_hash(path) != expected_hash:
            raise ValueError(f"robustness artifact hash mismatch: {name}")
    robustness_metadata = json.loads((artifacts / "run_metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "git_commit": provenance.git_commit,
        "preregistration_commit": provenance.preregistration_commit,
        "configuration_hash": provenance.configuration_hash,
        "environment_lock_hash": provenance.environment_lock_hash,
        "source_gate_hash": provenance.source_gate_hash,
        "input_hashes": provenance.input_hashes,
    }
    provenance_mismatch = [
        field
        for field, expected in expected_metadata.items()
        if robustness_metadata.get(field) != expected
    ]
    if provenance_mismatch:
        raise ValueError(f"robustness artifact provenance mismatch: {provenance_mismatch}")
    if primary_artifact_dir is not None:
        if primary_provenance is None:
            raise ValueError("primary provenance is required for composed report validation")
        primary = Path(primary_artifact_dir)
        validate_primary_artifacts(primary, primary_provenance)
        primary_metadata = json.loads((primary / "run_metadata.json").read_text(encoding="utf-8"))
        fields = (
            "git_commit",
            "preregistration_commit",
            "configuration_hash",
            "environment_lock_hash",
            "source_gate_hash",
            "input_hashes",
        )
        mismatched = [
            field
            for field in fields
            if primary_metadata.get(field) != robustness_metadata.get(field)
        ]
        if mismatched:
            raise ValueError(f"primary/robustness provenance mismatch: {mismatched}")


def generate_robustness_report(
    artifact_dir: str | Path,
    repository_root: str | Path,
    *,
    provenance: ArtifactProvenance,
    primary_artifact_dir: str | Path | None = None,
    primary_provenance: ArtifactProvenance | None = None,
) -> None:
    """Generate restricted local robustness tables and prose from artifacts."""

    artifacts = Path(artifact_dir)
    root = Path(repository_root)
    _validate_restricted_report_root(root)
    validate_robustness_artifacts(
        artifacts,
        provenance,
        primary_artifact_dir,
        primary_provenance,
    )
    summaries = pd.read_csv(artifacts / "robustness_summaries.csv")
    selections = pd.read_csv(artifacts / "robustness_selections.csv")
    required_groups = {
        "parameter_policy",
        "portfolio_cost",
        "missing_policy",
        "feature",
        "holding",
        "execution",
        "leave_one_out",
        "crisis_period",
        "turnover_convention",
    }
    if set(summaries["group"]) != required_groups:
        raise ValueError("robustness matrix is incomplete")
    table_dir = root / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table_dir / "robustness-summaries.csv", summaries, include_index=False)
    write_csv(table_dir / "robustness-selections.csv", selections, include_index=False)

    selected = summaries.loc[
        (summaries["group"] == "parameter_policy") & (summaries["variant"] == "selected")
    ].iloc[0]
    fixed = summaries.loc[
        (summaries["group"] == "parameter_policy") & (summaries["variant"] == "fixed_no_tuning")
    ].iloc[0]
    registered_turnover = summaries.loc[
        (summaries["group"] == "turnover_convention")
        & (summaries["variant"] == "registered_target_weight")
    ].iloc[0]
    drift_turnover = summaries.loc[
        (summaries["group"] == "turnover_convention")
        & (summaries["variant"] == "post_v1_drift_adjusted")
    ].iloc[0]
    leave_one_out = summaries.loc[summaries["group"] == "leave_one_out"]
    crisis = summaries.loc[
        (summaries["group"] == "crisis_period") & (summaries["variant"] == "crisis_component")
    ].iloc[0]
    pre_crisis = summaries.loc[
        (summaries["group"] == "crisis_period") & (summaries["variant"] == "pre_crisis_component")
    ].iloc[0]
    selected_valid = selections.loc[
        (selections["group"] == "parameter_policy") & (selections["variant"] == "selected")
    ].sort_values("fold_id")["candidates_valid"]
    long_only_costs = summaries.loc[
        (summaries["group"] == "portfolio_cost") & (summaries["portfolio_kind"] == "long_only")
    ].sort_values("cost_bps")
    portfolio_costs = summaries.loc[
        (summaries["group"] == "portfolio_cost") & summaries["cost_bps"].isin([0, 10])
    ].sort_values(["portfolio_kind", "cost_bps"])

    def variant_row(group: str, variant: str) -> pd.Series:
        matches = summaries.loc[(summaries["group"] == group) & (summaries["variant"] == variant)]
        if len(matches) != 1:
            raise ValueError(f"expected one robustness row for {group}/{variant}")
        return matches.iloc[0]

    specification_rows = [
        ("Missingness", "No fill", variant_row("missing_policy", "no_fill")),
        ("Missingness", "One-session forward fill", variant_row("missing_policy", "ffill_1")),
        ("Feature", "Ratio", variant_row("feature", "ratio")),
        ("Feature", "Log-ratio", variant_row("feature", "log_ratio")),
        ("Holding", "One session", variant_row("holding", "1_session")),
        ("Holding", "Three sessions", variant_row("holding", "3_sessions")),
        ("Execution", "Next adjusted open", variant_row("execution", "next_adjusted_open")),
        ("Execution", "Lagged adjusted close", variant_row("execution", "lagged_adjusted_close")),
    ]
    cost_table = "\n".join(
        f"| {int(float(str(row.cost_bps)))} | {_pct(row.strategy_total_return)} | "
        f"{float(str(row.transaction_cost_total)):.4f} |"
        for row in long_only_costs.itertuples(index=False)
    )
    portfolio_table = "\n".join(
        f"| {str(row.portfolio_kind).replace('_', ' ')} | "
        f"{int(float(str(row.cost_bps)))} | "
        f"{_pct(row.strategy_total_return)} | {_pct(row.gross_exposure_mean)} | "
        f"{int(float(str(row.active_days)))}/{int(float(str(row.observations)))} |"
        for row in portfolio_costs.itertuples(index=False)
    )
    specification_table = "\n".join(
        f"| {family} | {label} | {int(row['cost_bps'])} | "
        f"{_pct(row['strategy_total_return'])} | "
        f"{_pct(row['benchmark_total_return'])} |"
        for family, label, row in specification_rows
    )
    inference_path = root / "reports" / "tables" / "inference.json"
    beta_text = ""
    if inference_path.is_file():
        exposure = json.loads(inference_path.read_text(encoding="utf-8"))["benchmark_exposure"]
        beta_text = (
            f" At 10 bps one-way cost, full-sample beta was "
            f"{_decimal(exposure['beta'], 3)} "
            f"(HAC SE {_decimal(exposure['beta_hac_se'], 3)})."
        )
    crisis_unallocated = float(crisis["unused_gross_capacity_mean"])
    selection_difference = 100.0 * (
        float(selected["strategy_total_return"]) - float(fixed["strategy_total_return"])
    )
    report = f"""# Robustness summary

{LOCAL_REPORT_NOTICE}

This report is generated from `reports/tables/robustness-summaries.csv`; robustness specifications are descriptive and were not used to select the registered primary rule.

## Parameter selection

The selected policy returned {_pct(selected["strategy_total_return"])} at {int(selected["cost_bps"])} bps one-way cost, versus {_pct(fixed["strategy_total_return"])} at {int(fixed["cost_bps"])} bps one-way cost for the fixed no-tuning policy. The selected-minus-fixed difference was {_decimal(selection_difference, 2)} percentage points. Valid candidates by fold were {", ".join(str(int(value)) for value in selected_valid)} out of eight tested.

## Cost and portfolio variants

| Long-only one-way cost (bps) | Total return | Summed charges (NAV) |
|---:|---:|---:|
{cost_table}

| Portfolio | One-way cost (bps) | Total return | Mean gross exposure | Exposure-active days |
|---|---:|---:|---:|---:|
{portfolio_table}

## Post-v1.0 turnover and benchmark accounting

The registered target-weight convention returned {_pct(registered_turnover["strategy_total_return"])} at 10 bps one-way cost. Re-evaluating the same locked selected weights with cost-funded drift-adjusted turnover returned {_pct(drift_turnover["strategy_total_return"])}. This descriptive accounting analysis does not change training selection, the registered primary series, or confirmatory inference. The supplemental daily-rebalanced benchmark returned {_pct(drift_turnover["cost_aware_rebalanced_benchmark_total_return"])} after 10 bps one-way drift-adjusted costs; the registered comparison remains the frictionless benchmark.

## Feature, missingness, holding, and execution

| Family | Variant | One-way cost (bps) | Strategy total return | Corresponding benchmark |
|---|---|---:|---:|---:|
{specification_table}

The table reports the strategy and corresponding benchmark for each specification. These rows are descriptive comparisons and do not authorize post-result specification selection.

## Universe and crisis diagnostics

At 10 bps one-way cost, leave-one-asset-out strategy returns range from {_pct(leave_one_out["strategy_total_return"].min())} to {_pct(leave_one_out["strategy_total_return"].max())}. The return-end-based crisis component returned {_pct(crisis["strategy_total_return"])} versus {_pct(crisis["benchmark_total_return"])} for the benchmark, with {int(crisis["active_days"])}/{int(crisis["observations"])} exposure-active days, {_pct(crisis["gross_exposure_mean"])} mean risky exposure, and {_pct(crisis_unallocated)} mean unallocated long-only capital. The pre-crisis component returned {_pct(pre_crisis["strategy_total_return"])} versus {_pct(pre_crisis["benchmark_total_return"])}.{beta_text} Period components are descriptive and do not establish predictive attribution.

The complete table retains every cost, portfolio, turnover, feature, missingness, holding, execution, leave-one-out, and crisis row. Variant-level ignored artifacts retain targets, pre-trade weights, asset returns, gross/net returns, costs, folds, and selected parameters for independent reconciliation.
"""
    (root / "reports" / "robustness-summary.md").write_text(report, encoding="utf-8", newline="\n")
    if primary_artifact_dir is not None:
        _integrate_robustness_context(root, summaries)
        _generate_local_report_index(root, include_robustness=True)


def _integrate_robustness_context(root: Path, summaries: pd.DataFrame) -> None:
    """Idempotently add required adverse robustness context to generated prose."""

    selected = summaries.loc[
        (summaries["group"] == "parameter_policy") & (summaries["variant"] == "selected")
    ].iloc[0]
    fixed = summaries.loc[
        (summaries["group"] == "parameter_policy") & (summaries["variant"] == "fixed_no_tuning")
    ].iloc[0]
    leave_one_out = summaries.loc[summaries["group"] == "leave_one_out"]
    crisis = summaries.loc[
        (summaries["group"] == "crisis_period") & (summaries["variant"] == "crisis_component")
    ].iloc[0]
    pre_crisis = summaries.loc[
        (summaries["group"] == "crisis_period") & (summaries["variant"] == "pre_crisis_component")
    ].iloc[0]
    long_only_costs = summaries.loc[
        (summaries["group"] == "portfolio_cost") & (summaries["portfolio_kind"] == "long_only")
    ].set_index("cost_bps")

    def variant_row(group: str, variant: str) -> pd.Series:
        matches = summaries.loc[(summaries["group"] == group) & (summaries["variant"] == variant)]
        if len(matches) != 1:
            raise ValueError(f"expected one robustness row for {group}/{variant}")
        return matches.iloc[0]

    neutral_zero = variant_row("portfolio_cost", "market_neutral_0bps")
    neutral_ten = variant_row("portfolio_cost", "market_neutral_10bps")
    directional_zero = variant_row("portfolio_cost", "directional_0bps")
    directional_ten = variant_row("portfolio_cost", "directional_10bps")
    forward_fill = variant_row("missing_policy", "ffill_1")
    log_ratio = variant_row("feature", "log_ratio")
    three_sessions = variant_row("holding", "3_sessions")
    lagged_close = variant_row("execution", "lagged_adjusted_close")
    inference_path = root / "reports" / "tables" / "inference.json"
    exposure_text = ""
    if inference_path.is_file():
        inference = json.loads(inference_path.read_text(encoding="utf-8"))
        beta = inference["benchmark_exposure"]
        exposure_text = (
            f" At 10 bps one-way cost, full-sample benchmark beta was "
            f"{_decimal(beta['beta'], 3)} "
            f"(HAC SE {_decimal(beta['beta_hac_se'], 3)})."
        )
    crisis_unallocated = float(crisis["unused_gross_capacity_mean"])
    attribution_text = ""
    attribution_path = root / "reports" / "tables" / "per_asset_attribution.json"
    if attribution_path.is_file():
        attribution_payload = json.loads(attribution_path.read_text(encoding="utf-8"))
        values = attribution_payload.get("values", attribution_payload)
        ordered = sorted(values.items(), key=lambda item: float(item[1]))
        attribution_text = (
            f" Gross contribution was offsetting and asset-specific: {ordered[-1][0]} "
            f"{_pct(ordered[-1][1])} and {ordered[-2][0]} {_pct(ordered[-2][1])} were the "
            f"highest sums, while {ordered[0][0]} {_pct(ordered[0][1])} and "
            f"{ordered[1][0]} {_pct(ordered[1][1])} were the lowest sums."
        )
    coverage_text = ""
    coverage_path = root / "reports" / "tables" / "signal_coverage.json"
    if coverage_path.is_file():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        by_asset = coverage["by_asset"]
        sparse = sorted(by_asset.items(), key=lambda item: int(item[1]["score_available"]))[:2]
        coverage_text = (
            f" The score produced {int(coverage['aggregate_active_asset_days'])} active asset-days "
            f"({int(coverage['aggregate_long_asset_days'])} positive and "
            f"{int(coverage['aggregate_short_asset_days'])} negative); coverage was especially sparse "
            f"for {sparse[0][0]} ({int(sparse[0][1]['score_available'])} scores, "
            f"{int(sparse[0][1]['active_events'])} events) and {sparse[1][0]} "
            f"({int(sparse[1][1]['score_available'])} scores, "
            f"{int(sparse[1][1]['active_events'])} events)."
        )
    decay_text = ""
    decay_path = root / "reports" / "tables" / "signal_decay.json"
    if decay_path.is_file():
        decay = json.loads(decay_path.read_text(encoding="utf-8"))
        decay_text = (
            " Rank-IC decay at horizons 1/2/3 was "
            + "/".join(_decimal(item["mean_rank_ic"], 4) for item in decay)
            + "."
        )
    corrected_text = ""
    corrected_path = root / "reports" / "tables" / "corrected_fb_jpm.json"
    if corrected_path.is_file():
        corrected = json.loads(corrected_path.read_text(encoding="utf-8"))
        corrected_text = (
            " The corrected full-sample bridge at 10 bps one-way cost (descriptive, not OOS) "
            "returned "
            f"{_pct(corrected['FB']['regimes']['long_only']['10']['total_return'])} for FB "
            f"long-only and {_pct(corrected['FB']['regimes']['directional']['10']['total_return'])} "
            f"for FB directional, versus {_pct(corrected['FB']['buy_and_hold_benchmark']['total_return'])} "
            f"buy-and-hold; JPM returned "
            f"{_pct(corrected['JPM']['regimes']['long_only']['10']['total_return'])} long-only and "
            f"{_pct(corrected['JPM']['regimes']['directional']['10']['total_return'])} directional, "
            f"versus {_pct(corrected['JPM']['buy_and_hold_benchmark']['total_return'])} buy-and-hold."
        )
    legacy_text = ""
    legacy_path = root / "reports" / "tables" / "legacy-fb-jpm.json"
    if legacy_path.is_file():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy_map = {(item["ticker"], item["regime"]): item for item in legacy}
        legacy_text = (
            "The quarantined legacy reproduction at 100 bps reported arithmetic/compounded "
            f"returns of {_pct(legacy_map[('FB', 'long_only')]['strategy_arithmetic_return'])}/"
            f"{_pct(legacy_map[('FB', 'long_only')]['strategy_compounded_return'])} for FB long-only, "
            f"{_pct(legacy_map[('FB', 'long_short')]['strategy_arithmetic_return'])}/"
            f"{_pct(legacy_map[('FB', 'long_short')]['strategy_compounded_return'])} for FB directional, "
            f"{_pct(legacy_map[('JPM', 'long_only')]['strategy_arithmetic_return'])}/"
            f"{_pct(legacy_map[('JPM', 'long_only')]['strategy_compounded_return'])} for JPM long-only, "
            f"and {_pct(legacy_map[('JPM', 'long_short')]['strategy_arithmetic_return'])}/"
            f"{_pct(legacy_map[('JPM', 'long_short')]['strategy_compounded_return'])} for JPM "
            "directional. Those post-selected, current-inclusive, persistent-position figures are "
            "methodological evidence only, not empirical support."
        )
    selection_difference = 100.0 * (
        float(selected["strategy_total_return"]) - float(fixed["strategy_total_return"])
    )
    robustness_section = f"""## Robustness evidence

The fixed no-tuning policy returned {_pct(fixed["strategy_total_return"])} at {int(fixed["cost_bps"])} bps one-way cost, versus {_pct(selected["strategy_total_return"])} at {int(selected["cost_bps"])} bps one-way cost for the selected policy; the selected-minus-fixed difference was {_decimal(selection_difference, 2)} percentage points. Long-only returns at 0/5/10/25/50/100 bps one-way cost were {"/".join(_pct(long_only_costs.loc[cost, "strategy_total_return"]) for cost in (0, 5, 10, 25, 50, 100))}. Market-neutral returned {_pct(neutral_zero["strategy_total_return"])} at 0 bps one-way cost and {_pct(neutral_ten["strategy_total_return"])} at 10 bps one-way cost; directional returned {_pct(directional_zero["strategy_total_return"])} at 0 bps one-way cost and {_pct(directional_ten["strategy_total_return"])} at 10 bps one-way cost. At 10 bps one-way cost, one-session forward fill returned {_pct(forward_fill["strategy_total_return"])} and log-ratio returned {_pct(log_ratio["strategy_total_return"])}. Three-session and lagged-close strategy/benchmark pairs were {_pct(three_sessions["strategy_total_return"])}/{_pct(three_sessions["benchmark_total_return"])} and {_pct(lagged_close["strategy_total_return"])}/{_pct(lagged_close["benchmark_total_return"])}. All variants remain descriptive.

## Competing explanations

At 10 bps one-way cost, leave-one-asset-out returns ranged from {_pct(leave_one_out["strategy_total_return"].min())} to {_pct(leave_one_out["strategy_total_return"].max())}.{attribution_text} The bounded-forward-fill, fixed-policy, and selected-policy values are reported above without post-result relabeling. The crisis component returned {_pct(crisis["strategy_total_return"])} versus {_pct(crisis["benchmark_total_return"])} with {int(crisis["active_days"])}/{int(crisis["observations"])} exposure-active days, {_pct(crisis["gross_exposure_mean"])} mean risky exposure, and {_pct(crisis_unallocated)} mean unallocated long-only capital, while pre-crisis performance was {_pct(pre_crisis["strategy_total_return"])} versus {_pct(pre_crisis["benchmark_total_return"])}.{exposure_text} These decompositions are descriptive and do not establish predictive attribution.{coverage_text}{decay_text}

## Legacy and corrected FB/JPM bridge

{legacy_text or "Legacy aggregate values are stored in `reports/artifacts/legacy-fb-jpm.json` and the audit report; they are quarantined from production metrics."}{corrected_text}

These descriptive variants do not alter the registered primary decision rule and do not justify choosing a holding or execution rule after viewing results.

See `reports/robustness-summary.md` and `reports/tables/robustness-summaries.csv` for every variant.
"""
    for relative in (
        Path("reports/research-report.md"),
        Path("docs/RESULTS_INTERPRETATION.md"),
    ):
        destination = root / relative
        if destination.is_file():
            base = destination.read_text(encoding="utf-8").split(
                "\n## Robustness evidence", maxsplit=1
            )[0]
            destination.write_text(
                base.rstrip() + "\n\n" + robustness_section,
                encoding="utf-8",
                newline="\n",
            )


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _decimal(value: Any, places: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{places}f}"


def _conclusion(metrics: dict[str, Any], inference: dict[str, Any]) -> dict[str, Any]:
    strategy = metrics["strategy_net"]
    benchmark = metrics["rebalanced_equal_weight"]
    active = inference["active_return_hac"]
    rank_ic = inference["rank_ic_hac"]
    adjusted = inference["holm_adjusted_p_values"]
    predictive_confirmed = bool(rank_ic["mean"] > 0 and adjusted["rank_ic"] < 0.05)
    trading_confirmed = bool(active["mean"] > 0 and adjusted["active_return"] < 0.05)
    return {
        "cost_basis": metrics["cost_basis"],
        "predictive_hypothesis_confirmed": predictive_confirmed,
        "trading_hypothesis_confirmed": trading_confirmed,
        "strategy_total_return": strategy["total_return"],
        "benchmark_total_return": benchmark["total_return"],
        "mean_daily_active_spread": active["mean"],
        "active_holm_p_value": adjusted["active_return"],
        "mean_daily_rank_ic": rank_ic["mean"],
        "rank_ic_holm_p_value": adjusted["rank_ic"],
    }


def _hypothesis_sentence(conclusion: dict[str, Any]) -> str:
    predictive = bool(conclusion["predictive_hypothesis_confirmed"])
    trading = bool(conclusion["trading_hypothesis_confirmed"])
    if predictive and trading:
        return "Both preregistered hypotheses met their signed, Holm-adjusted 5% criteria."
    if not predictive and not trading:
        return "Neither preregistered hypothesis met its signed, Holm-adjusted 5% criterion."
    return "One preregistered hypothesis met its signed, Holm-adjusted 5% criterion."


LOCAL_REPORT_NOTICE = (
    "> **Restricted local empirical output.** Content-addressed provenance verifies the "
    "artifact lineage; it does not grant publication or redistribution rights. Do not copy "
    "this output into tracked files or public profiles without a separate disclosure review."
)


def _validate_restricted_report_root(root: Path) -> None:
    """Reject direct report writes to a Git checkout outside its ignored artifact tree."""

    resolved = root.resolve()
    probe = resolved
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=probe,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return
    checkout = Path(completed.stdout.strip()).resolve()
    artifact_root = (checkout / "reports" / "artifacts").resolve()
    if resolved == artifact_root or artifact_root not in resolved.parents:
        raise RuntimeError("generated empirical reports must remain under reports/artifacts/")


def _write_aggregate_tables(artifacts: Path, root: Path) -> dict[str, Any]:
    """Copy aggregate evidence into a restricted local report bundle."""

    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    inference = json.loads((artifacts / "inference.json").read_text(encoding="utf-8"))
    conclusion = _conclusion(metrics, inference)
    table_dir = root / "reports" / "tables"
    figure_dir = root / "reports" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    write_json(table_dir / "primary-summary.json", conclusion)
    aggregate_files = (
        "per_fold_metrics.json",
        "cost_scenarios.json",
        "portfolio_variants.json",
        "per_asset_attribution.json",
        "selected_parameters.json",
        "crisis_decomposition.json",
        "signal_coverage.json",
        "score_sorting.json",
        "signal_decay.json",
        "parameter_stability.json",
        "corrected_fb_jpm.json",
        "inference.json",
        "metrics.json",
    )
    for name in aggregate_files:
        source = artifacts / name
        if source.exists():
            shutil.copyfile(source, table_dir / name)

    # A daily wealth path is derived from restricted source observations and remains in
    # the ignored local artifact bundle. The local figure uses aggregate fold returns.
    old_daily_figure = figure_dir / "primary-wealth.png"
    if old_daily_figure.exists():
        old_daily_figure.unlink()
    fold_records = json.loads((artifacts / "per_fold_metrics.json").read_text(encoding="utf-8"))
    fold_ids = [int(item["fold_id"]) for item in fold_records]
    strategy_returns = [100.0 * float(item["strategy"]["total_return"]) for item in fold_records]
    benchmark_returns = [100.0 * float(item["benchmark"]["total_return"]) for item in fold_records]
    positions = np.arange(len(fold_ids), dtype=float)
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    width = 0.38
    axis.bar(positions - width / 2, strategy_returns, width, label="Strategy net")
    axis.bar(
        positions + width / 2,
        benchmark_returns,
        width,
        label="Daily-rebalanced equal weight",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, [f"F{fold_id}" for fold_id in fold_ids])
    axis.set_ylabel("Total return (%)")
    axis.set_xlabel("58 next-adjusted-open returns per fold (348 OOS returns total)")
    figure_title = (
        "Primary long-only net performance at 10 bps one-way cost (2019-01-07 to 2020-05-22)"
    )
    axis.set_title(figure_title)
    axis.legend()
    fig.tight_layout()
    fold_figure = figure_dir / "fold-performance.png"
    fig.savefig(
        fold_figure,
        dpi=150,
        metadata={"Software": "news-sentiment-trading-algo"},
    )
    plt.close(fig)
    write_json(
        figure_dir / "manifest.json",
        [
            {
                "cost_basis": metrics["cost_basis"],
                "file": fold_figure.name,
                "sha256": file_hash(fold_figure),
                "title": figure_title,
            }
        ],
    )
    return conclusion


def _generate_local_report_index(root: Path, *, include_robustness: bool = False) -> None:
    """Generate a restricted local evidence index from validated artifacts."""

    metrics_path = root / "reports" / "tables" / "metrics.json"
    inference_path = root / "reports" / "tables" / "inference.json"
    folds_path = root / "reports" / "tables" / "per_fold_metrics.json"
    if not (metrics_path.is_file() and inference_path.is_file() and folds_path.is_file()):
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    strategy = metrics["strategy_net"]
    gross = metrics["strategy_gross"]
    benchmark = metrics["rebalanced_equal_weight"]
    static_benchmark = metrics["static_equal_weight"]
    cost_aware_benchmark = metrics["cost_aware_rebalanced_equal_weight"]
    fixed = metrics["fixed_baseline_net"]
    active = inference["active_return_hac"]
    ic = inference["rank_ic_hac"]
    adjusted = inference["holm_adjusted_p_values"]
    beta = inference["benchmark_exposure"]
    conclusion = _conclusion(metrics, inference)
    hypothesis_sentence = _hypothesis_sentence(conclusion)
    fold_count = len(folds)
    negative_strategy_folds = sum(item["strategy"]["total_return"] < 0 for item in folds)
    negative_active_folds = sum(item["mean_active_return_annualized"] < 0 for item in folds)
    selection_path = root / "reports" / "tables" / "selected_parameters.json"
    selection_text = ""
    if selection_path.is_file():
        selection_records = json.loads(selection_path.read_text(encoding="utf-8"))
        valid_counts = ", ".join(str(int(item["candidates_valid"])) for item in selection_records)
        selection_text = f" Valid candidates by fold were {valid_counts} out of eight tested."
    robustness_path = root / "reports" / "tables" / "robustness-summaries.csv"
    robustness = "Robustness results are generated separately; run the locked matrix command below."
    if include_robustness and robustness_path.is_file():
        summaries = pd.read_csv(robustness_path)
        leave_one_out = summaries.loc[summaries["group"] == "leave_one_out"]
        crisis = summaries.loc[
            (summaries["group"] == "crisis_period") & (summaries["variant"] == "crisis_component")
        ].iloc[0]
        pre_crisis = summaries.loc[
            (summaries["group"] == "crisis_period")
            & (summaries["variant"] == "pre_crisis_component")
        ].iloc[0]
        crisis_unallocated = float(crisis["unused_gross_capacity_mean"])
        robustness = (
            f"The fixed no-tuning policy returned {_pct(fixed['total_return'])} net at "
            "10 bps one-way cost; the "
            f"leave-one-asset-out range was "
            f"{_pct(leave_one_out['strategy_total_return'].min())} to "
            f"{_pct(leave_one_out['strategy_total_return'].max())} at 10 bps one-way cost. "
            "At 10 bps one-way cost, the return-end-based "
            f"crisis component returned {_pct(crisis['strategy_total_return'])} versus "
            f"{_pct(crisis['benchmark_total_return'])} for the benchmark, with "
            f"{int(crisis['active_days'])}/{int(crisis['observations'])} exposure-active "
            f"days, {_pct(crisis['gross_exposure_mean'])} mean risky exposure, and "
            f"{_pct(crisis_unallocated)} mean unallocated long-only capital. At 10 bps "
            "one-way cost, the pre-crisis "
            "component returned "
            f"{_pct(pre_crisis['strategy_total_return'])} versus "
            f"{_pct(pre_crisis['benchmark_total_return'])}. At 10 bps one-way cost, "
            "full-sample beta was "
            f"{_decimal(beta['beta'], 3)} (HAC SE {_decimal(beta['beta_hac_se'], 3)}). "
            "Crisis and pre-crisis rows are reported separately and remain descriptive."
            f"{selection_text}"
        )
    report_index = f"""# Local empirical report index

{LOCAL_REPORT_NOTICE}

**Question.** Under the locked trading and predictive estimands, do daily news-sentiment extremes generalize into economically useful next-session equity returns?

**Local empirical result.** Across {strategy["observations"]} held-out returns, the cash-inclusive strategy returned {_pct(strategy["total_return"])} net at 10 bps one-way cost versus {_pct(benchmark["total_return"])} for the daily-rebalanced equal-weight benchmark. {hypothesis_sentence}

**Key correction.** The academic analysis selected and evaluated parameters on the same full sample and used ambiguous timing. This reconstruction locks shared parameters inside expanding training windows, uses strictly prior features, executes at the next adjusted open, and evaluates six non-overlapping outer folds once.

**Estimand boundary.** The trading comparison is a sparse, cash-inclusive rule against a fully invested benchmark, not an exposure-matched test of news information. The predictive test ranks every available sentiment score and does not condition the rank IC on the RVT activity mask. Neither test cleanly isolates intensity-conditioned news information.

**Other major limitations.** The 2018-2020 sample is short, overlaps the COVID shock, and contains daily aggregates without article timestamps. The evidence cannot support live trading or a claim of benchmark-adjusted skill.

## Compact results

| Evidence | Result |
|---|---:|
| Strategy gross / net total return (10 bps one-way cost) | {_pct(gross["total_return"])} / {_pct(strategy["total_return"])} |
| Rebalanced benchmark total return | {_pct(benchmark["total_return"])} |
| Static equal-weight benchmark total return | {_pct(static_benchmark["total_return"])} |
| Supplemental rebalanced benchmark net total return (10 bps one-way, drift-adjusted turnover) | {_pct(cost_aware_benchmark["total_return"])} |
| Mean gross exposure / unallocated long-only capital | {_pct(strategy["gross_exposure_mean"])} / {_pct(strategy["unused_gross_capacity_mean"])} |
| Mean daily active spread (10 bps one-way cost) | {_pct(active["mean"])} |
| Active HAC p / Holm p (10 bps one-way cost) | {_decimal(active["p_value_two_sided"])} / {_decimal(adjusted["active_return"])} |
| Mean daily rank IC | {_decimal(ic["mean"], 4)} |
| Rank-IC HAC p / Holm p | {_decimal(ic["p_value_two_sided"])} / {_decimal(adjusted["rank_ic"])} |
| Benchmark beta (HAC SE; strategy net at 10 bps one-way cost) | {_decimal(beta["beta"], 3)} ({_decimal(beta["beta_hac_se"], 3)}) |
| Negative strategy-return folds / negative mean-spread folds (10 bps one-way cost) | {negative_strategy_folds}/{fold_count} / {negative_active_folds}/{fold_count} |

At 10 bps one-way cost, summed transaction charges were {strategy["transaction_cost_total"]:.4f} NAV; gross and net results are reported together above without a hard-coded polarity interpretation.

## Robustness

{robustness}

## What changed from the academic version?

- Replaced same-sample optimization with nested walk-forward selection and an immutable preregistration.
- Replaced current-inclusive rolling features and filled/persistent signals with strict-prior, bounded events and explicit missingness.
- Replaced ambiguous same-day timing with a signal-date, next-open execution-date, and following-open return-end trace.
- Replaced summed returns and nonstandard risk statistics with compounded wealth, additive spread statistics, relative wealth, HAC/Holm inference, block bootstrap intervals, and exposure diagnostics.
- Expanded the two-stock illustration to the complete 10-equity supplied universe and supports the predefined robustness matrix without claiming it was supplied in a primary-only bundle.

## Reproduce

Original inputs are not redistributed. Users with authorised source CSVs can point `NEWS_SENTIMENT_SOURCE` to a compatible external directory, then run:

```text
uv sync --all-extras --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run news-sentiment-trading synthetic-demo
uv run news-sentiment-trading repository-scan
uv run news-sentiment-trading run-primary --output-dir reports/artifacts/primary-local
uv run news-sentiment-trading run-robustness --output-dir reports/artifacts/robustness-local
```

Daily empirical artifacts and this generated report remain ignored local material. Content-addressed provenance does not grant redistribution rights. See `docs/REPRODUCIBILITY.md`, `docs/PRE_REGISTRATION.md`, and `reports/research-report.md` inside this local report bundle.

## Data rights and scope

The code is MIT-licensed. Third-party sentiment and price data are not covered by that license and are excluded from Git because redistribution rights were not established. The repository includes a schema-only synthetic example, not empirical records. See `THIRD_PARTY_NOTICES.md` and `docs/DATA_PROVENANCE.md`.
"""
    (root / "LOCAL_REPORT_INDEX.md").write_text(report_index, encoding="utf-8", newline="\n")


def generate_markdown_reports(
    artifact_dir: str | Path,
    repository_root: str | Path,
    provenance: ArtifactProvenance,
    robustness_artifact_dir: str | Path | None = None,
    robustness_provenance: ArtifactProvenance | None = None,
    legacy_artifact: str | Path | None = None,
) -> None:
    """Generate prose and tables only from machine-readable artifacts."""

    artifacts = Path(artifact_dir)
    root = Path(repository_root)
    _validate_restricted_report_root(root)
    validate_primary_artifacts(artifacts, provenance)
    if robustness_artifact_dir is not None:
        if robustness_provenance is None:
            raise ValueError("robustness provenance is required for a composed report")
        validate_robustness_artifacts(
            robustness_artifact_dir,
            robustness_provenance,
            artifacts,
            provenance,
        )
    (root / "docs").mkdir(parents=True, exist_ok=True)
    if legacy_artifact is not None:
        legacy_path = Path(legacy_artifact)
        legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        expected_pairs = {
            ("FB", "long_only"),
            ("FB", "long_short"),
            ("JPM", "long_only"),
            ("JPM", "long_short"),
        }
        observed_pairs = {
            (str(item.get("ticker")), str(item.get("regime"))) for item in legacy_payload
        }
        if observed_pairs != expected_pairs:
            raise ValueError("legacy artifact does not contain the four quarantined FB/JPM rows")
        write_json(root / "reports" / "tables" / "legacy-fb-jpm.json", legacy_payload)
    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    inference = json.loads((artifacts / "inference.json").read_text(encoding="utf-8"))
    selections = json.loads((artifacts / "selected_parameters.json").read_text(encoding="utf-8"))
    strategy = metrics["strategy_net"]
    strategy_gross = metrics["strategy_gross"]
    benchmark = metrics["rebalanced_equal_weight"]
    static_benchmark = metrics["static_equal_weight"]
    cost_aware_benchmark = metrics["cost_aware_rebalanced_equal_weight"]
    active_spread = metrics["active_spread"]
    relative_wealth = metrics["relative_wealth"]
    active_hac = inference["active_return_hac"]
    ic_hac = inference["rank_ic_hac"]
    adjusted = inference["holm_adjusted_p_values"]
    conclusion = _write_aggregate_tables(artifacts, root)
    hypothesis_sentence = _hypothesis_sentence(conclusion)
    summary = f"""# Results summary

{LOCAL_REPORT_NOTICE}

This document is generated from write-once-by-CLI, hash-manifested local artifacts. Local aggregate copies are in `reports/tables/`.

| Result | Walk-forward strategy net (10 bps one-way cost) | Rebalanced equal weight | Static equal weight |
|---|---:|---:|---:|
| Total return | {_pct(strategy["total_return"])} | {_pct(benchmark["total_return"])} | {_pct(static_benchmark["total_return"])} |
| CAGR | {_pct(strategy["cagr"])} | {_pct(benchmark["cagr"])} | {_pct(static_benchmark["cagr"])} |
| Annualized volatility | {_pct(strategy["annualized_volatility"])} | {_pct(benchmark["annualized_volatility"])} | {_pct(static_benchmark["annualized_volatility"])} |
| Sharpe | {strategy["sharpe_ratio"] if strategy["sharpe_ratio"] is not None else "n/a"} | {benchmark["sharpe_ratio"] if benchmark["sharpe_ratio"] is not None else "n/a"} | {static_benchmark["sharpe_ratio"] if static_benchmark["sharpe_ratio"] is not None else "n/a"} |
| Maximum drawdown | {_pct(strategy["maximum_drawdown"])} | {_pct(benchmark["maximum_drawdown"])} | {_pct(static_benchmark["maximum_drawdown"])} |

At 10 bps one-way cost, summed transaction charges were {strategy["transaction_cost_total"]:.4f} NAV; gross/net total returns were {_pct(strategy_gross["total_return"])} / {_pct(strategy["total_return"])}. The additive active spread summed to {_pct(active_spread["arithmetic_sum"])}; the strategy-to-benchmark relative-wealth return was {_pct(relative_wealth["relative_total_return"])}. Neither quantity is presented as a compoundable active-return portfolio.

The registered comparison remains the frictionless daily-rebalanced benchmark. As a post-v1.0 descriptive robustness, charging that benchmark 10 bps one-way on drift-adjusted rebalancing gives {_pct(cost_aware_benchmark["total_return"])} total return; it does not replace or retroactively alter the registered benchmark.

| Confirmatory inference | Estimate | HAC SE | 95% HAC CI | HAC p | Holm p | Block-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Mean daily active spread (10 bps one-way cost) | {active_hac["mean"]:.6f} | {active_hac["standard_error"]:.6f} | [{active_hac["confidence_low"]:.6f}, {active_hac["confidence_high"]:.6f}] | {active_hac["p_value_two_sided"]:.3f} | {adjusted["active_return"]:.3f} | [{inference["active_mean_block_bootstrap_ci"][0]:.6f}, {inference["active_mean_block_bootstrap_ci"][1]:.6f}] |
| Mean daily rank IC | {ic_hac["mean"]:.4f} | {ic_hac["standard_error"]:.4f} | [{ic_hac["confidence_low"]:.4f}, {ic_hac["confidence_high"]:.4f}] | {ic_hac["p_value_two_sided"]:.3f} | {adjusted["rank_ic"]:.3f} | not preregistered |

At 10 bps one-way cost, benchmark exposure was beta {_decimal(inference["benchmark_exposure"]["beta"], 4)} (HAC SE {_decimal(inference["benchmark_exposure"]["beta_hac_se"], 4)}) with daily intercept {_decimal(inference["benchmark_exposure"]["intercept_daily"], 6)} (HAC SE {_decimal(inference["benchmark_exposure"]["intercept_hac_se"], 6)}). Mean risky exposure was {_pct(strategy["gross_exposure_mean"])} and mean unallocated long-only capital was {_pct(strategy["unused_gross_capacity_mean"])}. For the 10 bps one-way trading rule and the cost-independent rank-IC test, {hypothesis_sentence.lower()}

The trading test is the complete cash-inclusive strategy versus a fully invested benchmark; it is not exposure matched. The predictive test ranks each available sentiment-extreme score and does not condition the rank IC on the RVT activity mask. These estimand limitations prevent either test from isolating intensity-conditioned news information, and the uncertainty intervals do not establish an absence of predictive information.

Selected parameter records and all fold outcomes remain visible in `reports/tables/selected_parameters.json` and `per_fold_metrics.json`.
"""
    research_report = f"""# Research report

{LOCAL_REPORT_NOTICE}

## Question and answer

Under the preregistered cash-inclusive trading rule and available-score rank-IC estimand, do daily news-sentiment extremes contain stable, economically useful information about subsequent equity returns?

The preregistered {strategy["observations"]}-observation outer test produced a net total return at 10 bps one-way cost of {_pct(strategy["total_return"])} versus {_pct(benchmark["total_return"])} for the daily-rebalanced equal-weight universe and {_pct(static_benchmark["total_return"])} for static equal weight. The Holm-adjusted active-spread p-value at 10 bps one-way cost is {adjusted["active_return"]:.3f}; the cost-independent Holm-adjusted rank-IC p-value is {adjusted["rank_ic"]:.3f}. {hypothesis_sentence} A criterion not being met would not establish that predictive information is absent.

## Design

Features use only history strictly before each signal date. Daily aggregates on `t` execute at adjusted open `t+1` and earn the open `t+1` to open `t+2` return. Parameters are shared across all ten equities and chosen inside each expanding outer-training set from eight locked candidates. Outer blocks are non-overlapping and used once.

## Primary evidence

Gross/net total returns were {_pct(strategy_gross["total_return"])} / {_pct(strategy["total_return"])} at 10 bps one-way cost, with {strategy["transaction_cost_total"]:.4f} NAV in summed charges, {_pct(strategy["gross_exposure_mean"])} mean risky exposure, and {_pct(strategy["unused_gross_capacity_mean"])} mean unallocated long-only capital. Per-fold gross/net/cost/exposure results and per-asset gross contributions are generated in `reports/tables/per_fold_metrics.json` and `per_asset_attribution.json`. Full daily audit artifacts are stored under a write-once-by-CLI, hash-manifested ignored `reports/artifacts/<run-id>/` directory.

## Interpretation discipline

The trading comparison is cash-inclusive rather than exposure matched, and the predictive IC is not RVT-mask-conditioned. Together with the short sample, daily aggregate timing, uncertain data provenance, COVID overlap, and exclusion of market impact, these limitations prevent live-trading or benchmark-adjusted-skill claims.
"""
    methodology = f"""# Methodology report

{LOCAL_REPORT_NOTICE}

The source audit identified full-sample tuning, current-inclusive rolling thresholds, unrestricted fill and persistence, deleted price dates, arithmetic return aggregation, and nonstandard risk metrics. The reconstruction replaces these with a complete price calendar, explicit missing signals, strictly-prior rolling features, next-session execution, bounded one-interval events, shared nested walk-forward selection, auditable portfolio weights, weight-change costs, compounded wealth, and time-series uncertainty.

The locked design is `docs/PRE_REGISTRATION.md`; implementation details are `docs/RESEARCH_DESIGN.md`. Machine-readable folds, selections, weights, returns, costs, and variant audit records live under write-once-by-CLI, hash-manifested ignored `reports/artifacts/<run-id>/` directories. The trading test is explicitly cash-inclusive versus fully invested equal weight, and the predictive rank IC uses available scores without the RVT event mask; neither is treated as a pure intensity-conditioned information test.
"""
    limitations = f"""# Limitations

{LOCAL_REPORT_NOTICE}

- Daily sentiment aggregates lack article timestamps, so intraday point-in-time accuracy cannot be claimed.
- The 2018-2020 sample is short and includes the COVID shock; crisis timing can dominate estimates.
- Sentiment and price redistribution rights are unverified. Raw and daily derived data are excluded from Git.
- Adjusted-open construction relies on vendor adjustment factors and historical total-return conventions.
- Eight parameter configurations are tested inside each training set; the outer test protects evaluation but does not create more data.
- The registered primary costs use target-weight changes. Post-v1.0 robustness adds cost-funded drift-adjusted turnover, but market impact, borrow fees, taxes, explicit cash yield/borrow rates, and slippage remain absent.
- The universe has ten hand-provided equities and is not a point-in-time investable-universe study.
- At 10 bps one-way cost, the cash-inclusive trading rule averaged only {_pct(strategy["gross_exposure_mean"])} risky exposure versus a fully invested benchmark; without an exposure-matched control, active spread does not isolate news information.
- The predictive rank IC uses all available sentiment scores rather than conditioning on the RVT activity mask, and only {ic_hac["observations"]} dates were usable; no minimum usable-date threshold or small-sample block sensitivity was preregistered.
"""
    generated = {
        root / "reports" / "results-summary.md": summary,
        root / "reports" / "research-report.md": research_report,
        root / "reports" / "methodology-report.md": methodology,
        root / "reports" / "limitations.md": limitations,
    }
    for path, content in generated.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    selected_text = ", ".join(
        f"F{item['fold_id']}: w={item['selected']['window']}, "
        f"z={item['selected']['z_threshold']}, q={item['selected']['rvt_quantile']}"
        for item in selections
    )
    interpretation = f"""# Results interpretation

{LOCAL_REPORT_NOTICE}

Under the cash-inclusive trading and available-score predictive estimands, the primary net result at 10 bps one-way cost is {_pct(strategy["total_return"])}, compared with {_pct(benchmark["total_return"])} for the rebalanced universe. Mean daily active spread at 10 bps one-way cost is {_pct(active_hac["mean"])} with Holm-adjusted p={adjusted["active_return"]:.3f}; cost-independent mean daily rank IC is {_decimal(ic_hac["mean"], 4)} with Holm-adjusted p={adjusted["rank_ic"]:.3f}. {hypothesis_sentence} The trading comparison is not exposure matched, and the rank IC is not conditioned on the RVT activity mask.

Selected parameters were {selected_text}. Parameter movement, every fold outcome, cost sensitivity, beta, crisis decomposition, and leave-one-out results are retained in this restricted local evidence bundle.
"""
    (root / "docs" / "RESULTS_INTERPRETATION.md").write_text(
        interpretation, encoding="utf-8", newline="\n"
    )

    reproducibility = """# Reproducibility

The original CSV files and populated source gate are not redistributed. Users with authorised access can set `NEWS_SENTIMENT_SOURCE` to a compatible external data directory and keep the source gate outside this repository.

## Environment and fast validation

```text
uv sync --all-extras --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run pytest --cov=news_sentiment_trading --cov-report=term-missing
uv run news-sentiment-trading synthetic-demo
uv run news-sentiment-trading repository-scan
```

The default suite is synthetic-only and does not read empirical artifacts.

## Empirical reconstruction

Each output path below must be new. Use a different run ID on a repeat; artifact writers refuse overwrite.

```text
uv run news-sentiment-trading legacy-reproduce --output reports/artifacts/legacy-fb-jpm.json
uv run news-sentiment-trading run-primary --source-gate <authorised-source-gate.json> --output-dir reports/artifacts/primary-local
uv run news-sentiment-trading run-primary --source-gate <authorised-source-gate.json> --output-dir reports/artifacts/primary-local-repeat
uv run news-sentiment-trading run-robustness --source-gate <authorised-source-gate.json> --output-dir reports/artifacts/robustness-local
uv run news-sentiment-trading generate-report --source-gate <authorised-source-gate.json> --artifact-dir reports/artifacts/primary-local --provenance reports/artifacts/primary-local.provenance.json --robustness-artifact-dir reports/artifacts/robustness-local --robustness-provenance reports/artifacts/robustness-local.provenance.json --legacy-artifact reports/artifacts/legacy-fb-jpm.json --report-root reports/artifacts/generated-report-local
```

Compare every file hash in the two primary manifests for deterministic reproduction. `run-primary` and `run-robustness` refuse a dirty tree and record Git, preregistration, configuration, environment-lock, and input hashes. They write only ignored empirical artifacts; the explicit complete-report command validates matching primary/robustness provenance before generating a restricted local bundle under `reports/artifacts/`. Provenance verification is not disclosure permission. Robustness variant directories retain weights, returns, costs, folds, and selections. No notebook editing is required.
"""
    (root / "docs" / "REPRODUCIBILITY.md").write_text(
        reproducibility, encoding="utf-8", newline="\n"
    )

    folds = json.loads((artifacts / "per_fold_metrics.json").read_text(encoding="utf-8"))
    negative_strategy_folds = sum(item["strategy"]["total_return"] < 0 for item in folds)
    negative_active_folds = sum(item["mean_active_return_annualized"] < 0 for item in folds)
    fold_count = len(folds)
    robustness_context = "No matching robustness bundle was supplied; no robustness claim is made."
    robustness_path = (
        None
        if robustness_artifact_dir is None
        else Path(robustness_artifact_dir) / "robustness_summaries.csv"
    )
    if robustness_path is not None and robustness_path.is_file():
        summaries = pd.read_csv(robustness_path)
        leave_one_out = summaries.loc[summaries["group"] == "leave_one_out"]
        crisis = summaries.loc[
            (summaries["group"] == "crisis_period") & (summaries["variant"] == "crisis_component")
        ].iloc[0]
        pre_crisis = summaries.loc[
            (summaries["group"] == "crisis_period")
            & (summaries["variant"] == "pre_crisis_component")
        ].iloc[0]
        crisis_unallocated = float(crisis["unused_gross_capacity_mean"])
        robustness_context = (
            f"[At 10 bps one-way cost, leave-one-asset-out returns ranged from "
            f"{_pct(leave_one_out['strategy_total_return'].min())} to "
            f"{_pct(leave_one_out['strategy_total_return'].max())}. At 10 bps one-way cost, "
            f"the return-end-based crisis component was "
            f"{_pct(crisis['strategy_total_return'])} versus "
            f"{_pct(crisis['benchmark_total_return'])}, with "
            f"{int(crisis['active_days'])}/{int(crisis['observations'])} exposure-active "
            f"days, {_pct(crisis['gross_exposure_mean'])} mean risky exposure, and "
            f"{_pct(crisis_unallocated)} mean unallocated long-only capital; at 10 bps "
            "one-way cost, pre-crisis returns were "
            f"{_pct(pre_crisis['strategy_total_return'])} versus "
            f"{_pct(pre_crisis['benchmark_total_return'])}]"
            "(../reports/tables/robustness-summaries.csv)."
        )
    evidence_card = f"""# Local empirical evidence card

{LOCAL_REPORT_NOTICE}

- Primary strategy/benchmark total returns at 10 bps one-way cost: {_pct(strategy["total_return"])} / {_pct(benchmark["total_return"])}.
- Gross/net strategy total returns: {_pct(strategy_gross["total_return"])} / {_pct(strategy["total_return"])}.
- Confirmatory decision: {hypothesis_sentence}
- Holm-adjusted active-spread/rank-IC p-values: {adjusted["active_return"]:.3f} / {adjusted["rank_ic"]:.3f}.
- Negative strategy-return/mean-spread folds: {negative_strategy_folds}/{fold_count} and {negative_active_folds}/{fold_count}.
- Mean risky exposure: {_pct(strategy["gross_exposure_mean"])}; benchmark beta: {_decimal(inference["benchmark_exposure"]["beta"], 3)} (HAC SE {_decimal(inference["benchmark_exposure"]["beta_hac_se"], 3)}).
- Robustness status: {robustness_context}

Required interpretation boundaries: the trading comparison is cash-inclusive versus fully invested, the predictive IC uses available scores rather than the RVT activity mask, article timestamps are unavailable, and the short sample overlaps COVID. This card is restricted local evidence and is not part of the distributable repository.
"""
    (root / "docs" / "EMPIRICAL_EVIDENCE_CARD.md").write_text(
        evidence_card, encoding="utf-8", newline="\n"
    )
    _generate_local_report_index(root, include_robustness=False)
    if robustness_artifact_dir is not None:
        if robustness_provenance is None:
            raise ValueError("robustness provenance is required for a composed report")
        generate_robustness_report(
            robustness_artifact_dir,
            root,
            provenance=robustness_provenance,
            primary_artifact_dir=artifacts,
            primary_provenance=provenance,
        )
