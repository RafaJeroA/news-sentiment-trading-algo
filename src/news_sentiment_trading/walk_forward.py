"""Deterministic outer folds and training-only shared-parameter selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from news_sentiment_trading.config import PrimaryConfig, SignalParameters
from news_sentiment_trading.data import (
    ReturnFrame,
    align_signal_to_return_end,
    aligned_forward_returns,
)
from news_sentiment_trading.features import FeatureSet, build_features
from news_sentiment_trading.portfolio import (
    BenchmarkResult,
    PortfolioResult,
    benchmarks,
    build_weights,
    evaluate_portfolio,
)
from news_sentiment_trading.signals import bounded_holding, event_signal


@dataclass(frozen=True)
class OuterFold:
    fold_id: int
    test_start: str
    test_end: str
    test_observations: int
    first_signal_date: str
    first_execution_date: str
    selection_data_end: str


@dataclass(frozen=True)
class CandidateScore:
    parameters: SignalParameters
    valid: bool
    median_active_return: float
    worst_block_active_return: float
    annualized_turnover: float
    active_asset_days: int
    blocks_with_signals: int
    reason: str


@dataclass(frozen=True)
class SelectionRecord:
    fold_id: int
    selected: SignalParameters
    used_fixed_fallback: bool
    candidates_tested: int
    candidates_valid: int
    selected_median_active_return: float
    selected_worst_block_active_return: float
    selected_annualized_turnover: float
    selection_data_end: str


@dataclass(frozen=True)
class WalkForwardResult:
    folds: tuple[OuterFold, ...]
    selections: tuple[SelectionRecord, ...]
    candidate_scores: tuple[tuple[CandidateScore, ...], ...]
    parameters_by_date: pd.DataFrame
    signal: pd.DataFrame
    score: pd.DataFrame
    asset_returns: pd.DataFrame
    portfolio: PortfolioResult
    benchmark: BenchmarkResult
    fixed_baseline_portfolio: PortfolioResult
    return_frame: ReturnFrame

    def fold_manifest(self) -> list[dict[str, Any]]:
        return [asdict(fold) for fold in self.folds]

    def selection_manifest(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.selections]


def make_outer_slices(
    return_dates: pd.DatetimeIndex, initial: int, block_size: int
) -> tuple[pd.DatetimeIndex, ...]:
    """Partition every post-initial return into one non-overlapping test block."""

    if initial < 1 or block_size < 1:
        raise ValueError("initial and block_size must be positive")
    if initial >= len(return_dates):
        raise ValueError("initial history consumes every return")
    slices = tuple(
        return_dates[start : min(start + block_size, len(return_dates))]
        for start in range(initial, len(return_dates), block_size)
    )
    concatenated = pd.DatetimeIndex(np.concatenate([block.to_numpy() for block in slices]))
    if not concatenated.equals(return_dates[initial:]):
        raise AssertionError("outer blocks do not cover post-initial returns exactly once")
    return slices


def _parameter_key(parameters: SignalParameters) -> tuple[int, float, float, str, str, int]:
    return (
        parameters.window,
        parameters.z_threshold,
        parameters.rvt_quantile,
        parameters.feature_kind,
        parameters.missing_policy,
        parameters.holding_days,
    )


def _inner_blocks(
    eligible_dates: pd.DatetimeIndex, config: PrimaryConfig
) -> tuple[pd.DatetimeIndex, ...]:
    required = config.walk_forward.inner_initial_sessions
    if len(eligible_dates) < required:
        raise ValueError("not enough eligible history for the preregistered inner design")
    selected = eligible_dates[-required:]
    block = config.walk_forward.inner_test_sessions
    blocks = tuple(selected[start : start + block] for start in range(0, required, block))
    if len(blocks) != 3 or any(len(item) != block for item in blocks):
        raise AssertionError("inner design must contain three equal validation blocks")
    return blocks


def _score_candidate(
    parameters: SignalParameters,
    aligned_signal: pd.DataFrame,
    asset_returns: pd.DataFrame,
    benchmark_return: pd.Series,
    blocks: tuple[pd.DatetimeIndex, ...],
    config: PrimaryConfig,
) -> CandidateScore:
    block_active_returns: list[float] = []
    turnovers: list[float] = []
    active_asset_days = 0
    blocks_with_signals = 0
    validation_dates = pd.DatetimeIndex(np.concatenate([block.to_numpy() for block in blocks]))
    validation_signal = aligned_signal.loc[validation_dates]
    validation_weights = build_weights(
        validation_signal,
        "long_only",
        config.portfolio.long_only_weight_per_asset,
        config.portfolio.neutral_gross_limit,
    )
    validation_portfolio = evaluate_portfolio(
        validation_weights,
        asset_returns.loc[validation_dates],
        config.walk_forward.selection_cost_bps,
        liquidate=True,
    )
    for block_dates in blocks:
        block_weights = validation_weights.loc[block_dates]
        active_asset_days += int(block_weights.gt(0).sum().sum())
        if validation_portfolio.gross_exposure.loc[block_dates].gt(0).any():
            blocks_with_signals += 1
        active_return = (
            validation_portfolio.net_return.loc[block_dates] - benchmark_return.loc[block_dates]
        )
        block_active_returns.append(float(active_return.mean() * config.research.annualization))
        turnovers.append(
            float(
                validation_portfolio.turnover.loc[block_dates].mean()
                * config.research.annualization
            )
        )

    median_active_return = float(np.median(block_active_returns))
    worst_block_active_return = float(min(block_active_returns))
    annualized_turnover = float(np.mean(turnovers))
    reasons: list[str] = []
    if blocks_with_signals != len(blocks):
        reasons.append("no signal in an inner block")
    if active_asset_days < config.walk_forward.minimum_active_asset_days:
        reasons.append("insufficient active asset-days")
    total_asset_days = sum(len(block) * aligned_signal.shape[1] for block in blocks)
    if active_asset_days / total_asset_days < config.walk_forward.minimum_active_fraction:
        reasons.append("active fraction below minimum")
    if annualized_turnover > config.walk_forward.maximum_annualized_turnover:
        reasons.append("annualized turnover above maximum")
    return CandidateScore(
        parameters=parameters,
        valid=not reasons,
        median_active_return=median_active_return,
        worst_block_active_return=worst_block_active_return,
        annualized_turnover=annualized_turnover,
        active_asset_days=active_asset_days,
        blocks_with_signals=blocks_with_signals,
        reason="; ".join(reasons),
    )


def _selection_sort_key(score: CandidateScore) -> tuple[float, float, float, float, float, int]:
    """Higher objective/worst block, lower turnover, then conservative parameters."""

    return (
        score.median_active_return,
        score.worst_block_active_return,
        -score.annualized_turnover,
        score.parameters.z_threshold,
        score.parameters.rvt_quantile,
        score.parameters.window,
    )


def _select_candidate(
    scores: tuple[CandidateScore, ...], fixed: SignalParameters
) -> tuple[CandidateScore, bool]:
    """Apply the locked validity, tie-break, and fixed-fallback rule."""

    valid = [score for score in scores if score.valid]
    if valid:
        return max(valid, key=_selection_sort_key), False
    try:
        return next(score for score in scores if score.parameters == fixed), True
    except StopIteration as exc:
        raise AssertionError("fixed baseline is absent from the candidate grid") from exc


def run_walk_forward(panel: pd.DataFrame, config: PrimaryConfig) -> WalkForwardResult:
    """Run the preregistered nested walk-forward experiment."""

    return_frame = aligned_forward_returns(panel, config.execution.convention)
    return_dates = pd.DatetimeIndex(return_frame.returns.index)
    outer_slices = make_outer_slices(
        return_dates,
        config.walk_forward.outer_initial_sessions,
        config.walk_forward.outer_test_sessions,
    )
    benchmark_full = benchmarks(return_frame.returns)
    parameter_grid = config.features.parameters(config.execution)
    fixed = config.features.fixed(config.execution)

    @cache
    def artifacts(
        key: tuple[int, float, float, str, str, int],
    ) -> tuple[FeatureSet, pd.DataFrame, pd.DataFrame]:
        parameters = SignalParameters(
            window=key[0],
            z_threshold=key[1],
            rvt_quantile=key[2],
            feature_kind=key[3],  # type: ignore[arg-type]
            missing_policy=key[4],  # type: ignore[arg-type]
            holding_days=key[5],
        )
        features = build_features(panel, parameters)
        signal = bounded_holding(
            event_signal(features, parameters.z_threshold), parameters.holding_days
        )
        return (
            features,
            align_signal_to_return_end(signal, return_frame),
            align_signal_to_return_end(features.score, return_frame),
        )

    folds: list[OuterFold] = []
    selections: list[SelectionRecord] = []
    all_candidate_scores: list[tuple[CandidateScore, ...]] = []
    selected_signal_parts: list[pd.DataFrame] = []
    selected_score_parts: list[pd.DataFrame] = []
    parameter_rows: list[pd.DataFrame] = []

    for fold_id, test_dates in enumerate(outer_slices, start=1):
        first_test = pd.Timestamp(test_dates[0])
        first_signal_date = pd.Timestamp(return_frame.signal_dates.loc[first_test])
        first_execution_date = pd.Timestamp(return_frame.execution_dates.loc[first_test])
        eligible_dates = pd.DatetimeIndex(return_dates[return_dates <= first_signal_date])
        inner_blocks = _inner_blocks(eligible_dates, config)
        scores = tuple(
            _score_candidate(
                parameters,
                artifacts(_parameter_key(parameters))[1],
                return_frame.returns,
                benchmark_full.rebalanced_return,
                inner_blocks,
                config,
            )
            for parameters in parameter_grid
        )
        valid = [score for score in scores if score.valid]
        selected_score, used_fallback = _select_candidate(scores, fixed)
        selected = selected_score.parameters
        _, aligned_signal, aligned_score = artifacts(_parameter_key(selected))
        selected_signal_parts.append(aligned_signal.loc[test_dates])
        selected_score_parts.append(aligned_score.loc[test_dates])
        parameter_rows.append(
            pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "window": selected.window,
                    "z_threshold": selected.z_threshold,
                    "rvt_quantile": selected.rvt_quantile,
                    "feature_kind": selected.feature_kind,
                    "missing_policy": selected.missing_policy,
                    "holding_days": selected.holding_days,
                },
                index=test_dates,
            )
        )
        selection_data_end = pd.Timestamp(eligible_dates[-1])
        folds.append(
            OuterFold(
                fold_id=fold_id,
                test_start=str(first_test.date()),
                test_end=str(pd.Timestamp(test_dates[-1]).date()),
                test_observations=len(test_dates),
                first_signal_date=str(first_signal_date.date()),
                first_execution_date=str(first_execution_date.date()),
                selection_data_end=str(selection_data_end.date()),
            )
        )
        selections.append(
            SelectionRecord(
                fold_id=fold_id,
                selected=selected,
                used_fixed_fallback=used_fallback,
                candidates_tested=len(scores),
                candidates_valid=len(valid),
                selected_median_active_return=selected_score.median_active_return,
                selected_worst_block_active_return=selected_score.worst_block_active_return,
                selected_annualized_turnover=selected_score.annualized_turnover,
                selection_data_end=str(selection_data_end.date()),
            )
        )
        all_candidate_scores.append(scores)

    selected_signal = pd.concat(selected_signal_parts).sort_index()
    selected_score_frame = pd.concat(selected_score_parts).sort_index()
    parameters_by_date = pd.concat(parameter_rows).sort_index()
    oos_returns = return_frame.returns.loc[selected_signal.index]
    selected_weights = build_weights(
        selected_signal,
        config.portfolio.primary,
        config.portfolio.long_only_weight_per_asset,
        config.portfolio.neutral_gross_limit,
    )
    portfolio = evaluate_portfolio(
        selected_weights,
        oos_returns,
        config.research.primary_cost_bps,
        liquidate=True,
    )

    _, fixed_signal_full, _ = artifacts(_parameter_key(fixed))
    fixed_signal = fixed_signal_full.loc[selected_signal.index]
    fixed_weights = build_weights(
        fixed_signal,
        config.portfolio.primary,
        config.portfolio.long_only_weight_per_asset,
        config.portfolio.neutral_gross_limit,
    )
    fixed_portfolio = evaluate_portfolio(
        fixed_weights,
        oos_returns,
        config.research.primary_cost_bps,
        liquidate=True,
    )
    return WalkForwardResult(
        folds=tuple(folds),
        selections=tuple(selections),
        candidate_scores=tuple(all_candidate_scores),
        parameters_by_date=parameters_by_date,
        signal=selected_signal,
        score=selected_score_frame,
        asset_returns=oos_returns,
        portfolio=portfolio,
        benchmark=benchmarks(
            oos_returns,
            rebalanced_cost_bps=config.research.primary_cost_bps,
        ),
        fixed_baseline_portfolio=fixed_portfolio,
        return_frame=return_frame,
    )
