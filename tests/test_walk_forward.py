from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from news_sentiment_trading.config import SignalParameters, load_config
from news_sentiment_trading.data import aligned_forward_returns
from news_sentiment_trading.walk_forward import (
    CandidateScore,
    _inner_blocks,
    _score_candidate,
    _select_candidate,
    make_outer_slices,
    run_walk_forward,
)


def _config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "primary.toml")


def test_outer_blocks_cover_every_post_initial_observation_once(panel: pd.DataFrame) -> None:
    returns = aligned_forward_returns(panel)
    blocks = make_outer_slices(returns.returns.index, 252, 58)
    assert len(blocks) == 6
    assert all(len(block) == 58 for block in blocks)
    combined = pd.DatetimeIndex([date for block in blocks for date in block])
    assert combined.equals(returns.returns.index[252:])
    assert not combined.duplicated().any()


def test_every_outer_fold_uses_exact_prior_three_by_42_inner_blocks(
    panel: pd.DataFrame,
) -> None:
    config = _config()
    frame = aligned_forward_returns(panel)
    return_dates = pd.DatetimeIndex(frame.returns.index)
    outer = make_outer_slices(return_dates, 252, 58)
    for test_dates in outer:
        first_end = pd.Timestamp(test_dates[0])
        signal_date = pd.Timestamp(frame.signal_dates.loc[first_end])
        execution_date = pd.Timestamp(frame.execution_dates.loc[first_end])
        eligible = return_dates[return_dates <= signal_date]
        blocks = _inner_blocks(eligible, config)
        combined = pd.DatetimeIndex([date for block in blocks for date in block])
        assert tuple(map(len, blocks)) == (42, 42, 42)
        assert combined.equals(eligible[-126:])
        assert blocks[-1][-1] == signal_date
        assert signal_date < execution_date < first_end


def test_walk_forward_is_deterministic(panel: pd.DataFrame) -> None:
    config = _config()
    first = run_walk_forward(panel, config)
    second = run_walk_forward(panel, config)
    assert first.selections == second.selections
    pd.testing.assert_frame_equal(first.portfolio.weights, second.portfolio.weights)
    pd.testing.assert_series_equal(first.portfolio.net_return, second.portfolio.net_return)


def test_outer_test_mutation_cannot_change_its_selected_parameters(panel: pd.DataFrame) -> None:
    config = _config()
    original = run_walk_forward(panel, config)
    mutated = panel.copy()
    first_test_start = pd.Timestamp(original.folds[0].test_start)
    future_rows = mutated.index.get_level_values("date") >= first_test_start
    mutated.loc[future_rows, "Adj Close"] *= 1.7
    mutated.loc[future_rows, "Adjusted Open"] *= 1.7
    mutated.loc[future_rows, "positivePartscr"] *= 100.0
    changed = run_walk_forward(mutated, config)
    assert original.selections[0].selected == changed.selections[0].selected


def _candidate(
    parameters: SignalParameters,
    *,
    valid: bool = True,
    median: float = 1.0,
    worst: float = 0.5,
    turnover: float = 10.0,
) -> CandidateScore:
    return CandidateScore(parameters, valid, median, worst, turnover, 30, 3, "")


def test_selection_uses_locked_objective_and_conservative_tie_order() -> None:
    base = SignalParameters(20, 1.5, 0.5)
    high_median = _candidate(base, median=2.0, worst=-10.0, turnover=100.0)
    high_worst = _candidate(SignalParameters(50, 2.0, 0.75), median=1.0, worst=0.6)
    assert _select_candidate((high_worst, high_median), base) == (high_median, False)

    high_turnover = _candidate(SignalParameters(20, 1.5, 0.5), turnover=10.0)
    low_turnover = _candidate(SignalParameters(50, 1.5, 0.5), turnover=5.0)
    assert _select_candidate((high_turnover, low_turnover), base) == (low_turnover, False)

    z_preferred = _candidate(SignalParameters(20, 2.0, 0.5))
    q_and_window_alternative = _candidate(SignalParameters(50, 1.5, 0.75))
    assert _select_candidate((q_and_window_alternative, z_preferred), base) == (z_preferred, False)

    q_preferred = _candidate(SignalParameters(20, 2.0, 0.75))
    window_alternative = _candidate(SignalParameters(50, 2.0, 0.5))
    assert _select_candidate((window_alternative, q_preferred), base) == (q_preferred, False)

    long_window = _candidate(SignalParameters(50, 2.0, 0.75))
    short_window = _candidate(SignalParameters(20, 2.0, 0.75))
    assert _select_candidate((short_window, long_window), base) == (long_window, False)

    invalid_high_score = _candidate(
        SignalParameters(50, 2.0, 0.75),
        valid=False,
        median=100.0,
        worst=100.0,
    )
    valid_low_score = _candidate(base, median=-100.0, worst=-100.0)
    assert _select_candidate((invalid_high_score, valid_low_score), base) == (
        valid_low_score,
        False,
    )


def test_selection_uses_fixed_baseline_when_every_candidate_is_invalid() -> None:
    fixed = SignalParameters(20, 1.5, 0.5)
    scores = (
        _candidate(SignalParameters(50, 2.0, 0.75), valid=False),
        _candidate(fixed, valid=False),
    )

    assert _select_candidate(scores, fixed) == (scores[1], True)


def test_candidate_score_locks_block_objective_coverage_and_turnover_rules() -> None:
    dates = pd.date_range("2026-01-05", periods=6, freq="B")
    blocks = tuple(pd.DatetimeIndex(dates[start : start + 2]) for start in range(0, 6, 2))
    signal = pd.DataFrame({"AAA": 1.0, "BBB": 0.0}, index=dates)
    returns = pd.DataFrame({"AAA": [0.01, 0.03, 0.02, 0.04, -0.01, 0.01], "BBB": 0.0}, index=dates)
    benchmark = pd.Series(0.0, index=dates)
    config = _config()
    permissive = replace(
        config,
        walk_forward=replace(
            config.walk_forward,
            minimum_active_asset_days=1,
            minimum_active_fraction=0.01,
            maximum_annualized_turnover=10_000.0,
            selection_cost_bps=0,
        ),
    )
    parameters = SignalParameters(20, 1.5, 0.5)

    score = _score_candidate(parameters, signal, returns, benchmark, blocks, permissive)

    assert score.valid
    assert score.blocks_with_signals == 3
    assert score.active_asset_days == 6
    assert score.median_active_return == pytest.approx(1.008)
    assert score.worst_block_active_return == pytest.approx(0.0)

    sparse_signal = signal.copy()
    sparse_signal.loc[blocks[1], "AAA"] = 0.0
    strict = replace(
        permissive,
        walk_forward=replace(
            permissive.walk_forward,
            minimum_active_asset_days=6,
            minimum_active_fraction=0.60,
            maximum_annualized_turnover=0.0,
        ),
    )
    rejected = _score_candidate(parameters, sparse_signal, returns, benchmark, blocks, strict)
    assert not rejected.valid
    assert "no signal in an inner block" in rejected.reason
    assert "insufficient active asset-days" in rejected.reason
    assert "active fraction below minimum" in rejected.reason
    assert "annualized turnover above maximum" in rejected.reason
