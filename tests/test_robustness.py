from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from news_sentiment_trading.config import PrimaryConfig, load_config
from news_sentiment_trading.data import PRIMARY_TICKERS
from news_sentiment_trading.robustness import run_robustness
from news_sentiment_trading.synthetic import synthetic_panel
from news_sentiment_trading.walk_forward import run_walk_forward


def _compact_config() -> PrimaryConfig:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "primary.toml")
    tickers = PRIMARY_TICKERS[:3]
    return replace(
        config,
        universe=replace(config.universe, primary=tickers),
        walk_forward=replace(
            config.walk_forward,
            outer_initial_sessions=30,
            outer_test_sessions=10,
            inner_initial_sessions=15,
            inner_test_sessions=5,
            minimum_active_fraction=0.0,
            minimum_active_asset_days=0,
            maximum_annualized_turnover=10_000.0,
        ),
        features=replace(
            config.features,
            windows=(5,),
            z_thresholds=(0.5,),
            rvt_quantiles=(0.5,),
            fixed_window=5,
            fixed_z_threshold=0.5,
            fixed_rvt_quantile=0.5,
        ),
    )


def _synthetic_result():
    config = _compact_config()
    panel = synthetic_panel(sessions=66, tickers=config.universe.primary)
    primary = run_walk_forward(panel, config)
    crisis_start = pd.Timestamp(
        primary.portfolio.net_return.index[len(primary.portfolio.net_return) // 2]
    )
    return run_robustness(
        panel,
        config,
        primary_result=primary,
        crisis_start=crisis_start,
    )


def test_runner_covers_every_preregistered_robustness_family() -> None:
    result = _synthetic_result()
    expected_groups = {
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
    assert set(result.summaries["group"]) == expected_groups

    portfolio_cost = result.summaries[result.summaries["group"] == "portfolio_cost"]
    assert set(portfolio_cost["portfolio_kind"]) == {
        "long_only",
        "market_neutral",
        "directional",
    }
    assert set(portfolio_cost["cost_bps"]) == {0, 5, 10, 25, 50, 100}

    turnover = result.summaries[result.summaries["group"] == "turnover_convention"].set_index(
        "variant"
    )
    assert turnover.loc["registered_target_weight", "turnover_convention"] == "target_weight"
    assert turnover.loc["post_v1_drift_adjusted", "turnover_convention"] == "drift_adjusted"
    selected_return = result.summaries.query(
        "group == 'parameter_policy' and variant == 'selected'"
    )["strategy_total_return"].iloc[0]
    assert turnover.loc["registered_target_weight", "strategy_total_return"] == pytest.approx(
        selected_return
    )

    leave_one_out = result.summaries[result.summaries["group"] == "leave_one_out"]
    assert tuple(leave_one_out["excluded_ticker"]) == PRIMARY_TICKERS[:3]
    assert len(result.audit_records) == len(result.summaries)
    audit_keys = {(record.group, record.variant) for record in result.audit_records}
    summary_keys = set(zip(result.summaries["group"], result.summaries["variant"], strict=True))
    assert audit_keys == summary_keys


def test_costs_and_crisis_decomposition_reconcile() -> None:
    result = _synthetic_result()
    costs = result.summaries[
        (result.summaries["group"] == "portfolio_cost")
        & (result.summaries["portfolio_kind"] == "long_only")
    ].sort_values("cost_bps")
    assert costs["transaction_cost_total"].is_monotonic_increasing

    periods = result.summaries[result.summaries["group"] == "crisis_period"].set_index("variant")
    assert periods.loc["full_oos", "observations"] == (
        periods.loc["pre_crisis_component", "observations"]
        + periods.loc["crisis_component", "observations"]
    )
    assert (
        periods.loc["crisis_excluded_liquidated", "observations"]
        == periods.loc["pre_crisis_component", "observations"]
    )
    assert (
        periods.loc["crisis_excluded_liquidated", "transaction_cost_total"]
        >= periods.loc["pre_crisis_component", "transaction_cost_total"]
    )


def test_runner_is_deterministic_on_synthetic_data() -> None:
    first = _synthetic_result()
    second = _synthetic_result()
    pd.testing.assert_frame_equal(first.summaries, second.summaries)
    pd.testing.assert_frame_equal(first.selections, second.selections)


def test_full_rerun_variants_record_fold_selections() -> None:
    result = _synthetic_result()
    selection_pairs = set(
        zip(result.selections["group"], result.selections["variant"], strict=True)
    )
    assert ("parameter_policy", "selected") in selection_pairs
    assert ("missing_policy", "ffill_1") in selection_pairs
    assert ("feature", "log_ratio") in selection_pairs
    assert ("holding", "3_sessions") in selection_pairs
    assert ("execution", "lagged_adjusted_close") in selection_pairs
    assert {variant for group, variant in selection_pairs if group == "leave_one_out"} == {
        f"exclude_{ticker}" for ticker in PRIMARY_TICKERS[:3]
    }
