from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from news_sentiment_trading.config import SignalParameters
from news_sentiment_trading.features import (
    FeatureSet,
    build_features,
    bull_bear_components,
    sentiment_feature,
    strictly_prior_rolling,
)
from news_sentiment_trading.signals import bounded_holding, event_signal


def test_bull_bear_formula(panel: pd.DataFrame) -> None:
    components = bull_bear_components(panel)
    row = panel.iloc[0]
    assert components.iloc[0]["BULL"] == pytest.approx(
        row["positivePartscr"] + row["certaintyPartscr"] + row["finupPartscr"]
    )
    assert components.iloc[0]["BEAR"] == pytest.approx(
        row["negativePartscr"] + row["uncertaintyPartscr"] + row["findownPartscr"]
    )


def test_zero_denominator_is_missing_no_signal(panel: pd.DataFrame) -> None:
    mutated = panel.copy()
    date = mutated.index.get_level_values("date").unique()[90]
    columns = [
        "positivePartscr",
        "certaintyPartscr",
        "finupPartscr",
        "negativePartscr",
        "uncertaintyPartscr",
        "findownPartscr",
    ]
    mutated.loc[(date, "JPM"), columns] = 0.0
    ratio = sentiment_feature(mutated)
    assert pd.isna(ratio.loc[(date, "JPM")])
    parameters = SignalParameters(20, 1.5, 0.5)
    features = build_features(mutated, parameters)
    assert not features.available.loc[date, "JPM"]
    assert event_signal(features, 1.5).loc[date, "JPM"] == 0


def test_strictly_prior_rolling_excludes_current() -> None:
    values = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})
    mean = strictly_prior_rolling(values, window=3, statistic="mean", minimum_fraction=1.0)
    assert mean.loc[3, "x"] == pytest.approx(2.0)


def test_log_ratio_and_eighty_percent_history_boundary() -> None:
    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-01-02"), "A")])
    panel = pd.DataFrame(
        {
            "positivePartscr": [1.0],
            "certaintyPartscr": [2.0],
            "finupPartscr": [3.0],
            "negativePartscr": [4.0],
            "uncertaintyPartscr": [5.0],
            "findownPartscr": [6.0],
        },
        index=index,
    )
    value = sentiment_feature(panel, "log_ratio").iloc[0]
    assert value == pytest.approx(np.log1p(6.0) - np.log1p(15.0))

    values = pd.DataFrame({"x": np.arange(21, dtype=float)})
    exactly_eighty = values.copy()
    exactly_eighty.loc[[0, 1, 2, 3], "x"] = np.nan
    below_eighty = exactly_eighty.copy()
    below_eighty.loc[4, "x"] = np.nan
    exact_mean = strictly_prior_rolling(exactly_eighty, 20, "mean").loc[20, "x"]
    missing_mean = strictly_prior_rolling(below_eighty, 20, "mean").loc[20, "x"]
    assert exact_mean == pytest.approx(np.mean(np.arange(4, 20, dtype=float)))
    assert pd.isna(missing_mean)


def test_population_std_linear_quantile_and_signal_boundaries_are_locked() -> None:
    values = pd.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 999.0]})
    prior_std = strictly_prior_rolling(values, 4, "std", minimum_fraction=1.0).loc[4, "x"]
    prior_quantile = strictly_prior_rolling(
        values,
        4,
        "quantile",
        minimum_fraction=1.0,
        quantile=0.25,
    ).loc[4, "x"]
    assert prior_std == pytest.approx(np.sqrt(125.0))
    assert prior_quantile == pytest.approx(7.5)

    index = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06"])
    columns = ["A"]
    score = pd.DataFrame([1.5, -1.5, 1.500001], index=index, columns=columns)
    available = pd.DataFrame(True, index=index, columns=columns)
    features = FeatureSet(
        raw=score.copy(),
        score=score,
        prior_mean=score.copy(),
        prior_std=score.copy(),
        prior_rvt_threshold=score.copy(),
        active_news=available.copy(),
        available=available.copy(),
        imputed=pd.DataFrame(False, index=index, columns=columns),
    )
    assert event_signal(features, 1.5)["A"].tolist() == [0, 0, 1]


def test_rvt_threshold_is_inclusive_at_equality(panel: pd.DataFrame) -> None:
    mutated = panel.copy()
    dates = mutated.index.get_level_values("date").unique()
    mutated.loc[(dates[:21], "AAPL"), "RVT"] = 1.0
    features = build_features(mutated, SignalParameters(20, 1.5, 0.5))
    assert features.prior_rvt_threshold.loc[dates[20], "AAPL"] == pytest.approx(1.0)
    assert features.active_news.loc[dates[20], "AAPL"]


def test_future_mutation_does_not_change_past_features(panel: pd.DataFrame) -> None:
    parameters = SignalParameters(20, 1.5, 0.5)
    original = build_features(panel, parameters)
    cutoff = panel.index.get_level_values("date").unique()[250]
    mutated = panel.copy()
    future = mutated.index.get_level_values("date") > cutoff
    mutated.loc[future, "positivePartscr"] *= 10_000
    changed = build_features(mutated, parameters)
    pd.testing.assert_frame_equal(original.score.loc[:cutoff], changed.score.loc[:cutoff])
    pd.testing.assert_frame_equal(
        original.active_news.loc[:cutoff], changed.active_news.loc[:cutoff]
    )


def test_bounded_forward_fill_expires_after_one_session(panel: pd.DataFrame) -> None:
    mutated = panel.copy()
    dates = mutated.index.get_level_values("date").unique()
    columns = [
        "positivePartscr",
        "certaintyPartscr",
        "finupPartscr",
        "negativePartscr",
        "uncertaintyPartscr",
        "findownPartscr",
    ]
    mutated.loc[(dates[100], "DB"), columns] = np.nan
    mutated.loc[(dates[101], "DB"), columns] = np.nan
    parameters = SignalParameters(20, 1.5, 0.5, missing_policy="ffill_1")
    features = build_features(mutated, parameters)
    assert features.imputed.loc[dates[100], "DB"]
    assert not features.imputed.loc[dates[101], "DB"]
    assert pd.isna(features.raw.loc[dates[101], "DB"])


def test_bounded_holding_has_no_accidental_persistence() -> None:
    signal = pd.DataFrame({"A": [0, 1, 0, 0, -1, 0, 0]})
    one = bounded_holding(signal, 1)
    three = bounded_holding(signal, 3)
    assert one["A"].tolist() == [0, 1, 0, 0, -1, 0, 0]
    assert three["A"].tolist() == [0, 1, 1, 1, -1, -1, -1]


def test_bounded_holding_overlapping_event_resets_named_horizon() -> None:
    signal = pd.DataFrame({"A": [1, 0, -1, 0, 1, 1, 0, 0]})
    held = bounded_holding(signal, 3)
    assert held["A"].tolist() == [1, 1, -1, -1, 1, 1, 1, 1]


def test_adversarial_future_leak_fixture_rejects_leaked_story() -> None:
    rng = np.random.default_rng(7)
    values = pd.DataFrame({"x": rng.normal(size=2_000)})
    future_return = values["x"].shift(-1)
    correct = (values["x"] - strictly_prior_rolling(values, 20, "mean")["x"]).shift(0)
    leaked = values["x"].shift(-1)
    assert leaked.corr(future_return) > 0.999
    assert abs(correct.corr(future_return)) < 0.10
