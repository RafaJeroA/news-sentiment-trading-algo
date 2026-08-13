from __future__ import annotations

from pathlib import Path

import pandas as pd

from news_sentiment_trading.config import load_config
from news_sentiment_trading.diagnostics import (
    corrected_fb_jpm,
    crisis_decomposition,
    parameter_stability,
    score_sorting,
    signal_coverage,
    signal_decay,
)
from news_sentiment_trading.walk_forward import run_walk_forward


def _config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "primary.toml")


def test_descriptive_diagnostics_are_complete(panel: pd.DataFrame) -> None:
    config = _config()
    result = run_walk_forward(panel, config)
    corrected = corrected_fb_jpm(panel, config)
    assert set(corrected) == {"FB", "JPM"}
    assert corrected["FB"]["parameters"]["window"] == 50
    assert corrected["JPM"]["parameters"]["window"] == 20
    assert set(corrected["FB"]["regimes"]) == {"long_only", "directional"}

    crisis = crisis_decomposition(result, config)
    observations = sum(
        crisis[name]["strategy"]["observations"]
        for name in ("pre_crisis_and_crisis_excluded", "crisis")
    )
    assert observations == len(result.portfolio.net_return)

    coverage = signal_coverage(result)
    assert set(coverage["by_asset"]) == set(result.signal.columns)
    assert coverage["aggregate_active_asset_days"] == int(result.signal.ne(0).sum().sum())
    assert sum(parameter_stability(result).values()) == len(result.folds)


def test_sorting_and_decay_do_not_change_primary_result(panel: pd.DataFrame) -> None:
    config = _config()
    result = run_walk_forward(panel, config)
    before = result.portfolio.net_return.copy()
    sorting = score_sorting(result)
    decay = signal_decay(panel, result)
    pd.testing.assert_series_equal(before, result.portfolio.net_return)
    assert [row["score_group"] for row in sorting] == [1, 2, 3, 4, 5]
    assert [row["horizon_sessions"] for row in decay] == [1, 2, 3]
