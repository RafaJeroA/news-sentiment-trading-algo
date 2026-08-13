"""Point-in-time feature construction with strict-prior rolling statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from news_sentiment_trading.config import FeatureKind, MissingPolicy, SignalParameters

BULL_COLUMNS: tuple[str, ...] = ("positivePartscr", "certaintyPartscr", "finupPartscr")
BEAR_COLUMNS: tuple[str, ...] = ("negativePartscr", "uncertaintyPartscr", "findownPartscr")


@dataclass(frozen=True)
class FeatureSet:
    raw: pd.DataFrame
    score: pd.DataFrame
    prior_mean: pd.DataFrame
    prior_std: pd.DataFrame
    prior_rvt_threshold: pd.DataFrame
    active_news: pd.DataFrame
    available: pd.DataFrame
    imputed: pd.DataFrame


def bull_bear_components(panel: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the assignment's BULL and BEAR definitions exactly."""

    missing = (set(BULL_COLUMNS) | set(BEAR_COLUMNS)) - set(panel.columns)
    if missing:
        raise ValueError(f"missing sentiment components: {sorted(missing)}")
    output = pd.DataFrame(index=panel.index)
    output["BULL"] = panel.loc[:, list(BULL_COLUMNS)].sum(axis=1, min_count=3)
    output["BEAR"] = panel.loc[:, list(BEAR_COLUMNS)].sum(axis=1, min_count=3)
    output["components_available"] = (
        panel.loc[:, [*BULL_COLUMNS, *BEAR_COLUMNS]].notna().all(axis=1)
    )
    return output


def sentiment_feature(panel: pd.DataFrame, kind: FeatureKind = "ratio") -> pd.Series:
    """Create the predefined ratio or stable log-ratio without filling."""

    components = bull_bear_components(panel)
    denominator = components["BULL"] + components["BEAR"]
    valid = components["components_available"].astype(bool) & denominator.gt(0)
    feature = pd.Series(np.nan, index=panel.index, dtype=float, name=kind)
    if kind == "ratio":
        feature.loc[valid] = 100.0 * components.loc[valid, "BULL"] / denominator.loc[valid]
    elif kind == "log_ratio":
        feature.loc[valid] = np.log1p(components.loc[valid, "BULL"]) - np.log1p(
            components.loc[valid, "BEAR"]
        )
    else:  # pragma: no cover - protected by validated configuration
        raise ValueError(f"unsupported feature kind: {kind}")
    return feature


def _apply_missing_policy(
    feature: pd.DataFrame, policy: MissingPolicy
) -> tuple[pd.DataFrame, pd.DataFrame]:
    originally_missing = feature.isna()
    if policy == "no_fill":
        filled = feature.copy()
    elif policy == "ffill_1":
        filled = feature.ffill(limit=1)
    else:  # pragma: no cover - protected by validated configuration
        raise ValueError(f"unsupported missing policy: {policy}")
    imputed = originally_missing & filled.notna()
    return filled, imputed


def strictly_prior_rolling(
    values: pd.DataFrame,
    window: int,
    statistic: str,
    minimum_fraction: float = 0.80,
    quantile: float | None = None,
) -> pd.DataFrame:
    """Calculate a rolling statistic from observations strictly before each row."""

    if window < 2:
        raise ValueError("window must be at least two")
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be in (0, 1]")
    minimum = int(math.ceil(window * minimum_fraction))
    rolling = values.rolling(window=window, min_periods=minimum)
    if statistic == "mean":
        result = rolling.mean()
    elif statistic == "std":
        result = rolling.std(ddof=0)
    elif statistic == "quantile":
        if quantile is None or not 0 <= quantile <= 1:
            raise ValueError("a quantile in [0, 1] is required")
        result = rolling.quantile(quantile, interpolation="linear")
    else:
        raise ValueError(f"unsupported rolling statistic: {statistic}")
    return result.shift(1)


def build_features(
    panel: pd.DataFrame,
    parameters: SignalParameters,
    minimum_history_fraction: float = 0.80,
) -> FeatureSet:
    """Build the complete point-in-time feature set for every asset."""

    raw_long = sentiment_feature(panel, parameters.feature_kind)
    raw = raw_long.unstack("ticker").sort_index()
    filled, imputed = _apply_missing_policy(raw, parameters.missing_policy)
    rvt = panel["RVT"].unstack("ticker").sort_index()

    prior_mean = strictly_prior_rolling(filled, parameters.window, "mean", minimum_history_fraction)
    prior_std = strictly_prior_rolling(
        filled, parameters.window, "std", minimum_history_fraction
    ).replace(0.0, np.nan)
    prior_rvt_threshold = strictly_prior_rolling(
        rvt,
        parameters.window,
        "quantile",
        minimum_history_fraction,
        quantile=parameters.rvt_quantile,
    )
    score = filled.sub(prior_mean).div(prior_std)
    active_news = rvt.ge(prior_rvt_threshold)
    available = (
        filled.notna()
        & rvt.notna()
        & prior_mean.notna()
        & prior_std.notna()
        & prior_rvt_threshold.notna()
        & np.isfinite(score)
    )
    active_news = active_news & available
    score = score.where(available)
    return FeatureSet(
        raw=filled,
        score=score,
        prior_mean=prior_mean,
        prior_std=prior_std,
        prior_rvt_threshold=prior_rvt_threshold,
        active_news=active_news,
        available=available,
        imputed=imputed,
    )
