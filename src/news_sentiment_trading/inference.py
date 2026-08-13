"""Defensible time-series uncertainty and exposure diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


@dataclass(frozen=True)
class HACMeanResult:
    mean: float
    standard_error: float
    t_statistic: float
    p_value_two_sided: float
    confidence_low: float
    confidence_high: float
    lags: int
    observations: int


@dataclass(frozen=True)
class BetaResult:
    intercept_daily: float
    beta: float
    intercept_hac_se: float
    beta_hac_se: float
    lags: int
    observations: int


def _bartlett_hac_meat(scores: np.ndarray, lags: int) -> np.ndarray:
    """Return the unscaled Bartlett HAC meat while preserving explicit row gaps.

    Rows containing a non-finite score are excluded from contemporaneous products and
    from every lag pair that touches the row. Keeping those rows in place is important:
    dropping them first would incorrectly turn observations on opposite sides of a
    missing market date into adjacent observations.
    """

    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional array")
    valid_rows = np.isfinite(scores).all(axis=1)
    finite_scores = scores[valid_rows]
    meat = finite_scores.T @ finite_scores
    for lag in range(1, lags + 1):
        paired = valid_rows[lag:] & valid_rows[:-lag]
        if not paired.any():
            continue
        gamma = scores[lag:][paired].T @ scores[:-lag][paired]
        weight = 1.0 - lag / (lags + 1.0)
        meat += weight * (gamma + gamma.T)
    return np.asarray(meat, dtype=float)


def automatic_hac_lags(observations: int) -> int:
    if observations <= 1:
        return 0
    return max(0, int(math.floor(4.0 * (observations / 100.0) ** (2.0 / 9.0))))


def hac_mean(series: pd.Series, lags: int | None = None, confidence: float = 0.95) -> HACMeanResult:
    """Newey-West uncertainty for a sample mean using Bartlett weights.

    Explicit missing rows retain their position in the lag structure. The reported
    observation count and mean use only finite observations.
    """

    values = series.to_numpy(dtype=float)
    finite = np.isfinite(values)
    observations = int(finite.sum())
    if observations == 0:
        raise ValueError("HAC mean requires observations")
    selected_lags = automatic_hac_lags(observations) if lags is None else int(lags)
    if not 0 <= selected_lags < observations:
        raise ValueError("lags must be in [0, observations)")
    mean = float(values[finite].mean())
    centered = values - mean
    meat = _bartlett_hac_meat(centered[:, None], selected_lags)
    variance_of_mean = max(float(meat[0, 0]), 0.0) / observations**2
    standard_error = math.sqrt(variance_of_mean)
    t_statistic = mean / standard_error if standard_error > 0 else float("nan")
    p_value = float(2.0 * norm.sf(abs(t_statistic))) if math.isfinite(t_statistic) else float("nan")
    critical = float(norm.ppf(0.5 + confidence / 2.0))
    return HACMeanResult(
        mean=mean,
        standard_error=standard_error,
        t_statistic=t_statistic,
        p_value_two_sided=p_value,
        confidence_low=mean - critical * standard_error,
        confidence_high=mean + critical * standard_error,
        lags=selected_lags,
        observations=observations,
    )


def moving_block_bootstrap_mean_ci(
    series: pd.Series,
    samples: int = 10_000,
    block_length: int = 10,
    seed: int = 20260801,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Circular moving-block bootstrap confidence interval for a complete daily series.

    Missing rows must not be silently dropped because doing so would turn dates on
    opposite sides of a calendar gap into adjacent observations. Callers must supply
    the complete synchronized daily estimand used by the preregistered bootstrap.
    """

    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("bootstrap requires a complete finite series")
    observations = len(values)
    if observations == 0:
        raise ValueError("bootstrap requires observations")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if not 1 <= block_length <= observations:
        raise ValueError("block_length must be in [1, observations]")
    rng = np.random.default_rng(seed)
    blocks_needed = int(math.ceil(observations / block_length))
    bootstrap_means = np.empty(samples, dtype=float)
    offsets = np.arange(block_length)
    for index in range(samples):
        starts = rng.integers(0, observations, size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]) % observations
        sample = values[indices.ravel()[:observations]]
        bootstrap_means[index] = float(sample.mean())
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return float(low), float(high)


def daily_rank_ic(
    score: pd.DataFrame, forward_returns: pd.DataFrame, minimum_assets: int = 6
) -> pd.Series:
    """Daily Spearman rank information coefficient with a coverage rule."""

    score, forward_returns = score.align(forward_returns, join="inner", axis=0)
    rows: dict[pd.Timestamp, float] = {}
    for date in score.index:
        joined = pd.concat(
            [score.loc[date].rename("score"), forward_returns.loc[date].rename("return")], axis=1
        ).dropna()
        if len(joined) < minimum_assets or joined["score"].nunique() < 2:
            rows[pd.Timestamp(date)] = float("nan")
        else:
            correlation = spearmanr(joined["score"], joined["return"]).statistic
            rows[pd.Timestamp(date)] = float(correlation)
    return pd.Series(rows, name="rank_ic", dtype=float)


def hac_alpha_beta(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, lags: int | None = None
) -> BetaResult:
    """OLS intercept/beta with a gap-preserving Newey-West covariance matrix."""

    joined = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    )
    strategy = joined["strategy"].to_numpy(dtype=float)
    benchmark = joined["benchmark"].to_numpy(dtype=float)
    finite = np.isfinite(strategy) & np.isfinite(benchmark)
    observations = int(finite.sum())
    y = strategy[finite]
    x = np.column_stack([np.ones(observations), benchmark[finite]])
    if observations < 3 or np.linalg.matrix_rank(x) < 2:
        raise ValueError("alpha/beta regression requires at least three non-collinear rows")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    selected_lags = automatic_hac_lags(observations) if lags is None else int(lags)
    if not 0 <= selected_lags < observations:
        raise ValueError("lags must be in [0, observations)")
    bread = np.linalg.inv(x.T @ x)
    scores = np.full((len(joined), 2), np.nan, dtype=float)
    scores[finite] = x * residuals[:, None]
    meat = _bartlett_hac_meat(scores, selected_lags)
    covariance = bread @ meat @ bread
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return BetaResult(
        intercept_daily=float(coefficients[0]),
        beta=float(coefficients[1]),
        intercept_hac_se=float(standard_errors[0]),
        beta_hac_se=float(standard_errors[1]),
        lags=selected_lags,
        observations=observations,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjusted p-values in original order."""

    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[int(original_index)]))
        running = max(running, candidate)
        adjusted[int(original_index)] = running
    return adjusted.tolist()
