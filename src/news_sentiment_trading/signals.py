"""Explicit event-signal and bounded holding policies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from news_sentiment_trading.features import FeatureSet


def event_signal(features: FeatureSet, z_threshold: float) -> pd.DataFrame:
    """Emit +1/-1 only on a qualifying event; otherwise emit zero."""

    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")
    signal = pd.DataFrame(0, index=features.score.index, columns=features.score.columns, dtype=int)
    active = features.active_news & features.available
    signal = signal.mask(active & features.score.gt(z_threshold), 1)
    signal = signal.mask(active & features.score.lt(-z_threshold), -1)
    return signal


def bounded_holding(signal: pd.DataFrame, holding_days: int = 1) -> pd.DataFrame:
    """Persist each nonzero event for a named, finite number of signal sessions."""

    if holding_days < 1:
        raise ValueError("holding_days must be positive")
    if holding_days == 1:
        return signal.copy()

    output = pd.DataFrame(0, index=signal.index, columns=signal.columns, dtype=int)
    for column in signal.columns:
        remaining = 0
        position = 0
        values = signal[column].to_numpy(dtype=int)
        held = np.zeros(len(values), dtype=int)
        for index, event in enumerate(values):
            if event != 0:
                position = int(event)
                remaining = holding_days
            if remaining > 0:
                held[index] = position
                remaining -= 1
            else:
                position = 0
        output[column] = held
    return output
