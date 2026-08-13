"""Original synthetic data for tests and public demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from news_sentiment_trading.data import PRIMARY_TICKERS, SENTIMENT_COLUMNS


def synthetic_panel(
    sessions: int = 602,
    tickers: tuple[str, ...] = PRIMARY_TICKERS,
    seed: int = 20260801,
) -> pd.DataFrame:
    """Create a deterministic panel with no relationship to restricted source rows."""

    if sessions < 60:
        raise ValueError("synthetic panel requires at least 60 sessions")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=sessions)
    frames: list[pd.DataFrame] = []
    for ticker_index, ticker in enumerate(tickers):
        innovations = rng.normal(0.0002, 0.012 + ticker_index * 0.0003, sessions)
        close = (80.0 + 5.0 * ticker_index) * np.exp(np.cumsum(innovations))
        overnight = rng.normal(0.0, 0.003, sessions)
        open_price = close * np.exp(overnight)
        high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.0, 0.012, sessions))
        low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.0, 0.012, sessions))
        sentiment = rng.gamma(shape=1.5, scale=0.01, size=(sessions, len(SENTIMENT_COLUMNS)))
        sentiment[:, 0] = rng.uniform(0.0001, 0.005, sessions)
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": rng.integers(500_000, 30_000_000, sessions),
                **{column: sentiment[:, index] for index, column in enumerate(SENTIMENT_COLUMNS)},
                "Adjusted Open": open_price,
            }
        )
        if ticker_index % 3 == 0:
            frame.loc[37, ["RVT", *SENTIMENT_COLUMNS[1:]]] = np.nan
        if ticker_index % 4 == 0 and sessions > 73:
            frame.loc[
                73,
                [
                    "positivePartscr",
                    "certaintyPartscr",
                    "finupPartscr",
                    "negativePartscr",
                    "uncertaintyPartscr",
                    "findownPartscr",
                ],
            ] = 0.0
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    panel.index.names = ["date", "ticker"]
    return panel


def write_synthetic_csvs(destination: str | Path, sessions: int = 80) -> Path:
    """Write public demonstration CSVs created entirely by this package."""

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    panel = synthetic_panel(sessions=max(sessions, 60))
    for ticker in PRIMARY_TICKERS:
        asset = panel.xs(ticker, level="ticker").reset_index().rename(columns={"date": "Date"})
        asset = asset.drop(columns=["Adjusted Open"])
        asset.to_csv(destination_path / f"{ticker}.csv", index=False, lineterminator="\n")
    return destination_path
