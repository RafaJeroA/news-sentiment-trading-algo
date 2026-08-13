"""Immutable-source discovery, schema validation, and return alignment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_TICKERS: tuple[str, ...] = (
    "AAPL",
    "AMZN",
    "DB",
    "DIS",
    "FB",
    "GOOG",
    "HSBC",
    "JPM",
    "MSFT",
    "PFE",
)

PRICE_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
SENTIMENT_COLUMNS: tuple[str, ...] = (
    "RVT",
    "positivePartscr",
    "negativePartscr",
    "fearPartscr",
    "findownPartscr",
    "finupPartscr",
    "finhypePartscr",
    "certaintyPartscr",
    "uncertaintyPartscr",
)
REQUIRED_COLUMNS: tuple[str, ...] = ("Date", *PRICE_COLUMNS, *SENTIMENT_COLUMNS)


@dataclass(frozen=True)
class ManifestEntry:
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class AssetInventory:
    ticker: str
    relative_path: str
    sha256: str
    rows: int
    start: str
    end: str
    duplicate_dates: int
    sorted_dates: bool
    missing_price_cells: int
    missing_sentiment_cells: int
    undefined_ratio_rows: int


@dataclass(frozen=True)
class ReturnFrame:
    """Asset returns with explicit signal, execution, and return-end dates."""

    returns: pd.DataFrame
    signal_dates: pd.Series
    execution_dates: pd.Series


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_text(path: str | Path) -> str:
    """Hash UTF-8 text after canonical LF normalization across Git checkouts."""

    text = Path(path).read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_manifest(source_root: str | Path) -> list[ManifestEntry]:
    """Return a stable hash manifest without copying any source bytes."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source root is not a directory: {root}")
    entries = [
        ManifestEntry(
            relative_path=path.relative_to(root).as_posix(),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(p for p in root.rglob("*") if p.is_file())
    ]
    return sorted(entries, key=lambda item: item.relative_path)


def write_manifest(source_root: str | Path, destination: str | Path) -> Path:
    """Atomically write a deterministic manifest outside the immutable source root."""

    root = Path(source_root).resolve()
    output = Path(destination).resolve()
    if output.is_relative_to(root):
        raise ValueError("manifest destination must be outside the immutable source root")
    if output.exists():
        raise FileExistsError(f"manifest destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(entry) for entry in source_manifest(root)]
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def discover_asset_files(
    source_root: str | Path, tickers: Iterable[str] = PRIMARY_TICKERS
) -> dict[str, Path]:
    """Discover one canonical file per ticker and reject divergent duplicates."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source root is not a directory: {root}")

    discovered: dict[str, Path] = {}
    for ticker in tickers:
        candidates = sorted(path for path in root.rglob(f"{ticker}.csv") if path.is_file())
        if not candidates:
            raise FileNotFoundError(f"no {ticker}.csv under {root}")
        by_hash: dict[str, list[Path]] = {}
        for path in candidates:
            by_hash.setdefault(sha256_file(path), []).append(path)
        if len(by_hash) != 1:
            detail = {
                digest: [p.relative_to(root).as_posix() for p in paths]
                for digest, paths in by_hash.items()
            }
            raise ValueError(f"divergent duplicate files for {ticker}: {detail}")
        discovered[ticker] = min(
            candidates,
            key=lambda path: (len(path.relative_to(root).parts), path.as_posix()),
        )
    return discovered


def load_asset_csv(path: str | Path, ticker: str) -> pd.DataFrame:
    """Load one asset without deleting price dates or filling sentiment."""

    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{ticker}: missing columns {sorted(missing_columns)}")

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    try:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ticker}: invalid Date values") from exc
    if frame["Date"].duplicated().any():
        duplicates = frame.loc[frame["Date"].duplicated(keep=False), "Date"].dt.strftime("%Y-%m-%d")
        raise ValueError(f"{ticker}: duplicate dates {duplicates.tolist()}")
    if not frame["Date"].is_monotonic_increasing:
        raise ValueError(f"{ticker}: dates are not strictly increasing")

    numeric_columns = [*PRICE_COLUMNS, *SENTIMENT_COLUMNS]
    malformed_sentiment: dict[str, int] = {}
    for column in numeric_columns:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        if column in SENTIMENT_COLUMNS:
            malformed_count = int((original.notna() & converted.isna()).sum())
            if malformed_count:
                malformed_sentiment[column] = malformed_count
        frame[column] = converted
    if malformed_sentiment:
        raise ValueError(f"{ticker}: malformed/non-numeric sentiment cells {malformed_sentiment}")
    if frame.loc[:, list(PRICE_COLUMNS)].isna().any().any():
        missing = frame.loc[:, list(PRICE_COLUMNS)].isna().sum()
        raise ValueError(f"{ticker}: missing/non-numeric price cells {missing.to_dict()}")
    if not np.isfinite(frame.loc[:, list(PRICE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError(f"{ticker}: non-finite price")
    if (frame.loc[:, ["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive price")
    if (frame["Volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")
    sentiment = frame.loc[:, list(SENTIMENT_COLUMNS)]
    if np.isinf(sentiment.to_numpy(dtype=float)).any():
        raise ValueError(f"{ticker}: non-finite sentiment value")
    if (sentiment < 0).any().any():
        raise ValueError(f"{ticker}: negative RVT or sentiment component")

    adjustment_factor = frame["Adj Close"] / frame["Close"]
    if not np.isfinite(adjustment_factor).all() or (adjustment_factor <= 0).any():
        raise ValueError(f"{ticker}: invalid close adjustment factor")
    with np.errstate(over="ignore", invalid="ignore"):
        frame["Adjusted Open"] = frame["Open"] * adjustment_factor
    adjusted_open = frame["Adjusted Open"].to_numpy(dtype=float)
    if not np.isfinite(adjusted_open).all() or (adjusted_open <= 0).any():
        raise ValueError(f"{ticker}: invalid Adjusted Open")
    frame.insert(1, "Ticker", ticker)
    frame = frame.set_index("Date")
    frame.index.name = "date"
    return frame


def load_panel(
    source_root: str | Path,
    tickers: Iterable[str] = PRIMARY_TICKERS,
    require_common_calendar: bool = True,
) -> pd.DataFrame:
    """Load a sorted long panel and preserve explicit sentiment missingness."""

    ticker_tuple = tuple(tickers)
    files = discover_asset_files(source_root, ticker_tuple)
    assets = {ticker: load_asset_csv(files[ticker], ticker) for ticker in ticker_tuple}
    if require_common_calendar:
        reference_ticker = ticker_tuple[0]
        reference = assets[reference_ticker].index
        for ticker, frame in assets.items():
            if not frame.index.equals(reference):
                raise ValueError(f"{ticker}: price calendar differs from {reference_ticker}")

    panel = pd.concat(assets.values(), axis=0)
    panel = panel.reset_index().set_index(["date", "Ticker"]).sort_index()
    panel.index.names = ["date", "ticker"]
    if panel.index.duplicated().any():
        raise ValueError("panel contains duplicate (date, ticker) rows")
    return panel


def inventory(
    source_root: str | Path, tickers: Iterable[str] = PRIMARY_TICKERS
) -> list[AssetInventory]:
    root = Path(source_root).resolve()
    files = discover_asset_files(root, tickers)
    rows: list[AssetInventory] = []
    for ticker, path in files.items():
        validated = load_asset_csv(path, ticker)
        dates = validated.index
        bull = validated[["positivePartscr", "certaintyPartscr", "finupPartscr"]].sum(
            axis=1, min_count=3
        )
        bear = validated[["negativePartscr", "uncertaintyPartscr", "findownPartscr"]].sum(
            axis=1, min_count=3
        )
        undefined = (bull + bear).isna() | ((bull + bear) <= 0)
        rows.append(
            AssetInventory(
                ticker=ticker,
                relative_path=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                rows=len(validated),
                start=str(dates.min().date()),
                end=str(dates.max().date()),
                duplicate_dates=int(dates.duplicated().sum()),
                sorted_dates=bool(dates.is_monotonic_increasing),
                missing_price_cells=int(validated[list(PRICE_COLUMNS)].isna().sum().sum()),
                missing_sentiment_cells=int(validated[list(SENTIMENT_COLUMNS)].isna().sum().sum()),
                undefined_ratio_rows=int(undefined.sum()),
            )
        )
    return rows


def asset_prices(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in panel.columns:
        raise KeyError(column)
    return panel[column].unstack("ticker").sort_index()


def aligned_forward_returns(
    panel: pd.DataFrame,
    convention: str = "next_adjusted_open",
) -> ReturnFrame:
    """Map close-t signals to the next executable one-session return.

    A signal formed after close t executes at the next market session and earns the
    one-session return ending on t+2. The adjusted-close robustness uses the same
    two-session alignment, with execution at close t+1.
    """

    if convention == "next_adjusted_open":
        price_column = "Adjusted Open"
    elif convention == "lagged_adjusted_close":
        price_column = "Adj Close"
    else:
        raise ValueError(f"unsupported execution convention: {convention}")
    prices = asset_prices(panel, price_column)
    dates = prices.index
    if len(dates) < 3:
        raise ValueError("at least three market dates are required")
    values = prices.shift(-2).div(prices.shift(-1)).sub(1.0).iloc[:-2].copy()
    return_end = dates[2:]
    values.index = return_end
    values.index.name = "return_end_date"
    signal_dates = pd.Series(dates[:-2], index=return_end, name="signal_date")
    execution_dates = pd.Series(dates[1:-1], index=return_end, name="execution_date")
    if values.isna().any().any():
        raise ValueError("primary price returns contain missing values")
    return ReturnFrame(values, signal_dates, execution_dates)


def align_signal_to_return_end(signal: pd.DataFrame, returns: ReturnFrame) -> pd.DataFrame:
    """Align close-t signals to the return ending at t+2 without using future signal data."""

    aligned = signal.reindex(pd.DatetimeIndex(returns.signal_dates.to_numpy())).copy()
    aligned.index = returns.returns.index
    aligned.index.name = "return_end_date"
    return aligned
