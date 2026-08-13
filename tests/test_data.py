from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news_sentiment_trading.data import (
    PRIMARY_TICKERS,
    align_signal_to_return_end,
    aligned_forward_returns,
    discover_asset_files,
    inventory,
    load_asset_csv,
    load_panel,
    sha256_file,
    sha256_normalized_text,
    source_manifest,
    write_manifest,
)
from news_sentiment_trading.synthetic import write_synthetic_csvs


def test_adjusted_open_and_next_open_timing(panel: pd.DataFrame) -> None:
    returns = aligned_forward_returns(panel)
    prices = panel["Adjusted Open"].unstack("ticker")
    first_end = returns.returns.index[0]
    first_ticker = PRIMARY_TICKERS[0]
    expected = prices[first_ticker].iloc[2] / prices[first_ticker].iloc[1] - 1.0
    assert returns.returns.loc[first_end, first_ticker] == pytest.approx(expected)
    assert returns.signal_dates.loc[first_end] == prices.index[0]
    assert returns.execution_dates.loc[first_end] == prices.index[1]


def test_lagged_close_has_same_two_session_timing() -> None:
    dates = pd.bdate_range("2026-01-05", periods=4)
    index = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    panel = pd.DataFrame(
        {
            "Adjusted Open": [100.0, 200.0, 100.0, 100.0],
            "Adj Close": [100.0, 110.0, 121.0, 133.1],
        },
        index=index,
    )
    result = aligned_forward_returns(panel, "lagged_adjusted_close")
    assert result.returns.iloc[0, 0] == pytest.approx(121.0 / 110.0 - 1.0)
    assert result.returns.index[0] == dates[2]
    assert result.signal_dates.iloc[0] == dates[0]
    assert result.execution_dates.iloc[0] == dates[1]


def test_forward_returns_reject_unknown_execution_convention(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unsupported execution convention"):
        aligned_forward_returns(panel, "same_close")


def test_signal_alignment_waits_until_t_plus_two(panel: pd.DataFrame) -> None:
    returns = aligned_forward_returns(panel)
    dates = panel.index.get_level_values("date").unique()
    signal = pd.DataFrame(0, index=dates, columns=PRIMARY_TICKERS)
    signal.loc[dates[5], "AAPL"] = 1
    aligned = align_signal_to_return_end(signal, returns)
    assert aligned.loc[dates[7], "AAPL"] == 1
    assert aligned.loc[: dates[6], "AAPL"].sum() == 0


def test_schema_rejects_duplicate_dates(tmp_path: Path) -> None:
    csv_dir = write_synthetic_csvs(tmp_path / "data")
    path = csv_dir / "AAPL.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "Date"] = frame.loc[0, "Date"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate dates"):
        load_asset_csv(path, "AAPL")


def test_schema_rejects_nonpositive_price(tmp_path: Path) -> None:
    csv_dir = write_synthetic_csvs(tmp_path / "data")
    path = csv_dir / "AAPL.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "Open"] = 0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-positive price"):
        load_asset_csv(path, "AAPL")


def test_inventory_uses_canonical_schema_and_value_validation(tmp_path: Path) -> None:
    csv_dir = write_synthetic_csvs(tmp_path / "data")
    path = csv_dir / "AAPL.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "Open"] = -1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-positive price"):
        inventory(csv_dir)

    frame.loc[0, "Open"] = 100
    frame = frame.drop(columns=["Close"])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns.*Close"):
        inventory(csv_dir)


def test_schema_rejects_malformed_sentiment_without_treating_it_as_missing(
    tmp_path: Path,
) -> None:
    csv_dir = write_synthetic_csvs(tmp_path / "data")
    path = csv_dir / "AAPL.csv"
    frame = pd.read_csv(path)
    frame["positivePartscr"] = frame["positivePartscr"].astype(object)
    frame.loc[4, "positivePartscr"] = "not-a-number"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="malformed/non-numeric sentiment"):
        load_asset_csv(path, "AAPL")


def test_schema_rejects_adjusted_open_overflow(tmp_path: Path) -> None:
    csv_dir = write_synthetic_csvs(tmp_path / "data")
    path = csv_dir / "AAPL.csv"
    frame = pd.read_csv(path)
    frame.loc[4, ["Open", "High"]] = 1e308
    frame.loc[4, "Close"] = 1.0
    frame.loc[4, "Adj Close"] = 2.0
    frame.to_csv(path, index=False)

    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="invalid Adjusted Open"):
            load_asset_csv(path, "AAPL")


def test_missing_sentiment_does_not_delete_price_calendar(tmp_path: Path) -> None:
    csv_dir = write_synthetic_csvs(tmp_path / "data")
    loaded = load_panel(csv_dir)
    assert len(loaded.xs("AAPL", level="ticker")) == 80
    assert loaded.xs("AAPL", level="ticker")["RVT"].isna().sum() == 1


def test_divergent_duplicates_fail_closed(tmp_path: Path) -> None:
    csv_dir = write_synthetic_csvs(tmp_path / "root" / "one")
    duplicate = tmp_path / "root" / "two"
    duplicate.mkdir()
    original = pd.read_csv(csv_dir / "AAPL.csv")
    original.loc[0, "Close"] *= 2
    original.to_csv(duplicate / "AAPL.csv", index=False)
    with pytest.raises(ValueError, match="divergent duplicate"):
        discover_asset_files(tmp_path / "root", ["AAPL"])


def test_manifest_is_deterministic_and_byte_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("alpha", encoding="utf-8")
    first = source_manifest(tmp_path)
    second = source_manifest(tmp_path)
    assert first == second
    first_hash = sha256_file(path)
    path.write_text("alphb", encoding="utf-8")
    assert sha256_file(path) != first_hash


def test_manifest_write_is_outside_source_atomic_and_no_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("alpha", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the immutable source root"):
        write_manifest(source, source / "manifest.json")
    assert not (source / "manifest.json").exists()

    destination = tmp_path / "evidence" / "manifest.json"
    assert write_manifest(source, destination) == destination.resolve()
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        write_manifest(source, destination)
    assert destination.read_bytes() == original
    assert not list(destination.parent.glob(".*.tmp"))


def test_text_hash_is_invariant_to_checkout_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.lock"
    crlf = tmp_path / "crlf.lock"
    lf.write_bytes(b"version = 1\npackage = 'x'\n")
    crlf.write_bytes(b"version = 1\r\npackage = 'x'\r\n")
    assert sha256_file(lf) != sha256_file(crlf)
    assert sha256_normalized_text(lf) == sha256_normalized_text(crlf)
