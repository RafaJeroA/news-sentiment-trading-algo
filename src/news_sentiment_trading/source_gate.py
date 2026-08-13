"""Fail-closed identity checks for the preregistered empirical source."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from news_sentiment_trading.data import sha256_file

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CalendarGate:
    """The complete ordered primary market calendar identity."""

    sessions: int
    start: str
    end: str
    sha256: str


@dataclass(frozen=True)
class FoldGate:
    """An exact preregistered outer-test boundary."""

    fold_id: int
    test_start: str
    test_end: str
    test_observations: int


@dataclass(frozen=True)
class SourceGate:
    """Disclosure-restricted source bytes, calendar, and fold boundaries."""

    schema_version: int
    universe: tuple[str, ...]
    source_sha256: Mapping[str, str]
    calendar: CalendarGate
    outer_folds: tuple[FoldGate, ...]


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(raw)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def load_source_gate(path: str | Path) -> SourceGate:
    """Load and strictly validate an external empirical identity gate."""

    raw_value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("source gate must be a JSON object")
    raw = cast(dict[str, Any], raw_value)
    _exact_keys(
        raw,
        {"schema_version", "universe", "source_sha256", "calendar", "outer_folds"},
        "source gate",
    )
    if raw["schema_version"] != 1:
        raise ValueError("unsupported source-gate schema version")

    universe_value = raw["universe"]
    if not isinstance(universe_value, list) or not universe_value:
        raise ValueError("source-gate universe must be a non-empty list")
    if not all(isinstance(ticker, str) and ticker for ticker in universe_value):
        raise ValueError("source-gate tickers must be non-empty strings")
    universe = tuple(cast(list[str], universe_value))
    if len(universe) != len(set(universe)):
        raise ValueError("source-gate universe contains duplicate tickers")

    hashes_value = raw["source_sha256"]
    if not isinstance(hashes_value, dict):
        raise ValueError("source_sha256 must be an object")
    hashes_raw = cast(dict[str, Any], hashes_value)
    if set(hashes_raw) != set(universe):
        raise ValueError("source_sha256 tickers must exactly match the universe")
    hashes = {ticker: _sha256(hashes_raw[ticker], f"source_sha256.{ticker}") for ticker in universe}

    calendar_value = raw["calendar"]
    if not isinstance(calendar_value, dict):
        raise ValueError("calendar must be an object")
    calendar_raw = cast(dict[str, Any], calendar_value)
    _exact_keys(calendar_raw, {"sessions", "start", "end", "sha256"}, "calendar")
    sessions = calendar_raw["sessions"]
    start = calendar_raw["start"]
    end = calendar_raw["end"]
    if not isinstance(sessions, int) or sessions < 3:
        raise ValueError("calendar.sessions must be an integer of at least three")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("calendar start/end must be ISO date strings")
    pd.Timestamp(start)
    pd.Timestamp(end)
    calendar = CalendarGate(
        sessions, start, end, _sha256(calendar_raw["sha256"], "calendar.sha256")
    )

    folds_value = raw["outer_folds"]
    if not isinstance(folds_value, list) or not folds_value:
        raise ValueError("outer_folds must be a non-empty list")
    folds: list[FoldGate] = []
    for position, item_value in enumerate(folds_value, start=1):
        if not isinstance(item_value, dict):
            raise ValueError("each outer fold must be an object")
        item = cast(dict[str, Any], item_value)
        _exact_keys(item, {"fold_id", "test_start", "test_end", "test_observations"}, "fold")
        fold_id = item["fold_id"]
        test_start = item["test_start"]
        test_end = item["test_end"]
        observations = item["test_observations"]
        if fold_id != position:
            raise ValueError("outer fold IDs must be consecutive starting at one")
        if not isinstance(test_start, str) or not isinstance(test_end, str):
            raise ValueError("outer fold boundaries must be ISO date strings")
        if not isinstance(observations, int) or observations < 1:
            raise ValueError("outer fold observations must be positive integers")
        pd.Timestamp(test_start)
        pd.Timestamp(test_end)
        folds.append(FoldGate(fold_id, test_start, test_end, observations))

    return SourceGate(1, universe, hashes, calendar, tuple(folds))


def calendar_sha256(dates: Iterable[pd.Timestamp | datetime | date | str]) -> str:
    """Hash the complete ordered calendar as newline-delimited ISO dates."""

    payload = "\n".join(pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_source_identity(
    gate: SourceGate,
    files: Mapping[str, Path],
    dates: Iterable[pd.Timestamp | datetime | date | str],
) -> dict[str, str]:
    """Reject any empirical source bytes or ordered calendar not locked by the gate."""

    actual_hashes = validate_source_hashes(gate, files)

    calendar = pd.DatetimeIndex(list(dates))
    if len(calendar) != gate.calendar.sessions:
        raise RuntimeError("source calendar session count differs from the locked gate")
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise RuntimeError("source calendar must be unique and strictly increasing")
    if (
        str(calendar[0].date()) != gate.calendar.start
        or str(calendar[-1].date()) != gate.calendar.end
    ):
        raise RuntimeError("source calendar bounds differ from the locked gate")
    if calendar_sha256(calendar) != gate.calendar.sha256:
        raise RuntimeError("complete ordered source calendar differs from the locked gate")
    return actual_hashes


def validate_source_hashes(gate: SourceGate, files: Mapping[str, Path]) -> dict[str, str]:
    """Reject source-file order or bytes that differ from the committed identity."""

    if tuple(files) != gate.universe:
        raise RuntimeError("source files do not exactly match the locked ticker order")
    actual_hashes = {ticker: sha256_file(files[ticker]) for ticker in gate.universe}
    mismatched = [
        ticker for ticker in gate.universe if actual_hashes[ticker] != gate.source_sha256[ticker]
    ]
    if mismatched:
        raise RuntimeError(f"source SHA-256 mismatch for: {', '.join(mismatched)}")
    return actual_hashes


def validate_fold_manifest(gate: SourceGate, folds: Sequence[Mapping[str, object]]) -> None:
    """Require every outer block to match the locked return-end boundaries exactly."""

    if len(folds) != len(gate.outer_folds):
        raise RuntimeError("outer fold count differs from the locked gate")
    for expected, actual in zip(gate.outer_folds, folds, strict=True):
        fold_id = actual["fold_id"]
        test_start = actual["test_start"]
        test_end = actual["test_end"]
        observations = actual["test_observations"]
        if not isinstance(fold_id, int) or not isinstance(observations, int):
            raise RuntimeError("outer fold IDs and observation counts must be integers")
        if not isinstance(test_start, str) or not isinstance(test_end, str):
            raise RuntimeError("outer fold boundaries must be ISO date strings")
        observed = (
            fold_id,
            test_start,
            test_end,
            observations,
        )
        locked = (
            expected.fold_id,
            expected.test_start,
            expected.test_end,
            expected.test_observations,
        )
        if observed != locked:
            raise RuntimeError(f"outer fold {expected.fold_id} differs from the locked gate")


def planned_fold_manifest(
    dates: Iterable[pd.Timestamp | datetime | date | str], initial: int, block_size: int
) -> list[dict[str, object]]:
    """Derive outer return-end blocks from the calendar without reading empirical returns."""

    calendar = pd.DatetimeIndex(list(dates))
    return_end_dates = calendar[2:]
    if initial < 1 or block_size < 1 or initial >= len(return_end_dates):
        raise RuntimeError("invalid outer-fold plan for the locked calendar")
    test_dates = return_end_dates[initial:]
    blocks = [
        test_dates[start : start + block_size] for start in range(0, len(test_dates), block_size)
    ]
    return [
        {
            "fold_id": fold_id,
            "test_start": str(block[0].date()),
            "test_end": str(block[-1].date()),
            "test_observations": len(block),
        }
        for fold_id, block in enumerate(blocks, start=1)
    ]
