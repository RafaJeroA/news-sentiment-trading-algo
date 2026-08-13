from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from news_sentiment_trading.cli import _source_gate_path, preregistration_gate
from news_sentiment_trading.data import sha256_file
from news_sentiment_trading.source_gate import (
    CalendarGate,
    FoldGate,
    SourceGate,
    calendar_sha256,
    load_source_gate,
    planned_fold_manifest,
    validate_fold_manifest,
    validate_source_identity,
)


def _synthetic_gate(files: dict[str, Path], dates: pd.DatetimeIndex) -> SourceGate:
    return SourceGate(
        schema_version=1,
        universe=tuple(files),
        source_sha256={ticker: sha256_file(path) for ticker, path in files.items()},
        calendar=CalendarGate(
            sessions=len(dates),
            start=str(dates[0].date()),
            end=str(dates[-1].date()),
            sha256=calendar_sha256(dates),
        ),
        outer_folds=(FoldGate(1, "2026-01-07", "2026-01-09", 3),),
    )


def test_source_identity_accepts_exact_synthetic_bytes_and_calendar(tmp_path: Path) -> None:
    files = {"AAA": tmp_path / "AAA.csv", "BBB": tmp_path / "BBB.csv"}
    files["AAA"].write_text("date,value\n2026-01-05,1\n", encoding="utf-8")
    files["BBB"].write_text("date,value\n2026-01-05,2\n", encoding="utf-8")
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    gate = _synthetic_gate(files, dates)

    assert validate_source_identity(gate, files, dates) == dict(gate.source_sha256)


def test_source_identity_rejects_one_byte_value_mutation(tmp_path: Path) -> None:
    files = {"AAA": tmp_path / "AAA.csv"}
    files["AAA"].write_text("date,value\n2026-01-05,1\n", encoding="utf-8")
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    gate = _synthetic_gate(files, dates)
    files["AAA"].write_text("date,value\n2026-01-05,9\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch for: AAA"):
        validate_source_identity(gate, files, dates)


def test_source_identity_rejects_synchronized_interior_date_substitution(tmp_path: Path) -> None:
    files = {"AAA": tmp_path / "AAA.csv"}
    files["AAA"].write_text("synthetic", encoding="utf-8")
    dates = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-08", "2026-01-09", "2026-01-12"])
    gate = _synthetic_gate(files, dates)
    substituted = dates.to_list()
    substituted[2] = pd.Timestamp("2026-01-07")

    with pytest.raises(RuntimeError, match="complete ordered source calendar"):
        validate_source_identity(gate, files, substituted)


def test_fold_gate_rejects_wrong_exact_boundary(tmp_path: Path) -> None:
    file_path = tmp_path / "AAA.csv"
    file_path.write_text("synthetic", encoding="utf-8")
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    gate = _synthetic_gate({"AAA": file_path}, dates)
    wrong = [
        {
            "fold_id": 1,
            "test_start": "2026-01-08",
            "test_end": "2026-01-09",
            "test_observations": 3,
        }
    ]

    with pytest.raises(RuntimeError, match="fold 1"):
        validate_fold_manifest(gate, wrong)


def test_planned_fold_manifest_uses_calendar_only() -> None:
    dates = pd.date_range("2026-01-05", periods=10, freq="B")

    folds = planned_fold_manifest(dates, initial=2, block_size=3)

    assert folds == [
        {
            "fold_id": 1,
            "test_start": "2026-01-09",
            "test_end": "2026-01-13",
            "test_observations": 3,
        },
        {
            "fold_id": 2,
            "test_start": "2026-01-14",
            "test_end": "2026-01-16",
            "test_observations": 3,
        },
    ]


def test_source_gate_loader_rejects_unknown_keys(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "universe": ["AAA"],
        "source_sha256": {"AAA": "0" * 64},
        "calendar": {
            "sessions": 3,
            "start": "2026-01-05",
            "end": "2026-01-07",
            "sha256": "1" * 64,
        },
        "outer_folds": [
            {
                "fold_id": 1,
                "test_start": "2026-01-05",
                "test_end": "2026-01-07",
                "test_observations": 3,
            }
        ],
        "unexpected": True,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        load_source_gate(path)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _gate_repository(tmp_path: Path) -> Path:
    root = tmp_path / "gate-repository"
    (root / "docs").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "docs" / "PRE_REGISTRATION.md").write_text("locked\n", encoding="utf-8")
    (root / "configs" / "primary.toml").write_text("locked = true\n", encoding="utf-8")
    (root / "configs" / "source_gate.json").write_text("{}\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.name", "Test Maintainer")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "lock gate")
    return root


def test_preregistration_gate_rejects_dirty_tree(tmp_path: Path) -> None:
    root = _gate_repository(tmp_path)
    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean Git working tree"):
        preregistration_gate(root)


def test_preregistration_gate_rejects_shallow_history(tmp_path: Path) -> None:
    source = _gate_repository(tmp_path / "source")
    shallow = tmp_path / "shallow"
    _git(tmp_path, "clone", "--depth", "1", source.resolve().as_uri(), str(shallow))

    with pytest.raises(RuntimeError, match="complete, non-shallow Git history"):
        preregistration_gate(shallow)


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("docs/PRE_REGISTRATION.md", "preregistration"),
        ("configs/primary.toml", "primary configuration"),
    ],
)
def test_preregistration_gate_rejects_changed_locked_file(
    tmp_path: Path, relative_path: str, message: str
) -> None:
    root = _gate_repository(tmp_path)
    path = root / relative_path
    path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    _git(root, "add", relative_path)
    _git(root, "commit", "-m", "impermissible gate change")

    with pytest.raises(RuntimeError, match=message):
        preregistration_gate(root)


def test_empirical_run_requires_external_source_gate(tmp_path: Path) -> None:
    root = tmp_path / "research-repository"
    root.mkdir()
    internal = root / "source-gate.json"
    internal.write_text("{}\n", encoding="utf-8")
    external = tmp_path / "authorised-source-gate.json"
    external.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="outside the repository"):
        _source_gate_path(str(internal), root)

    assert _source_gate_path(str(external), root) == external.resolve()
