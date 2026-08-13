from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from news_sentiment_trading.cli import (
    _restricted_artifact_path,
    _restricted_report_root,
    build_parser,
)
from news_sentiment_trading.synthetic import write_synthetic_csvs


def test_synthetic_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "news_sentiment_trading.cli",
            "synthetic-demo",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["synthetic_only"] is True
    assert payload["outer_observations"] == 348


def test_legacy_cli_smoke_is_immutable(tmp_path: Path) -> None:
    source = write_synthetic_csvs(tmp_path / "source")
    checkout = tmp_path / "checkout"
    output = checkout / "reports" / "artifacts" / "legacy.json"
    output.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    command = [
        sys.executable,
        "-m",
        "news_sentiment_trading.cli",
        "legacy-reproduce",
        "--source-root",
        str(source),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, cwd=checkout, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert {(row["ticker"], row["regime"]) for row in payload} == {
        ("FB", "long_only"),
        ("FB", "long_short"),
        ("JPM", "long_only"),
        ("JPM", "long_short"),
    }
    original = output.read_bytes()
    repeated = subprocess.run(command, cwd=checkout, text=True, capture_output=True, check=False)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert output.read_bytes() == original


def test_cli_exposes_complete_empirical_and_report_interface() -> None:
    parser = build_parser()
    assert parser.parse_args(["run-primary", "--output-dir", "primary"]).command == "run-primary"
    assert (
        parser.parse_args(["run-robustness", "--output-dir", "robustness"]).command
        == "run-robustness"
    )
    report = parser.parse_args(
        [
            "generate-report",
            "--artifact-dir",
            "primary",
            "--provenance",
            "primary-provenance.json",
            "--robustness-artifact-dir",
            "robustness",
            "--robustness-provenance",
            "robustness-provenance.json",
            "--legacy-artifact",
            "legacy.json",
        ]
    )
    assert report.command == "generate-report"
    assert report.report_root == "reports/artifacts/generated-report-local"
    assert report.robustness_artifact_dir == "robustness"
    assert report.legacy_artifact == "legacy.json"


def test_empirical_report_root_is_new_and_ignored(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "reports" / "artifacts").mkdir(parents=True)
    expected = root / "reports" / "artifacts" / "generated-local"
    assert _restricted_report_root(root, "reports/artifacts/generated-local") == expected
    with pytest.raises(RuntimeError, match="must remain under reports/artifacts"):
        _restricted_report_root(root, ".")
    expected.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _restricted_report_root(root, expected)


def test_every_empirical_output_is_confined_to_a_new_artifact_path(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    allowed = root / "reports" / "artifacts" / "inventory.json"
    assert _restricted_artifact_path(root, allowed, "test output") == allowed
    with pytest.raises(RuntimeError, match="must remain under reports/artifacts"):
        _restricted_artifact_path(root, tmp_path / "outside.json", "test output")
    allowed.parent.mkdir(parents=True)
    allowed.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _restricted_artifact_path(root, allowed, "test output")


def test_cli_exposes_repository_scan_without_external_configuration() -> None:
    args = build_parser().parse_args(["repository-scan"])
    assert args.command == "repository-scan"


def test_repository_tracks_no_executable_notebooks() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""


def test_repository_tracks_no_empirical_results_or_attestations() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "reports/tables",
            "reports/figures",
            "attestations",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""
