"""Command-line research and reproducibility interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from news_sentiment_trading import __version__
from news_sentiment_trading.config import load_config
from news_sentiment_trading.data import (
    PRIMARY_TICKERS,
    discover_asset_files,
    inventory,
    load_panel,
    sha256_file,
    sha256_normalized_text,
    write_manifest,
)
from news_sentiment_trading.legacy import reproduce_fb_jpm
from news_sentiment_trading.metrics import compounded_total_return
from news_sentiment_trading.reporting import (
    ArtifactProvenance,
    generate_markdown_reports,
    load_artifact_provenance,
    write_artifact_provenance,
    write_json,
    write_primary_artifacts,
    write_robustness_artifacts,
)
from news_sentiment_trading.repository_scan import scan_repository
from news_sentiment_trading.robustness import run_robustness
from news_sentiment_trading.source_gate import (
    load_source_gate,
    planned_fold_manifest,
    validate_fold_manifest,
    validate_source_hashes,
    validate_source_identity,
)
from news_sentiment_trading.synthetic import synthetic_panel, write_synthetic_csvs
from news_sentiment_trading.walk_forward import run_walk_forward


def repository_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("repository-bound command must run inside a Git checkout")
    return Path(completed.stdout.strip()).resolve()


def _run_git(arguments: Sequence[str], root: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def preregistration_gate(root: Path) -> tuple[str, str]:
    """Require a clean committed preregistration before a real outer run."""

    if _run_git(["rev-parse", "--is-shallow-repository"], root) == "true":
        raise RuntimeError("primary outer run requires complete, non-shallow Git history")
    status = _run_git(["status", "--porcelain"], root)
    if status:
        raise RuntimeError("primary outer run requires a clean Git working tree")
    immutable_paths = {
        "docs/PRE_REGISTRATION.md": "preregistration",
        "configs/primary.toml": "primary configuration",
    }
    commits: dict[str, str] = {}
    for path, label in immutable_paths.items():
        _run_git(["ls-files", "--error-unmatch", path], root)
        path_commits = _run_git(["log", "--format=%H", "--", path], root).splitlines()
        if len(path_commits) != 1:
            raise RuntimeError(f"{label} must have exactly one immutable commit")
        commits[path] = path_commits[0]
        _run_git(["diff", "--exit-code", path_commits[0], "HEAD", "--", path], root)
    preregistration_commit = commits["docs/PRE_REGISTRATION.md"]
    if not preregistration_commit:
        raise RuntimeError("no commit contains docs/PRE_REGISTRATION.md")
    head = _run_git(["rev-parse", "HEAD"], root)
    return head, preregistration_commit


def _source_root(value: str | None) -> Path:
    supplied = value or os.environ.get("NEWS_SENTIMENT_SOURCE")
    if not supplied:
        raise RuntimeError("provide --source-root or set NEWS_SENTIMENT_SOURCE")
    path = Path(supplied).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _source_gate_path(value: str | None, root: Path) -> Path:
    supplied = value or os.environ.get("NEWS_SENTIMENT_SOURCE_GATE")
    if not supplied:
        raise RuntimeError(
            "provide --source-gate or set NEWS_SENTIMENT_SOURCE_GATE; source identities are not "
            "distributed with the repository"
        )
    path = Path(supplied).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return path
    raise RuntimeError("the empirical source gate must remain outside the repository")


def _restricted_artifact_path(root: Path, requested: str | Path, label: str) -> Path:
    artifact_root = (root / "reports" / "artifacts").resolve()
    output = Path(requested)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output == artifact_root or artifact_root not in output.parents:
        raise RuntimeError(f"{label} must remain under reports/artifacts/")
    if output.exists():
        raise FileExistsError(f"{label} already exists: {output}")
    return output


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def command_discover(args: argparse.Namespace) -> int:
    source_root = _source_root(args.source_root)
    files = discover_asset_files(source_root, PRIMARY_TICKERS)
    _json_print(
        {
            ticker: {
                "relative_path": path.relative_to(source_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for ticker, path in files.items()
        }
    )
    return 0


def command_inventory(args: argparse.Namespace) -> int:
    source_root = _source_root(args.source_root)
    rows = [dataclasses.asdict(item) for item in inventory(source_root)]
    if args.output:
        destination = _restricted_artifact_path(repository_root(), args.output, "inventory output")
        write_json(destination, rows)
    _json_print(rows)
    return 0


def command_manifest(args: argparse.Namespace) -> int:
    root = repository_root()
    source_root = _source_root(args.source_root)
    destination = _restricted_artifact_path(root, args.output, "source manifest")
    write_manifest(source_root, destination)
    print(destination.resolve())
    return 0


def command_legacy(args: argparse.Namespace) -> int:
    root = repository_root()
    source_root = _source_root(args.source_root)
    panel = load_panel(source_root)
    rows = [dataclasses.asdict(result) for result in reproduce_fb_jpm(panel)]
    destination = _restricted_artifact_path(root, args.output, "legacy artifact")
    write_json(destination, rows)
    _json_print(rows)
    return 0


def command_synthetic(args: argparse.Namespace) -> int:
    packaged_config = files("news_sentiment_trading").joinpath("resources/primary.toml")
    try:
        with as_file(packaged_config) as config_path:
            config = load_config(config_path) if config_path.is_file() else None
    except FileNotFoundError:
        config = None
    if config is None:
        checkout_config = repository_root() / "configs" / "primary.toml"
        if not checkout_config.is_file():
            raise FileNotFoundError(checkout_config)
        config = load_config(checkout_config)
    panel = synthetic_panel(seed=config.research.seed)
    result = run_walk_forward(panel, config)
    summary = {
        "synthetic_only": True,
        "outer_observations": len(result.portfolio.net_return),
        "folds": result.fold_manifest(),
        "net_total_return": compounded_total_return(result.portfolio.net_return),
    }
    if args.csv_dir:
        write_synthetic_csvs(args.csv_dir, sessions=80)
    if args.output:
        write_json(args.output, summary)
    _json_print(summary)
    return 0


def command_primary(args: argparse.Namespace) -> int:
    root = repository_root()
    head, preregistration_commit = preregistration_gate(root)
    locked_config = (root / "configs" / "primary.toml").resolve()
    if args.config and Path(args.config).resolve() != locked_config:
        raise RuntimeError("primary run accepts only the committed configs/primary.toml")
    config = load_config(locked_config)
    source_root = _source_root(args.source_root)
    source_gate_path = _source_gate_path(args.source_gate, root)
    source_gate = load_source_gate(source_gate_path)
    if source_gate.universe != config.universe.primary:
        raise RuntimeError("source gate universe differs from the locked primary configuration")
    files = discover_asset_files(source_root, config.universe.primary)
    input_hashes = validate_source_hashes(source_gate, files)
    panel = load_panel(source_root, config.universe.primary)
    dates = panel.index.get_level_values("date").unique()
    if validate_source_identity(source_gate, files, dates) != input_hashes:
        raise RuntimeError("source bytes changed while the primary panel was loading")
    validate_fold_manifest(
        source_gate,
        planned_fold_manifest(
            dates,
            config.walk_forward.outer_initial_sessions,
            config.walk_forward.outer_test_sessions,
        ),
    )
    lock_path = root / "uv.lock"
    if not lock_path.exists():
        raise RuntimeError("uv.lock is required")
    result = run_walk_forward(panel, config)
    if len(result.return_frame.returns) != 600:
        raise RuntimeError("primary source must yield exactly 600 executable returns")
    validate_fold_manifest(source_gate, result.fold_manifest())
    artifact_dir = _restricted_artifact_path(
        root,
        args.output_dir or root / "reports" / "artifacts" / "primary",
        "primary artifact directory",
    )
    manifest = write_primary_artifacts(
        result=result,
        panel=panel,
        config=config,
        output_dir=artifact_dir,
        input_hashes=input_hashes,
        git_commit=head,
        lock_hash=sha256_normalized_text(lock_path),
        preregistration_commit=preregistration_commit,
        source_gate_hash=sha256_normalized_text(source_gate_path),
    )
    provenance_path = _restricted_artifact_path(
        root,
        args.provenance_output or artifact_dir.parent / f"{artifact_dir.name}.provenance.json",
        "primary provenance",
    )
    write_artifact_provenance(artifact_dir, provenance_path)
    _json_print(
        {
            "artifact_dir": str(artifact_dir.resolve()),
            "provenance": str(provenance_path.resolve()),
            "files": manifest,
        }
    )
    return 0


def command_robustness(args: argparse.Namespace) -> int:
    root = repository_root()
    head, preregistration_commit = preregistration_gate(root)
    config = load_config(root / "configs" / "primary.toml")
    source_root = _source_root(args.source_root)
    source_gate_path = _source_gate_path(args.source_gate, root)
    source_gate = load_source_gate(source_gate_path)
    if source_gate.universe != config.universe.primary:
        raise RuntimeError("source gate universe differs from the locked primary configuration")
    files = discover_asset_files(source_root, config.universe.primary)
    input_hashes = validate_source_hashes(source_gate, files)
    panel = load_panel(source_root, config.universe.primary)
    dates = panel.index.get_level_values("date").unique()
    if validate_source_identity(source_gate, files, dates) != input_hashes:
        raise RuntimeError("source bytes changed while the robustness panel was loading")
    validate_fold_manifest(
        source_gate,
        planned_fold_manifest(
            dates,
            config.walk_forward.outer_initial_sessions,
            config.walk_forward.outer_test_sessions,
        ),
    )
    lock_path = root / "uv.lock"
    primary = run_walk_forward(panel, config)
    validate_fold_manifest(source_gate, primary.fold_manifest())
    result = run_robustness(panel, config, primary_result=primary)
    artifact_dir = _restricted_artifact_path(
        root,
        args.output_dir or root / "reports" / "artifacts" / "robustness",
        "robustness artifact directory",
    )
    manifest = write_robustness_artifacts(
        result,
        artifact_dir,
        git_commit=head,
        preregistration_commit=preregistration_commit,
        configuration_hash=config.digest(),
        lock_hash=sha256_normalized_text(lock_path),
        source_gate_hash=sha256_normalized_text(source_gate_path),
        input_hashes=input_hashes,
    )
    provenance_path = _restricted_artifact_path(
        root,
        args.provenance_output or artifact_dir.parent / f"{artifact_dir.name}.provenance.json",
        "robustness provenance",
    )
    write_artifact_provenance(artifact_dir, provenance_path)
    _json_print(
        {
            "artifact_dir": str(artifact_dir.resolve()),
            "provenance": str(provenance_path.resolve()),
            "files": manifest,
        }
    )
    return 0


def _validate_provenance_against_repository(
    root: Path, provenance: ArtifactProvenance, source_gate_path: Path
) -> None:
    head, preregistration_commit = preregistration_gate(root)
    config = load_config(root / "configs" / "primary.toml")
    source_gate = load_source_gate(source_gate_path)
    expected = {
        "git_commit": head,
        "preregistration_commit": preregistration_commit,
        "configuration_hash": config.digest(),
        "environment_lock_hash": sha256_normalized_text(root / "uv.lock"),
        "source_gate_hash": sha256_normalized_text(source_gate_path),
        "input_hashes": dict(source_gate.source_sha256),
    }
    mismatched = [field for field, value in expected.items() if getattr(provenance, field) != value]
    if mismatched:
        raise RuntimeError(f"artifact provenance does not match the repository: {mismatched}")


def _restricted_report_root(root: Path, requested: str | Path) -> Path:
    return _restricted_artifact_path(root, requested, "generated report root")


def command_report(args: argparse.Namespace) -> int:
    root = repository_root()
    source_gate_path = _source_gate_path(args.source_gate, root)
    provenance = load_artifact_provenance(args.provenance)
    _validate_provenance_against_repository(root, provenance, source_gate_path)
    robustness_provenance = (
        load_artifact_provenance(args.robustness_provenance) if args.robustness_provenance else None
    )
    if robustness_provenance is not None:
        _validate_provenance_against_repository(root, robustness_provenance, source_gate_path)
    report_root = _restricted_report_root(root, args.report_root)
    generate_markdown_reports(
        args.artifact_dir,
        report_root,
        provenance,
        robustness_artifact_dir=args.robustness_artifact_dir,
        robustness_provenance=robustness_provenance,
        legacy_artifact=args.legacy_artifact,
    )
    print("reports regenerated")
    return 0


def command_repository_scan(_args: argparse.Namespace) -> int:
    result = scan_repository(repository_root())
    _json_print(result.to_dict())
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="news-sentiment-trading")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, function in (("discover", command_discover), ("inventory", command_inventory)):
        child = subparsers.add_parser(name)
        child.add_argument("--source-root")
        if name == "inventory":
            child.add_argument("--output")
        child.set_defaults(function=function)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--source-root")
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(function=command_manifest)

    legacy = subparsers.add_parser("legacy-reproduce")
    legacy.add_argument("--source-root")
    legacy.add_argument("--output", default="reports/artifacts/legacy_summary.json")
    legacy.set_defaults(function=command_legacy)

    synthetic = subparsers.add_parser("synthetic-demo")
    synthetic.add_argument("--output")
    synthetic.add_argument("--csv-dir")
    synthetic.set_defaults(function=command_synthetic)

    primary = subparsers.add_parser("run-primary")
    primary.add_argument("--source-root")
    primary.add_argument("--source-gate")
    primary.add_argument("--config")
    primary.add_argument("--output-dir")
    primary.add_argument("--provenance-output")
    primary.set_defaults(function=command_primary)

    robustness = subparsers.add_parser("run-robustness")
    robustness.add_argument("--source-root")
    robustness.add_argument("--source-gate")
    robustness.add_argument("--output-dir")
    robustness.add_argument("--provenance-output")
    robustness.set_defaults(function=command_robustness)

    report = subparsers.add_parser("generate-report")
    report.add_argument("--source-gate")
    report.add_argument("--artifact-dir", required=True)
    report.add_argument("--provenance", required=True)
    report.add_argument(
        "--report-root",
        default="reports/artifacts/generated-report-local",
        help="local output root; keep empirical reports under the ignored artifact tree",
    )
    report.add_argument("--robustness-artifact-dir")
    report.add_argument("--robustness-provenance")
    report.add_argument("--legacy-artifact")
    report.set_defaults(function=command_report)

    scan = subparsers.add_parser("repository-scan")
    scan.set_defaults(function=command_repository_scan)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
