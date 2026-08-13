"""Public-safe scan for tracked files and reachable Git history."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_TRACKED_FILE_BYTES = 1_000_000

_REVIEWED_PATHS = frozenset(
    {
        ".gitattributes",
        ".github/workflows/ci.yml",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "configs/primary.toml",
        "configs/source_gate.example.json",
        "data/README.md",
        "data/synthetic/README.md",
        "data/synthetic/schema-example.csv",
        "docs/ARCHITECTURE.md",
        "docs/DATA_PROVENANCE.md",
        "docs/LIMITATIONS.md",
        "docs/PRE_REGISTRATION.md",
        "docs/PREREGISTRATION_AMENDMENTS.md",
        "docs/REPRODUCIBILITY.md",
        "docs/RESEARCH_DESIGN.md",
        "docs/RESEARCH_PROVENANCE.md",
        "pyproject.toml",
        "scripts/wheel_smoke.py",
        "src/news_sentiment_trading/__init__.py",
        "src/news_sentiment_trading/cli.py",
        "src/news_sentiment_trading/config.py",
        "src/news_sentiment_trading/data.py",
        "src/news_sentiment_trading/diagnostics.py",
        "src/news_sentiment_trading/features.py",
        "src/news_sentiment_trading/inference.py",
        "src/news_sentiment_trading/legacy.py",
        "src/news_sentiment_trading/metrics.py",
        "src/news_sentiment_trading/portfolio.py",
        "src/news_sentiment_trading/reporting.py",
        "src/news_sentiment_trading/repository_scan.py",
        "src/news_sentiment_trading/robustness.py",
        "src/news_sentiment_trading/signals.py",
        "src/news_sentiment_trading/source_gate.py",
        "src/news_sentiment_trading/synthetic.py",
        "src/news_sentiment_trading/walk_forward.py",
        "tests/conftest.py",
        "tests/test_adversarial.py",
        "tests/test_cli.py",
        "tests/test_config.py",
        "tests/test_data.py",
        "tests/test_diagnostics.py",
        "tests/test_features_signals.py",
        "tests/test_inference_known_answers.py",
        "tests/test_legacy.py",
        "tests/test_metrics_inference.py",
        "tests/test_packaging_identity.py",
        "tests/test_portfolio.py",
        "tests/test_reporting.py",
        "tests/test_repository_scan.py",
        "tests/test_robustness.py",
        "tests/test_source_gate.py",
        "tests/test_walk_forward.py",
        "uv.lock",
    }
)

_RESTRICTED_SUFFIXES = {
    ".7z",
    ".doc",
    ".docx",
    ".ipynb",
    ".npz",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".ppt",
    ".pptx",
    ".sqlite",
    ".xls",
    ".xlsx",
    ".zip",
}
_ALLOWED_CSV_PATHS = {"data/synthetic/schema-example.csv"}
_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]"),
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
)
_REVIEWED_REFS = frozenset({"refs/heads/main", "refs/tags/v1.1.1"})


@dataclass(frozen=True, order=True)
class Finding:
    """One stable repository-scan finding."""

    scope: str
    path: str
    code: str
    detail: str


@dataclass(frozen=True)
class ScanResult:
    """Complete repository-scan result."""

    passed: bool
    tracked_files: int
    reachable_commits: int
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "tracked_files": self.tracked_files,
            "reachable_commits": self.reachable_commits,
            "findings": [asdict(finding) for finding in self.findings],
        }


def _run_git(root: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return completed.stdout


def _run_git_bytes(root: Path, arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        stdout = completed.stdout.decode("utf-8", errors="replace")
        raise RuntimeError((stderr or stdout).strip())
    return completed.stdout


def _tracked_paths(root: Path) -> tuple[str, ...]:
    output = _run_git_bytes(root, ["ls-files", "-z"])
    return tuple(
        sorted(item.decode("utf-8", errors="strict") for item in output.split(b"\0") if item)
    )


def _path_findings(path: str, scope: str) -> list[Finding]:
    normalized = path.replace("\\", "/")
    findings: list[Finding] = []
    if normalized not in _REVIEWED_PATHS:
        findings.append(
            Finding(scope, normalized, "UNREVIEWED_PATH", "path is not in the reviewed tree")
        )
    suffix = Path(normalized).suffix.casefold()
    if suffix in _RESTRICTED_SUFFIXES:
        findings.append(
            Finding(scope, normalized, "RESTRICTED_FORMAT", f"restricted file type: {suffix}")
        )
    if suffix == ".csv" and normalized not in _ALLOWED_CSV_PATHS:
        findings.append(
            Finding(scope, normalized, "UNAPPROVED_CSV", "CSV is not the synthetic schema example")
        )
    if normalized.startswith("reports/artifacts/"):
        findings.append(
            Finding(scope, normalized, "EMPIRICAL_ARTIFACT", "local artifact path is tracked")
        )
    return findings


def _content_findings(data: bytes, scope: str, path: str) -> list[Finding]:
    findings: list[Finding] = []
    if len(data) > MAX_TRACKED_FILE_BYTES:
        findings.append(
            Finding(scope, path, "OVERSIZED_FILE", f"file exceeds {MAX_TRACKED_FILE_BYTES} bytes")
        )
        return findings
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        findings.append(Finding(scope, path, "NON_UTF8_CONTENT", "tracked content is not UTF-8"))
        return findings

    if any(pattern.search(text) for pattern in _ABSOLUTE_PATH_PATTERNS):
        findings.append(Finding(scope, path, "ABSOLUTE_PATH", "local absolute path"))
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        findings.append(Finding(scope, path, "POSSIBLE_SECRET", "high-confidence secret pattern"))
    return findings


def _validate_source_gate_example(root: Path, tracked: tuple[str, ...]) -> list[Finding]:
    path = "configs/source_gate.example.json"
    if path not in tracked:
        return []
    try:
        payload = json.loads((root / path).read_text(encoding="utf-8"))
        hashes = [*payload["source_sha256"].values(), payload["calendar"]["sha256"]]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return [Finding("current", path, "INVALID_SOURCE_GATE_EXAMPLE", str(exc))]
    if any(value != "0" * 64 for value in hashes):
        return [
            Finding(
                "current",
                path,
                "POPULATED_SOURCE_HASH",
                "source-gate example contains a non-placeholder hash",
            )
        ]
    return []


def _history_findings(root: Path, commits: tuple[str, ...]) -> list[Finding]:
    findings: list[Finding] = []
    seen_blobs: set[str] = set()
    for commit in commits:
        listing = _run_git_bytes(root, ["ls-tree", "-r", "-z", commit])
        for entry in listing.split(b"\0"):
            if not entry:
                continue
            tree_metadata, raw_path = entry.split(b"\t", 1)
            path = raw_path.decode("utf-8", errors="strict")
            object_id = tree_metadata.split()[2].decode("ascii")
            findings.extend(_path_findings(path, "history"))
            if object_id in seen_blobs:
                continue
            seen_blobs.add(object_id)
            data = _run_git_bytes(root, ["cat-file", "blob", object_id])
            findings.extend(_content_findings(data, "history", path))

    if not commits:
        return findings
    metadata = _run_git(
        root,
        ["log", "--all", "--format=%H%x1f%ae%x1f%ce%x1f%s%x1f%b%x1e"],
    )
    for record in metadata.split("\x1e"):
        record = record.strip("\r\n")
        if not record:
            continue
        commit, author_email, committer_email, subject, body = record.split("\x1f", 4)
        for email in (author_email, committer_email):
            if email and not (
                email.casefold().endswith("@users.noreply.github.com")
                or email.casefold().endswith(".invalid")
            ):
                findings.append(
                    Finding("history", commit, "UNAPPROVED_GIT_EMAIL", "non-public Git email")
                )
        findings.extend(
            _content_findings(f"{subject}\n{body}".encode(), "history", f"commit:{commit}")
        )
    return findings


def _ref_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    output = _run_git(
        root,
        [
            "for-each-ref",
            "--format=%(refname)%00%(objecttype)%00%(objectname)",
            "refs/heads",
            "refs/tags",
        ],
    )
    for record in output.splitlines():
        if not record:
            continue
        ref_name, object_type, object_id = record.split("\0", 2)
        if ref_name not in _REVIEWED_REFS:
            findings.append(
                Finding("repository", ref_name, "UNREVIEWED_REF", "ref is not part of the release")
            )
        if ref_name == "refs/tags/v1.1.1" and object_type != "tag":
            findings.append(
                Finding("repository", ref_name, "LIGHTWEIGHT_TAG", "release tag is not annotated")
            )
        if object_type != "tag":
            continue
        tag_data = _run_git_bytes(root, ["cat-file", "tag", object_id])
        findings.extend(_content_findings(tag_data, "repository", ref_name))
        tag_text = tag_data.decode("utf-8", errors="replace")
        tagger = re.search(r"(?m)^tagger .* <([^>]+)>", tag_text)
        if tagger is None or not tagger.group(1).casefold().endswith("@users.noreply.github.com"):
            findings.append(
                Finding(
                    "repository",
                    ref_name,
                    "UNAPPROVED_TAG_EMAIL",
                    "annotated tag does not use a public no-reply email",
                )
            )
    return findings


def scan_repository(root: str | Path) -> ScanResult:
    """Scan current tracked files and all reachable commits for unsafe repository content."""

    repository = Path(root).resolve()
    inside = _run_git(repository, ["rev-parse", "--is-inside-work-tree"])
    if inside.strip() != "true":
        raise RuntimeError("repository scan must run inside a Git worktree")

    tracked = _tracked_paths(repository)
    findings: list[Finding] = []
    for path in tracked:
        findings.extend(_path_findings(path, "current"))
        findings.extend(_content_findings((repository / path).read_bytes(), "current", path))
    findings.extend(_validate_source_gate_example(repository, tracked))
    findings.extend(_ref_findings(repository))

    commit_output = _run_git(repository, ["rev-list", "--all"])
    commits = tuple(line for line in commit_output.splitlines() if line)
    findings.extend(_history_findings(repository, commits))

    unique = tuple(sorted(set(findings)))
    return ScanResult(
        passed=not unique,
        tracked_files=len(tracked),
        reachable_commits=len(commits),
        findings=unique,
    )


__all__ = ["Finding", "ScanResult", "scan_repository"]
