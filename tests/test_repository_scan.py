from __future__ import annotations

import subprocess
from pathlib import Path

from news_sentiment_trading.repository_scan import scan_repository


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def _commit(root: Path, message: str = "test: safe snapshot") -> None:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Repository Test",
        "-c",
        "user.email=repository-test@example.invalid",
        "commit",
        "-qm",
        message,
    )


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "README.md").write_text("# Safe research repository\n", encoding="utf-8")
    _commit(root)
    return root


def test_safe_repository_passes(tmp_path: Path) -> None:
    result = scan_repository(_repository(tmp_path))
    assert result.passed
    assert result.tracked_files == 1
    assert result.reachable_commits == 1
    assert result.findings == ()


def test_unreviewed_path_is_rejected_from_reachable_history(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    blocked = root / "notes" / "operator-instructions.md"
    blocked.parent.mkdir()
    blocked.write_text("internal notes\n", encoding="utf-8")
    _commit(root, "test: add blocked path")
    blocked.unlink()
    _commit(root, "test: remove blocked path")

    result = scan_repository(root)
    assert not result.passed
    assert any(
        finding.scope == "history" and finding.code == "UNREVIEWED_PATH"
        for finding in result.findings
    )


def test_absolute_path_and_secret_are_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    token = "gh" + "p_" + "A" * 32
    (root / "notes.txt").write_text(
        "local path C:" + "\\Users\\example\\data and token " + token + "\n",
        encoding="utf-8",
    )
    _commit(root, "test: unsafe content")

    codes = {finding.code for finding in scan_repository(root).findings}
    assert {"ABSOLUTE_PATH", "POSSIBLE_SECRET"} <= codes


def test_unapproved_csv_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "prices.csv").write_text("date,value\n2026-01-01,1\n", encoding="utf-8")
    _commit(root, "test: unsafe csv")

    assert any(finding.code == "UNAPPROVED_CSV" for finding in scan_repository(root).findings)


def test_populated_source_gate_example_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    gate = root / "configs" / "source_gate.example.json"
    gate.parent.mkdir()
    gate.write_text(
        '{"source_sha256":{"SYNTH":"' + "1" * 64 + '"},"calendar":{"sha256":"' + "0" * 64 + '"}}\n',
        encoding="utf-8",
    )
    _commit(root, "test: populated gate")

    assert any(
        finding.code == "POPULATED_SOURCE_HASH" for finding in scan_repository(root).findings
    )


def test_non_public_git_email_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "README.md").write_text("# Test\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Example",
        "-c",
        "user.email=personal@example.com",
        "commit",
        "-qm",
        "test: unsafe identity",
    )

    assert any(finding.code == "UNAPPROVED_GIT_EMAIL" for finding in scan_repository(root).findings)


def test_unreviewed_branch_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _git(root, "branch", "scratch")

    assert any(finding.code == "UNREVIEWED_REF" for finding in scan_repository(root).findings)


def test_release_tag_must_be_annotated_with_public_email(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _git(root, "tag", "v1.1.1")
    assert any(finding.code == "LIGHTWEIGHT_TAG" for finding in scan_repository(root).findings)

    _git(root, "tag", "-d", "v1.1.1")
    _git(
        root,
        "-c",
        "user.name=Repository Test",
        "-c",
        "user.email=repository-test@users.noreply.github.com",
        "tag",
        "-a",
        "v1.1.1",
        "-m",
        "News Sentiment Trading Research v1.1.1",
    )
    assert scan_repository(root).passed
