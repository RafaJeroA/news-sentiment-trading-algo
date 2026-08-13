from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_and_entry_point_are_current() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "news-sentiment-trading-algo"
    assert project["version"] == "1.1.1"
    assert project["scripts"] == {"news-sentiment-trading": "news_sentiment_trading.cli:main"}
    assert project["requires-python"] == ">=3.12,<3.14"


def test_citation_and_license_metadata_are_consistent() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert 'title: "News Sentiment Trading Research"' in citation
    assert "version: 1.1.1" in citation
    assert "date-released: 2026-08-13" in citation
    assert 'family-names: "Jerónimo Aragón"' in citation
    assert 'given-names: "Rafael"' in citation
    assert "Copyright (c) 2026 Rafael Jerónimo Aragón" in license_text


def test_readme_has_a_self_contained_synthetic_quick_start() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = readme.split("## Quick start", 1)[1].split("## Repository structure", 1)[0]
    for command in (
        "uv sync --all-extras --locked",
        "uv --cache-dir .venv/uv-cache pip check",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run mypy src",
        "uv run pytest -q",
        "uv run news-sentiment-trading synthetic-demo",
        "uv --cache-dir .venv/uv-cache build --no-build-isolation",
    ):
        assert command in quick_start
    assert "NEWS_SENTIMENT_SOURCE" not in quick_start
    assert "--source-gate" not in quick_start


def test_ci_runs_the_public_package_gate_on_linux_and_windows() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "os: [ubuntu-latest, windows-latest]" in workflow
    for command in (
        "uv sync --all-extras --locked",
        "uv pip check",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run mypy src",
        "uv run pytest -q",
        "uv run news-sentiment-trading synthetic-demo",
        "uv build --out-dir dist/ci",
        "uv run news-sentiment-trading repository-scan",
    ):
        assert command in workflow
    assert "news_sentiment_trading_algo-1.1.1-py3-none-any.whl" in workflow
    assert "news_sentiment_trading_algo-1.1.1.tar.gz" in workflow
    jobs = workflow.split("jobs:\n", 1)[1]
    assert len(re.findall(r"^  [a-z][a-z0-9_-]*:\s*$", jobs, flags=re.MULTILINE)) == 1


def test_public_document_set_is_compact() -> None:
    assert {path.name for path in (ROOT / "docs").glob("*.md")} == {
        "ARCHITECTURE.md",
        "DATA_PROVENANCE.md",
        "LIMITATIONS.md",
        "PRE_REGISTRATION.md",
        "PREREGISTRATION_AMENDMENTS.md",
        "REPRODUCIBILITY.md",
        "RESEARCH_DESIGN.md",
        "RESEARCH_PROVENANCE.md",
    }


def test_wheel_smoke_targets_current_version() -> None:
    script = (ROOT / "scripts" / "wheel_smoke.py").read_text(encoding="utf-8")
    assert 'version("news-sentiment-trading-algo") == "1.1.1"' in script
    assert "resources/primary.toml" in script
