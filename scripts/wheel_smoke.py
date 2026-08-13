"""Install one built artifact in isolation and verify its portable interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from news_sentiment_trading.config import load_config


def _run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    return parser


def _verify_sdist_members(artifact: Path, root: Path) -> None:
    if not artifact.name.endswith(".tar.gz"):
        return
    tracked_output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    tracked = {item.decode("utf-8") for item in tracked_output.split(b"\0") if item}
    exact = {
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "configs/primary.toml",
        "pyproject.toml",
    }
    expected = exact | {path for path in tracked if path.startswith("src/news_sentiment_trading/")}
    expected.add("PKG-INFO")
    with tarfile.open(artifact, mode="r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {member.split("/", maxsplit=1)[0] for member in members}
    if len(roots) != 1:
        raise AssertionError(f"sdist must contain exactly one root directory: {sorted(roots)}")
    actual = {member.split("/", maxsplit=1)[1] for member in members}
    if actual != expected:
        raise AssertionError(
            f"sdist member allowlist mismatch; unexpected={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    artifact = args.artifact.resolve()
    if not artifact.is_file() or not (
        artifact.name.endswith(".whl") or artifact.name.endswith(".tar.gz")
    ):
        raise ValueError(f"expected a wheel or source distribution: {artifact}")
    _verify_sdist_members(artifact, root)
    locked_config = root / "configs" / "primary.toml"
    expected_config_digest = load_config(locked_config).digest()
    expected_config_sha256 = hashlib.sha256(locked_config.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="news-sentiment-package-") as temporary:
        temporary_root = Path(temporary)
        environment = temporary_root / "venv"
        run_root = temporary_root / "outside-checkout"
        run_root.mkdir()
        _run(
            "uv",
            "venv",
            str(environment),
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}",
        )
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        executable = environment / (
            "Scripts/news-sentiment-trading.exe"
            if sys.platform == "win32"
            else "bin/news-sentiment-trading"
        )
        _run("uv", "pip", "install", "--python", str(python), str(artifact), cwd=run_root)
        _run("uv", "pip", "check", "--python", str(python), cwd=run_root)
        check = f"""
import hashlib
from importlib.metadata import distribution, version
from importlib.resources import as_file, files

from news_sentiment_trading.config import load_config

assert version("news-sentiment-trading-algo") == "1.1.1"
installed_distribution = distribution("news-sentiment-trading-algo")
metadata = installed_distribution.metadata
assert metadata["Author"] == "Rafael Jerónimo Aragón"
assert metadata["Project-URL"].find("RafaJeroA/news-sentiment-trading-algo") >= 0
scripts = {{
    item.name: item.value
    for item in installed_distribution.entry_points
    if item.group == "console_scripts"
}}
assert scripts == {{"news-sentiment-trading": "news_sentiment_trading.cli:main"}}
resource = files("news_sentiment_trading").joinpath("resources/primary.toml")
with as_file(resource) as path:
    assert load_config(path).digest() == {expected_config_digest!r}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == {expected_config_sha256!r}
"""
        _run(str(python), "-c", check, cwd=run_root)
        repository_check = f"""
from pathlib import Path
from news_sentiment_trading.cli import repository_root
assert repository_root() == Path({str(root)!r}).resolve()
"""
        _run(str(python), "-c", repository_check, cwd=root)
        output = run_root / "synthetic.json"
        _run(str(executable), "--version", cwd=run_root)
        _run(str(executable), "synthetic-demo", "--output", str(output), cwd=run_root)
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("synthetic_only") is not True or payload.get("outer_observations") != 348:
            raise AssertionError("installed synthetic demonstration did not reconcile")
        foreign_repository = run_root / "foreign-repository"
        (foreign_repository / "configs").mkdir(parents=True)
        (foreign_repository / "configs" / "primary.toml").write_text(
            "foreign = true\n", encoding="utf-8"
        )
        _run("git", "init", "-q", cwd=foreign_repository)
        foreign_output = foreign_repository / "synthetic.json"
        _run(
            str(executable),
            "synthetic-demo",
            "--output",
            str(foreign_output),
            cwd=foreign_repository,
        )
        foreign_payload = json.loads(foreign_output.read_text(encoding="utf-8"))
        if foreign_payload.get("outer_observations") != 348:
            raise AssertionError("installed synthetic demo read a foreign checkout configuration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
