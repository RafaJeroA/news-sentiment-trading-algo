from __future__ import annotations

from pathlib import Path

import pytest

from news_sentiment_trading.config import load_config


def test_primary_config_has_locked_eight_candidate_grid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "primary.toml")
    assert len(config.features.parameters(config.execution)) == 8
    assert config.features.fixed(config.execution) in config.features.parameters(config.execution)
    assert len(config.universe.primary) == 10


def test_config_rejects_unknown_key(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "primary.toml").read_text(encoding="utf-8")
    path = tmp_path / "bad.toml"
    path.write_text(
        text.replace("seed = 20260801", "seed = 20260801\nleak = true"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="configuration keys"):
        load_config(path)
