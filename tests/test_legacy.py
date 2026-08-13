from __future__ import annotations

import pandas as pd
import pytest

from news_sentiment_trading.legacy import reproduce_fb_jpm


@pytest.mark.parametrize(
    (
        "ticker",
        "regime",
        "arithmetic_return",
        "compounded_return",
        "one_way_trades",
        "observations",
    ),
    [
        ("FB", "long_only", -0.3806060819251989, -0.3299124227166462, 18, 601),
        ("FB", "long_short", -1.0313087605555624, -0.6633463783518421, 38, 601),
        ("JPM", "long_only", 0.14531667036028217, 0.1261073191844988, 18, 601),
        ("JPM", "long_short", -0.25448953752156966, -0.2698821843723378, 38, 601),
    ],
)
def test_legacy_reproduction_synthetic_known_answers(
    panel: pd.DataFrame,
    ticker: str,
    regime: str,
    arithmetic_return: float,
    compounded_return: float,
    one_way_trades: int,
    observations: int,
) -> None:
    results = {(item.ticker, item.regime): item for item in reproduce_fb_jpm(panel)}
    result = results[(ticker, regime)]
    assert result.strategy_arithmetic_return == pytest.approx(arithmetic_return)
    assert result.strategy_compounded_return == pytest.approx(compounded_return)
    assert result.one_way_trades == one_way_trades
    assert result.observations == observations
