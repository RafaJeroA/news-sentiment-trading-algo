from __future__ import annotations

import pandas as pd
import pytest

from news_sentiment_trading.synthetic import synthetic_panel


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    return synthetic_panel()
