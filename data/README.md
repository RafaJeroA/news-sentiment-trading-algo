# Data contract

The original course dataset is not licensed for redistribution and is not included in this
repository. Users who independently possess authorised access may point the CLI to a compatible
directory with `--source-root` or `NEWS_SENTIMENT_SOURCE`.

The primary loader expects one CSV per ticker (`AAPL.csv`, ..., `PFE.csv`) with these columns:

`Date, Open, High, Low, Close, Adj Close, Volume, RVT, positivePartscr, negativePartscr, fearPartscr, findownPartscr, finupPartscr, finhypePartscr, certaintyPartscr, uncertaintyPartscr`

An unnamed integer export column is ignored. Dates must be unique and increasing; prices must be
positive; volume and RVT must be nonnegative; and the ten primary tickers must share the complete
price calendar. Missing sentiment remains missing. The loader never backward fills and never drops
a price row because sentiment is unavailable.

```text
uv run news-sentiment-trading inventory --source-root <data-directory>
```

This command applies the same schema, ordering, finite-value, price, volume, sentiment-domain, and
adjusted-open checks used by the empirical workflow.

`data/synthetic/schema-example.csv` and the programmatic fixtures in `synthetic.py` were created
from scratch. They contain no empirical or perturbed source rows.
