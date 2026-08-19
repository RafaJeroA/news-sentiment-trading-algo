# News Sentiment Trading Research

This project studies whether unusually positive or negative daily news sentiment, combined with elevated news activity, contains information about subsequent equity returns.

It started as a two-stock course exercise. I later expanded it to ten equities and replaced the original in-sample analysis with a walk-forward design, explicit execution timing, transaction costs and leakage checks.

## Research question

Do daily sentiment extremes predict subsequent equity returns after point-in-time feature construction, training-only parameter selection and realistic execution timing?

The purpose of the repository is to test that question carefully. It does not assume that the strategy must be profitable.

## Research design

- **Universe:** ten equities on a common 602-session calendar.
- **Timing:** news dated `t` is assumed available after the close, executes at the adjusted open on `t+1`, and earns the following adjusted-open return.
- **Features:** rolling means, dispersion and news-intensity thresholds use only observations before the signal date.
- **Selection:** eight parameter combinations are evaluated using ordered training blocks.
- **Evaluation:** six non-overlapping outer test blocks are used once each.
- **Portfolio:** the main specification is long-only, with a 20% cap per name and residual cash.
- **Costs:** the primary strategy applies a 10-basis-point one-way transaction cost.
- **Leakage checks:** tests verify that future observations cannot change earlier features, selected parameters or completed folds.

The full methodology is documented in [docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md).

## Data availability

The original course dataset is not included because redistribution rights are unavailable.

The public repository contains:

- the complete research implementation;
- the expected input schema;
- a deterministic synthetic demonstration;
- synthetic timing, accounting and leakage tests.

Researchers who independently have authorised access to compatible source files can run the empirical workflow locally. The repository does not distribute empirical rows, empirical results or modified versions of the restricted dataset.

See [data/README.md](data/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Quick start

Python 3.12 or 3.13 and `uv` are required.

~~~text
uv sync --all-extras --locked
uv --cache-dir .venv/uv-cache pip check
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run news-sentiment-trading synthetic-demo
uv --cache-dir .venv/uv-cache build --no-build-isolation
~~~

These commands use only synthetic data and require no credentials or private configuration.

## Repository structure

- `src/news_sentiment_trading/`: features, signals, walk-forward evaluation, portfolio accounting, inference and reporting.
- `tests/`: timing, leakage, accounting, inference and packaging tests.
- `configs/primary.toml`: primary research configuration.
- `data/synthetic/`: small synthetic schema example.
- `docs/`: methodology, limitations, data handling and reproduction notes.

## Interpretation

The public repository demonstrates the research process and implementation. It does not claim that news sentiment produces a profitable or statistically significant trading strategy.

The empirical sample is short, overlaps the COVID-19 crisis and uses daily aggregates without article timestamps. Transaction costs are simplified, and the strategy can hold cash while the benchmark remains fully invested.

The complete list of limitations is in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## License and data rights

Original code, tests, documentation and synthetic examples are released under the MIT License.

The license does not cover the unavailable course dataset, third-party data, course material or team submissions. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This repository is for research and educational use and is not investment advice.
