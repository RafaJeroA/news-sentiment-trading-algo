# News Sentiment Trading Research

This repository implements a walk-forward research framework for testing whether daily equity
news-sentiment extremes contain information about subsequent returns. It emphasizes observable
timing, training-only model selection, explicit portfolio accounting, transaction costs, and
reproducible synthetic verification.

## Research question

Do unusually positive or negative daily news signals, when accompanied by elevated news activity,
predict subsequent equity returns after realistic timing and out-of-sample evaluation?

## Why this project exists

The project began as a two-stock academic exercise. A later review found that the original analysis
used the same sample for parameter choice and evaluation, left execution timing ambiguous, and did
not fully reconcile portfolio returns and risk statistics. The implementation here turns that idea
into a falsifiable ten-equity study with a locked design and explicit failure criteria.

## Research design

- **Universe:** ten equities on a shared 602-session calendar, with missing sentiment represented as
  an explicit no-signal state.
- **Timing:** news dated `t` is treated as available after close, trades at adjusted open `t+1`, and
  earns the adjusted-open return ending `t+2`.
- **Features:** rolling means, dispersion, and news-intensity thresholds use observations strictly
  before the signal date.
- **Selection:** eight shared parameter combinations are evaluated inside ordered training blocks;
  six non-overlapping outer blocks are used once each.
- **Portfolio:** the primary strategy is long-only with 20% per-name caps, residual cash, and a
  10-basis-point one-way transaction-cost assumption.
- **Evaluation:** compounded performance, drawdowns, turnover, exposure, cross-sectional rank IC,
  HAC uncertainty, Holm adjustment, block-bootstrap intervals, and predefined sensitivity analyses.
- **Leakage controls:** mutation tests verify that future data cannot change earlier features,
  parameter choices, or completed fold outputs.

The full specification is in [Research design](docs/RESEARCH_DESIGN.md). The original locked plan
and subsequent amendments are retained in [Preregistration](docs/PRE_REGISTRATION.md) and
[Preregistration amendments](docs/PREREGISTRATION_AMENDMENTS.md).

## Data availability

The original course dataset is not included because redistribution rights are unavailable. Public
tests and examples use independently generated synthetic data. Empirical replication requires
separately authorised access to compatible source data; no empirical result or source-derived
aggregate is distributed here.

The contribution of this repository is the research design and implementation, not evidence that
the sentiment signal succeeds.

The expected schema and the boundary between public and restricted inputs are documented in
[Data provenance](docs/DATA_PROVENANCE.md) and [the local data contract](data/README.md).

## Quick start

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```text
uv sync --all-extras --locked
uv --cache-dir .venv/uv-cache pip check
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run news-sentiment-trading synthetic-demo
uv --cache-dir .venv/uv-cache build --no-build-isolation
```

These commands require no private data, credentials, or local configuration. The synthetic
demonstration is deterministic and exercises the same feature, signal, portfolio, and evaluation
modules used by the empirical workflow.

## Repository structure

- `src/news_sentiment_trading/` contains the research pipeline and command-line interface.
- `tests/` contains synthetic known-answer, timing, leakage, accounting, and packaging tests.
- `configs/primary.toml` contains the locked primary configuration.
- `data/synthetic/` contains a small artificial schema example.
- `docs/` covers architecture, methodology, provenance, limitations, and reproduction.

## Limitations

The sample is short and overlaps the COVID-19 crisis. Daily aggregates lack article timestamps,
the adjusted-open convention relies on retrospective adjustment factors, transaction costs are
stylized, and the strategy can have lower market exposure than its benchmark. See
[Limitations](docs/LIMITATIONS.md) for the complete interpretation boundary.

## Reproducibility

The public workflow verifies installation, code quality, tests, package builds, and the synthetic
demonstration without external data. Users who independently possess authorised access to
compatible source files can run the empirical commands described in
[Reproducibility](docs/REPRODUCIBILITY.md).

## License and attribution

Original code and documentation in this repository are available under the MIT License. The
license does not cover the excluded course dataset, course material, team submission, or other
third-party works. See [Third-party notices](THIRD_PARTY_NOTICES.md) and
[Citation metadata](CITATION.cff).

This repository is research software, not investment advice. It makes no claim of profitability,
statistical significance, or readiness for live trading.
