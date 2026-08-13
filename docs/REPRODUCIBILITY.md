# Reproducibility

## Synthetic workflow

The repository is self-contained for installation, static analysis, tests, package builds, and the
synthetic demonstration:

```text
uv sync --all-extras --locked --no-cache
uv --cache-dir .venv/uv-cache pip check
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run pytest --cov=news_sentiment_trading --cov-report=term-missing
uv run news-sentiment-trading synthetic-demo
uv run news-sentiment-trading repository-scan
uv --cache-dir .venv/uv-cache build --no-build-isolation --out-dir dist/validation
uv run python scripts/wheel_smoke.py dist/validation/news_sentiment_trading_algo-1.1.1-py3-none-any.whl
uv run python scripts/wheel_smoke.py dist/validation/news_sentiment_trading_algo-1.1.1.tar.gz
```

No external data or credentials are required. The tests use artificial inputs and include timing,
future-invariance, fold-isolation, accounting, inference, and package-installation checks.

## Empirical replication with authorised data

The original CSV files and their expected hashes are not included. Users who independently possess
authorised access to compatible files may supply a source directory and a populated source-gate
JSON outside the repository. `configs/source_gate.example.json` documents the format with zeroed
placeholder hashes; it cannot authorise an empirical run.

The following commands validate and run such a local copy:

```text
uv run news-sentiment-trading inventory --source-root <data-directory>
uv run news-sentiment-trading run-primary \
  --source-root <data-directory> \
  --source-gate <source-gate.json> \
  --output-dir reports/artifacts/primary-local
uv run news-sentiment-trading run-robustness \
  --source-root <data-directory> \
  --source-gate <source-gate.json> \
  --output-dir reports/artifacts/robustness-local
```

Empirical commands require a clean, complete Git checkout. They validate the locked configuration,
preregistration, source hashes, ordered calendar, and fold boundaries before evaluation. Output
directories are write-once and remain ignored under `reports/artifacts/`.

Report generation additionally checks content-addressed provenance against the current repository
and source gate. A full empirical reproduction claim therefore requires both this code and an
authorised copy of the unavailable data.
