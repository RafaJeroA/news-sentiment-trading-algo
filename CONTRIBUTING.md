# Contributing

Contributions that improve correctness, reproducibility, documentation, or tests are welcome. By
contributing, you confirm that you have the right to submit the material under the MIT License.

## Development setup

Install Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/), then run:

```text
uv sync --all-extras --locked
uv --cache-dir .venv/uv-cache pip check
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run news-sentiment-trading synthetic-demo
uv run news-sentiment-trading repository-scan
uv --cache-dir .venv/uv-cache build --no-build-isolation
```

Pull requests should explain the behavioral change, identify any affected research assumptions,
and include focused tests.

## Research safeguards

- Preserve the full price calendar and the explicit missing-sentiment state.
- Maintain close-`t` signal formation, next-open execution, and strictly prior rolling features.
- Keep parameter selection isolated from each outer test block.
- Reconcile weights, turnover, costs, returns, and compounded wealth.
- Keep negative and null findings visible alongside favorable findings.
- Do not edit `docs/PRE_REGISTRATION.md`; record later changes in
  `docs/PREREGISTRATION_AMENDMENTS.md` with their confirmatory or exploratory status.

Changes to formulas, timing, sample construction, parameter grids, inference, or empirical claims
must update the relevant research documentation.

## Data and third-party material

Do not commit empirical source files, source-derived observations, course or team material,
credentials, personal data, or local absolute paths. Tests and examples must use independently
created synthetic data. Third-party material must have documented provenance and a compatible
license; citation alone does not grant redistribution rights.
