# Preregistration amendments

The locked version-1 design in `docs/PRE_REGISTRATION.md` is unchanged. This file records later
corrections and descriptive extensions, including whether relevant results had already been seen.

## 2026-08-01 — normalized dependency-lock hashing

- **Reason:** Equivalent Windows checkouts produced different raw `uv.lock` hashes because of line
  endings, despite identical Git-normalized content and resolved dependencies.
- **Change:** The dependency-lock hash is computed after normalizing UTF-8 line endings to LF.
  Source-data hashes remain raw-byte hashes.
- **Result visibility:** The primary result had already been viewed.
- **Effect:** Provenance metadata only. Signals, selections, portfolios, metrics, and inference are
  unchanged.
- **Status:** Non-analytical reproducibility correction.

## 2026-08-01 — inference and reporting semantics

- **Reason:** Rank-IC HAC inference compressed missing calendar rows before forming lag pairs.
  Reporting also compounded an additive active-return spread and used ambiguous cash, hit-rate, and
  transaction-cost labels.
- **Change:** HAC calculations now preserve calendar gaps. Active performance is reported as
  additive spread statistics plus relative wealth; financing balance is separated from unused gross
  capacity; exposure-day and trading-day hit rates are distinct; transaction charges are not called
  return drag.
- **Result visibility:** The primary and sensitivity results had already been viewed. The correction
  was not used to select a new specification.
- **Effect:** Inference and reporting artifacts. Signals, selected parameters, weights, gross and
  net returns, costs, and the hypothesis decisions are unchanged.
- **Status:** Post-result implementation and reporting correction. No empirical artifact or outcome
  is distributed in this repository.

## 2026-08-08 — source identity and estimand limitations

- **Reason:** Recorded source hashes were not yet enforced against a fixed source/calendar/fold
  identity. The review also identified limitations in the cash-inclusive trading comparison, the
  unmasked rank-IC score, and the absence of a minimum usable-IC-date rule.
- **Change:** Empirical commands validate all ten source hashes, the complete ordered 602-session
  calendar, and the exact six fold boundaries before evaluation; they also recheck source bytes
  after loading and reject shallow history. The populated source gate remains external because its
  hashes cannot be distributed; `configs/source_gate.example.json` contains placeholders only.
- **Interpretation:** The trading statistic compares a potentially underinvested strategy with a
  fully invested benchmark. The predictive statistic ranks every available standardized BBr score
  and does not apply the RVT event mask. Neither cleanly isolates intensity-conditioned news
  information.
- **Future work:** A new confirmatory study should preregister exposure-matched controls, numerical
  coverage and concentration thresholds, an explicit predictive mask, a minimum valid-date count,
  and small-sample sensitivity analysis before viewing new results.
- **Result visibility:** Existing results had already been viewed; this change did not inspect or
  select among them.
- **Status:** Non-analytical input validation plus prospective research-design requirements. The
  locked version-1 rules remain unchanged.

## 2026-08-09 — drift-adjusted accounting analyses

- **Reason:** Target-to-target turnover is reproducible but differs from rebalancing after holdings
  drift with returns. Net exposure also has a financing interpretation for signed portfolios, and a
  frictionless rebalanced benchmark does not illustrate its own rebalancing cost.
- **Preserved primary:** Target-weight turnover, the 10-basis-point cost, signals, parameter grid,
  folds, selection, portfolios, registered benchmarks, inference, and hypothesis criteria are
  unchanged.
- **Added analyses:** Descriptive cost-funded drift-adjusted turnover; separate long, short, gross,
  and net exposure fields; financing balance and unused gross capacity; and a descriptive
  cost-aware rebalanced benchmark.
- **Failure semantics:** Non-finite inputs, returns below -100%, unsupported partial-bankruptcy
  rebalancing, resurrection after default, and nonpositive normalization value fail validation.
- **Result visibility:** The version-1 result had already been viewed. No empirical output was used
  to choose these additions.
- **Status:** Exploratory post-version-1 accounting analysis.

## 2026-08-09 — external source-gate packaging

- **Reason:** Empirical input hashes cannot accompany a distribution that excludes the underlying
  source data.
- **Change:** A populated source gate is supplied outside the repository through `--source-gate` or
  `NEWS_SENTIMENT_SOURCE_GATE`. Local provenance records its hash. The tracked example contains
  zero placeholders and cannot validate an empirical source.
- **Effect:** Packaging and provenance only. Feature, timing, selection, portfolio, cost, benchmark,
  and inference definitions are unchanged.
- **Status:** Non-analytical data-boundary correction.
