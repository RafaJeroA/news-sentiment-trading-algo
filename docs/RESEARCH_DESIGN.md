# Research design

## Question

Do daily news-sentiment extremes, conditional on abnormal news intensity, contain stable and economically useful information about subsequent equity returns after point-in-time feature construction, conservative execution, portfolio aggregation, and transaction costs?

The objective is an honest answer, not a profitable backtest. Weak or negative outer-test evidence is a valid outcome.

## Information and return timing

Daily aggregate sentiment dated `t` is assumed known only after close `t`. It forms a target at that close, executes at adjusted open `t+1`, and earns the one-session adjusted-open return:

`r(t) = AdjustedOpen(t+2) / AdjustedOpen(t+1) - 1`.

`AdjustedOpen = Open × AdjClose / Close`. The associated adjustment factor is a historical total-return convention, not proof of contemporaneous vendor availability. A close-based alternative uses the same two-session alignment and is robustness-only. No intraday claim is made without timestamps.

## Features

For complete current components:

```text
BULL(t) = positivePartscr(t) + certaintyPartscr(t) + finupPartscr(t)
BEAR(t) = negativePartscr(t) + uncertaintyPartscr(t) + findownPartscr(t)
BBr(t)  = 100 * BULL(t) / (BULL(t) + BEAR(t))
```

The primary ratio is missing unless all six components exist and the denominator is positive. No fill is applied. The log-ratio robustness is `log1p(BULL)-log1p(BEAR)`. A one-session forward fill is robustness-only; backward fill is impossible.

For fixed `w`, current feature and RVT are compared with mean, population SD, and linear-interpolated RVT quantile calculated from the prior `w` market sessions, never the current row. At least 80% of that window must be available. A qualifying high/low extreme creates +1/-1; otherwise the primary event is zero. Primary holding is exactly one executable interval.

## Walk-forward selection

The 602 market dates produce 600 signal/execution/return records. The first 252 are initial history. All remaining 348 are partitioned into six equal, non-overlapping 58-return outer blocks and used once.

At each outer boundary, selection uses only returns ending on or before the first test signal date. This one-open embargo excludes the return that ends at the first test execution open. The final 126 eligible returns are split into three ordered, non-overlapping 42-return inner validation blocks.

Eight shared panel-level candidates are locked: `w={20,50}`, `z={1.5,2.0}`, `q={0.5,0.75}`. The fixed baseline `(20,1.5,0.5)` is never tuned. Inner selection uses median annualized net active return versus the rebalanced universe benchmark, then worst block, lower turnover, and conservative parameter tie breaks. Coverage/turnover failures trigger the fixed baseline. Long-short is not separately tuned.

## Portfolios and costs

Long-only active names receive equal weights capped at 20%; gross exposure cannot exceed 100% and the remainder earns a 0% cash return. Balanced neutral books exist only when both sides are present. Each book gross is the minimum of 50%, 20% times the long count, and 20% times the short count; otherwise the portfolio is cash. Directional ± signals are labeled secondary and never called neutral.

The registered one-way cost is charged from target risky-asset weight changes:

`cost_t = rate × Σ_i |w(i,t) - w(i,t-1)|`.

Initial entry and final liquidation are included. Primary cost is 10 bps; 0, 5, 10, 25, 50, and 100 bps are reported. Borrow, financing, taxes, market impact, and slippage beyond this stylized rate are absent.

Version 1.1 preserves that registered definition and adds a post-v1.0 descriptive accounting
analysis. Prior targets earn their preceding row returns, rebalance costs are funded from the
financing account, and the next pre-trade risky weights are normalized by post-cost portfolio NAV.
Turnover is then measured from that state to the new targets. Final liquidation cost is converted
to start-of-row NAV units so `transaction_cost = rate × recorded_turnover` still reconciles. A
nonpositive normalization NAV, a return below -100%, or resurrection of a bankrupt asset fails
closed. This analysis does not enter training selection or confirmatory inference.

Reporting distinguishes long exposure, short exposure, gross exposure, net exposure, the
financing-account balance `1-net`, and unused gross capacity `1-gross`. For long-only portfolios,
unused gross capacity is also unallocated capital. For signed books the financing balance is not
described as unused cash.

## Benchmarks

- Daily-rebalanced equal weight resets every name to 10% each day; it continually sells relative winners and buys relative losers.
- Static equal-weight buy-and-hold begins at 10% each and permits weights to drift with cumulative asset performance.

Both use the same adjusted-open return convention. The primary trading comparison is the strategy's net active return versus the daily-rebalanced benchmark.

A post-v1.0 supplemental benchmark charges the daily-rebalanced equal-weight targets at the stated
one-way rate using drift-adjusted turnover, including entry and liquidation. The registered gross
benchmark remains the confirmatory comparison; the cost-aware version is descriptive only.

## Inference and interpretation

The stitched 348-return series is primary. For each outer fold, the registered predictive score is
the standardized BBr ratio `Z(t,w)` with no fill and that fold's selected window `w`. Daily
cross-sectional Spearman rank IC uses every finite selected score and subsequent return, with at
least six valid names; it is not filtered by the RVT event mask.
Trading evidence is daily net active return versus the rebalanced benchmark. Because the strategy
can hold cash while the benchmark remains fully invested, this is not an exposure-matched test of
news information.

Both statistics receive Newey-West uncertainty and Holm correction. A circular moving-block
bootstrap (10,000 samples, block length 10, synchronized dates, fixed seed) supplies mean confidence
intervals. Beta/exposure diagnostics use HAC covariance.

The locked version-1 rules do not specify a minimum number of usable rank-IC dates or numerical
thresholds for economic coverage, cost sensitivity, or segment dominance. These omissions limit
the strength of any interpretation; they are documented rather than repaired after results and
are prospective version-2 design requirements. No empirical result is distributed here.

Results are decomposed by fold, asset, cost, portfolio, pre-crisis/crisis period, leave-one-out universe, parameter stability, signal coverage/decay, feature, missingness, holding, and execution convention. Robustness is descriptive unless separately promoted by an amendment made without inspecting its result.
