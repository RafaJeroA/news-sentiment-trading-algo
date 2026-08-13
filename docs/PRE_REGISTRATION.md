# Preregistration: daily news sentiment and subsequent equity returns

Status: locked before the first real outer-test result. After the gate commit this file is immutable. Amendments belong only in `docs/PREREGISTRATION_AMENDMENTS.md`.

## 1. Research question and outcomes

Do daily news-sentiment extremes, conditional on abnormal news intensity, contain stable and economically useful information about subsequent equity returns after point-in-time feature construction, conservative execution, portfolio aggregation, and transaction costs?

Weak, null, negative, or unstable out-of-sample evidence is a valid outcome. The repository will not optimize its narrative for a favorable result.

### Predictive hypothesis

The cross-sectional rank correlation between the continuous point-in-time sentiment-extreme score and the subsequent adjusted-open return is positive out of sample.

### Trading hypothesis

The primary long-only strategy has positive mean net active return relative to the daily-rebalanced equal-weight universe benchmark at a 10 bps one-way cost.

### Null hypotheses

1. Mean daily cross-sectional rank IC is zero.
2. Mean daily net active return is zero.

Both confirmatory p-values use two-sided HAC tests and Holm family-wise adjustment. Positive point estimates without adequate uncertainty evidence are not treated as confirmation.

## 2. Universe and sample

Primary equities, in fixed order:

`AAPL, AMZN, DB, DIS, FB, GOOG, HSBC, JPM, MSFT, PFE`.

The verified common market calendar has 602 dates from 2018-01-02 through 2020-05-22. The historical source ticker remains `FB`; it is not relabeled META.

Currencies, indices, and commodities have shorter histories and different schemas/market structures. They are excluded from confirmatory work and may appear only in a clearly separated exploratory appendix.

## 3. Data boundary and missingness

Required data are unique, strictly increasing `(date,ticker)` rows with positive OHLC/adjusted close, nonnegative volume, RVT, and eight sentiment component columns. All primary assets must share the exact price calendar. An invalid price/schema fails the run.

Every price row is preserved. Sentiment availability, ratio availability, signal availability, and price availability are distinct. Primary missing treatment is no fill. Missing current components, missing RVT, a nonpositive Bull+Bear denominator, inadequate historical coverage, or undefined dispersion produces an explicit no-signal state. Backward filling is prohibited.

`AdjustedOpen(t) = Open(t) × AdjClose(t) / Close(t)`. The factor must be finite and positive. This uses the vendor's historical total-return adjustment and does not establish contemporaneous factor availability.

Every run stores source SHA-256 values, configuration hash, lock hash, Git commit, and preregistration commit. Source files and row-level empirical derivatives remain outside Git while redistribution rights are unresolved.

## 4. Feature formulas

For date `t`, only if all six components are present:

```text
BULL(t) = positivePartscr(t) + certaintyPartscr(t) + finupPartscr(t)
BEAR(t) = negativePartscr(t) + uncertaintyPartscr(t) + findownPartscr(t)
BBr(t)  = 100 × BULL(t) / (BULL(t) + BEAR(t))
```

`BBr(t)` is undefined unless `BULL(t)+BEAR(t)>0`.

For candidate window `w`, define from observations strictly before `t`:

```text
mu(t,w)    = mean(BBr in the prior w market sessions)
sigma(t,w) = population SD(BBr in the prior w market sessions, ddof=0)
Z(t,w)     = (BBr(t)-mu(t,w))/sigma(t,w)
Q(t,w,q)   = linear-interpolated q quantile of RVT in the prior w market sessions
```

At least `ceil(0.8w)` valid historical observations are required separately for BBr and RVT. The 80% threshold is fixed and untuned. Current `BBr(t)` and `RVT(t)` never enter `mu`, `sigma`, or `Q`.

The predefined stable alternative is `log1p(BULL)-log1p(BEAR)` with the same availability and rolling rules.

## 5. Signal, timing, and execution

A high event is `+1` when `Z(t,w)>z` and `RVT(t)>=Q(t,w,q)`. A low event is `-1` when `Z(t,w)<-z` and the same news filter holds. Otherwise the event is zero.

The primary holding policy lasts exactly one executable return interval. There is no implicit persistence. A three-interval bounded policy is robustness-only.

The complete daily aggregate for `t` is treated as known only after close `t`. A target formed then executes at adjusted open `t+1` and earns:

`AdjustedOpen(t+2)/AdjustedOpen(t+1)-1`.

The result is indexed by return-end date `t+2` and retains linked signal and execution dates. The final two market dates cannot form complete primary return records. No intraday point-in-time claim is made.

A lagged adjusted-close alternative uses the same two-session alignment (enter close `t+1`, exit close `t+2`) and is robustness-only.

## 6. Outer walk-forward experiment

The 602 market dates yield exactly 600 executable return records ending 2018-01-04 through 2020-05-22. The first 252 records are initial history. The remaining 348 records are used exactly once in six non-overlapping outer tests:

| Fold | Return-end dates | Observations |
|---:|---|---:|
| 1 | 2019-01-07 to 2019-03-29 | 58 |
| 2 | 2019-04-01 to 2019-06-21 | 58 |
| 3 | 2019-06-24 to 2019-09-13 | 58 |
| 4 | 2019-09-16 to 2019-12-05 | 58 |
| 5 | 2019-12-06 to 2020-03-02 | 58 |
| 6 | 2020-03-03 to 2020-05-22 | 58 |

For each fold, parameters are fixed before its first executable position. Selection may use only returns ending on or before that first test's signal date, strictly before the first test execution open. Thus the immediately preceding return ending at that execution open is embargoed from selection. Earlier outer-test observations may enter later expanding training sets only after observable; no outer observation can affect its own or an earlier choice.

## 7. Inner validation and locked grid

For each outer fold, take the final 126 eligible training returns and split them chronologically into three consecutive, non-overlapping 42-return validation blocks. No random split or shuffled cross-validation is permitted.

The shared panel-level grid has eight configurations:

- `w ∈ {20, 50}`;
- `z ∈ {1.5, 2.0}`;
- `q ∈ {0.50, 0.75}`.

Primary feature is BBr ratio, no fill, and one-interval holding for every candidate. The grid cannot be expanded after viewing any outer result. The completely fixed no-tuning baseline is `(w=20,z=1.5,q=0.50)`.

Candidate evaluation uses the primary long-only portfolio and 10 bps one-way costs. For each inner block calculate annualized mean net active return versus the daily-rebalanced equal-weight benchmark. The primary selection objective is the median across the three blocks.

Validity requires:

- at least one active portfolio day in every inner block;
- at least 20 pooled active asset-days across the three blocks;
- active asset-day fraction at least 2%;
- mean annualized one-way turnover no greater than 52 times NAV.

Among valid candidates, maximize median active return; then maximize the worst block; then minimize annualized turnover; then prefer higher `z`, higher `q`, and longer `w`, in that order. If no candidate is valid, use the fixed baseline. A no-trade outer fold remains in the stitched result with zero strategy return/cost and undefined active-day hit rate.

Parameters are shared across the entire universe. Neutral and directional variants inherit the long-only-selected parameters and are never separately tuned.

## 8. Portfolio rules

### Primary long-only

Each active positive name receives an equal weight of `min(20%, 1/n_long)`. Weights are nonnegative, per-name exposure is at most 20%, total gross is at most 100%, and the unallocated remainder is cash earning 0%. If no positive event exists, the portfolio is 100% cash.

### Market-neutral long-short robustness

The portfolio is active only when both positive and negative events exist. Each book's gross exposure is:

`min(50%, 20%×n_long, 20%×n_short)`.

Weights are equal inside each book, total gross is at most 100%, and net exposure is exactly zero. If either side is missing, hold cash. No leverage is hidden.

### Directional robustness

Signed events receive equal absolute weights capped at 20% and total gross at 100%. Net exposure may be nonzero; this specification is explicitly directional and never called market-neutral.

## 9. Transaction costs and benchmarks

Costs are one-way target risky-asset weight-change costs:

`cost_t = cost_rate × Σ_i |w(i,t)-w(i,t-1)|`.

Initial entry and final liquidation are charged. A reversal from +x to -x costs `2x×cost_rate`. Primary `cost_rate` is 10 bps. Report 0, 5, 10, 25, 50, and 100 bps; 100 bps is retained as the homework stress case. Financing, borrow fees, taxes, market impact, and unmodeled slippage are excluded and disclosed.

Two frictionless universe benchmarks use the primary return convention:

1. daily-rebalanced equal weight resets all names to 10% each day;
2. static equal-weight buy-and-hold begins at 10% and permits weights to drift.

The strategy is compared net of its costs against the gross benchmarks, a conservative asymmetry for the strategy. Benchmark turnover is reported conceptually but not charged in the confirmatory comparison.

## 10. Metrics

Every return series reports compounded wealth, total return, CAGR using 252 sessions/year, annualized arithmetic mean return, annualized sample volatility, Sharpe with 0% risk-free rate, Sortino, maximum percentage drawdown from wealth, drawdown duration, turnover, gross/net exposure, cash, active days/fraction, asset-level weight-change trade count, active-day hit rate, per-fold results, per-asset gross contribution, and gross/net-of-cost results.

Simple returns are never summed and labeled cumulative return. The homework's class Sharpe is legacy-only.

## 11. Confirmatory inference

The primary OOS sample is the stitched 348-return sequence, not an average of overlapping windows.

- Predictive statistic: daily Spearman rank IC between the continuous selected score and subsequent asset return, requiring at least six valid assets.
- Trading statistic: daily strategy net return minus daily-rebalanced equal-weight return.
- Newey-West/Bartlett uncertainty uses automatic lags `floor(4(T/100)^(2/9))`.
- The two two-sided p-values receive Holm family-wise adjustment at 5%.
- Circular moving-block bootstrap confidence intervals use 10,000 synchronized daily resamples, block length 10, and seeds 20260801 (strategy) / 20260802 (active return).
- Benchmark exposure is an OLS intercept/beta regression with Newey-West covariance. The intercept is not called alpha unless the design and evidence justify it.
- Eight configurations tested per fold and all researcher degrees of freedom are disclosed.

No unverified reality-check-style statistic will be added merely for sophistication.

## 12. Predefined robustness and diagnostics

All are secondary/descriptive unless an amendment promotes one before its result is inspected:

1. Fixed no-tuning baseline.
2. Long-only, balanced neutral, and labeled directional portfolios with identical selected parameters.
3. All six cost levels.
4. No fill versus exactly one-session forward fill; never backfill.
5. Ratio versus `log1p` log-ratio.
6. One versus three explicitly bounded holding intervals.
7. Next adjusted open versus lagged adjusted close.
8. Leave one asset out, rerunning the portfolio/selection without changing the grid.
9. Per-fold, per-asset, signal coverage, score/return sorting, and parameter stability.
10. Pre-crisis versus crisis beginning 2020-02-19; crisis-excluded result.
11. Daily rank IC, forward-return sorting, and signal decay where feasible.
12. Benchmark beta/exposure with HAC errors.

The final narrative cannot be selected from the most favorable variant.

## 13. Failure criteria and claim rules

The empirical run is invalid until fixed if any future-invariance, fold-isolation, timing, weight-reconciliation, cost, compounding, determinism, or schema test fails.

The predictive hypothesis is not confirmed unless mean rank IC is positive and its Holm-adjusted two-sided p-value is below 0.05. The trading hypothesis is not confirmed unless mean net active return is positive and its Holm-adjusted p-value is below 0.05. Economic usefulness also requires nontrivial signal coverage, tolerable cost sensitivity, and no single fold/asset/crisis segment dominating the result. Statistical evidence without economic magnitude, or vice versa, is reported as such.

Negative folds, failed hypotheses, benchmark underperformance, high costs, unstable parameters, and adverse robustness checks remain visible. “Alpha,” “robust,” “profitable,” and “live-trading ready” are prohibited unless directly supported.

## 14. Amendments

After the preregistration commit this file is never rewritten. Any necessary correction is appended to `docs/PREREGISTRATION_AMENDMENTS.md` with date, exact reason, affected artifacts, whether any relevant result was already viewed, and a confirmatory/exploratory label. A change after viewing relevant outer results cannot remain confirmatory.
