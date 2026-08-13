# Limitations

- Daily sentiment aggregates do not include article timestamps. Availability after close is an
  explicit assumption rather than verified point-in-time evidence.
- The fixed ten-equity sample covers 602 sessions from 2018 to 2020 and overlaps the COVID-19
  crisis. It is not a survivorship-safe investable-universe study.
- Adjusted-open returns use retrospective vendor adjustment factors rather than point-in-time
  vintages.
- The primary strategy can hold cash while the benchmark remains fully invested. Active return is
  therefore not an exposure-matched estimate of predictive skill.
- The registered rank-IC statistic uses available continuous scores and is not conditioned on the
  news-intensity event mask.
- The locked design requires at least six valid assets on a rank-IC date but sets no minimum number
  of usable dates. Its economic-usefulness criteria are qualitative rather than numerical.
- Transaction costs are stylized. Market impact, borrow fees, financing, taxes, and other slippage
  are not modeled.
- The short sample, six outer folds, sparse signals, multiple variants, and possible concentration
  in individual assets or regimes limit statistical power and generalizability.
- The original data and source-derived results cannot be redistributed, so the public workflow
  verifies the implementation with synthetic inputs rather than reproducing empirical findings.
