# Study 4: economic value of the forecasts. Pre-registration

Written on 19 October 2026, before any run.

## Question

Studies 1 and 3 found that the lab's best macro forecasts are hard to tell apart from a
univariate AR. This study asks the economic version of the question: do those forecasts
predict next-quarter excess returns on US equities, Treasuries and corporate bonds well
enough to be worth anything to an investor, following Welch and Goyal (2008) and Campbell
and Thompson (2008)?

## Design

- **Forecasts.** Study 1's eligible winner, CORE+OIL(x): a VARX in GDP growth, core PCE
  inflation, the fed funds rate and the real broad dollar, with WTI oil returns exogenous
  and AR-projected, lags by BIC up to 4. The benchmark is a univariate AR(BIC) for each of
  the four core variables. Both are re-estimated every quarter on ALFRED real-time
  vintages, reusing the vintage code and cache of Study 3. At a decision at the end of
  quarter t the vintage used is the one dated 45 days after quarter t-1 closed (Study 3's
  standpoint), so the data run to quarter t-1, and the one-quarter-ahead forecast is for
  quarter t. Every input series is taken at its vintage; the sample starts at the earliest
  date for which all five series have one, and no gap is filled with revised or final data.
- **Predictors.** The four core forecasts made at t (GDP growth, inflation, fed funds,
  dollar growth). Each model gives its own set.
- **Returns.** Bloomberg monthly total-return levels for the S&P 500 (SPXT), the Bloomberg
  US Treasury index (LUATTRUU) and the US corporate index (LUACTRUU), compounded to
  quarters. Excess returns are over the 3-month T-bill (FRED TB3MS, quarterly mean at the
  start of the quarter, de-annualised).
- **Predictive regression.** For each asset, an expanding-window OLS of the excess return
  in quarter s+1 on the forecasts made at s, using only pairs completed by the decision
  date; the first out-of-sample prediction needs 40 quarters. The benchmark forecast is the
  historical mean of excess returns to date.
- **Investor.** Mean-variance with risk aversion 3, weight on the asset in [0, 1.5], the
  rest in T-bills, variance from the trailing 40 quarters of excess returns. The same rule
  is run with the historical mean as the expected return. The certainty-equivalent gain is
  the difference in realised mean minus 1.5 x variance of the portfolio return, annualised
  (x4). No parameter is tuned.

## Hypotheses

- **H1.** The out-of-sample R-squared of CORE+OIL(x) excess-return forecasts against the
  historical mean is positive. Expected: no.
- **H2.** The certainty-equivalent gain of the CORE+OIL(x) investor over the historical-mean
  investor is positive. Expected: no.
- **H3.** CORE+OIL(x) beats the AR benchmark on both. Expected: no.

## Tests

- Clark-West test of each model against the historical mean, per asset: 2 x 3 = 6.
- Diebold-Mariano test of CORE+OIL(x) against the AR on squared forecast errors, per asset: 3.

Nine tests. Bonferroni: p < 0.05 / 9 = 0.0056. Certainty-equivalent gains are reported
as point estimates, as in Campbell and Thompson, and are not counted as tests.

## Amendment, 19 October 2026 (before any run)

The first ALFRED vintages are GDPC1 1991-12, FEDFUNDS 1996-12, PCEPILFE 2000-08,
RBUSBIS 2014-02 and WTISPLC 2016-10. Taking every series at its vintage would start the
sample in late 2016 and leave no out-of-sample period after the 40-quarter minimum, so,
following Study 3:

- **GDP and core PCE** are taken at their ALFRED vintages, as above.
- **The fed funds rate and WTI oil** are taken at final values. They are market prices and
  are not revised.
- **The BIS real broad dollar index** is taken at final values. It is lightly revised, so
  this is a small departure from a pure real-time information set.

Each final series is still cut at quarter t-1 at a decision at the end of quarter t. The
common start is 2000Q3, set by core PCE's first vintage, and the first out-of-sample
prediction comes about 40 quarters later, around 2010Q3. No other part of the design
changes.
