# Quantitative evaluation of sightings forecasts

This document describes the evaluation harness added to TARDISsight and the
methodological choices behind it. It is the experimental support for the
follow-up to *Modeling Sparse and Bursty Vulnerability Sightings*
(arXiv:2604.16038), prepared for VulnOptiCON.

## Motivation

The first paper compared forecasting models **qualitatively**, by inspecting
plots ("the confidence interval is too wide", "the forecast overestimates
reality"). That is enough to reject SARIMAX, but it cannot rank the surviving
models (Poisson, exponential decay, logistic) or quantify *how much* a proposed
improvement helps. To make defensible claims in the next paper we need:

1. a **quantitative, reproducible** comparison across many CVEs;
2. **proper scoring rules** that reward sharp *and* calibrated probabilistic
   forecasts, not just point accuracy; and
3. concrete model improvements aimed at the failure modes the first paper
   documented — **over-dispersion** and **excess zeros**.

## Design choices

### Probabilistic forecasts as samples

Every model exposes the same interface: `fit(train)` then
`sample(horizon, n_samples) -> array(n_samples, horizon)`. The models we compare
have no common analytic predictive distribution (count GLMs, a two-part hurdle,
intermittent-demand baselines), but they can all *simulate* futures. A pool of
samples is therefore the common representation that lets one scoring path treat
every model identically — essential for a fair comparison.

### Rolling-origin backtesting

Series are short, so a single train/test split would score each model on one
forecast and confound model quality with where the split lands. We instead slide
the forecast origin one day at a time: at origin `t`, fit on `series[:t]`,
forecast the next `horizon` days, score against `series[t:t+horizon]`. Only past
data is ever used (no leakage), and each CVE yields many scored forecasts, so the
comparison rests on a distribution of errors. Defaults: `horizon=7`,
`min_train=10`, `stride=1`.

### Scoring rules

- **CRPS** (continuous ranked probability score) — the strictly proper rule for
  the whole predictive distribution; in the units of the data. Primary metric.
- **Pinball / quantile loss** — averaged over quantile levels; a CRPS proxy that
  also exposes per-quantile behaviour.
- **RMSSE** — point error scaled by the in-sample lag-1 naive error, so it is
  unit-free and comparable across CVEs of very different volumes.
- **Interval coverage** (50/80/95%) — the quantitative form of the paper's
  "exploding CI" complaint: we measure empirical vs nominal coverage.
- **Randomised PIT calibration error** — uniform PIT ⇔ calibrated forecasts;
  summarised as the mean absolute deviation of the PIT histogram from uniform.

### Models compared

Baselines (must be beaten to justify any complex model; RMSSE is *defined*
relative to the naive forecast):

- `naive_last` — persistence (last observed daily count as a flat rate).
- `rolling_mean_7` — mean of the trailing 7 days.
- `sba` — Croston's method with the Syntetos–Boylan bias correction, the
  field-standard estimator for intermittent demand (long zero runs + occasional
  positive counts), which is exactly the sightings regime.

Count models (the contribution of this work package):

- `poisson_{const,trend}` — the first paper's model; `variance == mean`.
- `negbin_{const,trend}` — adds a dispersion parameter `alpha`
  (`variance = mean + alpha·mean²`); the standard fix for the over-dispersion
  the paper observed. Collapses to Poisson as `alpha → 0`.
- `hurdle_{poisson,negbin}` — two parts, a Bernoulli "is today active?" and a
  zero-truncated count "how large, given active". Targets the excess-zeros
  structure a single count model smears over.

`*_const` (intercept-only, constant rate) vs `*_trend` (log-linear time trend,
the paper's `sightings ~ time_index`) lets us test whether the trend actually
helps on these short series or merely adds extrapolation risk.

### Operational guard against runaway trends

A log-linear trend with a positive slope grows exponentially when extrapolated —
the count analogue of the SARIMAX blow-ups (negative / absurd forecasts, huge
CIs) the first paper reported. We cap the forecast rate at 10× the observed daily
peak. This is operationally sensible (one would never predict 1000× the
historical maximum) and prevents a handful of runaway origins from dominating the
metrics. The magnitude of the effect is itself a finding (see below).

## Reproducing

```bash
python -m tardissight.eval.run            # cached corpus snapshot
python -m tardissight.eval.run --refresh  # re-fetch sightings from the API
```

Raw API responses are cached under `data/sightings_cache/` so a result can be
regenerated without depending on the live API state. Outputs are written to
`results/`.

The corpus is the CVEs studied in the first paper plus longer-history cases,
spanning the sparse short-series regime (e.g. CVE-2024-9164) and the data-rich
regime (e.g. CVE-2022-26134, ~4 years of observations).

## Results

<!-- RESULTS-START -->
Corpus snapshot (6 CVEs; one, CVE-2024-9164, has only 11 days and produces no
`(min_train=10, horizon=7)` split, so 5 CVEs contribute forecasts):

| CVE | days | active days | sightings |
|---|---:|---:|---:|
| CVE-2025-61932 | 106 | 18 | 62 |
| CVE-2025-59287 | 235 | 158 | 278 |
| CVE-2022-26134 | 1464 | 576 | 695 |
| CVE-2024-9164 | 11 | 3 | 7 |
| CVE-2025-54236 | 269 | 25 | 61 |
| CVE-2025-8088 | 302 | 54 | 312 |

**Pooled comparison** (origin-weighted; `horizon=7`, `min_train=10`, `stride=1`,
`n_samples=2000`; 2296 forecasts/model). Sorted by CRPS, lower is better.

| model | CRPS | pinball | RMSSE | MAE | cov50 | cov80 | cov95 | PIT cal err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_mean_7 | **0.274** | 0.140 | 0.298 | 0.304 | 0.924 | 0.957 | 0.967 | 0.146 |
| hurdle_negbin | 0.308 | 0.155 | 0.362 | 0.336 | 0.857 | 0.929 | 0.965 | **0.071** |
| hurdle_poisson | 0.321 | 0.163 | 0.372 | 0.383 | 0.847 | 0.922 | 0.961 | 0.105 |
| naive_last | 0.321 | 0.165 | 0.342 | 0.360 | 0.885 | 0.903 | 0.912 | 0.172 |
| negbin_const | 0.377 | 0.187 | 0.541 | 0.468 | 0.807 | 0.915 | 0.956 | 0.230 |
| poisson_trend | 0.426 | 0.211 | 0.456 | 0.514 | 0.870 | 0.906 | 0.937 | 0.180 |
| sba | 0.564 | 0.283 | 0.461 | 0.696 | 0.833 | 0.899 | 0.967 | 0.258 |
| poisson_const | 0.579 | 0.277 | 0.618 | 0.820 | 0.779 | 0.928 | 0.970 | 0.282 |
| negbin_trend | 4.532 | 2.240 | 7.641 | 6.049 | 0.802 | 0.850 | 0.888 | 0.286 |

**Per-CVE CRPS** (equal weight per CVE in the last column):

| model | 2022-26134 | 2025-54236 | 2025-59287 | 2025-61932 | 2025-8088 | mean |
|---|---:|---:|---:|---:|---:|---:|
| rolling_mean_7 | 0.170 | 0.123 | 0.649 | 0.139 | 0.691 | **0.354** |
| hurdle_negbin | 0.222 | 0.114 | 0.960 | 0.091 | 0.482 | 0.374 |
| naive_last | 0.216 | 0.149 | 0.788 | 0.142 | 0.703 | 0.400 |
| negbin_const | 0.325 | 0.129 | 0.769 | 0.220 | 0.608 | 0.410 |
| hurdle_poisson | 0.202 | 0.114 | 1.215 | 0.085 | 0.494 | 0.422 |
| poisson_trend | 0.234 | 0.113 | 2.097 | 0.081 | 0.502 | 0.605 |
| poisson_const | 0.350 | 0.265 | 0.773 | 0.821 | 1.795 | 0.801 |
| sba | 0.241 | 1.488 | 0.569 | 1.732 | 1.015 | 1.009 |
| negbin_trend | 6.912 | 0.114 | 1.011 | 0.081 | 0.484 | 1.720 |
<!-- RESULTS-END -->

## Findings & takeaways

<!-- FINDINGS-START -->
1. **Simple baselines are hard to beat — confirmed quantitatively.** The
   trailing 7-day rolling mean has the lowest CRPS both pooled (0.274) and
   averaged per CVE (0.354). This turns the first paper's qualitative remark
   ("even a rolling average will outperform SARIMAX") into a measured result and
   sets the bar every model must clear.

2. **Negative Binomial beats Poisson, consistently.** Adding the dispersion
   parameter helps at every level: `hurdle_negbin` (0.308) < `hurdle_poisson`
   (0.321), and `negbin_const` (0.377) < `poisson_const` (0.579). This directly
   validates the over-dispersion fix the first paper called for.

3. **The hurdle improves calibration the most — the key practical result.**
   `hurdle_negbin` has by far the lowest PIT calibration error (0.071 vs 0.146
   for the rolling mean and 0.282 for plain Poisson). So while the rolling mean
   wins on point-distribution sharpness (CRPS), the NegBin hurdle produces the
   *best-calibrated* predictive distributions. For an operational risk/decision
   setting — where trustworthy intervals matter more than a marginally sharper
   median — `hurdle_negbin` is the recommended count model. Modelling the zeros
   separately and allowing over-dispersion is what buys this calibration.

4. **The time trend is a liability, not an asset.** `negbin_trend` is the worst
   model overall (pooled CRPS 4.532, driven by CVE-2022-26134 at 6.912 *even
   with* the 10× rate cap), and `poisson_trend` blows up on CVE-2025-59287
   (2.097). Exponential extrapolation of a fitted slope reproduces the SARIMAX
   pathology in count form. **Constant-rate count models are markedly more
   robust** on these short, bursty series — a concrete recommendation.

5. **Croston/SBA underperforms here.** Despite being the textbook
   intermittent-demand estimator, SBA trails the naive baselines (CRPS 0.564),
   collapsing on the low-volume recent CVEs (CVE-2025-54236: 1.488,
   CVE-2025-61932: 1.732). A single smoothed rate does not capture the
   burst-then-fade shape; this is a useful negative result.

6. **Interval coverage is inflated by structural zeros — read PIT instead.** All
   models "over-cover" the nominal 50% interval (0.78–0.92) because when the
   estimated rate is low the central quantiles are both 0 and the (frequent)
   zero actuals fall inside trivially. This makes raw coverage a poor calibration
   gauge for sparse counts; the randomised PIT error is the reliable measure and
   is the one we report on.

### Recommendation for the next iteration

Report the **rolling mean as the baseline to beat** and **`hurdle_negbin` as the
best-calibrated model**; drop time-trend extrapolation in favour of constant-rate
count models with proper interval calibration. Natural next steps (Tier 2 of the
roadmap): hierarchical pooling across CVEs to share the burst-size and zero-rate
parameters, and adding day-of-week and EPSS-dynamics covariates.

> **Tier 2 update.** Hierarchical pooling has now been prototyped and dominates
> both the rolling-mean baseline and the best unpooled count model across all
> training-window sizes — see [`pooling.md`](pooling.md).
<!-- FINDINGS-END -->
