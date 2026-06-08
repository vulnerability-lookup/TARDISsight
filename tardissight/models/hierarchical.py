"""
Hierarchical (partially pooled) hurdle model — Tier 2 prototype.

Tier 1 established that the *hurdle Negative-Binomial* is the best-calibrated
count model: split each day into "is there activity?" (Bernoulli) and "how large
is the burst, given activity?" (over-dispersed count). But Tier 1 fit each CVE in
isolation, and the first paper's central pain point is that a single CVE has very
little data (10-30 days). With so few observations the per-CVE activity rate and
burst size are estimated badly, and forecasts are unstable.

**Partial pooling** attacks this at the root. Instead of an independent fit per
CVE, we assume each CVE's parameters are drawn from a *population* distribution
shared across CVEs:

    activity rate    p_c   ~ Beta(a, b)
    burst-size rate  lam_c ~ Gamma(shape, rate)

The population hyperparameters are estimated once from a back-catalogue of past
CVEs. A new CVE with little data is then *shrunk toward the population mean*; a
CVE with a long history keeps its own estimate. This is exactly the leverage a
single-series model cannot have, and it is what should help the short-series
regime the most.

This prototype uses **empirical Bayes** (method-of-moments hyperparameters +
conjugate posterior means) rather than full MCMC. That keeps the dependency
footprint to numpy/scipy and runs instantly, which is the point of a prototype:
validate that pooling helps before investing in a full Bayesian (PyMC/NumPyro)
treatment with proper hyperparameter uncertainty.

Conjugacy used:
* Beta-Binomial: posterior mean of p_c = (k_c + a) / (n_c + a + b), with k_c
  active days out of n_c.
* Gamma-Poisson: posterior mean of lam_c = (shape + S_c) / (rate + A_c), with
  S_c total sightings over A_c active days. (We treat active-day counts as
  Poisson for the rate shrinkage; the marginal over-dispersion is handled
  separately by a pooled NB dispersion when sampling burst sizes. This is an
  acknowledged prototype approximation.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .count import _nb_samples

_EPS = 1e-6


@dataclass
class PopulationPrior:
    """Population-level hyperparameters shared across CVEs."""

    beta_a: float       # activity Beta prior
    beta_b: float
    gamma_shape: float  # burst-rate Gamma prior
    gamma_rate: float
    nb_alpha: float     # pooled over-dispersion of burst sizes
    n_cves: int         # how many CVEs the prior was estimated from

    @property
    def mean_activity(self) -> float:
        return self.beta_a / (self.beta_a + self.beta_b)

    @property
    def mean_burst_rate(self) -> float:
        return self.gamma_shape / self.gamma_rate


def _cve_stats(series: pd.Series) -> tuple[int, int, int, float]:
    """Return (n_days, active_days, total_sightings, mean_burst_rate) for a CVE."""
    y = series.to_numpy(dtype=float)
    n = len(y)
    active = int((y > 0).sum())
    total = float(y.sum())
    rate = total / active if active else 0.0  # mean count on active days (>= 1)
    return n, active, int(total), rate


def fit_population_prior(series_list: list[pd.Series]) -> PopulationPrior:
    """Estimate population hyperparameters from a set of CVE series (empirical Bayes).

    Activity Beta and burst-rate Gamma are fit by method of moments across the
    per-CVE empirical rates; the burst-size dispersion is a single pooled NB
    ``alpha`` from all positive daily counts. Falls back to weak/diffuse values
    when the moments are degenerate (e.g. too few CVEs, zero variance).
    """
    activity_rates = []
    burst_rates = []
    positive_counts: list[np.ndarray] = []

    for s in series_list:
        n, active, _total, rate = _cve_stats(s)
        if n == 0:
            continue
        activity_rates.append(active / n)
        if active > 0:
            burst_rates.append(rate)
            y = s.to_numpy(dtype=float)
            positive_counts.append(y[y > 0])

    n_cves = len(activity_rates)

    # --- Activity Beta(a, b) by method of moments ---
    p = np.asarray(activity_rates, dtype=float)
    m, v = (p.mean(), p.var()) if p.size else (0.5, 0.0)
    if p.size >= 2 and 0 < v < m * (1 - m):
        common = m * (1 - m) / v - 1.0
        beta_a, beta_b = m * common, (1 - m) * common
    else:
        # Diffuse but mean-centred prior worth ~1 pseudo-observation.
        m = min(max(m, _EPS), 1 - _EPS)
        beta_a, beta_b = m, 1 - m

    # --- Burst-rate Gamma(shape, rate) by method of moments ---
    lam = np.asarray(burst_rates, dtype=float)
    M, V = (lam.mean(), lam.var()) if lam.size else (1.0, 0.0)
    if lam.size >= 2 and V > 0:
        gamma_shape = M * M / V
        gamma_rate = M / V
    else:
        # Exponential-ish prior with the right mean, worth ~1 pseudo active day.
        gamma_shape = 1.0
        gamma_rate = 1.0 / max(M, _EPS)

    # --- Pooled burst-size over-dispersion (NB alpha) ---
    if positive_counts:
        allc = np.concatenate(positive_counts)
        mp, vp = allc.mean(), allc.var()
        nb_alpha = float(max((vp - mp) / max(mp * mp, _EPS), 0.0))
    else:
        nb_alpha = 0.0

    return PopulationPrior(
        beta_a=float(beta_a),
        beta_b=float(beta_b),
        gamma_shape=float(gamma_shape),
        gamma_rate=float(gamma_rate),
        nb_alpha=nb_alpha,
        n_cves=n_cves,
    )


class HierarchicalHurdle:
    """Hurdle forecaster whose per-CVE parameters are shrunk toward a population prior."""

    def __init__(self, prior: PopulationPrior, seed: int = 0) -> None:
        self.prior = prior
        self.name = "hier_hurdle"
        self._rng = np.random.default_rng(seed)
        self._p = prior.mean_activity
        self._lam = prior.mean_burst_rate

    def fit(self, train: pd.Series) -> "HierarchicalHurdle":
        n, active, total, _rate = _cve_stats(train)
        pr = self.prior
        # Conjugate posterior means: data pulls away from the prior in proportion
        # to how much of it there is. Short series stay near the population mean.
        self._p = (active + pr.beta_a) / (n + pr.beta_a + pr.beta_b) if n else pr.mean_activity
        self._lam = (total + pr.gamma_shape) / (active + pr.gamma_rate) if active else pr.mean_burst_rate
        return self

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        active = self._rng.random((n_samples, horizon)) < self._p
        # Burst size given activity is >= 1; model it as 1 + NB(mean=lam-1) so it
        # is strictly positive without rejection sampling. mean(size) == lam.
        excess_mean = np.full(horizon, max(self._lam - 1.0, _EPS))
        sizes = 1 + _nb_samples(excess_mean, self.prior.nb_alpha, n_samples, self._rng)
        return np.where(active, sizes, 0)
