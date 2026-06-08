"""
Fully Bayesian hierarchical hurdle — Tier 4.

Tier 2 pooled each CVE toward a population using *empirical Bayes*: the population
hyperparameters were point estimates (method of moments) and their uncertainty was
ignored. That is why, at the very shortest training windows, the empirical-Bayes
forecast was slightly over-confident — it acted as if the population were known
exactly. The paper's stated next step is a *fully Bayesian* treatment that
propagates hyperparameter uncertainty. This module delivers it, while keeping the
dependency footprint to numpy/scipy (no PyMC/Stan).

The trick that makes this cheap and exact is conjugacy. We keep the same
two-part structure (activity Beta–Binomial, burst-size Gamma–Poisson), and the
per-CVE parameters integrate out in closed form:

* Activity: ``k_c`` active days out of ``n_c``, with ``p_c ~ Beta(a, b)``. The
  marginal over ``p_c`` is Beta–Binomial, so the hyper-likelihood of ``(a, b)`` is
  available analytically.
* Burst: ``S_c`` total sightings over ``A_c`` active days, active-day counts
  ``~ Poisson(lambda_c)`` with ``lambda_c ~ Gamma(kappa, theta)`` (rate
  ``theta``). The marginal over ``lambda_c`` is Negative-Binomial, again giving a
  closed-form hyper-likelihood of ``(kappa, theta)``.

Because the per-CVE parameters are marginalised, only the four hyperparameters
remain, and we sample their joint posterior with a small random-walk Metropolis
sampler (in log-space, with weakly-informative log-Normal priors). Forecasting
then propagates three layers of uncertainty: hyperparameter posterior →
per-CVE conjugate posterior → count sampling. This is what should fix the
short-window over-confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.special import betaln, gammaln  # type: ignore[import-untyped]

_EPS = 1e-9
_PRIOR_SD = 3.0  # log-Normal(0, sd) weakly-informative prior on each log-hyperparameter


def _cve_activity_burst(series: pd.Series) -> tuple[int, int, int, int]:
    """(active_days k, total_days n, total_sightings S, active_days A) for one CVE."""
    y = series.to_numpy(dtype=float)
    n = len(y)
    active = int((y > 0).sum())
    total = int(y.sum())
    return active, n, total, active


def _betabinom_loglik(log_ab: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    """Beta-Binomial marginal log-likelihood of (a,b) over CVEs (binom coef dropped)."""
    a, b = np.exp(log_ab)
    if not (np.isfinite(a) and np.isfinite(b)):
        return -np.inf
    return float(np.sum(betaln(a + k, b + n - k) - betaln(a, b)))


def _gammapois_loglik(log_kt: np.ndarray, s: np.ndarray, a_days: np.ndarray) -> float:
    """Gamma-Poisson (NB) marginal log-likelihood of (kappa,theta) over CVEs."""
    kappa, theta = np.exp(log_kt)
    if not (np.isfinite(kappa) and np.isfinite(theta)):
        return -np.inf
    return float(
        np.sum(
            gammaln(kappa + s)
            - gammaln(kappa)
            - gammaln(s + 1)
            + kappa * (np.log(theta) - np.log(theta + a_days))
            + s * (np.log(a_days + _EPS) - np.log(theta + a_days))
        )
    )


def _log_prior(log_params: np.ndarray) -> float:
    """Weakly-informative N(0, _PRIOR_SD) prior on the log-hyperparameters."""
    return float(-0.5 * np.sum((log_params / _PRIOR_SD) ** 2))


def _metropolis(loglik, x0: np.ndarray, rng: np.random.Generator, n_iter: int, burn: int, step: float) -> np.ndarray:
    """Random-walk Metropolis; returns post-burn-in samples, shape (n_iter-burn, dim)."""
    x = np.asarray(x0, dtype=float)
    lp = loglik(x) + _log_prior(x)
    out = np.empty((n_iter, x.size))
    for i in range(n_iter):
        prop = x + rng.normal(0.0, step, size=x.size)
        lp_prop = loglik(prop) + _log_prior(prop)
        if np.log(rng.random() + _EPS) < lp_prop - lp:
            x, lp = prop, lp_prop
        out[i] = x
    return out[burn:]


@dataclass
class HyperPosterior:
    """Posterior samples of the four population hyperparameters."""

    a: np.ndarray
    b: np.ndarray
    kappa: np.ndarray
    theta: np.ndarray
    n_cves: int

    def __len__(self) -> int:
        return self.a.size


def sample_hyperposterior(
    series_list: list[pd.Series],
    *,
    n_iter: int = 3000,
    burn: int = 1000,
    step: float = 0.4,
    seed: int = 0,
) -> HyperPosterior:
    """Sample the joint posterior of (a,b,kappa,theta) from a corpus of CVE series."""
    stats = [_cve_activity_burst(s) for s in series_list if len(s) > 0]
    k = np.array([st[0] for st in stats], dtype=float)
    n = np.array([st[1] for st in stats], dtype=float)
    s = np.array([st[2] for st in stats if st[3] > 0], dtype=float)
    a_days = np.array([st[3] for st in stats if st[3] > 0], dtype=float)
    rng = np.random.default_rng(seed)

    # Activity (a,b): start from a method-of-moments-ish guess in log-space.
    ab = _metropolis(lambda lp: _betabinom_loglik(lp, k, n), np.array([0.0, 1.0]), rng, n_iter, burn, step)
    # Burst (kappa,theta).
    if s.size:
        kt = _metropolis(lambda lp: _gammapois_loglik(lp, s, a_days), np.array([0.0, 0.0]), rng, n_iter, burn, step)
    else:
        kt = np.zeros((n_iter - burn, 2))

    a, b = np.exp(ab[:, 0]), np.exp(ab[:, 1])
    kappa, theta = np.exp(kt[:, 0]), np.exp(kt[:, 1])
    return HyperPosterior(a=a, b=b, kappa=kappa, theta=theta, n_cves=len(stats))


class BayesianHierarchicalHurdle:
    """Hurdle forecaster that propagates full hyperparameter posterior uncertainty."""

    def __init__(self, hyper: HyperPosterior, seed: int = 0) -> None:
        self.hyper = hyper
        self.name = "bayes_hurdle"
        self._rng = np.random.default_rng(seed)
        self._k = self._n = self._s = self._a = 0

    def fit(self, train: pd.Series) -> "BayesianHierarchicalHurdle":
        self._k, self._n, self._s, self._a = _cve_activity_burst(train)
        return self

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        rng = self._rng
        H = self.hyper
        m = len(H)
        # 1) draw hyperparameters from their posterior (one per Monte-Carlo path)
        idx = rng.integers(0, m, size=n_samples)
        a, b = H.a[idx], H.b[idx]
        kappa, theta = H.kappa[idx], H.theta[idx]
        # 2) per-CVE conjugate posteriors given the (windowed) data
        p = rng.beta(a + self._k, b + (self._n - self._k) + _EPS)
        lam = rng.gamma(shape=kappa + self._s + _EPS, scale=1.0 / (theta + self._a + _EPS))
        # 3) count sampling: activity x (>=1) burst size
        active = rng.random((n_samples, horizon)) < p[:, None]
        excess = np.maximum(lam - 1.0, _EPS)[:, None]
        sizes = 1 + rng.poisson(np.broadcast_to(excess, (n_samples, horizon)))
        return np.where(active, sizes, 0)
