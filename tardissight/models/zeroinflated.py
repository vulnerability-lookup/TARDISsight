"""
Hierarchical Zero-Inflated Negative Binomial (ZINB) — Tier 5.

Tier 4 diagnosed that the binding problem in this regime is *excess dispersion*:
every model already over-covers, so the productive direction is a tighter
likelihood, not more inference machinery. The hurdle model of Tiers 2-3 treats
zeros and positives with two entirely separate processes; its positive part is a
heavy-tailed ``1 + NB`` whose pooled dispersion (alpha ~ 4.5) is what widens the
intervals. A *zero-inflated* model is the natural alternative: a single count
distribution generates the data, and an extra ``structural-zero'' gate absorbs the
excess zeros. Crucially, under ZINB the *same* NB component explains both the
small positive counts and (some of) the zeros, so its mean and dispersion are
estimated from more of the data and need not inflate to fit the bursts.

Model. For CVE ``c`` a daily count is

    y = 0                with probability  pi_c           (structural zero)
    y ~ NB2(mu_c, r)     with probability  1 - pi_c        (NB may also yield 0)

so P(y=0) = pi_c + (1-pi_c) * NB(0; mu_c, r) and, for k>0,
P(y=k) = (1-pi_c) * NB(k; mu_c, r). We pool across CVEs exactly as before:

    pi_c ~ Beta(a, b),     mu_c ~ Gamma(kappa, theta),     r shared (pooled).

Estimation is empirical Bayes. Unlike the hurdle, ZINB does not factorise (a zero
may come from either component), so there is no closed-form conjugate update.
Instead we fit ``(pi_c, mu_c)`` by **MAP expectation-maximisation**, with the
population Beta/Gamma priors entering as pseudo-counts -- which is exactly what
produces the shrinkage: a short window leans on the prior, a long one on its own
data. The shared dispersion ``r`` is fit once on the back-catalogue. This keeps
the numpy/scipy-only footprint and stays directly comparable to the hurdle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from scipy.special import gammaln  # type: ignore[import-untyped]

_EPS = 1e-9


def _nb_logpmf(k: np.ndarray, mu: float, r: float) -> np.ndarray:
    """Log PMF of NB2 (mean mu, size r): var = mu + mu^2/r."""
    mu = max(mu, _EPS)
    return (
        gammaln(k + r)
        - gammaln(r)
        - gammaln(k + 1)
        + r * (np.log(r) - np.log(r + mu))
        + k * (np.log(mu) - np.log(r + mu))
    )


def _nb_p0(mu: float, r: float) -> float:
    """NB2 probability of a zero, (r/(r+mu))^r."""
    mu = max(mu, _EPS)
    return float((r / (r + mu)) ** r)


def _zinb_loglik(y: np.ndarray, pi: float, mu: float, r: float) -> float:
    """Total ZINB log-likelihood of a count vector."""
    pi = min(max(pi, _EPS), 1 - _EPS)
    p0 = _nb_p0(mu, r)
    is_zero = y == 0
    ll0 = np.log(pi + (1 - pi) * p0 + _EPS)
    llk = np.log1p(-pi) + _nb_logpmf(y, mu, r)
    return float(ll0 * is_zero.sum() + llk[~is_zero].sum()) if is_zero.any() else float(llk.sum())


def fit_zinb_em(
    y: np.ndarray,
    r: float,
    *,
    a: float = 1.0,
    b: float = 1.0,
    kappa: float = 0.0,
    theta: float = 0.0,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """MAP-EM for (pi, mu) given fixed dispersion r and Beta(a,b)/Gamma(kappa,theta) priors.

    The priors act as pseudo-counts, so with the population hyperparameters this
    *is* the empirical-Bayes shrinkage. With weak priors (kappa=theta=0, a=b=1) it
    reduces to the plain MLE used to characterise the population.
    """
    n = y.size
    if n == 0:
        return a / (a + b), (kappa / theta if theta > 0 else 1.0)

    mean_y = float(y.mean())
    pi = float((y == 0).mean()) * 0.5  # init: half the zeros are structural
    mu = max(mean_y / (1 - pi + _EPS), _EPS)
    is_zero = y == 0

    prev = -np.inf
    for _ in range(n_iter):
        # E-step: responsibility that each zero is a *structural* zero.
        p0 = _nb_p0(mu, r)
        gamma = np.where(is_zero, pi / (pi + (1 - pi) * p0 + _EPS), 0.0)

        # M-step (MAP). Beta pseudo-counts for pi; Gamma pseudo-counts for mu.
        struct = gamma.sum()
        pi = (struct + a) / (n + a + b)
        w = 1.0 - gamma  # weight each obs by P(it came from the NB component)
        mu = (float((w * y).sum()) + kappa) / (float(w.sum()) + theta + _EPS)
        pi = min(max(pi, _EPS), 1 - _EPS)
        mu = max(mu, _EPS)

        ll = _zinb_loglik(y, pi, mu, r)
        if abs(ll - prev) < tol:
            break
        prev = ll
    return pi, mu


@dataclass
class ZINBPrior:
    """Population hyperparameters for the hierarchical ZINB."""

    beta_a: float
    beta_b: float
    gamma_shape: float
    gamma_rate: float
    r: float
    n_cves: int

    @property
    def mean_pi(self) -> float:
        return self.beta_a / (self.beta_a + self.beta_b)

    @property
    def mean_mu(self) -> float:
        return self.gamma_shape / self.gamma_rate


def _fit_shared_r(ys: list[np.ndarray], params: list[tuple[float, float]]) -> float:
    """Optimise the shared dispersion r over the corpus given per-CVE (pi, mu)."""

    def neg_ll(log_r: float) -> float:
        r = float(np.exp(log_r))
        return -sum(_zinb_loglik(y, pi, mu, r) for y, (pi, mu) in zip(ys, params))

    res = minimize_scalar(neg_ll, bounds=(np.log(0.05), np.log(200.0)), method="bounded")
    return float(np.exp(res.x))


def fit_zinb_population(series_list: list[pd.Series], *, outer: int = 2) -> ZINBPrior:
    """Empirical-Bayes population fit: per-CVE MLE (pi, mu), shared r, then moment-match priors."""
    ys = [s.to_numpy(dtype=float) for s in series_list if len(s) > 0]
    if not ys:
        return ZINBPrior(1.0, 1.0, 1.0, 1.0, 2.0, 0)

    r = 2.0
    params = [(0.5, 1.0)] * len(ys)
    for _ in range(outer):
        params = [fit_zinb_em(y, r) for y in ys]  # weak priors -> MLE
        r = _fit_shared_r(ys, params)

    pis = np.array([p[0] for p in params])
    mus = np.array([p[1] for p in params])

    # Beta(a,b) for pi by method of moments.
    m, v = pis.mean(), pis.var()
    if pis.size >= 2 and 0 < v < m * (1 - m):
        common = m * (1 - m) / v - 1.0
        beta_a, beta_b = m * common, (1 - m) * common
    else:
        m = min(max(m, _EPS), 1 - _EPS)
        beta_a, beta_b = m, 1 - m

    # Gamma(shape, rate) for mu by method of moments.
    M, V = mus.mean(), mus.var()
    if mus.size >= 2 and V > 0:
        gamma_shape, gamma_rate = M * M / V, M / V
    else:
        gamma_shape, gamma_rate = 1.0, 1.0 / max(M, _EPS)

    return ZINBPrior(float(beta_a), float(beta_b), float(gamma_shape), float(gamma_rate), float(r), len(ys))


class HierarchicalZINB:
    """Zero-Inflated NB forecaster with empirical-Bayes shrinkage toward a population prior."""

    def __init__(self, prior: ZINBPrior, seed: int = 0) -> None:
        self.prior = prior
        self.name = "zinb_hier"
        self._rng = np.random.default_rng(seed)
        self._pi = prior.mean_pi
        self._mu = prior.mean_mu

    def fit(self, train: pd.Series) -> "HierarchicalZINB":
        y = train.to_numpy(dtype=float)
        pr = self.prior
        # MAP-EM with the population prior as pseudo-counts (the shrinkage step).
        self._pi, self._mu = fit_zinb_em(
            y, pr.r, a=pr.beta_a, b=pr.beta_b, kappa=pr.gamma_shape, theta=pr.gamma_rate
        )
        return self

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        rng = self._rng
        r, mu = self.prior.r, self._mu
        structural = rng.random((n_samples, horizon)) < self._pi
        p = r / (r + max(mu, _EPS))
        counts = rng.negative_binomial(r, p, size=(n_samples, horizon))
        return np.where(structural, 0, counts)
