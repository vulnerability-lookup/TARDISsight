"""
Count-regression forecasters: Poisson, Negative Binomial, and Hurdle.

Motivation (the core of this work package):

* **Poisson** is the model the first paper settled on. It guarantees
  non-negative integer forecasts but assumes ``variance == mean``. The paper
  explicitly reports observing both over- and under-dispersion, which violates
  that assumption and is exactly what we want to fix and *measure*.

* **Negative Binomial (NB2)** adds a dispersion parameter ``alpha`` so
  ``variance = mean + alpha * mean^2``. This is the standard remedy for the
  overdispersion the paper documented; with ``alpha -> 0`` it collapses back to
  Poisson, so it can only help (at the cost of one parameter).

* **Hurdle** is a two-part model: a Bernoulli component for "is there any
  activity today?" and a zero-truncated count component for "how much, given
  there is some". Sightings are mostly zeros with occasional bursts, so the
  process generating zeros may differ from the one generating burst sizes. The
  hurdle lets those be modelled separately — directly targeting the excess-zeros
  problem a plain Poisson smears over.

Each model fits a log-linear time trend (``log mu = b0 + b1 * t``), mirroring the
paper's ``sightings ~ time_index`` specification, and falls back to an
intercept-only (constant-rate) fit when the series is too short or degenerate to
estimate a slope — the short-series fragility the paper warned about.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import statsmodels.api as sm  # type: ignore[import-untyped]

_EPS = 1e-6
_MAX_RATE = 1e6  # clamp exp() of an extrapolated trend so a steep slope can't overflow
_MAX_ALPHA = 1e6  # bound NB dispersion so r = 1/alpha stays strictly positive and finite
_LOG_MAX_RATE = float(np.log(_MAX_RATE))
_LOG_EPS = float(np.log(_EPS))


def _design(n: int, trend: bool) -> np.ndarray:
    """Design matrix: intercept, optionally a normalised time index."""
    if not trend:
        return np.ones((n, 1))
    t = np.arange(n, dtype=float)
    return np.column_stack([np.ones(n), t])


def _future_design(n_train: int, horizon: int, trend: bool) -> np.ndarray:
    if not trend:
        return np.ones((horizon, 1))
    t = np.arange(n_train, n_train + horizon, dtype=float)
    return np.column_stack([np.ones(horizon), t])


def _nb_samples(mu: np.ndarray, alpha: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample NB2 counts via the gamma-Poisson mixture.

    NB2 mean ``mu``, variance ``mu + alpha*mu^2``. In numpy's ``(r, p)``
    parameterisation ``r = 1/alpha`` and ``p = r / (r + mu)``.
    """
    mu = np.clip(mu, _EPS, _MAX_RATE)
    if not np.isfinite(alpha) or alpha <= _EPS:
        return rng.poisson(mu, size=(n_samples, mu.shape[0]))
    alpha = min(alpha, _MAX_ALPHA)
    r = 1.0 / alpha
    p = r / (r + mu)
    return rng.negative_binomial(r, p, size=(n_samples, mu.shape[0]))


class CountGLM:
    """Poisson or Negative-Binomial GLM with a log-linear time trend."""

    def __init__(self, family: str = "poisson", trend: bool = True, seed: int = 0) -> None:
        if family not in ("poisson", "negbin"):
            raise ValueError(f"unknown family {family!r}")
        self.family = family
        self.trend = trend
        self.name = f"{family}{'_trend' if trend else '_const'}"
        self._rng = np.random.default_rng(seed)
        self._coef: np.ndarray | None = None
        self._alpha = 0.0
        self._n_train = 0
        self._trend_used = trend
        self._rate_cap = _MAX_RATE

    def fit(self, train: pd.Series) -> "CountGLM":
        y = train.to_numpy(dtype=float)
        self._n_train = len(y)

        # Data-driven ceiling on the forecast rate. A log-linear trend with a
        # positive slope grows exponentially when extrapolated, which can produce
        # absurd forecasts (e.g. 10^6 sightings/day) — the count-model analogue
        # of the SARIMAX blow-ups the first paper reported. Capping at 10x the
        # observed daily peak is an operationally sensible guard and keeps the
        # trend variants from dominating the metrics with a few runaway origins.
        self._rate_cap = max(float(y.max()) * 10.0, 5.0) if self._n_train else _MAX_RATE

        # Degenerate cases: no data or all-zero history -> constant zero rate.
        if self._n_train == 0 or y.sum() == 0:
            self._coef = np.array([np.log(_EPS)])
            self._trend_used = False
            self._alpha = 0.0
            return self

        # Too few points to estimate a slope reliably -> intercept only.
        self._trend_used = self.trend and self._n_train >= 4
        X = _design(self._n_train, self._trend_used)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if self.family == "poisson":
                    res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                    self._coef = np.asarray(res.params, dtype=float)
                    self._alpha = 0.0
                else:
                    res = sm.NegativeBinomial(y, X).fit(disp=0, maxiter=200)
                    params = np.asarray(res.params, dtype=float)
                    # statsmodels appends ln(alpha) as the last parameter.
                    self._coef = params[:-1]
                    self._alpha = float(np.exp(params[-1]))
            except Exception:
                # Fall back to a moment-based constant rate; for NB also estimate
                # alpha from the sample mean/variance so dispersion isn't lost.
                self._coef = np.array([np.log(max(y.mean(), _EPS))])
                self._trend_used = False
                if self.family == "negbin":
                    m, v = y.mean(), y.var()
                    self._alpha = float(max((v - m) / max(m * m, _EPS), 0.0))
                else:
                    self._alpha = 0.0
        return self

    def _mu(self, horizon: int) -> np.ndarray:
        assert self._coef is not None
        X = _future_design(self._n_train, horizon, self._trend_used)
        eta = np.clip(X @ self._coef, _LOG_EPS, _LOG_MAX_RATE)
        return np.clip(np.exp(eta), _EPS, self._rate_cap)

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        return _nb_samples(self._mu(horizon), self._alpha, n_samples, self._rng)


class Hurdle:
    """Two-part hurdle model: P(active) x zero-truncated count size."""

    def __init__(self, family: str = "poisson", trend: bool = True, seed: int = 0) -> None:
        if family not in ("poisson", "negbin"):
            raise ValueError(f"unknown family {family!r}")
        self.family = family
        self.trend = trend
        self.name = f"hurdle_{family}"
        self._rng = np.random.default_rng(seed)
        self._count = CountGLM(family=family, trend=trend, seed=seed)
        self._p_coef: np.ndarray | None = None
        self._p_trend = trend
        self._n_train = 0

    def fit(self, train: pd.Series) -> "Hurdle":
        y = train.to_numpy(dtype=float)
        self._n_train = len(y)
        active = (y > 0).astype(float)

        # Bernoulli (activity) part. Needs both classes present to fit a logit;
        # otherwise use the empirical activity rate as a constant probability.
        self._p_trend = self.trend and self._n_train >= 4 and 0 < active.sum() < self._n_train
        if self._p_trend:
            X = _design(self._n_train, True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = sm.GLM(active, X, family=sm.families.Binomial()).fit()
                    self._p_coef = np.asarray(res.params, dtype=float)
                except Exception:
                    self._p_trend = False
        if not self._p_trend:
            rate = active.mean() if self._n_train else 0.0
            # store as logit-intercept so _p() is uniform
            rate = min(max(rate, _EPS), 1 - _EPS)
            self._p_coef = np.array([np.log(rate / (1 - rate))])

        # Count (size) part: fit on positive days only, so it models burst size
        # conditional on activity rather than being diluted by structural zeros.
        positive = train[train > 0]
        if len(positive) >= 1:
            self._count.fit(positive)
        else:
            self._count.fit(train)  # degenerate; yields ~zero rate
        return self

    def _p(self, horizon: int) -> np.ndarray:
        assert self._p_coef is not None
        X = _future_design(self._n_train, horizon, self._p_trend)
        logits = np.clip(X @ self._p_coef, -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        p = self._p(horizon)
        active = self._rng.random((n_samples, horizon)) < p

        # Zero-truncated count sizes: resample any zeros from the count part so
        # the size component is strictly positive (hurdle assumption).
        sizes = self._count.sample(horizon, n_samples)
        zeros = sizes == 0
        for _ in range(5):
            if not zeros.any():
                break
            sizes[zeros] = self._count.sample(horizon, n_samples)[zeros]
            zeros = sizes == 0
        sizes[zeros] = 1  # give up after a few tries; floor at 1

        return np.where(active, sizes, 0)
