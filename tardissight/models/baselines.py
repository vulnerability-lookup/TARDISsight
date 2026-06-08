"""
Simple, hard-to-beat baselines.

A central methodological claim of the first paper is that "for very short
windows, even a rolling average or exponential decay will outperform SARIMAX".
The evaluation harness must therefore include these cheap baselines explicitly:
a sophisticated model is only worth its complexity if it beats them on the
scoring rules. RMSSE in particular is *defined* relative to the naive forecast,
so a naive baseline is not optional — it is the denominator.

All baselines turn a point rate into Poisson-distributed samples so they produce
honest probabilistic forecasts comparable to the count models. A constant-rate
Poisson is itself a legitimate (if weak) model of the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

_EPS = 1e-6


def _poisson_samples(rate: float, horizon: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Poisson draws with a flat (time-invariant) rate over the horizon."""
    rate = max(float(rate), _EPS)
    return rng.poisson(rate, size=(n_samples, horizon))


class NaiveLast:
    """Persistence: future rate = last observed daily count."""

    name = "naive_last"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._rate = 0.0

    def fit(self, train: pd.Series) -> "NaiveLast":
        self._rate = float(train.iloc[-1]) if len(train) else 0.0
        return self

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        return _poisson_samples(self._rate, horizon, n_samples, self._rng)


class RollingMean:
    """Mean daily count over the trailing ``window`` days."""

    def __init__(self, window: int = 7, seed: int = 0) -> None:
        self.window = window
        self.name = f"rolling_mean_{window}"
        self._rng = np.random.default_rng(seed)
        self._rate = 0.0

    def fit(self, train: pd.Series) -> "RollingMean":
        self._rate = float(train.iloc[-self.window :].mean()) if len(train) else 0.0
        return self

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        return _poisson_samples(self._rate, horizon, n_samples, self._rng)


class Croston:
    """Croston's method for intermittent demand (and the SBA bias correction).

    Sightings are textbook *intermittent demand*: long runs of zeros punctuated
    by occasional positive counts. Croston (1972) is the field-standard estimator
    for exactly this regime — it smooths the non-zero demand sizes and the
    inter-arrival gaps separately, then forecasts rate = size / interval. The
    Syntetos-Boylan Approximation (``sba=True``) multiplies by ``1 - alpha/2`` to
    remove Croston's known positive bias. Including it tests whether the
    intermittent-demand literature beats the count-GLM approach the paper took.
    """

    def __init__(self, alpha: float = 0.1, sba: bool = True, seed: int = 0) -> None:
        self.alpha = alpha
        self.sba = sba
        self.name = "sba" if sba else "croston"
        self._rng = np.random.default_rng(seed)
        self._rate = 0.0

    def fit(self, train: pd.Series) -> "Croston":
        y = train.to_numpy(dtype=float)
        nz_idx = np.flatnonzero(y > 0)
        if nz_idx.size == 0:
            self._rate = 0.0
            return self

        # z = smoothed demand size, p = smoothed inter-arrival interval.
        z = y[nz_idx[0]]
        p = 1.0
        prev = nz_idx[0]
        for i in nz_idx[1:]:
            gap = i - prev
            z += self.alpha * (y[i] - z)
            p += self.alpha * (gap - p)
            prev = i

        rate = z / p if p > 0 else z
        if self.sba:
            rate *= 1.0 - self.alpha / 2.0
        self._rate = float(rate)
        return self

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        return _poisson_samples(self._rate, horizon, n_samples, self._rng)
