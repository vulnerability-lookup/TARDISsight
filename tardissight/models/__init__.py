"""
Forecaster interface shared by every model in the harness.

Design decision: **all models expose a sample-based probabilistic forecast**.
``fit(train)`` then ``sample(horizon, n_samples)`` returns an array of shape
``(n_samples, horizon)`` of simulated future counts.

Why sampling rather than (mean, variance) or closed-form quantiles? Because the
models we compare are heterogeneous — count GLMs (Poisson/NegBin), a two-part
hurdle, deterministic curve fits, and intermittent-demand baselines. They have
no common analytic predictive distribution. A pool of forecast samples is the
one representation all of them *can* produce, and it lets a single scoring path
compute CRPS, quantile/pinball loss, prediction-interval coverage, and PIT
calibration identically for every model. This keeps the model comparison fair,
which is the whole point of the evaluation harness.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


class Forecaster(Protocol):
    """Anything that can be fit on a daily series and sampled forward."""

    name: str

    def fit(self, train: pd.Series) -> "Forecaster": ...

    def sample(self, horizon: int, n_samples: int = 1000) -> np.ndarray:
        """Return simulated counts, shape ``(n_samples, horizon)``."""
        ...


def point_forecast(samples: np.ndarray) -> np.ndarray:
    """Median of the predictive samples — a robust point summary for skewed,
    bursty counts (the mean is dragged by rare large draws)."""
    return np.median(samples, axis=0)
