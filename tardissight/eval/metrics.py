"""
Proper scoring rules for probabilistic count forecasts.

The first paper assessed forecasts qualitatively ("the CI is too wide", "it
overestimates reality") by reading plots. To compare models for the follow-up we
need numbers that (a) reward sharp *and* calibrated predictive distributions and
(b) are comparable across CVEs of very different activity levels.

All scorers consume the common sample representation ``samples`` of shape
``(n_samples, horizon)`` and the realised ``actual`` of shape ``(horizon,)``.

* **CRPS** — the strictly proper scoring rule for the whole predictive
  distribution; generalises absolute error to distributions and is in the units
  of the data (sightings/day). Lower is better.
* **Pinball / quantile loss** — averaged over quantile levels; a discrete CRPS
  proxy that also lets us inspect performance at specific quantiles.
* **RMSSE** — point error scaled by the in-sample naive one-step error, so it is
  unit-free and can be averaged across CVEs without a high-volume CVE dominating.
* **Interval coverage** — empirical coverage of central prediction intervals vs
  nominal; the direct, quantitative version of the paper's "exploding CI"
  complaint (under- or over-coverage).
* **Randomised PIT** — for calibration; uniform PIT ⇔ calibrated. We summarise
  with the mean absolute deviation of the PIT histogram from uniform.
"""

from __future__ import annotations

import numpy as np

DEFAULT_QUANTILES = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
_EPS = 1e-12


def crps_samples(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per-step CRPS estimated from samples, shape ``(horizon,)``.

    Uses the sorted-sample identity
    ``CRPS = E|X-y| - (1/m^2) * sum_i (2i-m-1) x_(i)`` which is O(m log m) and
    avoids the O(m^2) pairwise double sum.
    """
    m = samples.shape[0]
    xs = np.sort(samples, axis=0)
    term1 = np.abs(samples - actual[None, :]).mean(axis=0)
    i = np.arange(1, m + 1)[:, None]
    term2 = (((2 * i - m - 1) * xs).sum(axis=0)) / (m * m)
    return term1 - term2


def pinball_loss(samples: np.ndarray, actual: np.ndarray, quantiles: np.ndarray = DEFAULT_QUANTILES) -> np.ndarray:
    """Per-step quantile loss averaged over ``quantiles``, shape ``(horizon,)``."""
    qhat = np.quantile(samples, quantiles, axis=0)  # (n_q, horizon)
    diff = actual[None, :] - qhat
    q = quantiles[:, None]
    loss = np.maximum(q * diff, (q - 1) * diff)
    return loss.mean(axis=0)


def interval_coverage(samples: np.ndarray, actual: np.ndarray, level: float = 0.8) -> np.ndarray:
    """Boolean per-step coverage of the central ``level`` prediction interval."""
    lo = (1 - level) / 2
    hi = 1 - lo
    q_lo, q_hi = np.quantile(samples, [lo, hi], axis=0)
    return (actual >= q_lo) & (actual <= q_hi)


def randomized_pit(samples: np.ndarray, actual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomised PIT values for count data, shape ``(horizon,)``.

    For discrete forecasts the PIT must be randomised to be uniform under
    calibration: ``u = F(y-1) + V * (F(y) - F(y-1))`` with ``V ~ U(0,1)``.
    """
    f_y = (samples <= actual[None, :]).mean(axis=0)
    f_ym1 = (samples <= (actual[None, :] - 1)).mean(axis=0)
    v = rng.random(actual.shape[0])
    return f_ym1 + v * (f_y - f_ym1)


def pit_calibration_error(pit: np.ndarray, bins: int = 10) -> float:
    """Mean absolute deviation of the PIT histogram from uniform.

    0 ⇒ perfectly calibrated; larger ⇒ mis-calibrated. Aggregated over all PIT
    values collected across the backtest.
    """
    if pit.size == 0:
        return float("nan")
    hist, _ = np.histogram(pit, bins=bins, range=(0, 1), density=False)
    expected = pit.size / bins
    return float(np.abs(hist - expected).sum() / pit.size)


def rmsse(point: np.ndarray, actual: np.ndarray, train: np.ndarray) -> float:
    """Root Mean Squared Scaled Error.

    Numerator: forecast MSE. Denominator: in-sample naive (lag-1) MSE on the
    training series. Scaling makes errors comparable across CVEs; a value < 1
    means the model beats the in-sample naive forecast.
    """
    if train.size < 2:
        return float("nan")
    scale = np.mean(np.diff(train) ** 2)
    scale = max(scale, _EPS)
    return float(np.sqrt(np.mean((point - actual) ** 2) / scale))
