"""
Rolling-origin backtesting over a corpus of CVEs.

Why rolling-origin (a.k.a. time-series cross-validation) rather than a single
train/test split? Sightings series are short, so a single split would score each
model on one forecast and confound model quality with the luck of where the cut
falls. Rolling-origin slides the forecast origin forward one day at a time: for
each origin ``t`` the model is fit on ``series[:t]`` and scored on the next
``horizon`` days ``series[t:t+horizon]``. This (a) uses only past data at every
step (no leakage) and (b) yields many scored forecasts per CVE, so the model
comparison rests on a distribution of errors, not a single draw.

Models are supplied as **factories** (zero-arg callables returning a fresh
forecaster) because every forecaster is stateful after ``fit`` and must be
re-created for each origin.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from . import metrics as M

ModelFactory = Callable[[], object]

DEFAULT_COVERAGE_LEVELS = (0.5, 0.8, 0.95)


def backtest_series(
    factories: dict[str, ModelFactory],
    series: pd.Series,
    *,
    vuln_id: str = "",
    horizon: int = 7,
    min_train: int = 10,
    stride: int = 1,
    n_samples: int = 2000,
    coverage_levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Backtest every model on one CVE series.

    Returns a tidy per-origin records DataFrame and, per model, the pooled
    randomised-PIT values (kept separately because calibration is assessed over
    all forecasts jointly, not per origin).
    """
    y = series.to_numpy(dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    records: list[dict[str, float | str | int]] = []
    pit_pool: dict[str, list[np.ndarray]] = {name: [] for name in factories}

    last_origin = n - horizon
    if last_origin < min_train:
        # Series too short to produce even one (min_train, horizon) split.
        return pd.DataFrame(records), {k: np.array([]) for k in factories}

    for origin in range(min_train, last_origin + 1, stride):
        train = series.iloc[:origin]
        actual = y[origin : origin + horizon]
        train_arr = y[:origin]

        for name, factory in factories.items():
            model = factory()
            model.fit(train)  # type: ignore[attr-defined]
            samples = model.sample(horizon, n_samples)  # type: ignore[attr-defined]
            point = np.median(samples, axis=0)

            rec: dict[str, float | str | int] = {
                "vuln_id": vuln_id,
                "model": name,
                "origin": origin,
                "crps": float(M.crps_samples(samples, actual).mean()),
                "pinball": float(M.pinball_loss(samples, actual).mean()),
                "rmsse": M.rmsse(point, actual, train_arr),
                "mae": float(np.abs(point - actual).mean()),
            }
            for lvl in coverage_levels:
                rec[f"cov_{int(lvl * 100)}"] = float(M.interval_coverage(samples, actual, lvl).mean())
            records.append(rec)

            pit_pool[name].append(M.randomized_pit(samples, actual, rng))

    pit = {name: (np.concatenate(v) if v else np.array([])) for name, v in pit_pool.items()}
    return pd.DataFrame(records), pit


def backtest_corpus(
    factories: dict[str, ModelFactory],
    corpus: dict[str, pd.Series],
    **kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest over many CVEs; return (per-origin records, summary table).

    The summary averages each metric per model across all origins and CVEs, and
    folds in PIT calibration error (pooled over the whole corpus per model).
    """
    all_records: list[pd.DataFrame] = []
    pit_pool: dict[str, list[np.ndarray]] = {name: [] for name in factories}

    for vuln_id, series in corpus.items():
        df, pit = backtest_series(factories, series, vuln_id=vuln_id, **kwargs)  # type: ignore[arg-type]
        if not df.empty:
            all_records.append(df)
        for name, vals in pit.items():
            if vals.size:
                pit_pool[name].append(vals)

    if not all_records:
        return pd.DataFrame(), pd.DataFrame()

    records = pd.concat(all_records, ignore_index=True)

    metric_cols = [c for c in records.columns if c not in ("vuln_id", "model", "origin")]
    summary = records.groupby("model")[metric_cols].mean().reset_index()
    summary["n_forecasts"] = records.groupby("model").size().to_numpy()

    cal_err = {
        name: M.pit_calibration_error(np.concatenate(vals)) if vals else float("nan")
        for name, vals in pit_pool.items()
    }
    summary["pit_cal_error"] = summary["model"].map(cal_err)

    summary = summary.sort_values("crps").reset_index(drop=True)
    return records, summary
