"""
Tier-3 experiment: modelling sighting *type*.

Three questions, in order of importance:

1. **Does pooling help the high-value `exploited` signal?** Per CVE the exploited
   stream is even scarcer than the total, so the Tier-2 pooling argument should
   apply with more force. We rerun the data-starvation backtest on the
   exploited-only series: pooled (type-specific prior) vs unpooled hurdle vs
   rolling-mean baseline.

2. **Does decomposing by type help forecast the total?** We compare the typed
   model (sum of per-type forecasts) against the single Tier-2 pooled model on
   the *total* daily count. Splitting the data could help (each type shrinks to
   its own population) or hurt (less data per component).

3. **Do PoC / seen sightings *precede* exploitation?** A descriptive lead-lag
   cross-correlation, supporting the first paper's goal of linking sightings to
   actual exploitation. (Descriptive here; predictive use is future work.)

    python -m tardissight.eval.run_typed 2>/dev/null
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tardissight.corpus import EXTENDED_CVES
from tardissight.data import SIGHTING_TYPES, get_sightings, daily_series, typed_daily_series
from tardissight.eval import metrics as M
from tardissight.models.baselines import RollingMean
from tardissight.models.count import Hurdle
from tardissight.models.hierarchical import HierarchicalHurdle, fit_population_prior
from tardissight.models.typed import TypedHierarchicalHurdle, fit_typed_population_priors

DEFAULT_WINDOWS = [5, 7, 10, 14, 21, 30]
COVERAGE_LEVELS = (0.5, 0.8, 0.95)


def _metrics(samples: np.ndarray, actual: np.ndarray, train_arr: np.ndarray) -> dict:
    point = np.median(samples, axis=0)
    rec = {
        "crps": float(M.crps_samples(samples, actual).mean()),
        "pinball": float(M.pinball_loss(samples, actual).mean()),
        "rmsse": M.rmsse(point, actual, train_arr),
        "mae": float(np.abs(point - actual).mean()),
    }
    for lvl in COVERAGE_LEVELS:
        rec[f"cov_{int(lvl * 100)}"] = float(M.interval_coverage(samples, actual, lvl).mean())
    return rec


def _origins(n: int, w: int, horizon: int, max_origins: int, base_stride: int):
    n_possible = n - horizon - w + 1
    if n_possible <= 0:
        return []
    stride = max(base_stride, math.ceil(n_possible / max_origins))
    return list(range(w, n - horizon + 1, stride))


def exploited_backtest(corpus_typed, *, windows, horizon, n_samples, max_origins, base_stride) -> pd.DataFrame:
    """Data-starvation backtest on the exploited-only series, leave-one-out prior."""
    records = []
    for target, typed in corpus_typed.items():
        series = typed["exploited"]
        y = series.to_numpy(dtype=float)
        if y.sum() == 0:
            continue
        prior = fit_population_prior(
            [t["exploited"] for c, t in corpus_typed.items() if c != target]
        )
        for w in windows:
            for origin in _origins(len(y), w, horizon, max_origins, base_stride):
                train = series.iloc[origin - w : origin]
                train_arr = y[origin - w : origin]
                actual = y[origin : origin + horizon]
                factories = {
                    "hier_exploited": lambda p=prior: HierarchicalHurdle(p),
                    "indep_hurdle_nb": lambda: Hurdle("negbin", trend=False),
                    "rolling_mean": lambda ww=w: RollingMean(window=ww),
                }
                for name, factory in factories.items():
                    model = factory().fit(train)
                    rec = {"vuln_id": target, "window": w, "model": name, "origin": origin}
                    rec.update(_metrics(model.sample(horizon, n_samples), actual, train_arr))
                    records.append(rec)
    return pd.DataFrame(records)


def decomposition_backtest(corpus_typed, *, windows, horizon, n_samples, max_origins, base_stride) -> pd.DataFrame:
    """Total-count forecast: typed (sum of per-type) vs single pooled (Tier 2)."""
    records = []
    totals = {c: sum(t.values()) for c, t in corpus_typed.items()}  # aligned per-type series sum
    for target, typed in corpus_typed.items():
        total = totals[target]
        y = total.to_numpy(dtype=float)
        if y.sum() == 0:
            continue
        # Leave-one-out priors: per-type set and the single total.
        typed_priors = fit_typed_population_priors(
            {c: t for c, t in corpus_typed.items() if c != target}
        )
        total_prior = fit_population_prior([totals[c] for c in corpus_typed if c != target])
        for w in windows:
            for origin in _origins(len(y), w, horizon, max_origins, base_stride):
                train_arr = y[origin - w : origin]
                actual = y[origin : origin + horizon]
                typed_train = {ty: s.iloc[origin - w : origin] for ty, s in typed.items()}
                total_train = total.iloc[origin - w : origin]

                typed_model = TypedHierarchicalHurdle(typed_priors).fit(typed_train)
                pooled_model = HierarchicalHurdle(total_prior).fit(total_train)

                for name, samples in (
                    ("typed_sum", typed_model.sample_total(horizon, n_samples)),
                    ("pooled_total", pooled_model.sample(horizon, n_samples)),
                ):
                    rec = {"vuln_id": target, "window": w, "model": name, "origin": origin}
                    rec.update(_metrics(samples, actual, train_arr))
                    records.append(rec)
    return pd.DataFrame(records)


def lead_lag_analysis(corpus_typed, *, max_lag: int = 14, min_active: int = 12) -> pd.DataFrame:
    """Mean cross-correlation of a precursor type with `exploited` at lags -L..+L.

    We compute ``corr(precursor[t], exploited[t+lag])``. A *positive* lag means
    the precursor *leads* exploitation by that many days; a negative lag means
    exploitation leads the precursor. Comparing the two halves tests
    directionality: if the positive-lag mass exceeds the negative-lag mass, the
    precursor genuinely carries lead information rather than merely co-occurring.
    Each CVE's series are z-scored; CVEs with too little activity are skipped.
    """

    def zscore(a: np.ndarray) -> np.ndarray | None:
        sd = a.std()
        return (a - a.mean()) / sd if sd > 0 else None

    rows = []
    for precursor in ("seen", "published-proof-of-concept"):
        per_lag: dict[int, list[float]] = {lag: [] for lag in range(-max_lag, max_lag + 1)}
        for typed in corpus_typed.values():
            exp = typed["exploited"].to_numpy(dtype=float)
            pre = typed[precursor].to_numpy(dtype=float)
            if (exp > 0).sum() < min_active or (pre > 0).sum() < min_active:
                continue
            ze, zp = zscore(exp), zscore(pre)
            if ze is None or zp is None:
                continue
            n = len(ze)
            for lag in range(-max_lag, max_lag + 1):
                if n - abs(lag) < min_active:
                    continue
                if lag >= 0:  # precursor leads: corr(pre[t], exp[t+lag])
                    per_lag[lag].append(float(np.mean(zp[: n - lag] * ze[lag:])))
                else:  # exploited leads: corr(pre[t-lag], exp[t])
                    k = -lag
                    per_lag[lag].append(float(np.mean(zp[k:] * ze[: n - k])))
        for lag, vals in per_lag.items():
            if vals:
                rows.append({"precursor": precursor, "lag": lag, "mean_xcorr": np.mean(vals), "n_cves": len(vals)})
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cves", nargs="*", default=EXTENDED_CVES)
    parser.add_argument("--windows", nargs="*", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--max-origins", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    print(f"Building typed corpus ({len(args.cves)} CVEs) ...")
    corpus_typed = {c: typed_daily_series(get_sightings(c)) for c in args.cves}

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    # --- 0. Per-type population priors (characterise the types) ---
    priors = fit_typed_population_priors(corpus_typed)
    print("\n=== Per-type population priors (full corpus) ===")
    prow = pd.DataFrame(
        {
            t: {
                "mean_activity": p.mean_activity,
                "mean_burst_rate": p.mean_burst_rate,
                "nb_alpha": p.nb_alpha,
            }
            for t, p in priors.items()
        }
    ).T
    print(prow.to_string(float_format=lambda x: f"{x:.3f}"))

    # --- 1. Exploited-signal data-starvation backtest (headline) ---
    print("\nRunning exploited-signal backtest ...")
    exp_df = exploited_backtest(
        corpus_typed,
        windows=args.windows,
        horizon=args.horizon,
        n_samples=args.n_samples,
        max_origins=args.max_origins,
        base_stride=args.stride,
    )
    print("\n=== Exploited-signal CRPS by training-window size (lower is better) ===")
    print(exp_df.pivot_table(index="model", columns="window", values="crps").to_string(float_format=lambda x: f"{x:.3f}"))

    # --- 2. Decomposition: typed-sum vs single pooled total ---
    print("\nRunning decomposition backtest ...")
    dec_windows = [w for w in args.windows if w in (7, 14, 30)] or args.windows
    dec_df = decomposition_backtest(
        corpus_typed,
        windows=dec_windows,
        horizon=args.horizon,
        n_samples=args.n_samples,
        max_origins=args.max_origins,
        base_stride=args.stride,
    )
    print("\n=== Total-count CRPS: typed decomposition vs single pooled ===")
    print(dec_df.pivot_table(index="model", columns="window", values="crps").to_string(float_format=lambda x: f"{x:.3f}"))

    # --- 3. Lead-lag (precursor -> exploited) ---
    ll_df = lead_lag_analysis(corpus_typed)
    print("\n=== Mean cross-correlation precursor->exploited (positive lag = precursor leads) ===")
    print(ll_df.pivot_table(index="precursor", columns="lag", values="mean_xcorr").to_string(float_format=lambda x: f"{x:.3f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    exp_df.to_csv(args.out / "typed_exploited_records.csv", index=False)
    dec_df.to_csv(args.out / "typed_decomposition_records.csv", index=False)
    ll_df.to_csv(args.out / "typed_lead_lag.csv", index=False)
    prow.to_csv(args.out / "typed_priors.csv")
    print(f"\nWrote results to {args.out}/typed_*.csv")


if __name__ == "__main__":
    main()
