"""
Tier-4 experiment: does full Bayesian pooling improve calibration?

The empirical-Bayes model of Tier 2 used point hyperparameters, which made it
slightly over-confident at the shortest training windows (its predictive
intervals were a touch too narrow because it treated the population as known).
The fully Bayesian model propagates hyperparameter uncertainty, so we expect:

* CRPS comparable to empirical Bayes (the posterior mean matches the EB point
  estimate, so sharpness should be similar), and
* better *calibration* (lower PIT error), most visibly at small windows.

Same data-starvation protocol as Tier 2/3: fixed trailing window of W days,
leave-one-out population fit, swept W. We compare:
  bayes_hurdle      — full Bayesian (this Tier),
  hier_hurdle       — empirical-Bayes pooled (Tier 2),
  indep_hurdle_nb   — best unpooled model (Tier 1).

    python -m tardissight.eval.run_bayes 2>/dev/null
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tardissight.corpus import EXTENDED_CVES
from tardissight.data import load_series
from tardissight.eval import metrics as M
from tardissight.models.bayesian import BayesianHierarchicalHurdle, sample_hyperposterior
from tardissight.models.count import Hurdle
from tardissight.models.hierarchical import HierarchicalHurdle, fit_population_prior

DEFAULT_WINDOWS = [5, 7, 10, 14, 21, 30]


def _origins(n: int, w: int, horizon: int, max_origins: int, base_stride: int):
    n_possible = n - horizon - w + 1
    if n_possible <= 0:
        return []
    stride = max(base_stride, math.ceil(n_possible / max_origins))
    return list(range(w, n - horizon + 1, stride))


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

    print(f"Loading corpus ({len(args.cves)} CVEs) ...")
    corpus = {c: load_series(c) for c in args.cves}

    records: list[dict] = []
    pit_pool: dict[tuple[int, str], list[np.ndarray]] = {}

    for target_idx, (target, series) in enumerate(corpus.items()):
        others = [s for c, s in corpus.items() if c != target]
        # Leave-one-out population fits (done once per target, not per origin).
        # Deterministic per-target seed (str hash() is randomised per process).
        hyper = sample_hyperposterior(others, seed=1000 + target_idx)
        eb_prior = fit_population_prior(others)
        y = series.to_numpy(dtype=float)

        for w in args.windows:
            for origin in _origins(len(y), w, args.horizon, args.max_origins, args.stride):
                train = series.iloc[origin - w : origin]
                train_arr = y[origin - w : origin]
                actual = y[origin : origin + args.horizon]
                factories = {
                    "bayes_hurdle": lambda h=hyper: BayesianHierarchicalHurdle(h),
                    "hier_hurdle": lambda p=eb_prior: HierarchicalHurdle(p),
                    "indep_hurdle_nb": lambda: Hurdle("negbin", trend=False),
                }
                for name, factory in factories.items():
                    model = factory().fit(train)
                    samples = model.sample(args.horizon, args.n_samples)
                    point = np.median(samples, axis=0)
                    rec = {
                        "vuln_id": target,
                        "window": w,
                        "model": name,
                        "origin": origin,
                        "crps": float(M.crps_samples(samples, actual).mean()),
                        "pinball": float(M.pinball_loss(samples, actual).mean()),
                        "rmsse": M.rmsse(point, actual, train_arr),
                        "cov_80": float(M.interval_coverage(samples, actual, 0.8).mean()),
                    }
                    records.append(rec)
                    pit_pool.setdefault((w, name), []).append(
                        M.randomized_pit(samples, actual, np.random.default_rng(origin))
                    )

    df = pd.DataFrame(records)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n=== CRPS by training-window size (lower is better) ===")
    print(df.pivot_table(index="model", columns="window", values="crps").to_string(float_format=lambda x: f"{x:.3f}"))

    cal_rows = [
        {"window": w, "model": name, "pit_cal_error": M.pit_calibration_error(np.concatenate(v))}
        for (w, name), v in pit_pool.items()
    ]
    cal = pd.DataFrame(cal_rows).pivot_table(index="model", columns="window", values="pit_cal_error")
    print("\n=== PIT calibration error by window (lower is better) ===")
    print(cal.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== 80% interval coverage by window (nominal 0.80) ===")
    print(df.pivot_table(index="model", columns="window", values="cov_80").to_string(float_format=lambda x: f"{x:.3f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "bayes_records.csv", index=False)
    cal.to_csv(args.out / "bayes_pit_by_window.csv")
    print(f"\nWrote results to {args.out}/bayes_records.csv and {args.out}/bayes_pit_by_window.csv")


if __name__ == "__main__":
    main()
