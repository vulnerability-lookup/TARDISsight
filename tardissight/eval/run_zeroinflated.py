"""
Tier-5 experiment: does a zero-inflated likelihood tighten the forecasts?

Tier 4 showed the binding problem is over-dispersion: every model over-covers
(empirical 80% coverage ~0.96 vs nominal 0.80), so the predictive distributions
are too wide. The hurdle's positive part is a heavy-tailed ``1 + NB`` with large
pooled dispersion; a zero-inflated NB instead lets one (tighter) NB component
explain both the small positive counts and some of the zeros. The question is
whether that yields sharper, better-calibrated forecasts at the same accuracy.

Same data-starvation protocol (fixed window W, leave-one-out population fit, 24
CVEs, H=7). We compare:
  zinb_hier        — hierarchical Zero-Inflated NB (this Tier),
  hier_hurdle      — empirical-Bayes hurdle (Tier 2, the model to beat),
  indep_hurdle_nb  — best unpooled model (Tier 1).

We report CRPS, PIT calibration error, and 80% interval coverage (the metric the
zero-inflated model is meant to improve).

    python -m tardissight.eval.run_zeroinflated 2>/dev/null
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
from tardissight.models.count import Hurdle
from tardissight.models.hierarchical import HierarchicalHurdle, fit_population_prior
from tardissight.models.zeroinflated import HierarchicalZINB, fit_zinb_population

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
    priors_report = []

    for target, series in corpus.items():
        others = [s for c, s in corpus.items() if c != target]
        # Leave-one-out population fits (once per target).
        zinb_prior = fit_zinb_population(others)
        hurdle_prior = fit_population_prior(others)
        priors_report.append({"target": target, "pi": zinb_prior.mean_pi, "mu": zinb_prior.mean_mu, "r": zinb_prior.r})
        y = series.to_numpy(dtype=float)

        for w in args.windows:
            for origin in _origins(len(y), w, args.horizon, args.max_origins, args.stride):
                train = series.iloc[origin - w : origin]
                train_arr = y[origin - w : origin]
                actual = y[origin : origin + args.horizon]
                factories = {
                    "zinb_hier": lambda p=zinb_prior: HierarchicalZINB(p),
                    "hier_hurdle": lambda p=hurdle_prior: HierarchicalHurdle(p),
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

    pr = pd.DataFrame(priors_report)
    print(
        f"\nZINB population prior (mean over LOO folds): structural-zero pi={pr['pi'].mean():.3f}, "
        f"NB mu={pr['mu'].mean():.3f}, shared r={pr['r'].mean():.3f}"
    )

    print("\n=== CRPS by training-window size (lower is better) ===")
    print(df.pivot_table(index="model", columns="window", values="crps").to_string(float_format=lambda x: f"{x:.3f}"))

    cal_rows = [
        {"window": w, "model": name, "pit_cal_error": M.pit_calibration_error(np.concatenate(v))}
        for (w, name), v in pit_pool.items()
    ]
    cal = pd.DataFrame(cal_rows).pivot_table(index="model", columns="window", values="pit_cal_error")
    print("\n=== PIT calibration error by window (lower is better) ===")
    print(cal.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== 80% interval coverage by window (nominal 0.80; closer is better) ===")
    print(df.pivot_table(index="model", columns="window", values="cov_80").to_string(float_format=lambda x: f"{x:.3f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "zinb_records.csv", index=False)
    cal.to_csv(args.out / "zinb_pit_by_window.csv")
    print(f"\nWrote results to {args.out}/zinb_records.csv and {args.out}/zinb_pit_by_window.csv")


if __name__ == "__main__":
    main()
