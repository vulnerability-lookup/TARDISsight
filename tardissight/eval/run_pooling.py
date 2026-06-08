"""
Tier-2 experiment: does partial pooling help in the data-scarce regime?

The hypothesis is that a hierarchical (partially pooled) model helps *most* when
a CVE has little data, because it can borrow strength from the population of past
CVEs. To test this directly we run a **data-starvation backtest**: the model is
only ever shown a short, fixed-length trailing window of ``W`` days, and we sweep
``W`` from very small (5) upward. If pooling helps, the hierarchical model should
beat the unpooled best count model most at small ``W`` and the gap should shrink
as ``W`` grows and the per-CVE data starts to speak for itself.

To keep the population prior honest (no leakage) it is estimated **leave-one-out**:
when forecasting CVE ``c`` the prior comes only from the *other* CVEs' full
histories — the realistic operational setting where a new CVE arrives and we
already hold a back-catalogue of past ones.

Models compared at each window size:
* ``hier_hurdle``     — hierarchical hurdle (this prototype);
* ``indep_hurdle_nb`` — Tier-1's best count model, fit on the window alone;
* ``rolling_mean``    — Tier-1's strongest baseline.

    python -m tardissight.eval.run_pooling
    python -m tardissight.eval.run_pooling --windows 5 7 10 14 21 30 --horizon 7
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tardissight.corpus import EXTENDED_CVES
from tardissight.data import build_corpus
from tardissight.eval import metrics as M
from tardissight.models.baselines import RollingMean
from tardissight.models.count import Hurdle
from tardissight.models.hierarchical import HierarchicalHurdle, fit_population_prior

DEFAULT_WINDOWS = [5, 7, 10, 14, 21, 30]
COVERAGE_LEVELS = (0.5, 0.8, 0.95)


def _starvation_backtest_cve(
    target: str,
    series: pd.Series,
    prior,
    *,
    windows: list[int],
    horizon: int,
    n_samples: int,
    max_origins: int,
    base_stride: int,
) -> tuple[list[dict], dict[tuple[int, str], list[np.ndarray]]]:
    y = series.to_numpy(dtype=float)
    n = len(y)
    records: list[dict] = []
    pit_pool: dict[tuple[int, str], list[np.ndarray]] = {}

    for w in windows:
        n_possible = n - horizon - w + 1
        if n_possible <= 0:
            continue
        stride = max(base_stride, math.ceil(n_possible / max_origins))

        for origin in range(w, n - horizon + 1, stride):
            train = series.iloc[origin - w : origin]
            train_arr = y[origin - w : origin]
            actual = y[origin : origin + horizon]

            factories = {
                "hier_hurdle": lambda: HierarchicalHurdle(prior),
                "indep_hurdle_nb": lambda: Hurdle("negbin", trend=False),
                "rolling_mean": lambda: RollingMean(window=w),
            }
            for name, factory in factories.items():
                model = factory()
                model.fit(train)
                samples = model.sample(horizon, n_samples)
                point = np.median(samples, axis=0)
                rec = {
                    "vuln_id": target,
                    "window": w,
                    "model": name,
                    "origin": origin,
                    "crps": float(M.crps_samples(samples, actual).mean()),
                    "pinball": float(M.pinball_loss(samples, actual).mean()),
                    "rmsse": M.rmsse(point, actual, train_arr),
                    "mae": float(np.abs(point - actual).mean()),
                }
                for lvl in COVERAGE_LEVELS:
                    rec[f"cov_{int(lvl * 100)}"] = float(M.interval_coverage(samples, actual, lvl).mean())
                records.append(rec)
                pit_pool.setdefault((w, name), []).append(
                    M.randomized_pit(samples, actual, np.random.default_rng(origin))
                )

    return records, pit_pool


def main() -> None:
    # Tiny training windows routinely trigger separation/convergence warnings in
    # the statsmodels fits; these are expected and handled by the models'
    # fallbacks. Filter here (after imports, which can reset the registry) so the
    # experiment output stays readable for the paper.
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cves", nargs="*", default=EXTENDED_CVES)
    parser.add_argument("--windows", nargs="*", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--max-origins", type=int, default=40, help="cap origins per CVE-window")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    print(f"Building corpus ({len(args.cves)} CVEs) ...")
    corpus = build_corpus(args.cves)

    all_records: list[dict] = []
    pit_pool: dict[tuple[int, str], list[np.ndarray]] = {}

    for target, series in corpus.items():
        # Leave-one-out population prior: everything except the target CVE.
        others = [s for c, s in corpus.items() if c != target]
        prior = fit_population_prior(others)
        recs, pits = _starvation_backtest_cve(
            target,
            series,
            prior,
            windows=args.windows,
            horizon=args.horizon,
            n_samples=args.n_samples,
            max_origins=args.max_origins,
            base_stride=args.stride,
        )
        all_records.extend(recs)
        for key, vals in pits.items():
            pit_pool.setdefault(key, []).extend(vals)

    if not all_records:
        print("No forecasts produced.")
        return

    records = pd.DataFrame(all_records)

    # Report a population prior fit on the whole corpus (for documentation).
    full_prior = fit_population_prior(list(corpus.values()))
    print(
        f"\nPopulation prior (full corpus, {full_prior.n_cves} CVEs): "
        f"mean activity={full_prior.mean_activity:.3f}, "
        f"mean burst rate={full_prior.mean_burst_rate:.3f}, "
        f"pooled NB alpha={full_prior.nb_alpha:.3f}"
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n=== CRPS by training-window size (lower is better) ===")
    crps_pivot = records.pivot_table(index="model", columns="window", values="crps", aggfunc="mean")
    print(crps_pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== 80% interval coverage by window (nominal 0.80) ===")
    cov_pivot = records.pivot_table(index="model", columns="window", values="cov_80", aggfunc="mean")
    print(cov_pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # PIT calibration error per (window, model), pooled across all forecasts.
    cal_rows = []
    for (w, name), vals in pit_pool.items():
        cal_rows.append({"window": w, "model": name, "pit_cal_error": M.pit_calibration_error(np.concatenate(vals))})
    cal = pd.DataFrame(cal_rows).pivot_table(index="model", columns="window", values="pit_cal_error")
    print("\n=== PIT calibration error by window (lower is better) ===")
    print(cal.to_string(float_format=lambda x: f"{x:.3f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.out / "pooling_records.csv", index=False)
    crps_pivot.to_csv(args.out / "pooling_crps_by_window.csv")
    print(f"\nWrote results to {args.out}/pooling_records.csv and {args.out}/pooling_crps_by_window.csv")


if __name__ == "__main__":
    main()
