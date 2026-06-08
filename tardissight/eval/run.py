"""
Run the forecasting evaluation and emit a model-comparison table.

    python -m tardissight.eval.run                 # use cached corpus
    python -m tardissight.eval.run --refresh       # re-fetch from the API
    python -m tardissight.eval.run --horizon 7 --min-train 10

Results (per-origin records and the summary table) are written to ``results/``
so they can be cited and regenerated for the paper. The corpus is the set of
CVEs used in the first paper plus a few high-volume, long-history cases, so the
comparison spans both the sparse short-series regime and the data-rich regime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from tardissight.data import build_corpus
from tardissight.eval.backtest import backtest_corpus
from tardissight.models.baselines import Croston, NaiveLast, RollingMean
from tardissight.models.count import CountGLM, Hurdle

# CVEs from the first paper (arXiv:2604.16038) plus extra long-history cases.
DEFAULT_CVES = [
    "CVE-2025-61932",
    "CVE-2025-59287",
    "CVE-2022-26134",
    "CVE-2024-9164",
    "CVE-2025-54236",
    "CVE-2025-8088",
]

# Factories: each returns a fresh, unfit forecaster. Both *_const and *_trend
# variants are included so the backtest can answer whether the log-linear time
# trend (the first paper's `sightings ~ time_index` spec) actually helps on
# these short series, or whether a constant rate is more robust.
MODEL_FACTORIES = {
    "naive_last": lambda: NaiveLast(),
    "rolling_mean_7": lambda: RollingMean(window=7),
    "sba": lambda: Croston(sba=True),
    "poisson_const": lambda: CountGLM("poisson", trend=False),
    "poisson_trend": lambda: CountGLM("poisson", trend=True),
    "negbin_const": lambda: CountGLM("negbin", trend=False),
    "negbin_trend": lambda: CountGLM("negbin", trend=True),
    "hurdle_poisson": lambda: Hurdle("poisson", trend=True),
    "hurdle_negbin": lambda: Hurdle("negbin", trend=True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cves", nargs="*", default=DEFAULT_CVES)
    parser.add_argument("--refresh", action="store_true", help="re-fetch sightings from the API")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--min-train", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    print(f"Building corpus ({len(args.cves)} CVEs, refresh={args.refresh}) ...")
    corpus = build_corpus(args.cves, refresh=args.refresh)
    for vuln_id, series in corpus.items():
        nz = int((series > 0).sum())
        print(f"  {vuln_id}: {len(series)} days, {nz} active days, {int(series.sum())} sightings")

    print(f"\nBacktesting (horizon={args.horizon}, min_train={args.min_train}, stride={args.stride}) ...")
    records, summary = backtest_corpus(
        MODEL_FACTORIES,
        corpus,
        horizon=args.horizon,
        min_train=args.min_train,
        stride=args.stride,
        n_samples=args.n_samples,
    )

    if summary.empty:
        print("No forecasts produced — corpus series may be too short for these settings.")
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== Model comparison (sorted by CRPS, lower is better) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Per-CVE CRPS. The pooled summary above is origin-weighted, so a
    # long-history CVE (many origins) dominates it. This breakdown shows whether
    # the ranking holds per CVE and across the sparse vs data-rich regimes.
    print("\n=== CRPS per CVE (lower is better) ===")
    pivot = records.pivot_table(index="model", columns="vuln_id", values="crps", aggfunc="mean")
    pivot["mean_of_cves"] = pivot.mean(axis=1)  # equal weight per CVE
    pivot = pivot.sort_values("mean_of_cves")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    args.out.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.out / "backtest_crps_per_cve.csv")
    records.to_csv(args.out / "backtest_records.csv", index=False)
    summary.to_csv(args.out / "backtest_summary.csv", index=False)
    print(f"\nWrote results to {args.out}/backtest_records.csv and {args.out}/backtest_summary.csv")


if __name__ == "__main__":
    main()
