"""
Generate publication figures from the evaluation results.

Tables are precise but hard to read; a paper needs charts. This script turns the
CSVs written by `tardissight.eval.run` / `run_pooling` (and the cached sighting
series) into a small set of figures for the VulnOptiCON follow-up.

    python -m tardissight.plots                 # write all figures to docs/img/eval/
    python -m tardissight.plots --out /tmp/fig  # elsewhere

Figures produced:
  1. sightings_examples.png       — raw daily series for a few CVEs (the sparse,
                                     bursty regime the paper is about).
  2. tier1_model_ranking.png      — CRPS and PIT calibration error per model.
  3. tier1_crps_per_cve.png       — per-CVE CRPS heatmap (ranking robustness).
  4. pooling_crps_vs_window.png   — THE Tier-2 result: CRPS vs training-window
                                     size, pooled vs unpooled vs baseline.
  5. forecast_example.png         — a probabilistic forecast (median + intervals)
                                     against the realised counts, for intuition.

Matplotlib only (already a dependency). Reading the comparison CSVs requires the
experiments to have been run first; missing inputs are skipped with a note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tardissight.corpus import EXTENDED_CVES, PAPER_CVES
from tardissight.data import load_series
from tardissight.models.count import Hurdle
from tardissight.models.hierarchical import HierarchicalHurdle, fit_population_prior

# Consistent, colourblind-friendly colours per model across all figures.
MODEL_COLORS = {
    "hier_hurdle": "#0072B2",
    "indep_hurdle_nb": "#D55E00",
    "rolling_mean": "#009E73",
    "hurdle_negbin": "#0072B2",
    "hurdle_poisson": "#56B4E9",
    "poisson_const": "#E69F00",
    "poisson_trend": "#F0E442",
    "negbin_const": "#CC79A7",
    "negbin_trend": "#999999",
    "naive_last": "#000000",
    "rolling_mean_7": "#009E73",
    "sba": "#882255",
}
MODEL_LABELS = {
    "hier_hurdle": "Hierarchical hurdle (pooled)",
    "indep_hurdle_nb": "Hurdle NB (unpooled)",
    "rolling_mean": "Rolling mean",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fig_sightings_examples(out: Path, cves: list[str]) -> None:
    """Small multiples of daily sighting counts — illustrates sparsity/burstiness."""
    fig, axes = plt.subplots(len(cves), 1, figsize=(8, 1.8 * len(cves)), sharex=False)
    if len(cves) == 1:
        axes = [axes]
    for ax, cve in zip(axes, cves):
        s = load_series(cve)
        ax.fill_between(s.index, s.values, step="mid", color="#0072B2", alpha=0.7)
        active = int((s > 0).sum())
        ax.set_title(f"{cve}  ({len(s)} days, {active} active, {int(s.sum())} sightings)", loc="left")
        ax.set_ylabel("sightings/day")
    fig.suptitle("Daily vulnerability sightings — sparse and bursty", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "sightings_examples.png")
    plt.close(fig)


def fig_tier1_model_ranking(out: Path, summary_csv: Path) -> None:
    """Two panels: CRPS (accuracy) and PIT calibration error per model."""
    if not summary_csv.exists():
        print(f"  skip tier1_model_ranking: {summary_csv} not found (run `python -m tardissight.eval.run`)")
        return
    df = pd.read_csv(summary_csv).sort_values("crps")
    colors = [MODEL_COLORS.get(m, "#777777") for m in df["model"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.barh(df["model"], df["crps"], color=colors)
    ax1.invert_yaxis()
    ax1.set_xlabel("CRPS (lower is better)")
    ax1.set_title("Forecast accuracy")
    # Clip the axis to the meaningful range so a single off-scale model (the
    # runaway negbin_trend) doesn't squash every other bar; annotate its value.
    vals = df["crps"].to_numpy()
    if len(vals) > 1:
        cap = float(np.sort(vals)[-2]) * 1.25
        ax1.set_xlim(0, cap)
        for i, v in enumerate(vals):
            if v > cap:
                ax1.text(cap * 0.99, i, f"{v:.2f} →", ha="right", va="center", color="white", fontsize=9)

    df2 = df.sort_values("pit_cal_error")
    ax2.barh(df2["model"], df2["pit_cal_error"], color=[MODEL_COLORS.get(m, "#777777") for m in df2["model"]])
    ax2.invert_yaxis()
    ax2.set_xlabel("PIT calibration error (lower is better)")
    ax2.set_title("Calibration")

    fig.suptitle("Tier 1: model comparison (rolling-origin backtest, 5 CVEs)")
    fig.tight_layout()
    fig.savefig(out / "tier1_model_ranking.png")
    plt.close(fig)


def fig_tier1_crps_per_cve(out: Path, per_cve_csv: Path) -> None:
    """Heatmap of CRPS per (model, CVE) — shows the ranking holds across CVEs."""
    if not per_cve_csv.exists():
        print(f"  skip tier1_crps_per_cve: {per_cve_csv} not found")
        return
    df = pd.read_csv(per_cve_csv, index_col=0).sort_values("mean_of_cves")
    cve_cols = [c for c in df.columns if c != "mean_of_cves"]
    data = df[cve_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(1.1 * len(cve_cols) + 3, 0.5 * len(df) + 1.5))
    # Cap the colour scale at the 85th percentile so outliers (negbin_trend's
    # blow-up) don't flatten the contrast across the normal range; true values
    # are still printed in each cell.
    vmax = float(np.percentile(data, 85))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmax=vmax)
    ax.set_xticks(range(len(cve_cols)))
    ax.set_xticklabels(cve_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    for i in range(len(df)):
        for j in range(len(cve_cols)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="CRPS", extend="max")
    ax.set_title("Tier 1: CRPS per CVE (models sorted by mean)")
    fig.tight_layout()
    fig.savefig(out / "tier1_crps_per_cve.png")
    plt.close(fig)


def fig_pooling_crps_vs_window(out: Path, records_csv: Path) -> None:
    """THE Tier-2 figure: CRPS vs training-window size, with ±SEM bands."""
    if not records_csv.exists():
        print(f"  skip pooling_crps_vs_window: {records_csv} not found (run `python -m tardissight.eval.run_pooling`)")
        return
    df = pd.read_csv(records_csv)
    g = df.groupby(["model", "window"])["crps"]
    mean = g.mean().unstack("model")
    sem = (g.std() / np.sqrt(g.count())).unstack("model")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model in ["rolling_mean", "indep_hurdle_nb", "hier_hurdle"]:
        if model not in mean.columns:
            continue
        x = mean.index.to_numpy()
        y = mean[model].to_numpy()
        e = sem[model].to_numpy()
        ax.plot(x, y, "-o", color=MODEL_COLORS[model], label=MODEL_LABELS[model], linewidth=2)
        ax.fill_between(x, y - e, y + e, color=MODEL_COLORS[model], alpha=0.15)

    ax.set_xlabel("training-window size (days)")
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title("Tier 2: partial pooling wins most when data is scarce")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pooling_crps_vs_window.png")
    plt.close(fig)


def fig_forecast_example(out: Path, cve: str, window: int = 14, horizon: int = 7, seed: int = 0) -> None:
    """Probabilistic forecast (median + 80/95% intervals) vs realised counts.

    Chooses the forecast origin at the most active stretch of the series so the
    figure shows a burst rather than a flat zero region. Compares the pooled
    hierarchical hurdle with the unpooled hurdle on the same window.
    """
    s = load_series(cve)
    y = s.to_numpy(dtype=float)
    if len(y) < window + horizon + 5:
        print(f"  skip forecast_example: {cve} too short")
        return

    # Pick the origin whose preceding `window` has the most activity (a burst).
    valid = range(window, len(y) - horizon + 1)
    origin = max(valid, key=lambda o: y[o - window : o].sum())
    train = s.iloc[origin - window : origin]
    hist = s.iloc[max(0, origin - 3 * window) : origin]
    future_idx = s.index[origin : origin + horizon]
    actual = y[origin : origin + horizon]

    prior = fit_population_prior([load_series(c) for c in EXTENDED_CVES if c != cve])
    models = {
        "hier_hurdle": HierarchicalHurdle(prior, seed=seed).fit(train),
        "indep_hurdle_nb": Hurdle("negbin", trend=False, seed=seed).fit(train),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (name, model) in zip(axes, models.items()):
        samples = model.sample(horizon, 4000)
        med = np.median(samples, axis=0)
        lo80, hi80 = np.quantile(samples, [0.1, 0.9], axis=0)
        lo95, hi95 = np.quantile(samples, [0.025, 0.975], axis=0)
        c = MODEL_COLORS[name]

        ax.plot(hist.index, hist.values, color="#444444", marker=".", linewidth=1, label="observed")
        ax.fill_between(future_idx, lo95, hi95, color=c, alpha=0.15, label="95% interval")
        ax.fill_between(future_idx, lo80, hi80, color=c, alpha=0.30, label="80% interval")
        ax.plot(future_idx, med, color=c, marker="o", linewidth=2, label="forecast median")
        ax.plot(future_idx, actual, color="black", marker="x", linestyle="", markersize=8, label="actual")
        ax.axvline(s.index[origin - 1], color="grey", linestyle=":", linewidth=1)
        ax.set_title(MODEL_LABELS.get(name, name))
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("sightings/day")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"{horizon}-day forecast for {cve} (trained on {window} days)")
    fig.tight_layout()
    fig.savefig(out / "forecast_example.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/img/eval"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--example-cve", default="CVE-2025-0282")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {args.out}/ ...")

    fig_sightings_examples(args.out, [PAPER_CVES[1], "CVE-2025-0282", "CVE-2024-9164"])
    fig_tier1_model_ranking(args.out, args.results / "backtest_summary.csv")
    fig_tier1_crps_per_cve(args.out, args.results / "backtest_crps_per_cve.csv")
    fig_pooling_crps_vs_window(args.out, args.results / "pooling_records.csv")
    fig_forecast_example(args.out, args.example_cve)

    print("Done.")


if __name__ == "__main__":
    main()
