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
MODEL_COLORS["hier_exploited"] = "#0072B2"
MODEL_COLORS["typed_sum"] = "#0072B2"
MODEL_COLORS["pooled_total"] = "#D55E00"
MODEL_LABELS = {
    "hier_hurdle": "Hierarchical hurdle (pooled)",
    "hier_exploited": "Pooled (type-specific prior)",
    "indep_hurdle_nb": "Hurdle NB (unpooled)",
    "rolling_mean": "Rolling mean",
    "typed_sum": "Typed decomposition (sum)",
    "pooled_total": "Single pooled total",
}

# Colourblind-friendly colours per sighting type.
TYPE_COLORS = {
    "seen": "#56B4E9",
    "published-proof-of-concept": "#E69F00",
    "exploited": "#D55E00",
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


def _crps_vs_window(records_csv: Path, models: list[str], title: str, outfile: Path, missing_hint: str) -> None:
    """Shared line plot: mean CRPS vs training-window size with ±SEM bands."""
    if not records_csv.exists():
        print(f"  skip {outfile.name}: {records_csv} not found ({missing_hint})")
        return
    df = pd.read_csv(records_csv)
    g = df.groupby(["model", "window"])["crps"]
    mean = g.mean().unstack("model")
    sem = (g.std() / np.sqrt(g.count())).unstack("model")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model in models:
        if model not in mean.columns:
            continue
        x = mean.index.to_numpy()
        y = mean[model].to_numpy()
        e = sem[model].to_numpy()
        ax.plot(x, y, "-o", color=MODEL_COLORS.get(model, "#777"), label=MODEL_LABELS.get(model, model), linewidth=2)
        ax.fill_between(x, y - e, y + e, color=MODEL_COLORS.get(model, "#777"), alpha=0.15)

    ax.set_xlabel("training-window size (days)")
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def fig_pooling_crps_vs_window(out: Path, records_csv: Path) -> None:
    """THE Tier-2 figure: CRPS vs training-window size, with ±SEM bands."""
    _crps_vs_window(
        records_csv,
        ["rolling_mean", "indep_hurdle_nb", "hier_hurdle"],
        "Tier 2: partial pooling wins most when data is scarce",
        out / "pooling_crps_vs_window.png",
        "run `python -m tardissight.eval.run_pooling`",
    )


def fig_typed_priors(out: Path, priors_csv: Path) -> None:
    """Per-type population priors — characterises how the sighting types differ."""
    if not priors_csv.exists():
        print(f"  skip typed_priors: {priors_csv} not found (run `python -m tardissight.eval.run_typed`)")
        return
    df = pd.read_csv(priors_csv, index_col=0)
    metrics = [
        ("mean_activity", "Activity rate\n(P(day has the type))"),
        ("mean_burst_rate", "Burst rate\n(mean count | active)"),
        ("nb_alpha", "Over-dispersion (NB α)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (col, title) in zip(axes, metrics):
        colors = [TYPE_COLORS.get(t, "#777") for t in df.index]
        ax.bar(range(len(df)), df[col].to_numpy(), color=colors)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([t.replace("published-proof-of-concept", "PoC") for t in df.index], rotation=15)
        ax.set_title(title)
    fig.suptitle("Tier 3: sighting types have distinct dynamics (population priors)")
    fig.tight_layout()
    fig.savefig(out / "typed_priors.png")
    plt.close(fig)


def fig_typed_exploited_crps(out: Path, records_csv: Path) -> None:
    """Headline Tier-3 figure: pooling helps the scarce, high-value exploited signal."""
    _crps_vs_window(
        records_csv,
        ["rolling_mean", "indep_hurdle_nb", "hier_exploited"],
        "Tier 3: pooling the exploited signal across CVEs",
        out / "typed_exploited_crps.png",
        "run `python -m tardissight.eval.run_typed`",
    )


def fig_typed_lead_lag(out: Path, lead_lag_csv: Path) -> None:
    """Cross-correlation of precursor types with exploited activity vs lag."""
    if not lead_lag_csv.exists():
        print(f"  skip typed_lead_lag: {lead_lag_csv} not found (run `python -m tardissight.eval.run_typed`)")
        return
    df = pd.read_csv(lead_lag_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    for precursor, sub in df.groupby("precursor"):
        sub = sub.sort_values("lag")
        label = precursor.replace("published-proof-of-concept", "PoC")
        ax.plot(sub["lag"], sub["mean_xcorr"], "-o", color=TYPE_COLORS.get(precursor, "#777"), label=label, ms=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.axvspan(0, df["lag"].max(), color="green", alpha=0.05)
    ax.text(df["lag"].max(), ax.get_ylim()[1] * 0.9, "precursor leads →", ha="right", fontsize=9, color="green")
    ax.set_xlabel("lag (days);  positive = precursor leads exploitation")
    ax.set_ylabel("mean cross-correlation with exploited")
    ax.set_title("Tier 3: do PoC / seen sightings precede exploitation?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "typed_lead_lag.png")
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
    fig_typed_priors(args.out, args.results / "typed_priors.csv")
    fig_typed_exploited_crps(args.out, args.results / "typed_exploited_records.csv")
    fig_typed_lead_lag(args.out, args.results / "typed_lead_lag.csv")

    print("Done.")


if __name__ == "__main__":
    main()
