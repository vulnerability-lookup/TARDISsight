# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TARDISsight is a **research/experimentation repository** for forecasting vulnerability
*sightings* (when and how often a CVE is observed/reported). It is the code companion to the
paper *"Modeling Sparse and Bursty Vulnerability Sightings: Forecasting Under Data Constraints"*
(Bonhomme & Dulaunoy, 2026, arXiv:2604.16038). Treat each module as a self-contained experiment
exploring a different forecasting model, not as a deployable library or service.

The central finding (see `docs/arima.md`) shapes the whole repo: sightings are **short-lived,
bursty event sequences that decay or saturate after initial visibility** — not long stationary
time series. ARIMA/SARIMAX is documented as *the wrong paradigm* for this data; the decay/growth
curve models are the favored direction.

## Environment & commands

Python `>=3.12`, dependencies managed with **Poetry** (`poetry.lock`, `pyproject.toml`).

```bash
poetry install                       # set up the environment
poetry run python tardissight/decay/adaptive.py     # run an experiment script
poetry run mypy tardissight          # type-check (scripts carry `# type: ignore` annotations)
```

There is no test suite, linter config, build step, or CI. `scipy` is used (via `curve_fit`) but
comes in transitively through `statsmodels` rather than being a direct dependency.

## How the scripts work (shared pattern)

Every model script under `poisson/`, `decay/`, and `arima/` follows the same skeleton — when
adding or modifying one, mirror it:

1. **Prompt** for a vulnerability id interactively: `vuln_id = input("Vulnerability id: ")`.
   This makes scripts non-importable as modules; they are meant to be run directly.
2. **Fetch** all sightings via paginated GET against the Vulnerability-Lookup API:
   `https://vulnerability.circl.lu/api/sighting?page={page}&per_page={per_page}&vuln_id={vuln_id}`,
   looping until `page * per_page >= data["metadata"]["count"]`.
3. **Aggregate** `creation_timestamp` into a daily count series (`pandas`). Some scripts switch to
   weekly aggregation when data is sparse (e.g. Poisson uses weekly if the daily zero-ratio > 0.5).
4. **Fit** the model, **forecast** ~10 future periods, and **plot** observed vs. forecast with
   `matplotlib` (typically blue = observed, red dashed = forecast, orange = forecast region).

A sighting record shape: `{author, creation_timestamp, source, type, uuid, vulnerability, vulnerability_lookup_origin}`.

## Model modules

- `tardissight/decay/` — the preferred curve-fitting approach for burst-and-fade dynamics:
  - `logistic_growth.py` — logistic curve `L / (1 + e^-k(t-t0))`, for newly published/trending CVEs.
  - `exponential_decay.py` — `a·e^-bt + c`, for vulnerabilities past their peak.
  - `adaptive.py` — detects trend slope, then auto-selects logistic (rising) vs. exponential decay.
- `tardissight/poisson/poisson.py` — Poisson GLM regression with adaptive daily/weekly aggregation.
- `tardissight/arima/` — `sarimax.py` / `sarimax1.py`, log1p-transformed SARIMAX experiments.
  Kept for the record; documented as ill-suited to this data (needs long stationary series).
- `tardissight/evolution/` — Jupyter notebooks tracking how sightings evolve over time, plus
  snapshot JSON exports (`sightings_*.json`).

Per-model rationale and example forecasts live in `docs/` (`decay.md`, `poisson.md`, `arima.md`,
`adaptive.md`).
