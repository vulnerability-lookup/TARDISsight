"""
Reusable data access layer for sightings forecasting.

The original experiment scripts (``poisson/``, ``decay/``, ``arima/``) each
re-implemented the same fetch-and-aggregate logic inline and prompted for a
vulnerability id via ``input()``. That made them impossible to drive
programmatically and impossible to backtest.

This module factors that logic out into pure, importable functions so the
evaluation harness can fetch many CVEs, cache them on disk for reproducibility
(so a paper result can be regenerated without depending on the live API state),
and turn raw sightings into the daily count series every model consumes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import requests  # type: ignore[import-untyped]

API_URL = "https://vulnerability.circl.lu/api/sighting"

# On-disk cache of raw API responses. Keeping the raw payload (not just the
# aggregated series) means we can later experiment with sighting `type` and
# `source` without re-fetching, and pin a corpus to a fixed snapshot.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "sightings_cache"


def fetch_sightings(vuln_id: str, per_page: int = 1000, timeout: int = 30) -> list[dict[str, Any]]:
    """Fetch all sightings for ``vuln_id`` from the Vulnerability-Lookup API.

    Paginates until the reported ``metadata.count`` is exhausted. Raises on HTTP
    errors rather than silently returning partial data, because a truncated
    series would quietly corrupt every downstream forecast and metric.
    """
    sightings: list[dict[str, Any]] = []
    page = 1
    while True:
        response = requests.get(
            API_URL,
            params={"page": page, "per_page": per_page, "vuln_id": vuln_id},
            headers={"accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        sightings.extend(payload["data"])
        total = payload["metadata"]["count"]
        if page * per_page >= total:
            break
        page += 1
    return sightings


def get_sightings(vuln_id: str, *, use_cache: bool = True, refresh: bool = False) -> list[dict[str, Any]]:
    """Return sightings for ``vuln_id``, reading/writing the on-disk cache.

    ``refresh=True`` forces a re-fetch and overwrites the cache. With
    ``use_cache=False`` the network is always hit and nothing is persisted.
    """
    if not use_cache:
        return fetch_sightings(vuln_id)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{vuln_id.upper()}.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())

    sightings = fetch_sightings(vuln_id)
    cache_file.write_text(json.dumps(sightings, indent=2))
    return sightings


def daily_series(sightings: list[dict[str, Any]]) -> pd.Series:
    """Aggregate raw sightings into a gap-free daily count series.

    Days with no sightings are filled with 0 (``asfreq('D', fill_value=0)``) so
    the index is contiguous — count models and backtests both depend on a
    regular calendar with explicit zeros rather than missing dates.
    """
    if not sightings:
        return pd.Series(dtype="int64", name="sightings")

    df = pd.DataFrame(sightings)
    ts = pd.to_datetime(df["creation_timestamp"], utc=True, errors="coerce")
    counts = ts.dt.floor("D").value_counts().sort_index()
    series = counts.asfreq("D", fill_value=0)
    series.name = "sightings"
    series.index.name = "date"
    return series


def load_series(vuln_id: str, *, use_cache: bool = True, refresh: bool = False) -> pd.Series:
    """Convenience: fetch (cached) sightings and return the daily count series."""
    return daily_series(get_sightings(vuln_id, use_cache=use_cache, refresh=refresh))


def build_corpus(vuln_ids: list[str], *, refresh: bool = False, pause: float = 0.5) -> dict[str, pd.Series]:
    """Fetch and cache a set of CVEs, returning ``{vuln_id: daily_series}``.

    ``pause`` throttles requests to be polite to the public API when populating
    the cache for the first time.
    """
    corpus: dict[str, pd.Series] = {}
    for vuln_id in vuln_ids:
        cached = (CACHE_DIR / f"{vuln_id.upper()}.json").exists()
        corpus[vuln_id] = load_series(vuln_id, refresh=refresh)
        if refresh or not cached:
            time.sleep(pause)
    return corpus
