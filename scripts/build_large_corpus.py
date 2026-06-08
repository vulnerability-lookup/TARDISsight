"""
Build a larger CVE corpus for scale-validation of the pooling result.

Uses the Vulnerability-Lookup `most_sighted` statistics endpoint to take the
top-N most-sighted CVEs over a date range, then keeps those with enough daily
history for the data-starvation backtest (span >= MIN_SPAN days and >= MIN_ACTIVE
active days). The backtest itself induces data scarcity (it restricts training to
a fixed W-day window), so high-volume CVEs are fine here. Caches each CVE's
sightings (via tardissight.data) and writes the qualifying CVE list to
data/large_corpus.json for reproducible re-runs.

    python scripts/build_large_corpus.py --target 180 --limit 350
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests  # type: ignore[import-untyped]

from tardissight.data import load_series

MOST_SIGHTED = "https://vulnerability.circl.lu/api/stats/vulnerability/most_sighted"
MIN_SPAN = 45      # days from first to last sighting
MIN_ACTIVE = 8     # days with >= 1 sighting
OUT = Path("data/large_corpus.json")


# The endpoint caps `limit` at 100, so we union the top-100 over several date
# windows (different eras + an exploited-only slice) to get a larger, more diverse
# candidate pool.
WINDOWS = [
    ("2015-01-01", "2019-12-31", None),
    ("2020-01-01", "2021-12-31", None),
    ("2022-01-01", "2023-12-31", None),
    ("2024-01-01", "2024-12-31", None),
    ("2025-01-01", "2025-12-31", None),
    ("2026-01-01", "2026-06-08", None),
    ("2015-01-01", "2026-06-08", "exploited"),
]


def _query(date_from: str, date_to: str, sighting_type: str | None) -> list[str]:
    params = {"date_from": date_from, "date_to": date_to, "limit": 100, "output": "json"}
    if sighting_type:
        params["sighting_type"] = sighting_type
    r = requests.get(MOST_SIGHTED, params=params, headers={"accept": "application/json"}, timeout=60)
    r.raise_for_status()
    return [rec["vulnerability"].upper() for rec in r.json()
            if rec.get("vulnerability", "").upper().startswith("CVE-")]


def gather_candidates() -> list[str]:
    """Union of the most-sighted CVEs across several date windows (order-preserving)."""
    seen: dict[str, None] = {}
    for date_from, date_to, stype in WINDOWS:
        try:
            for cve in _query(date_from, date_to, stype):
                seen.setdefault(cve, None)
        except Exception as e:  # noqa: BLE001
            print(f"  window {date_from}..{date_to} ({stype}) failed: {e}")
        time.sleep(0.3)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=200, help="number of qualifying CVEs to collect")
    args = ap.parse_args()

    print("Gathering most-sighted CVEs across date windows ...")
    candidates = gather_candidates()
    print(f"  {len(candidates)} distinct candidate CVEs")

    kept: list[str] = []
    for cve in candidates:
        if len(kept) >= args.target:
            break
        try:
            s = load_series(cve)  # fetches + caches
        except Exception as e:  # noqa: BLE001
            print(f"  {cve}: fetch failed ({e})")
            continue
        if len(s) == 0:
            continue
        span = (s.index.max() - s.index.min()).days + 1
        active = int((s > 0).sum())
        if span >= MIN_SPAN and active >= MIN_ACTIVE:
            kept.append(cve)
            if len(kept) % 25 == 0:
                print(f"  kept {len(kept)} ...")
        time.sleep(0.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sorted(kept), indent=2))
    print(f"\nKept {len(kept)} CVEs -> {OUT}")


if __name__ == "__main__":
    main()
