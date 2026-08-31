"""How many gauges does CHIRPS blend inside the study domain, and when?

The manuscript concedes that scoring a climatology against CHIRPS is "partly
circular" because CHIRPS blends station data into a satellite field, and it
cannot say how partly. The Climate Hazards Center publishes, for every month
since 1981, the list of stations it actually used. That turns the concession
into a measurement.

This counts them inside the model domain for the months that matter: the 44
scored months, plus a sample of the training period for comparison. Two numbers
decide what can be claimed:

  - If CHIRPS blends many gauges here, the anchor is partly an interpolation of
    those gauges, and a station scored against CHIRPS is not independent.
  - If it blends few, or if the count collapses in recent years (CHIRPS's
    near-real-time stream carries fewer stations than its retrospective one),
    then the evaluation window is mostly satellite, and an independent gauge
    comparison over that window is both possible and informative.

Nothing here needs a GPU, an account or a download of the precipitation data
itself. It reads one small CSV per month from a public directory.

Usage:
  python models/scripts/analysis/chirps_station_overlap.py
  python models/scripts/analysis/chirps_station_overlap.py --all-months
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
import io
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "provenance" / "chirps_station_overlap.csv"
BASE = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/diagnostics/"
        "list_of_stations_used/monthly/")
UA = {"User-Agent": "Mozilla/5.0 (AnchorGate station-overlap audit)"}

# The model domain, from the NetCDF the study is built on.
LAT = (4.375, 7.375)
LON = (-74.925, -71.725)
# The elevation strata the manuscript reports on cannot be assigned here without
# a DEM lookup; that is done downstream once the station list is fixed.

# The 44 scored months, and a matched sample of training months for contrast.
SCORED = [(y, m) for y in range(2021, 2026) for m in range(1, 13)
          if (y, m) >= (2021, 7) and (y, m) <= (2025, 2)]
TRAINING_SAMPLE = [(y, 6) for y in (1985, 1990, 1995, 2000, 2005, 2010, 2015)]


def fetch(year, month):
    url = f"{BASE}global.stationsUsed.{year}.{month:02d}.csv"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode("utf-8", "replace")


def count_in_domain(text):
    """Stations inside the box, and how many of those are labelled Colombia."""
    rd = csv.DictReader(io.StringIO(text))
    inbox = colombia = 0
    names = []
    for row in rd:
        try:
            la = float(row["latitude"])
            lo = float(row["longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        if LAT[0] <= la <= LAT[1] and LON[0] <= lo <= LON[1]:
            inbox += 1
            if (row.get("country_name") or "").strip().lower().startswith("colomb"):
                colombia += 1
                if len(names) < 6:
                    names.append((row.get("station_name") or "").strip())
    return inbox, colombia, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-months", action="store_true",
                    help="every scored month; otherwise one per year")
    args = ap.parse_args()

    scored = SCORED if args.all_months else [
        m for m in SCORED if m[1] in (1, 7)]
    plan = [("training", y, m) for y, m in TRAINING_SAMPLE] + \
           [("scored", y, m) for y, m in scored]

    print(f"domain lat {LAT}, lon {LON}")
    print(f"{len(SCORED)} scored months (2021-07 to 2025-02); "
          f"querying {len(plan)} months\n")
    print(f"{'period':<10}{'month':<10}{'in box':>8}{'Colombia':>10}   sample")
    print("-" * 78)

    rows = []
    for period, y, m in plan:
        try:
            txt = fetch(y, m)
        except urllib.error.HTTPError as e:
            print(f"{period:<10}{y}-{m:02d}   HTTP {e.code}")
            continue
        except Exception as e:                                 # noqa: BLE001
            print(f"{period:<10}{y}-{m:02d}   {type(e).__name__}")
            continue
        inbox, col, names = count_in_domain(txt)
        rows.append(dict(period=period, year=y, month=m,
                         in_domain=inbox, colombia=col))
        print(f"{period:<10}{y}-{m:02d}    {inbox:>8}{col:>10}   "
              f"{', '.join(names[:3])}")

    if not rows:
        print("\nnothing fetched")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tr = [r["in_domain"] for r in rows if r["period"] == "training"]
    sc = [r["in_domain"] for r in rows if r["period"] == "scored"]
    print()
    print("=" * 78)
    if tr:
        print(f"training sample : {min(tr)} to {max(tr)} stations in domain")
    if sc:
        print(f"scored window   : {min(sc)} to {max(sc)} stations in domain")
    # The observed pattern is neither "holds up" nor "collapses", so the summary
    # reports the split it finds rather than choosing between two stories that
    # the data does not support.
    if sc:
        thin = [r for r in rows if r["period"] == "scored"
                and r["in_domain"] <= 20]
        thick = [r["in_domain"] for r in rows if r["period"] == "scored"
                 and r["in_domain"] > 20]
        print(f"\n{len(thin)} scored months carry 20 gauges or fewer in the "
              f"domain, and {len(thick)}\ncarry more, at a median of "
              f"{int(np.median(thick)) if thick else 0}. The evaluation target is "
              f"therefore not one\nproduct but two regimes: months where CHIRPS "
              f"here is close to pure\nsatellite, and months where it is pulled "
              f"by roughly two hundred gauges.")
        if thin:
            print("Thin months: " +
                  ", ".join(f"{r['year']}-{r['month']:02d}" for r in thin))
        print("\nThat split is a free experiment. If the anchor's standing came "
              "from\nCHIRPS's climatological backbone, it should look relatively "
              "stronger in\nthe thin months; skill_vs_gauge_density.py tests "
              "exactly that.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
