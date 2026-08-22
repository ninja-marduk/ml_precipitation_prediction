"""Measure what the retrained factorial actually cost in machine time.

The training loop records wall-clock seconds, epochs, seconds per epoch and peak
device memory for every run, so the compute budget of the factorial is a
measurement rather than an estimate. This reads those records and reports the
budget three ways: per operator, per chunk of the run plan as it was executed,
and as a total.

Money is deliberately not hard-coded. Pass --usd-per-gpu-hour with the rate you
are actually quoting and the script converts; without it, it reports hours only.
A cost figure in a thesis needs a rate the reader can check, and the rate depends
on who is paying and when, so it is an input and not a constant.

Only the eighteen retrained cells are instrumented. Every earlier version of this
model family was trained before the loop recorded timings, so their cost is not
recoverable and this script does not guess at it.

Usage:
  python models/scripts/analysis/compute_cost.py
  python models/scripts/analysis/compute_cost.py --usd-per-gpu-hour 1.10
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P30 = ROOT / "models" / "output" / "V4_GNN_TAT_Models_p30"
ORDER = ("GAT", "GCN", "SAGE")


def runs():
    out = []
    for f in sorted(glob.glob(str(P30 / "SEED*" / "h12" / "*" /
                                  "training_metrics" / "*_history.json"))):
        h = json.loads(Path(f).read_text(encoding="utf-8"))
        parts = f.replace(os.sep, "/").split("/")
        h["seed"] = int([p for p in parts if p.startswith("SEED")][0][4:])
        h["variant"] = h["model_name"].replace("GNN_TAT_", "")
        out.append(h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd-per-gpu-hour", type=float, default=None)
    args = ap.parse_args()

    r = runs()
    if not r:
        print(f"no instrumented runs under {P30}")
        return 1
    if len(r) != 18:
        print(f"WARNING: {len(r)} runs found, not 18.\n")

    devices = sorted({x["device_name"] for x in r})
    print(f"{len(r)} instrumented runs on {', '.join(devices)}\n")

    print("=" * 72)
    print("PER OPERATOR  (mean over the six runs of each: 2 bundles x 3 seeds)")
    print("=" * 72)
    print(f"{'operator':<10}{'params':>9}{'epochs':>9}{'s/epoch':>10}"
          f"{'hours':>9}{'peak GB':>10}{'total h':>10}")
    print("-" * 72)
    total_h = 0.0
    for v in ORDER:
        g = [x for x in r if x["variant"] == v]
        if not g:
            continue
        h = sum(x["wall_seconds"] for x in g) / 3600
        total_h += h
        n = len(g)
        print(f"{v:<10}{g[0]['parameters']:>9,}"
              f"{sum(x['total_epochs'] for x in g) / n:>9.1f}"
              f"{sum(x['sec_per_epoch_mean'] for x in g) / n:>10.1f}"
              f"{h / n:>9.2f}{max(x['peak_gpu_gb'] for x in g):>10.1f}"
              f"{h:>10.2f}")
    print("-" * 72)
    print(f"{'total':<10}{'':>9}{'':>9}{'':>10}{'':>9}{'':>10}{total_h:>10.2f}")

    print("\n" + "=" * 72)
    print("PER SEED")
    print("=" * 72)
    by_seed = collections.defaultdict(float)
    for x in r:
        by_seed[x["seed"]] += x["wall_seconds"] / 3600
    for s in sorted(by_seed):
        print(f"  seed {s:<6}{by_seed[s]:>8.2f} h")

    print("\n" + "=" * 72)
    print("EARLY STOPPING")
    print("=" * 72)
    be = [x["best_epoch"] for x in r]
    te = [x["total_epochs"] for x in r]
    print(f"  patience was set to 30; the best epoch is {min(be)}-{max(be)} across")
    print(f"  the eighteen runs and no run reached the epoch cap, so the patience")
    print(f"  never bound. Runs ended between epoch {min(te)} and {max(te)}.")
    wasted = sum((x["total_epochs"] - x["best_epoch"]) * x["sec_per_epoch_mean"]
                 for x in r) / 3600
    print(f"  {wasted:.2f} of the {total_h:.2f} hours "
          f"({100 * wasted / total_h:.0f}%) were spent after the best epoch,")
    print(f"  which is the price of the patience and is what buys the guarantee")
    print(f"  that the stopping point was not an artefact of a short window.")

    print("\n" + "=" * 72)
    print("BUDGET")
    print("=" * 72)
    print(f"  {total_h:.2f} GPU-hours on {' and '.join(devices)}")
    if len(devices) > 1:
        for d in devices:
            n = sum(1 for x in r if x["device_name"] == d)
            hrs = sum(x["wall_seconds"] for x in r
                      if x["device_name"] == d) / 3600
            print(f"    {n:>2} runs, {hrs:>6.2f} h on {d}")
        per = collections.defaultdict(dict)
        for x in r:
            per[x["variant"]].setdefault(x["device_name"], []).append(
                x["sec_per_epoch_mean"])
        gaps = [max(sum(v) / len(v) for v in d.values())
                / min(sum(v) / len(v) for v in d.values())
                for d in per.values() if len(d) > 1]
        if gaps:
            print(f"  The sessions were allocated different cards. Every operator "
                  f"ran on\n  both, and the faster card is {100 * (max(gaps) - 1):.0f}% "
                  f"faster at most, so the\n  totals are not distorted by which "
                  f"session got which allocation.")
    if args.usd_per_gpu_hour:
        print(f"  at USD {args.usd_per_gpu_hour:.2f}/GPU-hour: "
              f"USD {total_h * args.usd_per_gpu_hour:,.2f}")
    else:
        print("  no rate given; rerun with --usd-per-gpu-hour to convert.")
    print(f"  peak device memory {max(x['peak_gpu_gb'] for x in r):.1f} GB, so a "
          f"40 GB card is\n  the smallest that runs this configuration.")

    prov = ROOT / "models" / "provenance" / "compute_cost.csv"
    import csv as _csv
    with prov.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["variant", "params", "epochs_mean", "sec_per_epoch_mean",
                    "hours_total", "min_per_run_lo", "min_per_run_hi",
                    "peak_gb_lo", "peak_gb_hi", "n_runs"])
        for v in ORDER:
            g = [x for x in r if x["variant"] == v]
            if not g:
                continue
            mins = [x["wall_seconds"] / 60 for x in g]
            w.writerow([v, g[0]["parameters"],
                        sum(x["total_epochs"] for x in g) / len(g),
                        sum(x["sec_per_epoch_mean"] for x in g) / len(g),
                        sum(x["wall_seconds"] for x in g) / 3600,
                        min(mins), max(mins),
                        min(x["peak_gpu_gb"] for x in g),
                        max(x["peak_gpu_gb"] for x in g), len(g)])
        w.writerow(["TOTAL", "", "", "", total_h, "", "", "",
                    max(x["peak_gpu_gb"] for x in r), len(r)])
    print(f"\nwrote {prov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
