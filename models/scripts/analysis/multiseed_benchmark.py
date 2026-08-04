"""Consolidate every architecture family into one multi-seed benchmark table.

The manuscript's results tables quote single numbers taken from the best seed,
which is the reporting practice the protocol sets out to correct. This script
rebuilds the comparison from the per-seed metric files so that every entry is a
mean and a spread over the seeds that were actually run, and so that entries with
only one seed are labelled as such rather than presented alongside the others as
if they were comparable.

Two aggregations are reported for each configuration, because they answer
different questions and the manuscript has conflated them:

  mean over horizons  what the model does across the forecast range
  peak horizon        the single best horizon, which is what a maximum-selecting
                      report quotes and what inflates with the number of seeds
                      and horizons searched

Usage: python models/scripts/analysis/multiseed_benchmark.py [--csv out.csv]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models" / "output"


def r2col(df):
    for c in df.columns:
        if c.lower() in ("r^2", "r2"):
            return c
    return None


def seed_of(path):
    m = re.search(r"SEED(\d+)", str(path))
    return int(m.group(1)) if m else None


def collect(pattern, model_col=None, feat_col=None, label=None, feat=None):
    """Return {(model, feat): {seed: DataFrame of 12 horizons}}."""
    out = {}
    for p in sorted(glob.glob(str(OUT / pattern))):
        s = seed_of(p)
        if s is None:
            continue
        df = pd.read_csv(p)
        rc = r2col(df)
        if rc is None or "H" not in df.columns:
            continue
        if model_col and model_col in df.columns:
            for (m, f), g in df.groupby([model_col, feat_col]):
                out.setdefault((m, f), {})[s] = g.sort_values("H")
        else:
            out.setdefault((label, feat), {})[s] = df.sort_values("H")
    return out


def summarise(entries):
    rows = []
    for (model, feat), per_seed in sorted(entries.items()):
        means, peaks = [], []
        for s, g in sorted(per_seed.items()):
            rc = r2col(g)
            means.append(g[rc].mean())
            peaks.append(g[rc].max())
        n = len(means)
        rows.append({
            "model": model, "features": feat, "seeds": n,
            "R2_mean": np.mean(means),
            "R2_mean_sd": np.std(means, ddof=1) if n > 1 else np.nan,
            "R2_peak": np.mean(peaks),
            "R2_peak_sd": np.std(peaks, ddof=1) if n > 1 else np.nan,
            "R2_peak_best_seed": max(peaks),
            "inflation": max(peaks) - np.mean(peaks),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    tables = {}

    # ConvLSTM family, three seeds where present
    tables["ConvLSTM (V2)"] = collect(
        "V2_Enhanced_Models/SEED*/metrics_spatial_v2_refactored_h12.csv",
        model_col="Model", feat_col="Experiment")

    # GNN-TAT, archived factorial: three seeds, GAT/GCN/SAGE x BASIC/PAFC.
    # Usable despite the leaked graph: the paired ablation measures that defect
    # at -0.002 R2, an order of magnitude below the seed spread.
    fac = OUT / "V4_GNN_TAT_Models/_archive_2026-04_leaked_graph/metrics_factorial_consolidated.csv"
    if fac.exists():
        d = pd.read_csv(fac)
        ent = {}
        for (v, f), g in d.groupby(["variant", "feat"]):
            for s, gg in g.groupby("seed"):
                ent.setdefault((f"GNN-TAT ({v})", f), {})[s] = gg.sort_values("H")
        tables["GNN-TAT (V4, archived graph)"] = ent

    # GNN-TAT, corrected graph: three seeds, GAT + BASIC only
    tables["GNN-TAT (V4, corrected graph)"] = collect(
        "V4_GNN_TAT_Models/SEED*/metrics_spatial_v4_gnn_tat_h12.csv",
        model_col="Model", feat_col="Experiment")

    # Late Fusion, three seeds, no model/feature columns
    tables["Late Fusion (V10)"] = collect(
        "V10_Late_Fusion/SEED*/v10_metrics.csv",
        label="Late Fusion", feat="ensemble")

    frames = []
    for family, ent in tables.items():
        if not ent:
            print(f"[skip] {family}: no per-seed files found")
            continue
        df = summarise(ent)
        df.insert(0, "family", family)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    print("Multi-seed benchmark. 'best seed' is what a maximum-selecting report quotes;")
    print("'inflation' is how much that exceeds the multi-seed expectation.\n")
    hdr = (f"{'family':<32}{'model':<18}{'feat':<8}{'n':>3}"
           f"{'R2 mean':>16}{'R2 peak':>16}{'best seed':>11}{'infl.':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in all_df.sort_values("R2_peak", ascending=False).iterrows():
        ms = f"{r.R2_mean:.4f}" + (f" +/-{r.R2_mean_sd:.3f}" if r.seeds > 1 else "  single ")
        ps = f"{r.R2_peak:.4f}" + (f" +/-{r.R2_peak_sd:.3f}" if r.seeds > 1 else "  single ")
        print(f"{r.family:<32}{str(r.model):<18}{str(r.features):<8}{int(r.seeds):>3}"
              f"{ms:>16}{ps:>16}{r.R2_peak_best_seed:>11.4f}{r.inflation:>8.4f}")

    multi = all_df[all_df.seeds > 1]
    print(f"\nAcross the {len(multi)} multi-seed configurations:")
    print(f"  median seed spread on peak R2: {multi.R2_peak_sd.median():.4f}")
    print(f"  median inflation from quoting the best seed: {multi.inflation.median():.4f}")
    print(f"  largest inflation: {multi.inflation.max():.4f} "
          f"({multi.loc[multi.inflation.idxmax(), 'model']} / "
          f"{multi.loc[multi.inflation.idxmax(), 'features']})")
    print("\nA difference between two architectures smaller than the seed spread is not")
    print("evidence about architecture. Read the ranking below that threshold as unresolved.")

    if args.csv:
        all_df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
