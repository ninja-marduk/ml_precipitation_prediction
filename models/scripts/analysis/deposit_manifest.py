"""Determine what the archive must contain for every reported number to be recomputable.

The Code and data availability statement excludes the prediction arrays, the
trained checkpoints and the engineered feature sets "because of their size", and
offers them on request. Neither half of that holds: the excluded material totals
about 2 GB against a Zenodo limit of 50 GB per record, and GMD's policy does not
accept "on request" for material needed to reproduce results. Here the excluded
material is the input to most of the analyses.

Rather than deposit the repository wholesale, this walks the analysis scripts,
resolves the paths they open, and reports the set actually required, with sizes.
The output is the manifest the deposit should follow and the statement should
describe.

Usage:
    python models/scripts/analysis/deposit_manifest.py
    python models/scripts/analysis/deposit_manifest.py --assemble ../deposit
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# scripts whose outputs appear in the manuscript
CONSUMERS = [
    "models/scripts/analysis/anchoring_protocol.py",
    "models/scripts/analysis/graph_edge_budget_audit.py",
    "models/scripts/analysis/graph_structure_diagnostic.py",
    "models/scripts/analysis/multiseed_benchmark.py",
    "models/scripts/analysis/graph_leakage_audit.py",
    "models/scripts/analysis/statistical_tests_audit.py",
    "models/scripts/analysis/factorial_blocked_analysis.py",
    "models/scripts/figures/benchmark/fusion_decomposition.py",
    "models/scripts/figures/benchmark/beyond_aggregate_corrected.py",
    "models/scripts/figures/benchmark/naive_baselines.py",
]

# path fragments the scripts compose at runtime, which a static scan cannot resolve
RUNTIME = [
    ("prediction arrays and targets, three seeds, three models",
     ["models/output/V2_Enhanced_Models/SEED*/map_exports/H12/BASIC/ConvLSTM_Bidirectional",
      "models/output/V4_GNN_TAT_Models/SEED*/map_exports/H12/BASIC/GNN_TAT_GAT",
      "models/output/V10_Late_Fusion/SEED*"]),
    ("per-seed metric files behind the multi-seed factorial",
     ["models/output/V2_Enhanced_Models/SEED*/*.csv",
      "models/output/V4_GNN_TAT_Models/SEED*/*.csv",
      "models/output/V10_Late_Fusion/SEED*/*.csv",
      "models/output/V4_GNN_TAT_Models/_archive_2026-04_leaked_graph/*.csv"]),
    ("engineered feature set, the input to the CPU analyses",
     ["data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs_with_onehot_elevation_clean.nc"]),
    ("provenance record",
     ["models/provenance/*"]),
]


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


def scan_literals():
    """Paths written literally in the consumer scripts."""
    found = set()
    for rel in CONSUMERS:
        p = ROOT / rel
        if not p.exists():
            print(f"  [missing script] {rel}")
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'"([^"]*\.(?:nc|npy|csv))"', src):
            found.add(m.group(1))
        for m in re.finditer(r"'([^']*\.(?:nc|npy|csv))'", src):
            found.add(m.group(1))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble", type=Path, default=None,
                    help="copy the required set into this directory")
    args = ap.parse_args()

    print("=" * 78)
    print("FILE NAMES WRITTEN LITERALLY IN THE ANALYSIS SCRIPTS")
    print("=" * 78)
    for f in scan_literals():
        print(f"  {f}")

    print()
    print("=" * 78)
    print("REQUIRED SET, RESOLVED")
    print("=" * 78)
    grand, manifest = 0, []
    for label, patterns in RUNTIME:
        files, size = [], 0
        for pat in patterns:
            for p in ROOT.glob(pat):
                if p.is_dir():
                    for q in p.rglob("*"):
                        if q.is_file():
                            files.append(q); size += q.stat().st_size
                elif p.is_file():
                    files.append(p); size += p.stat().st_size
        grand += size
        manifest.append((label, files, size))
        print(f"\n{label}")
        print(f"  {len(files)} files, {human(size)}")
        for q in files[:4]:
            print(f"    {q.relative_to(ROOT)}")
        if len(files) > 4:
            print(f"    ... and {len(files)-4} more")

    print()
    print("-" * 78)
    print(f"TOTAL REQUIRED: {human(grand)}")
    print("Zenodo accepts 50 GB per record, so size is not a reason to exclude any of it.")

    ckpt = list(ROOT.glob("models/output/**/*.pt"))
    csize = sum(p.stat().st_size for p in ckpt)
    print(f"\nTrained checkpoints, not required to recompute any reported number:")
    print(f"  {len(ckpt)} files, {human(csize)}. Deposit them anyway if the record has room;")
    print("  they are what a reader needs to run the models rather than to check the numbers.")

    if args.assemble:
        dst = args.assemble.resolve()
        print(f"\nassembling into {dst}")
        for label, files, _ in manifest:
            for q in files:
                target = dst / q.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(q, target)
        print(f"done: {sum(len(f) for _, f, _ in manifest)} files")


if __name__ == "__main__":
    main()
