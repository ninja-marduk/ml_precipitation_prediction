"""Assert every load-bearing number in the manuscript against the file that produces it.

The acceptance criterion for this revision was that each reported quantity trace to a
named script or provenance record. That criterion was stated but never enforced, and
five reviewers converged on the same three tables as a result. Enforcement is what this
module is: it is meant to be run until it is clean, not read.

Three kinds of check run here.

  ANCHORS      A registry binds a regular expression over the LaTeX sources to a key in
               a value store built from the provenance files. Every match of the
               expression must equal the stored value within a stated tolerance. An
               expression that matches nothing is a failure too, because it means the
               sentence it was written against has moved and the binding no longer
               guards anything.

  UNIQUENESS   An anchor may match many places. If two of them disagree, the manuscript
               states one quantity two ways, which is the defect that survived both
               previous revisions. Each anchor therefore reports the set of distinct
               values it found.

  STRUCTURE    Every tabular is parsed and any summary row (mean, median, average,
               overall) is recomputed from the data rows above it, unweighted and, where
               a count column exists, weighted by it. A summary row that matches neither
               is reported with both candidates so the author can see which aggregation
               was intended.

Nothing here is inferred from the prose. A number the registry does not mention is not
checked, so the registry is the statement of what has been verified, and its coverage
is reported at the end.

Usage:
    python models/scripts/analysis/manuscript_numbers_audit.py
    python models/scripts/analysis/manuscript_numbers_audit.py --verbose
    python models/scripts/analysis/manuscript_numbers_audit.py --only fusion
Exit status is 1 if any check fails, so it can gate a build.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROV = ROOT / "models" / "provenance"
OUT = ROOT / "models" / "output"
PAPERS = ROOT / ".docs" / "papers" / "5"

TEX_FILES = {
    "paper": PAPERS / "paper_gmd.tex",
    "supp": PAPERS / "supplement.tex",
}


# --------------------------------------------------------------------------- #
# 1. The value store: everything the manuscript is allowed to claim            #
# --------------------------------------------------------------------------- #

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _num(tok):
    """Parse a number written by any of our scripts or by LaTeX."""
    if tok is None:
        return None
    tok = (tok.replace("\u2212", "-").replace("$", "").replace("{", "").replace("}", "")
           .replace("\\,", "").replace("\\%", "").replace("%", "").replace(",", "")
           .replace("+", "").strip())
    tok = re.sub(r"\\text(bf|it|rm)?", "", tok)
    try:
        return float(tok)
    except ValueError:
        return None


def _cols(line: str):
    """Trailing numeric fields of a fixed-width report line."""
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+", line)]


def store() -> dict:
    """Build the store from the provenance records. Missing files fail loudly."""
    s: dict[str, float] = {}
    missing: list[str] = []

    # -- fusion decomposition, window-blocked folds --------------------------
    p = PROV / "fusion_decomposition_multiseed.txt"
    if p.exists():
        txt = _read(p)
        rows = {
            "conv_raw": "ConvLSTM raw", "gnn_raw": "GNN-TAT raw",
            "average": "simple average", "conv_recal": "ConvLSTM recalibrated",
            "gnn_recal": "GNN-TAT recalibrated", "ridge": "Ridge over both",
            "ridge_shuffled": "Ridge, folds shuffled",
            "published": "Late Fusion as published",
            "best_recal": "best recalibrated",
        }
        for key, label in rows.items():
            m = re.search(re.escape(label) + r"\s+(.*)", txt)
            if m:
                v = _cols(m.group(1))
                if len(v) >= 5:
                    s[f"blocked.{key}.s42"] = v[0]
                    s[f"blocked.{key}.s123"] = v[1]
                    s[f"blocked.{key}.s456"] = v[2]
                    s[f"blocked.{key}.mean"] = v[3]
                    s[f"blocked.{key}.sd"] = v[4]
        for key, label in (("comb_only", "combination only"),
                           ("cal_only", "calibration only"),
                           ("both", "both (Ridge) over best raw"),
                           ("incr", "INCREMENTAL value of combining")):
            m = re.search(re.escape(label) + r"[^\n]*?\s([-+]\d\.\d+)\s+(\d\.\d+)", txt)
            if m:
                s[f"blocked.d_{key}.mean"] = float(m.group(1))
                s[f"blocked.d_{key}.sd"] = float(m.group(2))
        m = re.search(r"(?:paired difference|total, published minus blocked refit)"
                      r"\s+:\s+([-+]\d\.\d+)\s+\+/-\s+(\d\.\d+)", txt)
        if m:
            s["blocked.pub_minus_refit.mean"] = float(m.group(1))
            s["blocked.pub_minus_refit.sd"] = float(m.group(2))
        m = re.search(r"paired t \(n=3, 2 d\.f\.\)\s+:\s+([-\d.]+)", txt)
        if m:
            s["blocked.incr_t"] = float(m.group(1))
        m = re.search(r"calibration accounts for (\d+)% ", txt)
        if m:
            s["blocked.cal_share_pct"] = float(m.group(1))
        # per-seed detail of the blocked refit: R2, RMSE, w_Conv, w_GNN, b, bias
        for lbl, key in (("42", "s42"), ("123", "s123"), ("456", "s456"),
                         ("mean", "mean"), ("s.d.", "sd")):
            m = re.search(r"^\s*" + re.escape(lbl) + r"\s+(-?\d\.\d{4}\s.*)$", txt, re.M)
            if m:
                v = _cols(m.group(1))
                if len(v) >= 6:
                    for i, f in enumerate(["r2", "rmse", "wconv", "wgnn", "b", "bias"]):
                        s[f"ridge.{key}.{f}"] = v[i]
        m = re.search(r"convolutional share of the mean weight: ([\d.]+)%", txt)
        if m:
            s["ridge.conv_share_pct"] = float(m.group(1))
        m = re.search(r"graph ([\d.]+)%\)", txt)
        if m:
            s["ridge.gnn_share_pct"] = float(m.group(1))
        m = re.search(r"seed 42: ConvLSTM slope ([\d.]+) intercept ([-+][\d.]+) mm \| "
                      r"GNN-TAT slope ([\d.]+) intercept ([-+][\d.]+)", txt)
        if m:
            s["cal.s42.conv_slope"] = float(m.group(1))
            s["cal.s42.conv_intercept"] = float(m.group(2))
            s["cal.s42.gnn_slope"] = float(m.group(3))
            s["cal.s42.gnn_intercept"] = float(m.group(4))
        for label, key in (("of which the fold scheme", "fold"),
                           ("of which not the folds", "unattributed")):
            m = re.search(re.escape(label) + r"\s+:\s+([-+]\d\.\d+)\s+\+/-\s+(\d\.\d+)", txt)
            if m:
                s[f"split.{key}.mean"] = float(m.group(1))
                s[f"split.{key}.sd"] = float(m.group(2))
    else:
        missing.append(str(p))

    # -- the significance-test audit -----------------------------------------
    p = PROV / "statistical_tests_audit.txt"
    if p.exists():
        txt = _read(p)
        m = re.search(r"family size: (\d+) pairwise tests", txt)
        if m:
            s["tests.family_declared_in_audit"] = float(m.group(1))
        for label, key in (("GNN-TAT vs ConvLSTM (RMSE)", "gnn_conv_rmse"),
                           ("GNN-TAT vs ConvLSTM (R2)", "gnn_conv_r2"),
                           ("FNO vs FNO-ConvLSTM", "fno"),
                           ("KCE vs BASIC (GNN)", "kce_gnn"),
                           ("KCE vs BASIC (ConvLSTM)", "kce_conv"),
                           ("Stacking vs Best Individual", "stacking"),
                           ("Stratified vs GNN-TAT", "stratified")):
            m = re.search(re.escape(label) + r"\s+(\d\.\d{4})\s+(\d\.\d{4})", txt)
            if m:
                s[f"tests.{key}.p_raw"] = float(m.group(1))
                s[f"tests.{key}.holm8"] = float(m.group(2))
        m = re.search(r"(\d+) of 8 rows report a value the named test cannot produce", txt)
        if m:
            s["tests.n_inadmissible"] = float(m.group(1))
        m = re.search(r"(\d+) of 8 survive Holm", txt)
        if m:
            s["tests.n_surviving"] = float(m.group(1))
        m = re.search(r"^\s+8\s+33\s+[\d.]+\s+([\d.]+)", txt, re.M)
        if m:
            s["tests.cd_k8_n33"] = float(m.group(1))
        # Holm over the family that remains once the three inadmissible rows are
        # withdrawn. The supplement corrects over five; the audit script corrects
        # over eight. Both are computed so the manuscript cannot quote a third.
        admissible = ["gnn_conv_rmse", "gnn_conv_r2", "fno", "stacking", "stratified"]
        ps = [(k, s[f"tests.{k}.p_raw"]) for k in admissible
              if f"tests.{k}.p_raw" in s]
        ps.sort(key=lambda kv: kv[1])
        run = 0.0
        for i, (k, pv) in enumerate(ps):
            run = max(run, pv * (len(ps) - i))
            s[f"tests.{k}.holm5"] = min(run, 1.0)
        s["tests.family_admissible"] = float(len(ps))
        s["tests.n_withdrawn"] = 8.0 - len(ps)
        s["tests.n_surviving_holm5"] = float(sum(
            1 for k, _ in ps if s[f"tests.{k}.holm5"] < 0.05))
    else:
        missing.append(str(p))

    # -- fusion, purged holdout ----------------------------------------------
    p = PROV / "fusion_purged_holdout.txt"
    if p.exists():
        txt = _read(p)
        for label, key in (("mean", "mean"), ("s.d.", "sd")):
            m = re.search(r"^\s*" + re.escape(label) + r"\s+(.*)$", txt, re.M)
            if m:
                v = _cols(m.group(1))
                if len(v) >= 4:
                    s[f"purged.raw_best.{key}"] = v[0]
                    s[f"purged.recal.{key}"] = v[1]
                    s[f"purged.ridge.{key}"] = v[2]
                    s[f"purged.ridge_insample.{key}"] = v[3]
        m = re.search(r"([-+]\d\.\d+) \+/- (\d\.\d+) \(paired across seeds\)", txt)
        if m:
            s["purged.incr.mean"] = float(m.group(1))
            s["purged.incr.sd"] = float(m.group(2))
        m = re.search(r"recalibration minus the same raw base learner:\s*\n\s*"
                      r"([-+]\d\.\d+) \+/- (\d\.\d+)", txt)
        if m:
            s["purged.d_recal.mean"] = float(m.group(1))
            s["purged.d_recal.sd"] = float(m.group(2))
        m = re.search(r"fusion minus the raw base learner picked on training data:\s*\n"
                      r"\s*([-+]\d\.\d+) \+/- (\d\.\d+)", txt)
        if m:
            s["purged.d_fusion.mean"] = float(m.group(1))
            s["purged.d_fusion.sd"] = float(m.group(2))
        m = re.search(r"in-sample minus out-of-sample for the Ridge: ([-+]\d\.\d+)", txt)
        if m:
            s["purged.insample_gap"] = float(m.group(1))
        m = re.search(r"purged split: (\d+) training windows, (\d+) test windows", txt)
        if m:
            s["purged.n_train"] = float(m.group(1))
            s["purged.n_test"] = float(m.group(2))
        m = re.search(r"embargo of (\d+) windows", txt)
        if m:
            s["purged.embargo"] = float(m.group(1))
        for seed, key in ((42, "s42"), (123, "s123"), (456, "s456")):
            m = re.search(r"^\s*%d\s+(.*)$" % seed, txt, re.M)
            if m:
                v = _cols(m.group(1))
                if len(v) >= 4:
                    s[f"purged.ridge.{key}"] = v[2]

    else:
        missing.append(str(p))

    # -- window overlap -------------------------------------------------------
    p = PROV / "window_overlap_audit.txt"
    if p.exists():
        txt = _read(p)
        m = re.search(r"training window: (\d+) of (\d+)\s+\(([\d.]+)%\)", txt)
        if m:
            s["overlap.random_pct"] = float(m.group(3))
        m = re.search(r"contiguous folds: \d+ of \d+\s+\(([\d.]+)%\)", txt)
        if m:
            s["overlap.contiguous_pct"] = float(m.group(1))
        m = re.search(r"pairs where t\[s,1:\] == t\[s\+1,:-1\] : (\d+) of (\d+)", txt)
        if m:
            s["overlap.consecutive_pairs"] = float(m.group(1))
        m = re.search(r"root-level.*?R2 = ([\d.]+)", txt, re.S)
        if m:
            s["gnn.array_released.pooled"] = float(m.group(1))
        m = re.search(r"SEED42/.*?R2 = ([\d.]+)", txt, re.S)
        if m:
            s["gnn.array_corrected.pooled"] = float(m.group(1))
    else:
        missing.append(str(p))

    # -- zero-cost references on the 33 released windows ----------------------
    p = PROV / "naive_baselines.txt"
    if p.exists():
        txt = _read(p)
        blocks = {"Climatology": "clim", "Persistence": "pers", "Seasonal-naive": "seas"}
        for label, key in blocks.items():
            m = re.search(r"\[" + label + r"[^\]]*\]\s*\n\s*POOLED \(all H\): "
                          r"R2=([-+][\d.]+)\s+RMSE=\s*([\d.]+)\s+MAE=\s*([\d.]+)\s+"
                          r"per-cell NSE=([-+][\d.]+)", txt)
            if m:
                s[f"anchor33.{key}.pooled"] = float(m.group(1))
                s[f"anchor33.{key}.rmse"] = float(m.group(2))
                s[f"anchor33.{key}.mae"] = float(m.group(3))
                s[f"anchor33.{key}.percell"] = float(m.group(4))
        for label, key in (("Climatology", "clim"), ("Seasonal-naive", "seas"),
                           ("Persistence", "pers"),
                           ("ConvLSTM-Bidir (V2)", "conv"),
                           ("GNN-TAT-GAT (V4, as released)", "gnn_released"),
                           ("GNN-TAT-GAT (V4, corrected rerun)", "gnn_corrected"),
                           ("Late Fusion (V10, root, superseded)", "lf_root"),
                           ("Late Fusion (V10, seed 42)", "lf")):
            m = re.search(r"^" + re.escape(label) + r"\s+(.*)$", txt, re.M)
            if m:
                v = _cols(m.group(1))
                if len(v) >= 5:
                    s[f"pooled33.{key}.r2"] = v[0]
                    s[f"pooled33.{key}.h5"] = v[1]
                    s[f"pooled33.{key}.h12"] = v[2]
                    s[f"pooled33.{key}.rmse"] = v[3]
                    s[f"pooled33.{key}.percell"] = v[4]
        m = re.search(r"targets: S=(\d+) windows, H=(\d+) horizons", txt)
        if m:
            s["design.n_val_windows"] = float(m.group(1))
            s["design.horizons"] = float(m.group(2))
    else:
        missing.append(str(p))

    # -- the same anchoring engine on every admissible origin -----------------
    p = PROV / "anchoring_case_study.txt"
    if p.exists():
        txt = _read(p)
        m = re.search(r"Boyaca \(case study\)\s+\S+[^\d\-]*(.*)$", txt, re.M)
        if m:
            v = _cols(m.group(1))
            if len(v) >= 6:
                for i, k in enumerate(["clim.pooled", "seas.pooled", "pers.pooled",
                                       "clim.percell", "seas.percell", "pers.percell"]):
                    s[f"anchorAll.{k}"] = v[i]
        m = re.search(r"origins=(\d+)\s+cells=(\d+)", txt)
        if m:
            s["anchorAll.n_origins"] = float(m.group(1))
            s["anchorAll.n_cells"] = float(m.group(2))
    else:
        missing.append(str(p))

    # -- eight regimes --------------------------------------------------------
    p = PROV / "anchoring_eight_regimes.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        for r in rows:
            slug = re.sub(r"[^a-z]", "", r["region"].lower().split("(")[0].strip())[:12]
            for col in ("pooled_climatology", "pooled_seasonal_naive", "pooled_persistence",
                        "percell_climatology", "percell_seasonal_naive", "percell_persistence"):
                if r.get(col):
                    s[f"regime.{slug}.{col}"] = float(r[col])
            if r.get("n_cells"):
                s[f"regime.{slug}.cells"] = float(r["n_cells"])
        for col in ("pooled_climatology", "pooled_seasonal_naive", "pooled_persistence",
                    "percell_climatology", "percell_seasonal_naive", "percell_persistence"):
            vals = [float(r[col]) for r in rows if r.get(col)]
            if vals:
                s[f"regime.median.{col}"] = statistics.median(vals)
                s[f"regime.min.{col}"] = min(vals)
                s[f"regime.max.{col}"] = max(vals)
        pc = [float(r["percell_climatology"]) for r in rows if r.get("percell_climatology")]
        po = [float(r["pooled_climatology"]) for r in rows if r.get("pooled_climatology")]
        if len(pc) == len(po) and len(pc) > 2:
            s["regime.clim_pooled_percell_r"] = statistics.correlation(po, pc)
        s["regime.n"] = float(len(rows))
        # A row the engine wrote carries every field it emits. A row missing
        # mean_mm, or carrying values rounded to three decimals where the engine
        # writes full precision, was entered by hand and is not a measurement.
        suspect = [r["region"] for r in rows
                   if not r.get("mean_mm")
                   or all(len(v.split(".")[-1]) <= 3
                          for k, v in r.items()
                          if k.startswith(("pooled_", "percell_")) and v)]
        s["regime.hand_entered"] = float(len(suspect))
        s["_regime_suspect_names"] = suspect
    else:
        missing.append(str(p))

    # -- the corrected factorial, which supersedes the archived one -----------
    p = PROV / "benchmark_p30.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        for r in rows:
            k = f"p30.{r['variant']}.{r['features']}"
            for f in ("r2_mean", "r2_mean_sd", "r2_peak", "r2_peak_sd",
                      "rmse", "rmse_sd", "bias", "inflation"):
                s[f"{k}.{f}"] = float(r[f])
        s["p30.median_inflation"] = statistics.median(
            float(r["inflation"]) for r in rows)
        s["p30.median_peak_sd"] = statistics.median(
            float(r["r2_peak_sd"]) for r in rows)
        s["p30.max_inflation"] = max(float(r["inflation"]) for r in rows)
        s["p30.n_configs"] = float(len(rows))
        # the bundle effect as the text states it: each bundle averaged over
        # its three operators, and the two averages differenced
        by_bundle = {b: [float(r["r2_mean"]) for r in rows if r["features"] == b]
                     for b in ("BASIC", "PAFC")}
        for b, v in by_bundle.items():
            s[f"p30.{b}.mean_over_variants"] = statistics.mean(v)
        s["p30.bundle_gap"] = (s["p30.PAFC.mean_over_variants"]
                               - s["p30.BASIC.mean_over_variants"])
        s["p30.pafc_spread"] = max(by_bundle["PAFC"]) - min(by_bundle["PAFC"])
        s["p30.max_mean_sd"] = max(float(r["r2_mean_sd"]) for r in rows)
        s["p30.min_mean_sd"] = min(float(r["r2_mean_sd"]) for r in rows)
    else:
        missing.append(str(p))

    # -- measured compute, from the instrumented training loop ---------------
    p = PROV / "compute_cost.csv"
    if p.exists():
        for r in csv.DictReader(p.open(encoding="utf-8")):
            k = f"cost.{r['variant']}"
            for f in ("hours_total", "min_per_run_lo", "min_per_run_hi",
                      "peak_gb_lo", "peak_gb_hi", "sec_per_epoch_mean"):
                if r[f]:
                    s[f"{k}.{f}"] = float(r[f])
    else:
        missing.append(str(p))

    # -- corrected factorial at the twelfth lead, for the supplement table ----
    p = PROV / "factorial_p30_lead12.csv"
    if p.exists():
        for r in csv.DictReader(p.open(encoding="utf-8")):
            k = f"p30l12.{r['variant']}.{r['feat']}"
            for f in ("R^2_mean", "R^2_std", "RMSE_mean", "RMSE_std"):
                s[f"{k}.{f}"] = float(r[f])
    else:
        missing.append(str(p))

    # -- multi-seed factorial, archived: kept for the historical claims -------
    p = PROV / "benchmark_multiseed.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        for r in rows:
            fam = "gnncorr" if "corrected" in r["family"] else \
                  "gnn" if "GNN" in r["family"] else \
                  "conv" if "ConvLSTM" in r["family"] else "lf"
            op = re.sub(r"[^A-Z]", "", r["model"]) or "LF"
            key = f"ms.{fam}.{op}.{r['features']}"
            s[f"{key}.mean"] = float(r["R2_mean"])
            s[f"{key}.mean_sd"] = float(r["R2_mean_sd"])
            s[f"{key}.peak"] = float(r["R2_peak"])
            s[f"{key}.peak_sd"] = float(r["R2_peak_sd"])
            s[f"{key}.infl"] = float(r["inflation"])
        s["ms.median_inflation"] = statistics.median(float(r["inflation"]) for r in rows)
        s["ms.median_peak_sd"] = statistics.median(float(r["R2_peak_sd"]) for r in rows)
        s["ms.n_configs"] = float(len(rows))
    else:
        missing.append(str(p))

    # -- blocked factorial, corrected pipeline --------------------------------
    p = PROV / "factorial_p30_blocked.txt"
    if p.exists():
        txt = _read(p)
        for src, key in (("feature bundle", "feat"), ("variant", "variant"),
                         ("feature x variant", "inter"), ("block (seed)", "block")):
            m = re.search(r"^" + re.escape(src) + r"\s+(.*)$", txt, re.M)
            if m:
                v = _cols(m.group(1))
                if len(v) >= 3:
                    s[f"p30rcb.{key}.F"] = v[-2]
                    s[f"p30rcb.{key}.p"] = v[-1]
        for src, key in (("bundle", "feat"), ("variant", "variant")):
            m = re.search(r"^\s+" + src + r"\s+observed spread [\d.]+\s+p = ([\d.]+)",
                          txt, re.M)
            if m:
                s[f"p30perm.{key}.p"] = float(m.group(1))
        m = re.search(r"block absorbs ([\d.]+)% of the total", txt)
        if m:
            s["p30rcb.block_ss_pct"] = float(m.group(1))
    else:
        missing.append(str(p))

    # -- blocked factorial, archived: kept for the historical comparison ------
    p = PROV / "factorial_blocked_analysis.txt"
    if p.exists():
        txt = _read(p)
        for src, key in (("feature bundle", "feat"), ("variant", "variant"),
                         ("feature x variant", "inter"), ("block (seed)", "block")):
            m = re.search(r"^" + re.escape(src) + r"\s+(.*)$", txt, re.M)
            if m:
                v = _cols(m.group(1))
                if len(v) >= 3:
                    s[f"rcb.{key}.F"] = v[-2]
                    s[f"rcb.{key}.p"] = v[-1]
        for src, key in (("feat", "feat"), ("variant", "variant")):
            m = re.search(r"^\s+" + src + r"\s+p = ([\d.]+)", txt, re.M)
            if m:
                s[f"perm.{key}.p"] = float(m.group(1))
        m = re.search(r"absorbs ([\d.]+)% of the total sum of squares", txt)
        if m:
            s["rcb.block_ss_pct"] = float(m.group(1))
    else:
        missing.append(str(p))

    # -- stratified comparison, one array per model --------------------------
    p = PROV / "stratified_comparison.txt"
    if p.exists():
        txt = _read(p)
        cols = ["conv", "gnn", "gnnpre", "lf"]
        for label, key in (("ALL", "all"), ("Low <1500 m", "low"),
                           ("Medium 1500-2800 m", "mid"), ("High >2800 m", "high"),
                           ("Short H1-4", "hshort"), ("Medium H5-8", "hmid"),
                           ("Long H9-12", "hlong")):
            m = re.search(r"^" + re.escape(label) + r"\s+(.*)$", txt, re.M)
            if m:
                v = _cols(m.group(1))
                if len(v) >= 4:
                    for i, c in enumerate(cols):
                        s[f"strat.{key}.{c}"] = v[i]
        m = re.search(r"LateFusion\s+[\d.]+ -> [\d.]+\s+\+([\d.]+)%", txt)
        if m:
            s["strat.lf_degradation_pct"] = float(m.group(1))
        m = re.search(r"degradation over the sweep: ([\d.]+)% to ([\d.]+)%", txt)
        if m:
            s["strat.sweep_min_pct"] = float(m.group(1))
            s["strat.sweep_max_pct"] = float(m.group(2))
        m = re.search(r"beats the convolutional one: (\d+) of (\d+)\s+\(([\d.]+)%\)", txt)
        if m:
            s["strat.gnn_wins_cells"] = float(m.group(1))
            s["strat.n_cells"] = float(m.group(2))
            s["strat.gnn_wins_pct"] = float(m.group(3))
        m = re.search(r"below 0\.2 and the fusion reaches 0\.5: (\d+) of", txt)
        if m:
            s["strat.rescue_cells"] = float(m.group(1))
        m = re.search(r"both base learners are below 0\.2: (\d+) ", txt)
        if m:
            s["strat.both_low_cells"] = float(m.group(1))
    else:
        missing.append(str(p))

    # -- anomaly skill, connectivity and conformal intervals -----------------
    p = PROV / "beyond_aggregate.txt"
    if p.exists():
        txt = _read(p)
        for name, key in (("ConvLSTM-Bidir", "conv"), ("GNN-TAT-GAT", "gnn"),
                          ("Late Fusion", "lf")):
            for h in (1, 12):
                m = re.search(re.escape(name) + r"\s+" + str(h) +
                              r"\s+([-\d.]+)\s+([\d.]+)%\s+([-\d.]+)\s+([-\d.]+)", txt)
                if m:
                    s[f"acc.{key}.h{h}"] = float(m.group(1))
                    s[f"acc.{key}.h{h}.pct"] = float(m.group(2))
                    s[f"acc.{key}.h{h}.pattern"] = float(m.group(3))
                    s[f"acc.{key}.h{h}.pooled"] = float(m.group(4))
        for tag, key in (("RAW", "raw"), ("DESEASONALIZED", "deseas")):
            blk = txt.split(f"[{tag}]")[-1][:600] if f"[{tag}]" in txt else ""
            m = re.search(r"rho>0\.8 : [\d,]+ \(([\d.]+)% of far pairs\)", blk)
            if m:
                s[f"conn.{key}.far_pct"] = float(m.group(1))
            m = re.search(r"beta_\|dElev\|=([-\d.]+)\s+R2=([\d.]+)", blk)
            if m:
                s[f"conn.{key}.beta_elev"] = float(m.group(1))
                s[f"conn.{key}.r2"] = float(m.group(2))
        m = re.search(r"mean truncated width ([\d.]+) mm", txt)
        if m:
            s["conformal.lf_width"] = float(m.group(1))
        m = re.search(r"Climatology : coverage [\d.]+ \| mean truncated width ([\d.]+)", txt)
        if m:
            s["conformal.clim_width"] = float(m.group(1))
        m = re.search(r"below zero in ([\d.]+)% of cells", txt)
        if m:
            s["conformal.neg_lower_pct"] = float(m.group(1))
        m = re.search(r"(\d+) calibration windows, (\d+) test windows", txt)
        if m:
            s["conformal.n_cal"] = float(m.group(1))
            s["conformal.n_test"] = float(m.group(2))
    else:
        missing.append(str(p))

    # -- variograms of the residual field ------------------------------------
    p = ROOT / "scripts" / "benchmark" / "output" / "variogram_results.csv"
    if p.exists():
        rows = {r["Model"]: r for r in csv.DictReader(p.open(encoding="utf-8"))}
        alias = {"V2_ConvLSTM": "conv", "V4_GNN_TAT": "gnn", "V10_Late_Fusion": "lf"}
        for k, a in alias.items():
            if k in rows:
                s[f"vario.{a}.sill"] = float(rows[k]["error_sill"])
                s[f"vario.{a}.range"] = float(rows[k]["error_range_km"])
                s[f"vario.{a}.nugget"] = float(rows[k]["error_nugget"])
                s[f"vario.{a}.fit"] = float(rows[k]["error_r2_fit"])
        if "V2_ConvLSTM" in rows and "V10_Late_Fusion" in rows:
            s["vario.lf_vs_conv_pct"] = 100 * (1 - s["vario.lf.sill"] / s["vario.conv.sill"])
        if "V2_ConvLSTM" in rows:
            s["vario.obs_range"] = float(rows["V2_ConvLSTM"]["observation_range_km"])
    else:
        missing.append(str(p))

    # -- Late Fusion per horizon, three seeds --------------------------------
    p = OUT / "V10_Late_Fusion" / "metrics_multiseed_consolidated.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        r2 = {int(r["H"]): float(r["R^2_mean"]) for r in rows}
        sd = {int(r["H"]): float(r["R^2_std"]) for r in rows}
        for h, v in r2.items():
            s[f"lfh.h{h}.mean"] = v
            s[f"lfh.h{h}.sd"] = sd[h]
        s["lfh.min"] = min(r2.values())
        s["lfh.max"] = max(r2.values())
        s["lfh.argmin"] = float(min(r2, key=r2.get))
        s["lfh.argmax"] = float(max(r2, key=r2.get))
        s["lfh.max_sd"] = max(sd.values())
    else:
        missing.append(str(p))

    if missing:
        print("MISSING PROVENANCE (regenerate before trusting this run):")
        for m in missing:
            print("   ", m)
        print()
    return s


# --------------------------------------------------------------------------- #
# 2. The registry: what the manuscript is allowed to say                       #
# --------------------------------------------------------------------------- #
# id, source key, regex (one capture group), tolerance, where it may appear
# The regex is searched over the whole file with re.S so it may span line breaks.
#
# sign : -1 when the manuscript writes the minus outside the captured group, as
#        LaTeX does with $-$0.510, so the capture is a magnitude.
# cmp  : "eq" (default) the stated value must equal the source;
#        "bound_ge" the manuscript states an upper bound, which must hold and is
#        warned about when it is loose by more than the tolerance.

def A(i, k, p, t=5e-4, f=("paper", "supp"), note="", sign=1, cmp="eq"):
    return dict(id=i, key=k, pat=p, tol=t, files=f, note=note, sign=sign, cmp=cmp)

R2 = r"R\^?\{?2\}?"
PM = r"(?:\$?\\pm\$?|\+/-)"
SEP = r"[\s~]*"

ANCHORS = [
    # ---- the anchoring step, on the 33 released windows --------------------
    A("clim.pooled", "anchor33.clim.pooled",
      r"climatology attains \$?R\^\{?2\}?\$?=([\d.]+)"),
    A("clim.pooled.repeat", "anchor33.clim.pooled",
      r"climatology, (?:which|attains)[^.]*?\$?R\^\{?2\}?\$?=([\d.]+) pooled"),
    A("clim.rmse", "anchor33.clim.rmse", r"RMSE ([\d.]+)\\,mm, MAE"),
    A("clim.mae", "anchor33.clim.mae", r"MAE ([\d.]+)\\,mm\)"),
    A("seas.pooled", "anchor33.seas.pooled",
      r"seasonal-naive predictor \$?R\^\{?2\}?\$?=([\d.]+)"),
    A("pers.pooled", "anchor33.pers.pooled",
      r"persistence \$?R\^\{?2\}?\$?=\$?[-\u2212]\$?(\d+\.\d+)", 5e-4, sign=-1),
    A("clim.percell", "anchor33.clim.percell",
      r"per-cell NSE \$?\\approx\$?([\d.]+) versus", 5e-3),
    A("anchorAll.origins", "anchorAll.n_origins",
      r"admissible forecast origin in the record \((\d+) of them\)", 0.5),
    A("anchorAll.clim", "anchorAll.clim.pooled",
      r"moves climatology to ([\d.]+)"),
    A("anchorAll.pers", "anchorAll.pers.pooled",
      r"persistence to \$?[-\u2212]\$?(\d+\.\d+)", sign=-1),
    A("design.windows", "design.n_val_windows",
      r"yields 343 training and (\d+) validation windows", 0.5),

    # ---- the fusion ladder --------------------------------------------------
    A("pub.mean", "blocked.published.mean",
      r"as published\) & \$?\\approx\$?100\\% & ([\d.]+)"),
    A("pub.sd", "blocked.published.sd",
      r"as published\) & \$?\\approx\$?100\\% & [\d.]+ \$\\pm\$ ([\d.]+)"),
    A("ridge.mean", "blocked.ridge.mean",
      r"\$?R\^\{?2\}? ?\$?=? ?\\?m?a?t?h?b?f?\{?([\d.]+) \\pm 0\.006"),
    A("ridge.mean.text", "blocked.ridge.mean",
      r"late fusion at ([\d.]+)\."),
    A("purged.ridge", "purged.ridge.mean",
      r"purged split (?:the figure )?is ([\d.]+)"),
    A("purged.ridge.tab", "purged.ridge.mean",
      r"embargo\}? & \\?t?e?x?t?b?f?\{?0\\%\}? & \\textbf\{([\d.]+)"),
    A("purged.rawbest", "purged.raw_best.mean",
      r"best base learner on the purged split: \d+\.\d+ against (\d+\.\d+)"),
    A("purged.ridge2", "purged.ridge.mean",
      r"best base learner on the purged split: (\d+\.\d+) against"),
    A("purged.incr", "purged.incr.mean",
      r"paired within seed, is \$\+\$(\d\.\d+)\\,\$\\pm\$\\,0\.036", 5e-4),
    A("purged.incr.tab", "purged.incr.mean",
      r"combining\} & & \\textbf\{\$\+\$0\.017 \$\\pm\$ 0\.013\} & & "
      r"\\textbf\{\$\+\$(\d\.\d+)", 5e-4),
    A("purged.recal", "purged.recal.mean",
      r"Recalibrated base learner, same purged split & 0\\% & (\d\.\d+)", 5e-4),
    A("purged.recal.cost", "purged.d_recal.mean",
      r"costs (\d\.\d+) out of sample rather than gaining", 5e-4, sign=-1),
    A("purged.recal.cost2", "purged.d_recal.mean",
      r"recalibration costs (\d\.\d+) and the ordering", 5e-4, sign=-1),
    A("purged.insample", "purged.ridge_insample.mean",
      r"In-sample the Ridge reaches ([\d.]+)"),
    A("decomp.bestrecal", "blocked.best_recal.mean",
      r"calibration only & (\d\.\d+) \$\\pm\$ \d\.\d+", 5e-4),
    A("decomp.bestrecal.sd", "blocked.best_recal.sd",
      r"calibration only & \d\.\d+ \$\\pm\$ (\d\.\d+)", 5e-4),
    A("purged.gap", "purged.insample_gap",
      r"which is (\d+\.\d+) above the purged estimate", 5e-4),
    A("purged.embargo", "purged.embargo",
      r"costs (\d+) windows per boundary", 0.5),
    A("purged.ntrain", "purged.n_train",
      r"leaves (\d+) training and \d+ test windows", 0.5),
    A("purged.ntest", "purged.n_test",
      r"leaves \d+ training and (\d+) test windows", 0.5),
    A("overlap.random", "overlap.random_pct",
      r"still leaves, for ([\d.]+)\\% of held-out", 5e-2),
    A("overlap.contig", "overlap.contiguous_pct",
      r"Contiguous folds reach ([\d.]+)\\%", 5e-2),

    # ---- the decomposition --------------------------------------------------
    A("dec.comb", "blocked.d_comb_only.mean",
      r"combination only[^&]*& [\d.]+ \$\\pm\$ [\d.]+ & \$[-\u2212]\$(\d+\.\d+)", sign=-1),
    A("dec.cal", "blocked.d_cal_only.mean",
      r"calibration only[^&]*& [\d.]+ \$\\pm\$ [\d.]+ & \$\+\$([\d.]+)"),
    A("dec.both", "blocked.d_both.mean",
      r"calibration \$\+\$ combination & [\d.]+ \$\\pm\$ [\d.]+ & \$\+\$([\d.]+)"),
    A("dec.incr.blocked", "blocked.d_incr.mean",
      r"Incremental value of combining\}? & & \\textbf\{\$\+\$([\d.]+)"),
    # ---- the seed-resolved factorial ---------------------------------------
    A("p30.gat.pafc.peak", "p30.GAT.PAFC.r2_peak",
      r"GAT with PAFC features \(\$?R\^\{?2\}?_\{?\\text\{peak\}\}?\$?=(\d+\.\d+)\)", 5e-4),
    A("p30.median.infl", "p30.median_inflation",
      r"the median across the six configurations is (\d+\.\d+)", 5e-4),
    A("p30.max.infl", "p30.max_inflation",
      r"with the worst at (\d+\.\d+) for SAGE with PAFC", 5e-4),
    A("p30.gatbasic.infl", "p30.GAT.BASIC.inflation",
      r"Quoting its best seed overstates it by (\d+\.\d+),", 5e-4),
    A("p30.median.peaksd", "p30.median_peak_sd",
      r"median seed spread on \$?R\^\{?2\}?_\{?\\text\{peak\}\}?\$? across configurations is (\d+\.\d+)", 5e-4),
    A("p30rcb.feat.p", "p30rcb.feat.p", r"\$F_\{1,10\}=6\.40\$, \$p=(\d\.\d+)\$", 5e-4),
    A("p30rcb.variant.p", "p30rcb.variant.p",
      r"\$F_\{2,10\}=0\.29\$, \$p=(\d\.\d+)\$", 5e-4),
    A("p30rcb.block", "p30rcb.block_ss_pct",
      r"block absorbs (\d+\.\d+)\\% of the total sum of squares\.", 5e-2),
    A("p30perm.feat", "p30perm.feat.p",
      r"at \$p=(\d\.\d+)\$ against \$p=\d\.\d+\$ for the variant", 5e-4),
    A("p30perm.variant", "p30perm.variant.p",
      r"against \$p=(\d\.\d+)\$ for the variant", 5e-4),
    A("p30.bundle.gap", "p30.bundle_gap",
      r"advantage of PAFC over BASIC is \$(\d\.\d+)\$ in \$R\^2\$", 5e-4),
    A("p30.pafc.mean", "p30.PAFC.mean_over_variants",
      r"\(\$(\d\.\d+)\$ against \$\d\.\d+\$ on the horizon mean\)", 5e-4),
    A("p30.basic.mean", "p30.BASIC.mean_over_variants",
      r"\(\$\d\.\d+\$ against \$(\d\.\d+)\$ on the horizon mean\)", 5e-4),
    A("p30.spread.min", "p30.min_mean_sd",
      r"Seed spreads run from \$\\pm\$(\d\.\d+) for GAT with PAFC", 5e-4),
    A("p30.spread.max", "p30.max_mean_sd",
      r"to \$\\pm\$(\d\.\d+) for GraphSAGE with PAFC", 5e-4),
    A("p30.pafc.spread", "p30.pafc_spread",
      r"against a (\d\.\d+) gap between the best and worst cell", 5e-4),

    # ---- measured compute ---------------------------------------------------
    A("cost.total", "cost.TOTAL.hours_total",
      r"factorial's (\d+\.\d+) GPU-hours", 5e-3),
    A("cost.total2", "cost.TOTAL.hours_total",
      r"The factorial cost (\d+\.\d+) GPU-hours in total", 5e-3),
    A("cost.gat.h", "cost.GAT.hours_total",
      r"GAT consumes (\d+\.\d+) of the factorial", 5e-3),
    A("cost.sage.h", "cost.SAGE.hours_total",
      r"GPU-hours against (\d+\.\d+) for GraphSAGE", 5e-3),
    A("cost.gat.sec", "cost.GAT.sec_per_epoch_mean",
      r"takes (\d+)\\,s per epoch", 0.5),
    A("cost.gat.lo", "cost.GAT.min_per_run_lo",
      r"(\d+) to \d+\\,min per run", 0.5),
    A("cost.gat.hi", "cost.GAT.min_per_run_hi",
      r"\d+ to (\d+)\\,min per run", 0.5),
    A("cost.gat.gblo", "cost.GAT.peak_gb_lo",
      r"holds (\d+\.\d+) to \d+\.\d+\\,GB of GPU memory", 5e-2),
    A("cost.gat.gbhi", "cost.GAT.peak_gb_hi",
      r"holds \d+\.\d+ to (\d+\.\d+)\\,GB of GPU memory", 5e-2),
    A("cost.s18.gat.lo", "cost.GAT.min_per_run_lo",
      r"\\textbf\{GNN-TAT \(GAT\)\}.*?\\textbf\{(\d+)-\d+ min\}", 0.5,
      f=("supp",)),
    A("cost.s18.gat.hi", "cost.GAT.min_per_run_hi",
      r"\\textbf\{GNN-TAT \(GAT\)\}.*?\\textbf\{\d+-(\d+) min\}", 0.5,
      f=("supp",)),
    A("cost.s18.gcn.lo", "cost.GCN.min_per_run_lo",
      r"\\textbf\{GNN-TAT \(GCN\)\}.*?\\textbf\{(\d+)-\d+ min\}", 0.5,
      f=("supp",)),
    A("cost.s18.gcn.hi", "cost.GCN.min_per_run_hi",
      r"\\textbf\{GNN-TAT \(GCN\)\}.*?\\textbf\{\d+-(\d+) min\}", 0.5,
      f=("supp",)),
    A("cost.s18.sage.lo", "cost.SAGE.min_per_run_lo",
      r"\\textbf\{GNN-TAT \(GraphSAGE\)\}.*?\\textbf\{(\d+)-\d+ min\}", 0.5,
      f=("supp",)),
    A("cost.s18.sage.hi", "cost.SAGE.min_per_run_hi",
      r"\\textbf\{GNN-TAT \(GraphSAGE\)\}.*?\\textbf\{\d+-(\d+) min\}", 0.5,
      f=("supp",)),
    A("p30perm.feat.supp", "p30perm.feat.p",
      r"Bundle main effect & permutation & 18 cells & - & ([\d.]+)", 5e-4,
      f=("supp",)),
    A("p30perm.variant.supp", "p30perm.variant.p",
      r"Variant main effect & permutation & 18 cells & - & ([\d.]+)", 5e-4,
      f=("supp",)),
    A("p30rcb.feat.p.supp", "p30rcb.feat.p",
      r"Bundle main effect & RCB ANOVA & 18 cells & \$F\$=[\d.]+ & ([\d.]+)",
      5e-4, f=("supp",)),
    A("p30rcb.variant.p.supp", "p30rcb.variant.p",
      r"Variant main effect & RCB ANOVA & 18 cells & \$F\$=[\d.]+ & ([\d.]+)",
      5e-4, f=("supp",)),
    A("p30rcb.feat.F.supp", "p30rcb.feat.F",
      r"Bundle main effect & RCB ANOVA & 18 cells & \$F\$=([\d.]+)", 5e-3,
      f=("supp",)),
    A("p30rcb.variant.F.supp", "p30rcb.variant.F",
      r"Variant main effect & RCB ANOVA & 18 cells & \$F\$=([\d.]+)", 5e-3,
      f=("supp",)),

    # ---- Late Fusion per horizon, three seeds -------------------------------
    A("lfh.min", "lfh.min", r"ranging from (\d+\.\d+) at H=12"),
    A("lfh.max", "lfh.max", r"ranging from \d+\.\d+ at H=12 to (\d+\.\d+) at H="),
    A("lfh.max_sd", "lfh.max_sd",
      r"inter-seed standard deviations below (\d+\.\d+) for every horizon",
      5e-3, cmp="bound_ge"),
    A("lfh.h1", "lfh.h1.mean",
      r"Late Fusion (\d+\.\d+) at H=1 against \d+\.\d+ at H=12, three-seed means"),
    A("lfh.h12", "lfh.h12.mean",
      r"Late Fusion \d+\.\d+ at H=1 against (\d+\.\d+) at H=12, three-seed means"),
    A("lfh.stability", "lfh.max_sd",
      r"\\l[et]q\$? ?(\d+\.\d+)\$? (?:across all horizons|at every horizon)",
      5e-3, cmp="bound_ge"),

    # ---- the blocked refit, per seed ---------------------------------------
    A("ridge.s42.r2", "ridge.s42.r2", r"^\s*42 & (\d\.\d+) & 79", 5e-4),
    A("ridge.s42.rmse", "ridge.s42.rmse", r"^\s*42 & \d\.\d+ & (\d+\.\d+)", 5e-3),
    A("ridge.mean.rmse", "ridge.mean.rmse",
      r"RMSE \$?=? ?\$?\\?m?a?t?h?b?f?\{?(\d+\.\d+) \\pm 0\.68", 5e-3),
    A("ridge.wconv", "ridge.mean.wconv",
      r"w_\{\\mathrm\{Conv(?:LSTM)?\}\}\$?=(\d\.\d+)\$?\\pm\$?0\.213"),
    A("ridge.wgnn", "ridge.mean.wgnn",
      r"w_\{\\mathrm\{GNN\}\}\$?=(\d\.\d+)\$?\\pm\$?0\.289"),
    A("ridge.b", "ridge.mean.b",
      r"bias of \$?\{?[-−]\}?(\d+\.\d+)\$?\\,mm", 5e-3, sign=-1),
    A("ridge.convshare", "ridge.conv_share_pct",
      r"convolutional branch taking the larger share \((\d+)\\% against", 0.5),
    A("ridge.gnnshare", "ridge.gnn_share_pct",
      r"larger share \(\d+\\% against (\d+)\\%\)", 0.5),
    A("cal.conv_slope", "cal.s42.conv_slope",
      r"fitted slopes are (\d\.\d+) for the convolutional", 5e-3),
    A("cal.gnn_slope", "cal.s42.gnn_slope",
      r"and (\d\.\d+) for the graph model", 5e-3),
    A("cal.conv_intercept", "cal.s42.conv_intercept",
      r"intercepts of \$?[-−]\$?(\d\.\d+) and", 5e-2, sign=-1),

    # ---- what the fold defect actually costs -------------------------------
    A("split.fold", "split.fold.mean",
      r"fold scheme accounts for (\d\.\d+) of the", 5e-4),
    A("split.total", "blocked.pub_minus_refit.mean",
      r"sits (\d\.\d+) above this refit", 5e-4),
    A("split.shuffled", "blocked.ridge_shuffled.mean",
      r"shuffled over flattened scalars gives \$?(\d\.\d+) \\pm 0\.006\$?", 5e-4),

    # ---- the significance tests --------------------------------------------
    A("tests.family", "tests.family_admissible",
      r"(?:the|across|over) (\d+) (?:admissible )?pairwise (?:tests|comparisons)", 0.5),
    A("tests.surviving", "tests.n_surviving_holm5",
      r"(\d+) of the (?:five|\d+) survive Holm", 0.5),
    A("tests.gnnconv.holm", "tests.gnn_conv_rmse.holm5",
      r"survives Holm over five \(\$?p\$?=(\d\.\d+)\)", 5e-4),
    A("tests.gnnconv.holm8", "tests.gnn_conv_rmse.holm8",
      r"but not over eight \(\$?p\$?=(\d\.\d+)\)", 5e-4),
    A("tests.cd", "tests.cd_k8_n33", r"CD\\,=\\,(\d\.\d+)", 5e-3),

    # ---- the two prediction arrays for one model ---------------------------
    A("gnn.pooled.released", "pooled33.gnn_released.r2",
      r"standalone GNN-TAT figure of (\d\.\d+) that an earlier version",
      5e-4, f=("supp",)),
    # the anchor comparison now quotes the three-seed blocked mean, which is the
    # estimate computed on the anchor's own targets; the released fit (0.672)
    # survives only in the figure caption that records the earlier comparison
    A("lf.blocked.anchor", "blocked.ridge.mean",
      r"Late Fusion, reaches pooled \$R\^2\$=(\d\.\d+)\$\\pm\$", 5e-4),
    A("lf.pooled33", "pooled33.lf.r2",
      r"the Late Fusion point is the released fit at (\d\.\d+)", 5e-3),
    A("lf.percell.anchor", "pooled33.lf.percell",
      r"versus (\d\.\d+) for Late Fusion, Table", 6e-3),

    # ---- eight regimes ------------------------------------------------------
    A("regime.boyaca.percell", "regime.boyaca.percell_climatology",
      r"pooled 0\.745 becomes a per-cell (\d\.\d+)"),
    A("regime.boyaca.pooled", "regime.boyaca.pooled_climatology",
      r"pooled (\d\.\d+) becomes a per-cell \d\.\d+"),
    A("regime.corr", "regime.clim_pooled_percell_r",
      r"two metrics correlate at \$?r\$?=(\d\.\d+)", 5e-3),
    A("regime.min", "regime.min.percell_climatology",
      r"[Pp]er-cell climatology scores between (\d\.\d+) and \d\.\d+"),
    A("regime.max", "regime.max.percell_climatology",
      r"[Pp]er-cell climatology scores between \d\.\d+ and (\d\.\d+)"),

    # ---- the stratified tables, from one array per model -------------------
    A("strat.lf.low", "strat.low.lf",
      r"Late Fusion drops from \$?R\^\{?2\}?\$?=(\d\.\d+) \(Low\)"),
    A("strat.lf.high", "strat.high.lf",
      r"drops from \$?R\^\{?2\}?\$?=\d\.\d+ \(Low\) to (\d\.\d+) \(High\)"),
    A("strat.degradation", "strat.lf_degradation_pct",
      r"to \d\.\d+ \(High\), a (\d+\.\d+)\\% reduction", 5e-2),
    A("strat.degradation2", "strat.lf_degradation_pct",
      r"The (\d+\.\d+)\\% \$?R\^\{?2\}?\$? degradation from low to high", 5e-2),
    A("strat.sweep.min", "strat.sweep_min_pct",
      r"remains in the (\d+\.\d+)-\d+\.\d+\\% range", 5e-2),
    A("strat.sweep.max", "strat.sweep_max_pct",
      r"remains in the \d+\.\d+-(\d+\.\d+)\\% range", 5e-2),
    A("strat.overall.conv", "strat.all.conv",
      r"and overall \((\d\.\d+) versus \d\.\d+\)"),
    A("strat.overall.gnn", "strat.all.gnn",
      r"and overall \(\d\.\d+ versus (\d\.\d+)\)"),
    A("strat.overall.lf", "strat.all.lf",
      r"above both everywhere \((\d\.\d+) overall\)"),
    A("strat.gnn_wins", "strat.gnn_wins_cells",
      r"higher per-cell \$?R\^\{?2\}?\$? in (\d+) of the 3,965 cells", 0.5),
    A("strat.gnn_wins_pct", "strat.gnn_wins_pct",
      r"that is (\d\.\d+)\\% of them", 5e-2),
    A("strat.rescue", "strat.rescue_cells",
      r"reach \$?R\^\{?2\}? \\geq 0\.5\$? in (\d+) cells", 0.5),
    A("strat.both_low", "strat.both_low_cells",
      r"the (\d+) cells where \\emph\{both\} fall below 0\.2", 0.5),

    # ---- percentages the manuscript derives from its own tables ------------
    A("dem.d10", "dem.lf_d10_pct",
      r"BASIC\\_D10 loses (\d+\.\d+)\\%", 5e-2),
    A("dem.pca6", "dem.lf_pca6_pct",
      r"BASIC\\_PCA6 (\d+\.\d+)\\%", 5e-2),
    A("dem.d10stats", "dem.lf_d10stats_pct",
      r"BASIC\\_D10\\_STATS (\d+\.\d+)\\%", 5e-2),
    A("dem.range.min", "dem.lf_d10_pct",
      r"by (\d+\.\d+)-\d+\.\d+\\% on the fusion", 5e-2),
    A("dem.range.max", "dem.lf_d10stats_pct",
      r"by \d+\.\d+-(\d+\.\d+)\\% on the fusion", 5e-2),
    A("dem.gnn.min", "dem.gnn_min_pct",
      r"degradation of (\d+)-\d+\\% against", 0.5),
    A("dem.gnn.max", "dem.gnn_max_pct",
      r"degradation of \d+-(\d+)\\% against", 0.5),
    A("dem.conv.min", "dem.conv_min_pct",
      r"against (\d+)-\d+\\% for ConvLSTM", 0.5),
    A("dem.conv.max", "dem.conv_max_pct",
      r"against \d+-(\d+)\\% for ConvLSTM", 0.5),
    A("stack.loss", "stack.loss_pct",
      r"scored (\d+)\\% below the best individual model", 0.5),

    # ---- variograms ---------------------------------------------------------
    A("vario.lf.sill", "vario.lf.sill",
      r"lowest error sill, (\d+\.\d+) against", 5e-2),
    A("vario.conv.sill", "vario.conv.sill",
      r"lowest error sill, \d+\.\d+ against (\d+\.\d+) for ConvLSTM", 5e-2),
    A("vario.gnn.sill", "vario.gnn.sill",
      r"and (\d+\.\d+) for GNN-TAT, so", 5e-2),
    A("vario.reduction", "vario.lf_vs_conv_pct",
      r"a (\d+)\\% lower error sill than ConvLSTM", 0.5),
    A("vario.gnn.range", "vario.gnn.range",
      r"longest of the three \((\d+\.\d+)\\,km against", 5e-2),
    A("vario.conv.range", "vario.conv.range",
      r"longest of the three \(\d+\.\d+\\,km against (\d+\.\d+)\\,km\)", 5e-2),
    A("vario.conv.nugget", "vario.conv.nugget",
      r"non-zero nugget \((\d+\.\d+)\) reveals", 5e-2),
    A("vario.obs.range", "vario.obs_range",
      r"observed precipitation field has a spatial range of (\d+\.\d+)\\,km", 5e-2),

    # ---- anomaly, connectivity, conformal ----------------------------------
    A("acc.lf.h1", "acc.lf.h1",
      r"per-cell ACC of \$\+\$(\d\.\d+) at a one-month lead", 5e-4),
    A("acc.lf.h12", "acc.lf.h12",
      r"and \$-\$(\d\.\d+) at twelve months", 5e-4, sign=-1),
    A("acc.conv.h1", "acc.conv.h1",
      r"ConvLSTM \$-\$(\d\.\d+) and \$-\$\d\.\d+", 5e-4, sign=-1),
    A("acc.conv.h12", "acc.conv.h12",
      r"ConvLSTM \$-\$\d\.\d+ and \$-\$(\d\.\d+)", 5e-4, sign=-1),
    A("acc.gnn.h1", "acc.gnn.h1",
      r"GNN-TAT \$-\$(\d\.\d+) and \$-\$\d\.\d+\)", 5e-4, sign=-1),
    A("acc.gnn.h12", "acc.gnn.h12",
      r"GNN-TAT \$-\$\d\.\d+ and \$-\$(\d\.\d+)\)", 5e-4, sign=-1),
    A("acc.pooled.max", "acc.lf.h1.pooled",
      r"pooled correlation, by contrast, appears as high as (\d\.\d+)", 5e-4),
    A("conn.raw.far", "conn.raw.far_pct",
      r"falls from (\d+\.\d+)\\% on raw series", 5e-2),
    A("conn.deseas.far", "conn.deseas.far_pct",
      r"to (\d+\.\d+)\\% once the seasonal cycle is removed", 5e-2),
    A("conn.deseas.beta", "conn.deseas.beta_elev",
      r"elevation coefficient of \$-\$(\d\.\d+) deseasonalized", 5e-5, sign=-1),
    A("conn.raw.beta", "conn.raw.beta_elev",
      r"deseasonalized against \$-\$(\d\.\d+) raw", 5e-5, sign=-1),
    A("conn.raw.r2", "conn.raw.r2",
      r"regression \$?R\^\{?2\}?\$? rising from (\d\.\d+) to \d\.\d+", 5e-4),
    A("conn.deseas.r2", "conn.deseas.r2",
      r"rising from \d\.\d+ to (\d\.\d+)", 5e-4),
    A("conformal.width", "conformal.lf_width",
      r"approximately (\d+)\\,mm wide after truncation", 0.5),
    A("conformal.clim", "conformal.clim_width",
      r"zero-cost climatology, which yields (\d+)\\,mm", 0.5),
    A("conformal.neg", "conformal.neg_lower_pct",
      r"and (\d+)\\% of the untruncated lower bounds", 0.5),
    A("conformal.ncal", "conformal.n_cal",
      r"33 windows split into (\d+) for calibration", 0.5),
    A("conformal.ntest", "conformal.n_test",
      r"for calibration and (\d+) for evaluation", 0.5),
]

# The supplement's lead-12 factorial table, one anchor per cell per metric. The
# rows are generated rather than written out because there are twenty-four of
# them and a hand-typed list is exactly the thing this tool exists to catch.
_M = r"(?:\\mathbf\{)?"       # the best cell in each column is bolded
_S17 = _M + r"%s \\pm %s\}?\$\s*& \$" + _M + r"%s \\pm %s\}?\$"
_NUM = (r"\d\.\d+", r"\d\.\d+", r"\d+\.\d+", r"\d+\.\d+")
_FIELDS = ("R^2_mean", "R^2_std", "RMSE_mean", "RMSE_std")

for _b, _v, _lab in (("BASIC", "GAT", "GAT"), ("BASIC", "GCN", "GCN"),
                     ("BASIC", "SAGE", "GraphSAGE"), ("PAFC", "GAT", "GAT"),
                     ("PAFC", "GCN", "GCN"), ("PAFC", "SAGE", "GraphSAGE")):
    for _i, _f in enumerate(_FIELDS):
        # capture the i-th number, match the other three
        _slots = [f"({n})" if j == _i else n for j, n in enumerate(_NUM)]
        _pat = (r"^" + _b + r"\s*& " + _lab + r"\s*& \$"
                + _S17 % tuple(_slots))
        ANCHORS.append(A(f"s17.{_v}.{_b}.{_f}", f"p30l12.{_v}.{_b}.{_f}", _pat,
                         5e-3 if "RMSE" in _f else 5e-4, f=("supp",)))


def derived(texts, s):
    """Quantities the manuscript derives from its own tables.

    A percentage quoted in prose against a table is not independent evidence; it is
    arithmetic on that table, and the only honest source for it is the table itself.
    Deriving it here means the prose and the table cannot drift apart, which is how
    a ratio survives a revision that changed its denominator.
    """
    tabs = {}
    for name, tex in texts.items():
        for label, rows, _cap in parse_tables(tex):
            tabs[f"{name}:{label}"] = rows

    def rowkey(r):
        return r[0].replace("\\_", "_").strip().upper()

    def cell(label, row_name, col):
        for r in tabs.get(label, []):
            if rowkey(r) == row_name.upper() and len(r) > col:
                return _num(r[col])
        return None

    # Table S9: the prose quotes the Late Fusion column's relative change, and the
    # per-architecture ranges, both of which are arithmetic on this table alone
    dem = "supp:tab:dem-negative"
    for col, who in ((2, "conv"), (3, "gnn"), (4, "lf")):
        base = cell(dem, "BASIC", col)
        if not base:
            continue
        vals = {}
        for r in tabs.get(dem, []):
            k = rowkey(r)
            if k.startswith("BASIC_") and len(r) > col:
                x = _num(r[col])
                if x is not None:
                    vals[k] = 100 * (1 - x / base)
        if not vals:
            continue
        s[f"dem.{who}_min_pct"] = min(vals.values())
        s[f"dem.{who}_max_pct"] = max(vals.values())
        if who == "lf":
            for k, short in (("BASIC_D10", "d10"), ("BASIC_PCA6", "pca6"),
                             ("BASIC_D10_STATS", "d10stats")):
                if k in vals:
                    s[f"dem.lf_{short}_pct"] = vals[k]

    # The decomposition table must reconcile with itself: each Delta is the paired
    # difference against the better raw learner, and the mean of differences is the
    # difference of means, so every Delta has to equal its own row's level minus the
    # comparator's. A row whose level came from one series and whose Delta came from
    # another passes every anchor and still cannot be added up, which is the defect
    # this recovers. The comparator is the better raw learner per seed; here that is
    # the convolutional one in all three, which the store lets us verify.
    dec = "paper:tab:fusion-decomposition"

    def lead(label, row_name, col):
        """The leading number of a 'mean +- sd' cell; _num wants a bare token."""
        for r in tabs.get(dec, []):
            if rowkey(r) == row_name.upper() and len(r) > col:
                mm = re.match(r"\s*([+-]?\d+\.\d+)", r[col].replace("$", ""))
                return float(mm.group(1)) if mm else None
        return None

    raw = lead(dec, "ConvLSTM-Bidir (raw)", 1)
    if raw is not None:
        for row in ("Simple average, combination only",
                    "Best single + affine, calibration only",
                    "Ridge over both, calibration + combination"):
            lvl, dlt = lead(dec, row, 1), lead(dec, row, 2)
            if lvl is not None and dlt is not None:
                short = row.split(",")[0].strip()
                s[f"decomp.reconcile.{short}"] = lvl - raw
                s[f"decomp.stated.{short}"] = dlt

    # Table 'ensemble-failure': the stacking loss against its own baseline row
    b = cell("supp:tab:ensemble-failure", "ConvLSTM (individual)", 1)
    v = cell("supp:tab:ensemble-failure", "Stacking Ensemble", 1)
    if b and v:
        s["stack.loss_pct"] = 100 * (1 - v / b)
    return s


# --------------------------------------------------------------------------- #
# 2b. Prohibitions: what the manuscript withdrew and must therefore not say     #
# --------------------------------------------------------------------------- #
# A claim withdrawn in one section and left standing in another is the defect
# that survived both previous revisions, and no equality check finds it, because
# the number is internally consistent with the sentence that should be gone.

FORBIDDEN = [
    dict(id="headline.unlabelled_ensemble",
         pat=r"(?:pooled \$R\^2\$=0\.67\b|0\.67 for the best ensemble)",
         why="the ensemble has four defensible values at four aggregations "
             "(0.672 released fit, 0.647 seed 42 blocked, 0.640 three-seed "
             "blocked mean, 0.598 purged). 0.67 is a rounding of the "
             "superseded one. Quote 0.640 against the anchor and label any "
             "other value with its aggregation at first use"),
    dict(id="withdrawn.archived_bundle_p",
         pat=r"\$?p\$?\s*=\s*0\.015 under a blocked analysis",
         why="the archived factorial's bundle p-value; the retrained factorial "
             "gives 0.030 blocked and 0.008 by permutation, and the paper "
             "reports the pipeline it released"),
    dict(id="withdrawn.kce_wilcoxon", pat=r"\$?p\$?\s*=\s*0\.036",
         why="the KCE-versus-BASIC Wilcoxon is withdrawn in Table S10 as inadmissible; "
             "cite the blocked factorial's bundle effect instead"),
    dict(id="withdrawn.kce_conv_wilcoxon", pat=r"\$?p\$?\s*=\s*0\.395",
         why="the KCE-versus-BASIC ConvLSTM Wilcoxon is withdrawn as inadmissible"),
    dict(id="withdrawn.holm075", pat=r"Holm-adjusted \$?p\$?=0\.075",
         why="computed over a family that no longer exists; the admissible family is "
             "five and this comparison is not in it"),
    dict(id="withdrawn.irreproducible",
         pat=r"could not be reconstructed from the released artefacts",
         why="retracted: the value traces to ConvLSTM-Attention/BASIC at a six-month "
             "total horizon, 0.633 +- 0.024, and Section 4.1.1 now names that cell"),
    dict(id="withdrawn.retrain_pending",
         pat=r"required before any of the GNN-TAT results",
         why="the retrain was done: three seeds, corrected graph, effect -0.002"),
    dict(id="withdrawn.fusion_beats_base",
         pat=r"improved on its stronger base learner by combining final predictions",
         why="withdrawn in Section 4.2.4: on the purged split the fusion does not beat "
             "its own best base learner"),
    dict(id="withdrawn.fusion_mean_advantage",
         pat=r"outperforms its base models not only in mean \$?R\^\{?2\}?\$?",
         why="same withdrawal; only the variance statement survives"),
    dict(id="stale.decomposition_numbers",
         pat=r"is worth only \$?\+\$?0\.002 \$?R\^\{?2\}?\$?",
         why="pre-rewrite output of fusion_decomposition.py; the paired term is "
             "-0.025 +- 0.046 and the table already says so"),
    dict(id="submission.doi_placeholder",
         pat=r"zenodo\.X+|to be inserted on release",
         why="GMD requires a persistent DOI at submission, not at acceptance, and "
             "does not accept an embargo. Make the manual Zenodo deposit (the release "
             "route cannot carry the inputs, see deposit_manifest.py --build), then "
             "put the version DOI in \\dataavailabilityDOI"),
    dict(id="submission.on_request",
         pat=r"available (?:from|upon) (?:the )?(?:corresponding )?author.{0,20}request|"
             r"upon reasonable request|available on request",
         unless=r"Nothing is|not available on request|rather than deposited",
         why="GMD's code and data policy prohibits this outright for anything needed "
             "to reproduce a result"),
    dict(id="stale.marginal_threshold",
         pat=r"smaller than the inter-seed standard deviation of the fused model itself",
         why="compares a paired difference against a marginal spread, which is the "
             "error the protocol's own component 2 tells the reader not to make"),
]


# --------------------------------------------------------------------------- #
# 3. Structural checks over every tabular                                      #
# --------------------------------------------------------------------------- #

SUMMARY_WORDS = ("mean", "median", "average", "overall", "total")

# A summary row that is deliberately not the aggregate of the rows shown. Each entry
# states why, and the audit prints the reason instead of the failure so the exemption
# is visible rather than silent. The manuscript must carry the same statement.
SUMMARY_EXEMPT = {
    "supp:tab:late-fusion-horizon": dict(
        why="the table lists 5 of the 12 leads; the Average row is over all 12, and "
            "the Delta column is the ratio of the two averages, not the mean of the "
            "ratios",
        caption_must_contain=r"Average row is over all twelve.*ratio of the two averages"),
    "supp:tab:bimamba-results": dict(
        why="the table lists 3 of the 12 leads; the Average row is over all 12",
        caption_must_contain=r"Average row is over all twelve"),
}

CLEAN = [
    (r"\\textbf|\\mathbf|\\textit|\\emph", ""),
    (r"\\footnotesize|\\centering|\\toprule|\\midrule|\\bottomrule|\\cmidrule\(?[lr]*\)?\{[\d-]+\}", ""),
    (r"\\text\{[^}]*\}", ""), (r"\\,", ""), (r"\\%", ""), (r"\\ddagger|\\dagger", ""),
    (r"\$\\pm\$", " +- "), (r"\\pm", " +- "), (r"\$\{?-\}?\$", "-"), (r"\u2212", "-"),
    (r"\\multirow\{\d+\}\{[^}]*\}", ""), (r"[{}$]", ""), (r"\\[a-zA-Z]+", " "),
]


def clean_cell(c: str) -> str:
    for pat, rep in CLEAN:
        c = re.sub(pat, rep, c)
    return c.strip()


def split_cells(raw: str):
    """Split a LaTeX row into cells, expanding \\multicolumn{n} into n slots.

    Without the expansion a spanning label shifts every column to its left and the
    summary-row check compares the wrong pairs, which is how a correct median row
    can be reported as wrong.
    """
    cells = []
    for part in raw.split("&"):
        m = re.search(r"\\multicolumn\{(\d+)\}", part)
        span = int(m.group(1)) if m else 1
        cells.append(clean_cell(re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", part)))
        cells.extend([""] * (span - 1))
    return cells


def parse_tables(tex: str):
    """Yield (label, header_cells, rows) for each tabular in the file."""
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S):
        block = m.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", block)
        label = lab.group(1) if lab else "(unlabelled)"
        cap = re.search(r"\\caption\{(.*?)\}\s*\n", block, re.S)
        caption = cap.group(1) if cap else ""
        tm = re.search(r"\\begin\{tabular\*?\}(.*?)\\end\{tabular\*?\}", block, re.S)
        if not tm:
            continue
        body = tm.group(1)
        body = re.sub(r"^\s*\{[^\n]*\}\s*\{[^\n]*\}", "", body)      # tabular* args
        rows = []
        for raw in body.split(r"\\"):
            if "multicolumn" in raw and raw.count("&") == 0:
                continue
            cells = split_cells(raw)
            if len(cells) < 2:
                continue
            rows.append(cells)
        yield label, rows, caption


def check_summary_rows(label: str, rows, caption, report):
    """Recompute any summary row from the rows above it."""
    if label in SUMMARY_EXEMPT:
        ex = SUMMARY_EXEMPT[label]
        if re.search(ex["caption_must_contain"], caption, re.S):
            report.ok(f"[{label}] summary row exempt and the caption says why")
        else:
            report.fail(f"[{label}] summary row is not the aggregate of the rows shown "
                        f"({ex['why']}), and the caption does not say so. A reader who "
                        f"adds up the column gets a different answer with no warning.")
        return
    data, summary = [], None
    for r in rows:
        head = r[0].lower()
        if any(w in head for w in SUMMARY_WORDS) and data:
            summary = r
            break
        if any(_num(c) is not None for c in r[1:]):
            data.append(r)
    if summary is None or not data:
        return
    ncol = min(len(summary), min(len(r) for r in data))
    # a count column, if one exists, is the candidate weight
    weights = None
    for j in range(1, ncol):
        vals = [_num(r[j]) for r in data]
        if all(v is not None and v == int(v) and v > 50 for v in vals):
            weights = vals
            break
    for j in range(1, ncol):
        got = _num(summary[j])
        if got is None:
            continue
        vals = [_num(r[j]) for r in data]
        if any(v is None for v in vals):
            continue
        if weights and vals is weights:
            continue
        cands = {"mean": statistics.fmean(vals), "median": statistics.median(vals)}
        if weights and len(weights) == len(vals):
            cands["weighted mean"] = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        if len(vals) > 1:
            cands["s.d."] = statistics.stdev(vals)
            cands["sum"] = sum(vals)
        hit = [k for k, v in cands.items() if abs(v - got) <= max(5e-4, abs(got) * 2e-3)]
        wanted = "median" if "median" in summary[0].lower() else None
        if hit and (wanted is None or wanted in hit):
            continue
        if hit:
            report.warn(f"[{label}] summary col {j} = {got} matches {hit} "
                        f"but the row is labelled '{summary[0]}'")
            continue
        report.fail(f"[{label}] summary row '{summary[0]}' col {j}: states {got}, "
                    + ", ".join(f"{k} of the rows above is {v:.4f}" for k, v in cands.items()))


# --------------------------------------------------------------------------- #
# 4. Runner                                                                    #
# --------------------------------------------------------------------------- #

class Report:
    def __init__(self):
        self.fails, self.warns, self.oks = [], [], []

    def fail(self, m):
        self.fails.append(m)

    def warn(self, m):
        self.warns.append(m)

    def ok(self, m):
        self.oks.append(m)


def line_of(tex: str, pos: int) -> int:
    return tex.count("\n", 0, pos) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="list the passing anchors too")
    ap.add_argument("--only", default=None, help="substring filter on anchor id")
    ap.add_argument("--no-structure", action="store_true")
    args = ap.parse_args()

    S = store()
    rep = Report()
    texts = {}
    for name, path in TEX_FILES.items():
        if not path.exists():
            rep.fail(f"missing manuscript source {path}")
            continue
        texts[name] = _read(path)
    S = derived(texts, S)

    print("=" * 78)
    print("ANCHORS  (a value in the text against the file that produces it)")
    print("=" * 78)
    unresolved = 0
    for a in ANCHORS:
        if args.only and args.only not in a["id"]:
            continue
        if a["key"] not in S:
            rep.warn(f"{a['id']}: source key '{a['key']}' not in the value store")
            unresolved += 1
            continue
        want = S[a["key"]]
        found = []
        for fname in a["files"]:
            if fname not in texts:
                continue
            for m in re.finditer(a["pat"], texts[fname], re.S | re.M):
                got = _num(m.group(1))
                found.append((fname, line_of(texts[fname], m.start()), got))
        if not found:
            rep.fail(f"{a['id']}: pattern matches nothing. The sentence it guarded has "
                     f"moved; rebind it or drop it. (expected {want:.4g})")
            continue
        found = [(f, ln, None if g is None else g * a["sign"]) for f, ln, g in found]
        distinct = sorted({round(g, 6) for _, _, g in found if g is not None})
        if len(distinct) > 1:
            rep.fail(f"{a['id']}: the manuscript states this quantity {len(distinct)} "
                     f"different ways: {distinct} at "
                     + ", ".join(f"{f}:{ln}" for f, ln, _ in found))
            continue
        where = ", ".join(f"{f}:{l}" for f, l, _ in found)
        got = found[0][2]
        if got is None:
            rep.fail(f"{a['id']}: {where} captured a token that is not a number")
        elif a["cmp"] == "bound_ge":
            if got < want:
                rep.fail(f"{a['id']}: {where} claims every value is below {got}, "
                         f"but {a['key']} reaches {want:.4g}")
            elif got - want > a["tol"]:
                rep.warn(f"{a['id']}: {where} states a true but loose bound {got}; "
                         f"{a['key']} is {want:.4g}")
            else:
                rep.ok(f"{a['id']}: bound {got} at {where}")
        elif abs(got - want) > a["tol"]:
            rep.fail(f"{a['id']}: {where} states {got}, {a['key']} is {want:.4g}")
        else:
            rep.ok(f"{a['id']}: {got} at {where}")

    print()
    print("=" * 78)
    print("SOURCES  (every row of a released table came from the engine that made it)")
    print("=" * 78)
    # GMD asks a methods-for-assessment paper to name and version its software tool
    # in the title, supply the code for review, and carry a code availability
    # paragraph. The first is checkable here; the others are checked by reading it.
    paper = texts.get("paper", "")
    m = re.search(r"\\title\{(.*?)\}\s*\n", paper, re.S)
    title = m.group(1) if m else ""
    if not re.search(r"[A-Z][A-Za-z]+ v\d+\.\d+", title):
        rep.fail("the title names no software tool with a version. GMD requires "
                 "'name and version must be identified in the title' for a methods "
                 "for assessment of models paper. Title is: " + title[:90])
    else:
        rep.ok("title carries a tool name and version: "
               + re.search(r"[A-Z][A-Za-z]+ v\d+\.\d+", title).group(0))
    if not re.search(r"\\codedataavailability\{", paper):
        rep.fail("no code and data availability section; GMD requires one")

    # GMD asks for vector graphics with embedded fonts. Every generator in this
    # repository wrote PNG only until the figures were revectorised, so the
    # regression to guard against is a new figure arriving as a raster, or a
    # regenerated PDF losing its vector twin.
    figdir = TEX_FILES["paper"].parent / "figures"
    raster = []
    for name, tex in texts.items():
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
            stem = m.group(1)
            if stem.endswith(".png"):
                raster.append(f"{name}:{stem}")
            elif stem.endswith(".pdf") and not (figdir / stem).exists():
                rep.fail(f"{name} includes {stem}, which is not in figures/")
    if raster:
        rep.fail(f"{len(raster)} figure(s) included as raster PNG: "
                 f"{', '.join(raster[:4])}. Regenerate with the vector twin and "
                 f"switch the include, or state why the figure cannot vectorise")
    else:
        rep.ok("every included figure is vector PDF")

    # The DOI now appears in two places that must be updated together: the macro in
    # the manuscript and the data citation in the bibliography. Fixing one and not
    # the other yields a paper whose availability section resolves and whose
    # reference list does not, which is worse than fixing neither.
    bib = TEX_FILES["paper"].parent / "refs.bib"
    if bib.exists():
        n = len(re.findall(r"zenodo\.X+", _read(bib)))
        if n:
            rep.fail(f"refs.bib still carries {n} placeholder Zenodo DOI; the data "
                     f"citation and \\dataavailabilityDOI must be set together")
        else:
            rep.ok("refs.bib carries no placeholder DOI")

        # A fabricated reference passes every check in this file, because
        # nothing here reads the bibliography's own claims. Two of them did:
        # one DOI resolved to an unrelated paper and was cited four times, and
        # the other did not resolve at all. The resolver check lives in
        # verify_bibliography.py because it needs the network; this records
        # whether it has been run against the current file and whether it
        # passed, so a stale pass cannot be mistaken for a current one.
        rec = PROV / "bibliography_check.txt"
        if not rec.exists():
            rep.fail("no bibliography_check.txt; run verify_bibliography.py, "
                     "which resolves every DOI against its registrant record")
        elif rec.stat().st_mtime < bib.stat().st_mtime:
            rep.fail("bibliography_check.txt predates refs.bib; the DOI "
                     "resolution has not been run since the file last changed")
        else:
            txt = _read(rec)
            m = re.search(r"(\d+) mismatched, (\d+) unresolved", txt)
            if not m:
                rep.fail("bibliography_check.txt has no summary line")
            elif int(m.group(1)):
                rep.fail(f"{m.group(1)} bibliography entries have a DOI that "
                         f"describes a different paper")
            else:
                rep.ok(f"bibliography resolved: {m.group(2)} unresolved "
                       f"(the pending deposit DOI), 0 mismatched")
    else:
        rep.fail(f"no bibliography at {bib}")

    # The decomposition table has to add up: the mean of paired differences is the
    # difference of means, so each Delta must equal its row's level minus the
    # comparator's. This caught a row whose level and whose Delta came from two
    # different series, which no per-value anchor can see.
    for k in [x for x in S if x.startswith("decomp.reconcile.")]:
        row = k[len("decomp.reconcile."):]
        stated = S.get(f"decomp.stated.{row}")
        if stated is None:
            continue
        if abs(S[k] - stated) > 1.5e-3:
            rep.fail(f"decomposition row '{row}' does not reconcile: the table's "
                     f"own levels give {S[k]:+.4f} but the row states "
                     f"{stated:+.4f}. One of the two is from a different series.")
        else:
            rep.ok(f"decomposition row '{row}' reconciles ({stated:+.4f})")

    # Two defects that survive a clean compile and are invisible in the source.
    # A doubled backslash before a letter is a LaTeX command that will not run, or a
    # line break that has been eaten; either way the source and the intent differ.
    # An unresolved reference renders as [?] or ?? in the PDF while the log carries
    # only a warning, so a document can be submitted with every citation missing.
    # A doubled backslash before a percent sign is the worst of these, because it
    # reads as a line break followed by a comment: it deletes the rest of the source
    # line from the rendered document and compiles without an error.
    COMMANDS = ("subsection|subsubsection|section|paragraph|citep|citet|ref|label|"
                "textbf|emph|texttt|begin|end|caption|footnotesize|item|url")
    for name, tex in texts.items():
        pct = re.findall(r"\\\\(?=%)", tex)
        cmd = re.findall(r"\\\\(?=(?:" + COMMANDS + r")\b)", tex)
        if pct:
            rep.fail(f"{name}: {len(pct)} doubled backslashes before a percent sign. "
                     f"Each one comments out the rest of its source line and leaves "
                     f"no error in the log.")
        if cmd:
            rep.fail(f"{name}: {len(cmd)} doubled backslashes before a command name, "
                     f"which will not run")
        if not pct and not cmd:
            rep.ok(f"{name}: no doubled command or comment sequences")
    for name, path in TEX_FILES.items():
        log = path.with_suffix(".log")
        if not log.exists():
            rep.warn(f"{name}: no build log, so unresolved references are unchecked")
            continue
        txt = _read(log)
        undef = set(re.findall(r"Citation `([^']*)' on page", txt)) | \
            set(re.findall(r"Reference `([^']*)' on page", txt))
        nobbl = "No file " + path.stem + ".bbl" in txt
        if undef or nobbl:
            rep.fail(f"{name}: the last build left "
                     + (f"{len(undef)} unresolved: {sorted(undef)[:6]}"
                        if undef else "no bibliography")
                     + ". These render as [?] or ?? in the PDF and the log records "
                       "them only as warnings.")
        else:
            rep.ok(f"{name}: every reference and citation resolved in the last build")

    suspect = S.get("_regime_suspect_names") or []
    if suspect:
        rep.fail("anchoring_eight_regimes.csv: " + ", ".join(suspect)
                 + " carry no mean_mm and are rounded to three decimals, so they were "
                   "not written by anchoring_protocol.py. A table row that did not come "
                   "from the engine its caption names is not a measurement. Re-run "
                   "`anchoring_protocol.py --regions` or drop the row.")
    else:
        rep.ok("anchoring_eight_regimes.csv: every row carries the engine's full output")

    print()
    print("=" * 78)
    print("PROHIBITIONS  (claims withdrawn elsewhere that are still standing here)")
    print("=" * 78)
    for f in FORBIDDEN:
        if args.only and args.only not in f["id"]:
            continue
        hits = []
        for fname, tex in texts.items():
            for m in re.finditer(f["pat"], tex, re.S):
                # a prohibition can have an exemption for the sentence that denies
                # the very thing it looks for, which otherwise reads as a violation
                ctx = tex[max(0, m.start() - 120):m.end() + 40]
                if f.get("unless") and re.search(f["unless"], ctx, re.S):
                    continue
                hits.append(f"{fname}:{line_of(tex, m.start())}")
        if hits:
            rep.fail(f"{f['id']}: still present at " + ", ".join(hits)
                     + f"\n        {f['why']}")
        else:
            rep.ok(f"{f['id']}: absent")

    if not args.no_structure:
        print()
        print("=" * 78)
        print("STRUCTURE  (every summary row recomputed from the rows above it)")
        print("=" * 78)
        for name, tex in texts.items():
            for label, rows, caption in parse_tables(tex):
                check_summary_rows(f"{name}:{label}", rows, caption, rep)

    print()
    for m in rep.fails:
        print("FAIL  " + m)
    for m in rep.warns:
        print("WARN  " + m)
    if args.verbose:
        for m in rep.oks:
            print("ok    " + m)

    checked = len(rep.oks) + len(rep.fails)
    print()
    print("-" * 78)
    print(f"{len(rep.oks)} passed, {len(rep.fails)} failed, {len(rep.warns)} warnings "
          f"({checked} anchors resolved, {unresolved} unresolved)")
    print(f"value store holds {len(S)} named quantities")
    print("Coverage is the registry, not the manuscript: a number no anchor names is")
    print("not checked. Add an anchor when you add a claim.")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
