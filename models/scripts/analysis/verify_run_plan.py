"""Resolve what a run plan will actually do, without a GPU.

Run this before launching. A GPU job that turns out to have been configured wrong
costs hours and, worse, produces numbers that look like results. The notebook
announces its plan in a banner, but a banner is a print statement: this reads the
plan table, the CONFIG literal and the override logic, applies them in the same
order the notebook does, and checks the values the training loop will see against
what the plan is supposed to mean.

Usage:
    python models/scripts/analysis/verify_run_plan.py               # the active plan
    python models/scripts/analysis/verify_run_plan.py --plan smoke_p30
    python models/scripts/analysis/verify_run_plan.py --all         # every plan
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "models" / "base_models_gnn_tat_v4.ipynb"

# measured on the corrected GAT/BASIC runs, A100-SXM4-80GB
SEC_PER_EPOCH = 321.7
# best epoch observed per cell in the archived runs, used to project the stop
BEST_EPOCH = {
    ("GAT", "BASIC"): [3, 4, 5], ("GAT", "PAFC"): [7, 25],
    ("GCN", "BASIC"): [1, 3], ("GCN", "PAFC"): [3, 6],
    ("SAGE", "BASIC"): [1, 5], ("SAGE", "PAFC"): [2, 4],
}


def notebook_cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def config_literal(cell8, ns):
    """Evaluate the CONFIG dict against the plan namespace it reads from."""
    m = re.search(r"^CONFIG = \{", cell8, re.M)
    i, depth = m.start() + len("CONFIG = "), 0
    while i < len(cell8):
        if cell8[i] == "{":
            depth += 1
        elif cell8[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    text = cell8[m.start() + len("CONFIG = "):i + 1]
    text = re.sub(r"Path\(BASE_PATH\)", "'<base>'", text)
    return eval(text, {"__builtins__": {"list": list}}, dict(ns))


def resolve(plan_name=None):
    cells = notebook_cells()
    cell1 = cells[1]
    if plan_name:
        cell1 = re.sub(r"^RUN_PLAN = '[^']*'", f"RUN_PLAN = {plan_name!r}",
                       cell1, count=1, flags=re.M)
    ns = {}
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cell1, ns)

    cfg = config_literal(cells[8], ns)
    out_dir = "V4_GNN_TAT_Models"
    if ns.get("PLAN_OUT_SUFFIX"):
        out_dir = f"V4_GNN_TAT_Models_{ns['PLAN_OUT_SUFFIX']}"
    for k, v in (ns.get("PLAN_CONFIG") or {}).items():
        if k not in cfg:
            raise KeyError(f"plan overrides CONFIG[{k!r}], which does not exist")
        cfg[k] = v
    # the ablation block runs after the plan overrides and takes precedence over
    # the output tree, exactly as cell 8 orders them
    abl = ns["ABLATION"]
    if abl == "leaked_graph":
        out_dir = "V4_GNN_TAT_Models_leaked_graph"
    elif abl == "batch2":
        cfg["batch_size"] = 2
        out_dir = "V4_GNN_TAT_Models_batch2"
    elif abl == "no_tf32":
        out_dir = "V4_GNN_TAT_Models_no_tf32"
    cfg["_out_dir"] = out_dir
    cfg["_sched"] = cfg.get("lr_patience", cfg["patience"] // 2)
    return ns, cfg, cells


INVARIANTS = [
    ("a diagnostic run is labelled as one and writes to its own tree",
     lambda ns, c: (ns["ABLATION"] is None) == (not c["_out_dir"].endswith(
         ("_leaked_graph", "_batch2", "_no_tf32"))))
    ,
    ("the graph fix is active unless this is the leakage diagnostic",
     lambda ns, c: ns["ABLATION"] != "leaked_graph"
     or c["_out_dir"].endswith("_leaked_graph")),
    ("edge budget is 500,000",
     lambda ns, c: c["gnn_config"]["max_edges"] == 500_000),
    ("batch size is 4, as the completed runs used, unless batch2 says otherwise",
     lambda ns, c: c["batch_size"] == (2 if ns["ABLATION"] == "batch2" else 4)),
    ("gnn chunk is 30",
     lambda ns, c: c["gnn_chunk_size"] == 30),
    ("gradient checkpointing on",
     lambda ns, c: c["gnn_grad_checkpoint"] is True),
    ("scheduler patience pinned at 7, not derived from early stopping",
     lambda ns, c: c["_sched"] == 7),
    ("input window 60, horizon 12",
     lambda ns, c: c["input_window"] == 60 and c["horizon"] == 12),
    ("hidden dim 64, 2 layers, dropout 0.1, 4 heads",
     lambda ns, c: (c["gnn_config"]["hidden_dim"] == 64
                    and c["gnn_config"]["num_gnn_layers"] == 2
                    and c["gnn_config"]["dropout"] == 0.1
                    and c["gnn_config"]["num_heads"] == 4)),
    ("a plan that changes CONFIG writes to its own tree",
     lambda ns, c: (not ns.get("PLAN_CONFIG")) or c["_out_dir"] != "V4_GNN_TAT_Models"),
    ("light mode off unless the plan is a smoke test",
     lambda ns, c: (c["light_mode"] is False)
     or ns["RUN_PLAN"].startswith("smoke")),
]


def project_cost(ns, cfg):
    """Projected A100-hours, from the measured rate and the observed stopping epochs."""
    if cfg["light_mode"]:
        return None
    total = 0.0
    rows = []
    for feat in ns["PLAN_FEATURES"]:
        for var in ns["PLAN_VARIANTS"]:
            be = BEST_EPOCH.get((var, feat))
            if not be:
                continue
            ep = min(cfg["epochs"], sum(b + cfg["patience"] + 1 for b in be) / len(be))
            h = ep * SEC_PER_EPOCH / 3600
            n = len(ns["PLAN_SEEDS"])
            rows.append((f"{var}/{feat}", ep, h, h * n))
            total += h * n
    return rows, total


def report(plan_name=None, verbose=True):
    ns, cfg, _ = resolve(plan_name)
    name = ns["RUN_PLAN"]
    runs = len(ns["PLAN_SEEDS"]) * len(ns["PLAN_FEATURES"]) * len(ns["PLAN_VARIANTS"])

    failures = [d for d, ok in ((d, f(ns, cfg)) for d, f in INVARIANTS) if not ok]

    if verbose:
        print("=" * 78)
        print(f"RUN PLAN: {name}")
        print("=" * 78)
        print(f"  seeds            {ns['PLAN_SEEDS']}")
        print(f"  bundles          {ns['PLAN_FEATURES']}")
        print(f"  variants         {ns['PLAN_VARIANTS']}")
        print(f"  runs             {runs}")
        print(f"  skip existing    {ns['PLAN_SKIP_EXISTING']}")
        print(f"  output tree      {cfg['_out_dir']}")
        print(f"  patience         {cfg['patience']}  (scheduler {cfg['_sched']})")
        print(f"  epoch cap        {cfg['epochs']}")
        print(f"  light mode       {cfg['light_mode']}"
              + (f"  ({cfg['light_grid_size']}x{cfg['light_grid_size']} grid)"
                 if cfg["light_mode"] else ""))
        for k, v in sorted((ns.get("PLAN_CONFIG") or {}).items()):
            print(f"  override         CONFIG[{k!r}] = {v!r}")

        cost = project_cost(ns, cfg)
        if cost:
            rows, total = cost
            print()
            print(f"  {'cell':<12}{'epochs':>8}{'h/seed':>9}{'h total':>10}")
            for label, ep, h, ht in rows:
                print(f"  {label:<12}{ep:>8.0f}{h:>9.1f}{ht:>10.1f}")
            print(f"  {'':<12}{'':>8}{'TOTAL':>9}{total:>10.1f} A100-hours")
            print("  (upper bound: the measured rate is GAT's; GCN and SAGE are cheaper)")
        elif cfg["light_mode"]:
            print()
            print("  light mode: 25 nodes instead of 3,965, so this checks the code")
            print("  paths and not the cost. The 500,000-edge budget never binds at")
            print("  25 nodes and peak memory is nothing like a full run's 27 GB.")

        print()
        for desc, fn in INVARIANTS:
            print(("  ok   " if fn(ns, cfg) else "  FAIL ") + desc)
    return name, runs, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.all:
        nb = json.loads(NB.read_text(encoding="utf-8"))
        ns = {}
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            exec("".join(nb["cells"][1]["source"]), ns)
        bad = 0
        print(f"{'plan':<26}{'runs':>6}{'pat':>5}  {'output tree':<32}status")
        print("-" * 96)
        for p in sorted(k for k, v in ns["_PLANS"].items() if v):
            name, runs, fails = report(p, verbose=False)
            _, cfg, _ = resolve(p)
            bad += bool(fails)
            print(f"{p:<26}{runs:>6}{cfg['patience']:>5}  {cfg['_out_dir']:<32}"
                  + ("ok" if not fails else "FAIL: " + fails[0]))
        return 1 if bad else 0

    _, _, fails = report(a.plan)
    print()
    print(f"{len(fails)} invariant(s) violated")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
