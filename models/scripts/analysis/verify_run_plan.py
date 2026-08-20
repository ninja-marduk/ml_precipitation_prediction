"""Resolve the configuration the active run plan produces, without a GPU.

Run this before launching. A GPU job that turns out to have been configured
wrong costs hours and, worse, produces numbers that look like results. The
notebook already announces its plan in a banner, but a banner is a print
statement: this reads the plan, the CONFIG literal and the override logic and
checks the values the training loop will actually see against what the plan is
supposed to mean, exiting non-zero on any mismatch.

Cell 8 depends on Colab globals, so rather than execute it whole this replays the
parts that decide the run: the CONFIG literal, the ablation block and the plan
overrides. What it prints is what the training loop will read.
"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "models" / "base_models_gnn_tat_v4.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
src = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])}

# 1. the plan
g = {}
exec(src[1], g)

# 2. the CONFIG literal, read rather than executed (it references Colab paths)
m = re.search(r"^CONFIG = \{", src[8], re.M)
depth, i = 0, m.start() + len("CONFIG = ")
while i < len(src[8]):
    if src[8][i] == "{":
        depth += 1
    elif src[8][i] == "}":
        depth -= 1
        if depth == 0:
            break
    i += 1
literal = src[8][m.start() + len("CONFIG = "):i + 1]
literal = re.sub(r"Path\(BASE_PATH\)", "'<base>'", literal)
# CONFIG reads the plan for enabled_variants and enabled_features, so it is
# evaluated against the plan's namespace rather than parsed as a bare literal
CONFIG = eval(literal, {"__builtins__": {"list": list}}, dict(g))

# 3. the plan overrides, as cell 8 applies them
out_dir = "V4_GNN_TAT_Models"
if g.get("PLAN_OUT_SUFFIX"):
    out_dir = f"V4_GNN_TAT_Models_{g['PLAN_OUT_SUFFIX']}"
if g.get("PLAN_PATIENCE") is not None:
    CONFIG["patience"] = int(g["PLAN_PATIENCE"])

# 4. the scheduler, as cell 21 reads it
sched = CONFIG.get("lr_patience", CONFIG["patience"] // 2)

REQUIRED = [
    ("run plan", g["RUN_PLAN"], "factorial_p30"),
    ("ablation (None keeps the graph fix)", g["ABLATION"], None),
    ("seeds", g["PLAN_SEEDS"], [42, 123, 456]),
    ("feature bundles", g["PLAN_FEATURES"], ["BASIC", "PAFC"]),
    ("variants", g["PLAN_VARIANTS"], ["GAT", "GCN", "SAGE"]),
    ("skip existing", g["PLAN_SKIP_EXISTING"], False),
    ("early-stopping patience", CONFIG["patience"], 30),
    ("scheduler patience", sched, 7),
    ("output tree", out_dir, "V4_GNN_TAT_Models_p30"),
    ("epoch cap", CONFIG["epochs"], 150),
    ("batch size", CONFIG["batch_size"], 4),
    ("gnn chunk", CONFIG["gnn_chunk_size"], 30),
    ("grad checkpoint", CONFIG["gnn_grad_checkpoint"], True),
    ("learning rate", CONFIG["learning_rate"], 1e-3),
    ("weight decay", CONFIG["weight_decay"], 1e-5),
    ("input window", CONFIG["input_window"], 60),
    ("horizon", CONFIG["horizon"], 12),
    ("edge budget", CONFIG["gnn_config"]["max_edges"], 500_000),
    ("hidden dim", CONFIG["gnn_config"]["hidden_dim"], 64),
    ("gnn layers", CONFIG["gnn_config"]["num_gnn_layers"], 2),
    ("dropout", CONFIG["gnn_config"]["dropout"], 0.1),
    ("attention heads", CONFIG["gnn_config"]["num_heads"], 4),
]

print(f"{'setting':<38}{'value':<26}{'required':<22}")
print("-" * 88)
bad = 0
for label, got, want in REQUIRED:
    ok = got == want
    bad += not ok
    print(f"{label:<38}{str(got):<26}{str(want):<22}{'' if ok else '  <-- MISMATCH'}")

runs = len(g["PLAN_SEEDS"]) * len(g["PLAN_FEATURES"]) * len(g["PLAN_VARIANTS"])
print("-" * 88)
print(f"runs: {runs}")

# the graph fix is a code path, not a setting: confirm it is the one that will run
graph = src[13]
assert "_corr_series = precip_series[:split_idx]" in graph
assert "ABLATION_LEAKED_GRAPH" in graph
print("graph correlation window: training period only, unless ABLATION='leaked_graph'")
print(f"           this run has ABLATION={g['ABLATION']!r}, so the fix is active")

print()
print("MISMATCHES:", bad)
raise SystemExit(1 if bad else 0)
