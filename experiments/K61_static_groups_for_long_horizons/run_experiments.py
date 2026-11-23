"""K61: which static feature group does the history readout need at long
horizons? (strong base, hydrogen bonds)

K59 showed that a history-only correction recovers 0.89 of the gain at
1 ns but almost nothing at long horizons. This experiment asks which
static feature group the history readout needs at long horizons.

Static features (K06(b) score_b.py, from tseries structural()):
  col 0: frozen T1I2006 probability                      ("other")
  col 1: origin donor-acceptor distance (frame 10)      (distance)
  col 2: CA-CA distance between donor and acceptor residues (distance)
  col 3: donor is backbone N (1) vs sidechain donor (0) (kind)
  col 4: acceptor is backbone O (1) vs sidechain O (0)  (kind)
Groups, dropped one at a time (all conditions keep the full real
history10):
  - full:        static[0,1,2,3,4] + history (reference, = K06(b) history)
  - drop_dist:   static[0,3,4] + history
  - drop_kind:   static[0,1,2] + history
  - drop_frozen: static[1,2,3,4] + history
The static-only baseline (full static, no history) is trained per
horizon in the same script and verified against K06(b). The gain of each
condition is its AUPRC minus the full-static baseline's AUPRC at the
same horizon (pooled and per-protein macro). The reference gain is the
real-history gain (+0.0712 macro / +0.0702 pooled at 1 ns).

Pre-fixed reading: if dropping a group clearly lowers the gain at long
horizons, that group is needed for the long-horizon readout. If no drop
lowers it, the combination of static features is what matters, not any
one group.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED_MODEL = 20261121

# static column groups: 0=frozen, 1=origin distance, 2=CA-CA distance,
# 3=donor backbone?, 4=acceptor backbone?
CONDITIONS = {
    "full": [0, 1, 2, 3, 4],
    "drop_dist": [0, 3, 4],
    "drop_kind": [0, 1, 2],
    "drop_frozen": [1, 2, 3, 4],
}

assert not (set(CAL) & set(TEST))
print("no overlap: cal", len(CAL), "test", len(TEST), flush=True)

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({"static": z[f"r{r}_static"], "history": z[f"r{r}_history"],
                             "label": z[f"r{r}_label"]})
    return rows

def learner(X, y):
    return HistGradientBoostingClassifier(learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED_MODEL, max_iter=160).fit(X, y)

def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

cal = load(CAL); test = load(TEST)
Str = np.concatenate([r["static"] for r in cal]).astype(np.float64)
Ste = np.concatenate([r["static"] for r in test]).astype(np.float64)
Htr = np.concatenate([r["history"] for r in cal]).astype(np.float64)
Hte = np.concatenate([r["history"] for r in test]).astype(np.float64)
Ytr = np.concatenate([r["label"] for r in cal]); Yte = np.concatenate([r["label"] for r in test])
print("n_cal", len(Ytr), "n_test", len(Yte), flush=True)

spans, cur = [], 0
for p in TEST:
    with np.load(SAVED / f"{p}.npz") as z:
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
    spans.append((p, cur, cur + n)); cur += n

def macro_auprc(y, p, j):
    vals = [average_precision_score(y[s:e, j], p[s:e])
            for _, s, e in spans if 0 < y[s:e, j].sum() < e - s]
    return float(np.mean(vals))

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())

FEATS = {"static": (Str, Ste)}
for name, cols in CONDITIONS.items():
    FEATS[name] = (np.column_stack([Str[:, cols], Htr]), np.column_stack([Ste[:, cols], Hte]))

for feat_name, (Xtr, Xte) in FEATS.items():
    results.setdefault(feat_name, {})
    for j, h in enumerate(HORIZONS):
        if str(h) in results[feat_name]:
            print(feat_name, h, "already done, skipped", flush=True); continue
        m = learner(Xtr, Ytr[:, j])
        p = m.predict_proba(Xte)[:, 1]
        row = metrics(Yte[:, j], p)
        row["macro_auprc"] = macro_auprc(Yte, p, j)
        results[feat_name][str(h)] = row
        print(feat_name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        PARTIAL.write_text(json.dumps(results, indent=2))

for h in HORIZONS:
    for feat_name, k in (("static", "static"), ("full", "history")):
        ref = k06b["by_horizon"][str(h)][k]
        got = results[feat_name][str(h)]
        print("verify vs K06b", feat_name, h, "auprc diff", round(abs(got["auprc"] - ref["auprc"]), 6),
              "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

summary = {}
for h in HORIZONS:
    s = results["static"][str(h)]
    row = {"static_auprc": s["auprc"], "static_macro_auprc": s["macro_auprc"]}
    for name in CONDITIONS:
        a = results[name][str(h)]
        gain = float(a["auprc"] - s["auprc"])
        gain_m = float(a["macro_auprc"] - s["macro_auprc"])
        row[f"{name}_auprc"] = a["auprc"]
        row[f"{name}_macro_auprc"] = a["macro_auprc"]
        row[f"{name}_gain_pooled"] = gain
        row[f"{name}_gain_macro"] = gain_m
    full_gain = row["full_gain_pooled"]; full_gain_m = row["full_gain_macro"]
    for name in CONDITIONS:
        row[f"{name}_retention_pooled"] = row[f"{name}_gain_pooled"] / full_gain if full_gain else None
        row[f"{name}_retention_macro"] = row[f"{name}_gain_macro"] / full_gain_m if full_gain_m else None
    summary[str(h)] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_conditions"] = CONDITIONS
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for feat_name in FEATS:
        for h in HORIZONS:
            r = results[feat_name][str(h)]
            w.writerow([feat_name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
