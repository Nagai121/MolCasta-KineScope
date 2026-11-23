"""K52: does a stronger correction model increase the real-history gain?
(strong base, hydrogen bonds)

So far the correction model has been a fixed-setting HistGBM. This
experiment asks whether increasing its capacity increases the
real-history gain.

Method: K06(b) conditions (frozen T1I2006 + correction model, 8
calibration proteins for training, 100 unseen proteins x 3 replicas for
evaluation, 163,074 candidates). Four capacities, all with the same
learning rate (0.065), min_samples_leaf (20) and L2 (0.15) as K06(b):
  base:     max_iter=160, max_leaf_nodes=15   (K06(b) setting)
  iter2x:   max_iter=320, max_leaf_nodes=15
  leaves63: max_iter=160, max_leaf_nodes=63
  big:      max_iter=320, max_leaf_nodes=63
For each capacity, both the static-only and the real-history (static +
real history10) conditions are trained, so the gain (history minus
static) is defined at matched capacity. The gain is measured at all
horizons (pooled and per-protein macro AUPRC).

Pre-fixed reading: if a larger capacity clearly increases the gain,
the readout model was the limit. If the gain is unchanged, the current
model already extracts the information.
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

CAPACITIES = {
    "base": dict(max_iter=160, max_leaf_nodes=15),
    "iter2x": dict(max_iter=320, max_leaf_nodes=15),
    "leaves63": dict(max_iter=160, max_leaf_nodes=63),
    "big": dict(max_iter=320, max_leaf_nodes=63),
    # NOTE (added after the first run): the four capacities above all use
    # early_stopping='auto' (the default), so iter2x stopped at the same
    # 78 iterations as base and big at the same 55 as leaves63 -- doubling
    # max_iter changed nothing. These two extra capacities disable early
    # stopping so the doubled iterations actually run.
    "iter2x_noes": dict(max_iter=320, max_leaf_nodes=15, early_stopping=False),
    "big_noes": dict(max_iter=320, max_leaf_nodes=63, early_stopping=False),
}

assert not (set(CAL) & set(TEST))
print("no overlap: cal", len(CAL), "test", len(TEST), "common", 0, flush=True)

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({"static": z[f"r{r}_static"], "history": z[f"r{r}_history"],
                             "label": z[f"r{r}_label"]})
    return rows

def learner(X, y, cap):
    return HistGradientBoostingClassifier(learning_rate=.065, min_samples_leaf=20,
                                          l2_regularization=.15, random_state=SEED_MODEL,
                                          **cap).fit(X, y)

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

FEATS = {"static": (Str, Ste),
         "history": (np.column_stack([Str, Htr]), np.column_stack([Ste, Hte]))}

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

for cap_name, cap in CAPACITIES.items():
    for feat_name, (Xtr, Xte) in FEATS.items():
        key = f"{cap_name}_{feat_name}"
        results.setdefault(key, {})
        for j, h in enumerate(HORIZONS):
            if str(h) in results[key]:
                print(key, h, "already done, skipped", flush=True); continue
            m = learner(Xtr, Ytr[:, j], cap)
            p = m.predict_proba(Xte)[:, 1]
            row = metrics(Yte[:, j], p)
            row["macro_auprc"] = macro_auprc(Yte, p, j)
            results[key][str(h)] = row
            print(key, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))

for h in HORIZONS:
    for feat_name, k in (("static", "static"), ("history", "history")):
        ref = k06b["by_horizon"][str(h)][k]
        got = results[f"base_{feat_name}"][str(h)]
        print("verify base", feat_name, h, "auprc diff", round(abs(got["auprc"] - ref["auprc"]), 6),
              "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

summary = {}
for h in HORIZONS:
    row = {}
    for cap_name in CAPACITIES:
        s = results[f"{cap_name}_static"][str(h)]
        a = results[f"{cap_name}_history"][str(h)]
        row[f"{cap_name}_static_auprc"] = s["auprc"]
        row[f"{cap_name}_static_macro_auprc"] = s["macro_auprc"]
        row[f"{cap_name}_history_auprc"] = a["auprc"]
        row[f"{cap_name}_history_macro_auprc"] = a["macro_auprc"]
        row[f"{cap_name}_gain_pooled"] = float(a["auprc"] - s["auprc"])
        row[f"{cap_name}_gain_macro"] = float(a["macro_auprc"] - s["macro_auprc"])
    summary[str(h)] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}
results["_capacities"] = {k: v for k, v in CAPACITIES.items()}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["capacity", "condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for cap_name in CAPACITIES:
        for feat_name in FEATS:
            for h in HORIZONS:
                r = results[f"{cap_name}_{feat_name}"][str(h)]
                w.writerow([cap_name, feat_name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
