"""K56: does a readout trained at one horizon work at another horizon?
(strong base, hydrogen bonds)

So far the correction model has been trained separately for each horizon.
This experiment asks whether a readout trained at one horizon also works
at another horizon.

Method: K06(b) conditions (frozen T1I2006 + correction model,
HistGradientBoosting max_iter=160, lr .065, max_leaf_nodes 15,
min_samples_leaf 20, L2 .15, seed 20261121; 8 calibration proteins for
training, 100 unseen proteins x 3 replicas for evaluation, 163,074
candidates). The features are horizon-independent (static 5 + real
history10); only the training label differs by horizon. So one model
trained at horizon t can be evaluated against every horizon's labels.
  - primary: train at 1 ns, evaluate at 5/10/30/50/90 ns
  - reverse: train at 90 ns, evaluate at 1 ns
  - supporting: the full 6x6 train-x-eval matrix (same fits, free)
The static-only condition is trained per horizon in the same script and
verified against K06(b); the gain at evaluation horizon e is
(transferred history AUPRC(e) - static AUPRC(e)), and the reference is
the matched-horizon gain at e. Recovery = transferred gain / matched gain.

Pre-fixed reading: if the transferred gain recovers 80% or more of the
matched gain, the mean readout is the same across horizons. If it does
not, the readout differs per horizon.
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
results.setdefault("static", {}); results.setdefault("history", {})

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())

# Train one model per horizon for static and for history; evaluate each
# trained model against every horizon's labels (transfer is free: the
# features do not depend on the horizon, only the label does).
for feat_name, Xtr, Xte in (("static", Str, Ste),
                            ("history", np.column_stack([Str, Htr]), np.column_stack([Ste, Hte]))):
    for t, h_train in enumerate(HORIZONS):
        key = f"{feat_name}_train{h_train}"
        results[feat_name].setdefault(key, {})
        if len(results[feat_name][key]) == len(HORIZONS):
            print(key, "already done, skipped", flush=True); continue
        m = learner(Xtr, Ytr[:, t])
        for e, h_eval in enumerate(HORIZONS):
            if str(h_eval) in results[feat_name][key]:
                continue
            p = m.predict_proba(Xte)[:, 1]
            row = metrics(Yte[:, e], p)
            row["macro_auprc"] = macro_auprc(Yte, p, e)
            results[feat_name][key][str(h_eval)] = row
            print(key, "->", h_eval, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))

for h in HORIZONS:
    for feat_name, k in (("static", "static"), ("history", "history")):
        ref = k06b["by_horizon"][str(h)][k]
        got = results[feat_name][f"{feat_name}_train{h}"][str(h)]
        print("verify matched", feat_name, h, "auprc diff", round(abs(got["auprc"] - ref["auprc"]), 6),
              "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

# Gains and recovery ratios.
summary = {}
for e in HORIZONS:
    s = results["static"][f"static_train{e}"][str(e)]
    a = results["history"][f"history_train{e}"][str(e)]
    matched_gain = float(a["auprc"] - s["auprc"])
    matched_gain_macro = float(a["macro_auprc"] - s["macro_auprc"])
    row = {"matched_gain_pooled": matched_gain, "matched_gain_macro": matched_gain_macro,
           "static_auprc": s["auprc"], "matched_history_auprc": a["auprc"]}
    for t in HORIZONS:
        tr = results["history"][f"history_train{t}"][str(e)]
        g = float(tr["auprc"] - s["auprc"])
        gm = float(tr["macro_auprc"] - s["macro_auprc"])
        row[f"train{t}_auprc"] = tr["auprc"]
        row[f"train{t}_gain_pooled"] = g
        row[f"train{t}_recovery_pooled"] = g / matched_gain if matched_gain else None
        row[f"train{t}_gain_macro"] = gm
        row[f"train{t}_recovery_macro"] = gm / matched_gain_macro if matched_gain_macro else None
    summary[str(e)] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train_horizon_ns", "eval_horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for feat_name in ("static", "history"):
        for t in HORIZONS:
            for e in HORIZONS:
                r = results[feat_name][f"{feat_name}_train{t}"][str(e)]
                w.writerow([feat_name, t, e, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
