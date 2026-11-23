"""K59: does the correction recover the gain with history features alone?
(strong base, hydrogen bonds)

So far the correction model has taken both static and history features.
This experiment asks whether the gain returns with history features alone,
without any static features.

Method: K06(b) conditions (frozen T1I2006 + correction model,
HistGradientBoosting max_iter=160, lr .065, max_leaf_nodes 15,
min_samples_leaf 20, L2 .15, seed 20261121; 8 calibration proteins for
training, 100 unseen proteins x 3 replicas for evaluation, 163,074
candidates). Conditions, all trained per horizon:
  - static: static 5 features only (reference baseline)
  - real6+static: static 5 + real history 6 statistics
    ([first, mean, std, min, last-first, frac<=3.5] = history[:, [0,1,2,3,5,7]])
  - hist6_only: the 6 history statistics only, no static features
The gain of a condition is its AUPRC minus the static condition's AUPRC
at the same horizon (pooled and per-protein macro). The reference gain is
the real-history gain (+0.0712 macro / +0.0702 pooled at 1 ns). Recovery
= hist6_only gain / real6+static gain, computed per horizon.

Pre-fixed reading: if history alone recovers 80% or more of the gain,
the history features carry signal without the static features. If it
does not, the readout needs the combination with the static features.
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
SIX = [0, 1, 2, 3, 5, 7]

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
Htr = np.concatenate([r["history"] for r in cal]).astype(np.float64)[:, SIX]
Hte = np.concatenate([r["history"] for r in test]).astype(np.float64)[:, SIX]
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
k12b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K10_readout_mismatch/results_k12b.json").read_text())

FEATS = {"static": (Str, Ste),
         "real6_static": (np.column_stack([Str, Htr]), np.column_stack([Ste, Hte])),
         "hist6_only": (Htr, Hte)}

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
    for feat_name, k in (("static", "static"), ("real6_static", "history")):
        ref = k06b["by_horizon"][str(h)][k]
        got = results[feat_name][str(h)]
        print("verify vs K06b", feat_name, h, "auprc diff", round(abs(got["auprc"] - ref["auprc"]), 6),
              "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

summary = {}
for h in HORIZONS:
    s = results["static"][str(h)]
    a = results["real6_static"][str(h)]
    o = results["hist6_only"][str(h)]
    gain_full = float(a["auprc"] - s["auprc"])
    gain_full_macro = float(a["macro_auprc"] - s["macro_auprc"])
    gain_only = float(o["auprc"] - s["auprc"])
    gain_only_macro = float(o["macro_auprc"] - s["macro_auprc"])
    summary[str(h)] = {
        "static_auprc": s["auprc"], "static_macro_auprc": s["macro_auprc"],
        "real6_static_auprc": a["auprc"], "real6_static_macro_auprc": a["macro_auprc"],
        "hist6_only_auprc": o["auprc"], "hist6_only_macro_auprc": o["macro_auprc"],
        "gain_full_pooled": gain_full, "gain_full_macro": gain_full_macro,
        "gain_only_pooled": gain_only, "gain_only_macro": gain_only_macro,
        "recovery_pooled": gain_only / gain_full if gain_full else None,
        "recovery_macro": gain_only_macro / gain_full_macro if gain_full_macro else None,
        "k06b_history_auprc_ref": k06b["by_horizon"][str(h)]["history"]["auprc"],
        "k12b_a6_auprc_ref": k12b["conditions"]["a6_train_real6_eval_real6"][str(h)]["auprc"],
        "k12b_a6_macro_auprc_ref": k12b["conditions"]["a6_train_real6_eval_real6"][str(h)]["macro_auprc"],
    }
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
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
