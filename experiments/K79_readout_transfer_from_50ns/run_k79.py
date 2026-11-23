"""K79: does a readout trained at 50 ns work at other horizons?
(strong base, hydrogen bonds)

K56 found that a readout trained at 1 ns transfers weakly to 50 ns
(recovery 0.608), while a readout trained at 50 ns transfers well to
other horizons (0.96-1.33 in K56's 6x6 matrix). K77 showed that at
50 ns the carriers are mean (0.933) and fraction (0.888), so the
weakness of the 1 ns -> 50 ns transfer is not a missing carrier at
50 ns. This experiment measures the reverse direction directly:
train the correction at 50 ns, evaluate at 1 ns and 90 ns.

Method: K06(b) conditions (frozen T1I2006 + correction model,
HistGradientBoosting max_iter=160, lr .065, max_leaf_nodes 15,
min_samples_leaf 20, L2 .15, seed 20261121; 8 calibration proteins for
training, 100 unseen proteins x 3 replicas for evaluation, 163,074
candidates). Features are horizon-independent (static 5 + real
history 10); only the training label differs by horizon, so a model
trained at 50 ns can be evaluated against the 1 ns and 90 ns labels.
The static-only condition is trained per horizon in the same script.
Gain at evaluation horizon e is (transferred history AUPRC(e) -
static AUPRC(e)); the reference is the matched-horizon gain at e.
Recovery = transferred gain / matched gain.

This script re-runs only the fits K79 needs (train 50 ns, plus
matched-horizon fits at 1 ns and 90 ns for the references) with the
same code path as K56, and cross-checks the results against K56's
stored 6x6 matrix (train-50 row).

Pre-fixed reading: if the reverse transfer recovers 80% or more of
the matched gain at both 1 ns and 90 ns, the 50 ns readout works at
other horizons. If it does not, the 50 ns readout itself differs
from the others. If only one of the two horizons recovers, say so.
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
EVAL = (1, 90)
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
k56 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K56_readout_transfer_across_horizons/results.json").read_text())

# Fits needed: static and history trained at 50 ns (the transfer source)
# and at 1 ns / 90 ns (matched-horizon references). Evaluate each at
# 1 ns and 90 ns.
for feat_name, Xtr, Xte in (("static", Str, Ste),
                            ("history", np.column_stack([Str, Htr]), np.column_stack([Ste, Hte]))):
    for h_train in (50, 1, 90):
        key = f"{feat_name}_train{h_train}"
        results[feat_name].setdefault(key, {})
        if len(results[feat_name][key]) == len(EVAL):
            print(key, "already done, skipped", flush=True); continue
        t = HORIZONS.index(h_train)
        m = learner(Xtr, Ytr[:, t])
        for h_eval in EVAL:
            if str(h_eval) in results[feat_name][key]:
                continue
            e = HORIZONS.index(h_eval)
            p = m.predict_proba(Xte)[:, 1]
            row = metrics(Yte[:, e], p)
            row["macro_auprc"] = macro_auprc(Yte, p, e)
            results[feat_name][key][str(h_eval)] = row
            print(key, "->", h_eval, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))

# Verify the matched-horizon fits against K06(b).
for h in EVAL:
    for feat_name in ("static", "history"):
        ref = k06b["by_horizon"][str(h)][feat_name]
        got = results[feat_name][f"{feat_name}_train{h}"][str(h)]
        print("verify matched", feat_name, h, "auprc diff", round(abs(got["auprc"] - ref["auprc"]), 6),
              "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

# Gains and recovery ratios for the 50 ns -> 1 ns / 90 ns transfer.
summary = {}
for e in EVAL:
    s = results["static"][f"static_train{e}"][str(e)]
    a = results["history"][f"history_train{e}"][str(e)]
    tr = results["history"]["history_train50"][str(e)]
    matched_gain = float(a["auprc"] - s["auprc"])
    matched_gain_macro = float(a["macro_auprc"] - s["macro_auprc"])
    g = float(tr["auprc"] - s["auprc"])
    gm = float(tr["macro_auprc"] - s["macro_auprc"])
    summary[str(e)] = {
        "static_auprc": s["auprc"], "static_macro_auprc": s["macro_auprc"],
        "matched_history_auprc": a["auprc"], "matched_history_macro_auprc": a["macro_auprc"],
        "transfer_history_auprc": tr["auprc"], "transfer_history_macro_auprc": tr["macro_auprc"],
        "matched_gain_pooled": matched_gain, "matched_gain_macro": matched_gain_macro,
        "transfer_gain_pooled": g, "transfer_gain_macro": gm,
        "recovery_pooled": g / matched_gain if matched_gain else None,
        "recovery_macro": gm / matched_gain_macro if matched_gain_macro else None,
        # K56's stored values for the same cells, for cross-checking.
        "k56_recovery_pooled": k56["_summary"][str(e)]["train50_recovery_pooled"],
        "k56_recovery_macro": k56["_summary"][str(e)]["train50_recovery_macro"],
    }
    print("eval", e, "ns: transfer gain pooled", round(g, 4), "macro", round(gm, 4),
          "| recovery pooled", round(summary[str(e)]["recovery_pooled"], 4),
          "macro", round(summary[str(e)]["recovery_macro"], 4),
          "| K56 recovery pooled", round(summary[str(e)]["k56_recovery_pooled"], 4),
          "macro", round(summary[str(e)]["k56_recovery_macro"], 4), flush=True)

results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history")} for h in EVAL}
results["_references_K56"] = {str(e): k56["_summary"][str(e)] for e in EVAL}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train_horizon_ns", "eval_horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for feat_name in ("static", "history"):
        for t in (50, 1, 90):
            for e in EVAL:
                r = results[feat_name][f"{feat_name}_train{t}"][str(e)]
                w.writerow([feat_name, t, e, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
