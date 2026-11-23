"""K42: can guidance be explained by the mean residual alone?
(strong base, hydrogen bonds)

K39 found that 63% of the real mean is predictable from the static
features, and the remaining part (the residual) relates to the label.
This experiment asks whether guidance itself is explained by that
residual alone.

Method: K06(b) conditions (frozen T1I2006 + correction model, 8
calibration proteins for training, 100 unseen proteins x 3 replicas for
evaluation, 163,074 candidates). The correction model is trained on the
static features plus ONLY the real mean's residual (mean - prediction
from the static features), and the gain is measured at all horizons.
The residual fit is done on calibration candidates only (as in K39:
OLS and HistGB regressor, same settings), and no test statistics are
used in the fit. References: real history gain (+0.0712 at 1 ns) and
static-only. Extra reference: static + the real mean itself (K21's
mean-alone condition, re-verified here).

Pre-fixed reading: if the residual-only condition recovers 80% or more
of the gain, guidance is the part of the mean that the static features
cannot predict. If it does not, other statistics also carry guidance.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED_MODEL = 20261121

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

def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED_MODEL).fit(X, y)

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

# residual fits on calibration candidates only (K39 settings)
mr_cal, mr_te = Htr[:, 1], Hte[:, 1]
ols = LinearRegression().fit(Str, mr_cal)
gbm = HistGradientBoostingRegressor(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                    min_samples_leaf=20, l2_regularization=.15,
                                    random_state=SEED_MODEL).fit(Str, mr_cal)
resid_ols_cal, resid_ols_te = mr_cal - ols.predict(Str), mr_te - ols.predict(Ste)
resid_gbm_cal, resid_gbm_te = mr_cal - gbm.predict(Str), mr_te - gbm.predict(Ste)
print("residual SD cal/test: ols", round(float(resid_ols_cal.std()), 4), round(float(resid_ols_te.std()), 4),
      "gbm", round(float(resid_gbm_cal.std()), 4), round(float(resid_gbm_te.std()), 4), flush=True)

def col(v):
    return v.reshape(-1, 1)
CONDS = {
    "static": (Str, Ste),
    "resid_ols": (np.column_stack([Str, resid_ols_cal]), np.column_stack([Ste, resid_ols_te])),
    "resid_gbm": (np.column_stack([Str, resid_gbm_cal]), np.column_stack([Ste, resid_gbm_te])),
    "mean_only": (np.column_stack([Str, mr_cal]), np.column_stack([Ste, mr_te])),
    "real_history": (np.column_stack([Str, Htr]), np.column_stack([Ste, Hte])),
}

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

for name, (Xtr, Xte) in CONDS.items():
    results.setdefault(name, {})
    for j, h in enumerate(HORIZONS):
        if str(h) in results[name]:
            print(name, h, "already done, skipped", flush=True); continue
        m = learner(Xtr, Ytr[:, j])
        p = m.predict_proba(Xte)[:, 1]
        row = metrics(Yte[:, j], p)
        row["macro_auprc"] = macro_auprc(Yte, p, j)
        results[name][str(h)] = row
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        PARTIAL.write_text(json.dumps(results, indent=2))

for h in HORIZONS:
    for name, key in (("static", "static"), ("real_history", "history")):
        ref = k06b["by_horizon"][str(h)][key]
        got = results[name][str(h)]
        print("verify", name, h, "auprc diff", round(abs(got["auprc"] - ref["auprc"]), 6),
              "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

summary = {}
for h in HORIZONS:
    s = results["static"][str(h)]
    a10 = results["real_history"][str(h)]
    row = {"static_auprc": s["auprc"], "static_macro_auprc": s["macro_auprc"],
           "a10_gain_pooled": float(a10["auprc"] - s["auprc"]),
           "a10_gain_macro": float(a10["macro_auprc"] - s["macro_auprc"])}
    for name in CONDS:
        if name in ("static", "real_history"):
            continue
        r = results[name][str(h)]
        row[f"{name}_gain_pooled"] = float(r["auprc"] - s["auprc"])
        row[f"{name}_gain_macro"] = float(r["macro_auprc"] - s["macro_auprc"])
        row[f"{name}_recovery_pooled"] = float((r["auprc"] - s["auprc"]) / row["a10_gain_pooled"])
        row[f"{name}_recovery_macro"] = float((r["macro_auprc"] - s["macro_auprc"]) / row["a10_gain_macro"])
    summary[str(h)] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for name in CONDS:
        for h in HORIZONS:
            r = results[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
