"""K44: where is the remaining 20-30% of guidance? (strong base, H-bonds)

K42 found the real mean's residual alone recovers 0.68-0.94 of the gain,
leaving 20-30% in other statistics. This experiment asks which
statistic holds the remainder.

Method: K06(b) conditions (frozen T1I2006 + correction model, 8
calibration proteins for training, 100 unseen proteins x 3 replicas for
evaluation, 163,074 candidates). The correction model is trained on the
static features plus the real mean's residual (K42's GBM version) and,
one at a time, one more statistic's residual: frac<=3.5, min, std.
Each added residual is the statistic minus its prediction from the
static features (K39's method), fitted on calibration candidates only
with a HistGB regressor (same settings as the correction model); no
test statistics are used in the fits. An "all together" condition
(mean residual + all three added residuals) is also measured. The gain
is measured at all horizons and compared with K42's residual-only
condition (re-run here). References: static-only and real history.

Pre-fixed reading: if adding one statistic brings the recovery to
about 1.0, the remainder is in that statistic. If no addition reaches
1.0, the remainder is dispersed across several statistics.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, r2_score

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

# GBM residual fits on calibration candidates only (K39/K42 settings)
# history cols: 1=mean, 2=std, 3=min, 7=frac<=3.5
STATS = {"mean": 1, "std": 2, "min": 3, "frac35": 7}
resid = {}
fit_info = {}
for name, col in STATS.items():
    reg = HistGradientBoostingRegressor(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                        min_samples_leaf=20, l2_regularization=.15,
                                        random_state=SEED_MODEL).fit(Str, Htr[:, col])
    r_cal = Htr[:, col] - reg.predict(Str)
    r_te = Hte[:, col] - reg.predict(Ste)
    resid[name] = (r_cal, r_te)
    fit_info[name] = {"r2_on_test": float(r2_score(Hte[:, col], reg.predict(Ste))),
                      "residual_sd_cal": float(r_cal.std()), "residual_sd_test": float(r_te.std())}
    print("fit", name, "R2", round(fit_info[name]["r2_on_test"], 4),
          "resid_sd", round(fit_info[name]["residual_sd_test"], 4), flush=True)

def stack(base_tr, base_te, *names):
    cols_tr = [base_tr] + [resid[n][0].reshape(-1, 1) for n in names]
    cols_te = [base_te] + [resid[n][1].reshape(-1, 1) for n in names]
    return np.column_stack(cols_tr), np.column_stack(cols_te)

CONDS = {
    "static": (Str, Ste),
    "resid_mean": stack(Str, Ste, "mean"),
    "resid_mean_plus_frac35": stack(Str, Ste, "mean", "frac35"),
    "resid_mean_plus_min": stack(Str, Ste, "mean", "min"),
    "resid_mean_plus_std": stack(Str, Ste, "mean", "std"),
    "resid_all4": stack(Str, Ste, "mean", "frac35", "min", "std"),
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

results = {"_fit_info": fit_info}
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
