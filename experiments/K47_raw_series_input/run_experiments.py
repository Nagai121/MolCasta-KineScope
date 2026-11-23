"""K47: does feeding the raw series recover more gain than the statistics?
(strong base, hydrogen bonds)

So far the history features summarized each series into 6 (or 10)
statistics. This experiment asks whether feeding the raw 11-point
distance series directly to the correction model recovers more gain
than the statistics.

Method: K06(b) conditions (frozen T1I2006 + correction model, 8
calibration proteins for training, 100 unseen proteins x 3 replicas for
evaluation, 163,074 candidates). The correction model is trained on the
static features plus the raw 11-point real series (frames 0..10, in
order, origin = frame 10), and the gain is measured at all horizons.
The series comes from the K22 cache, which was verified against K06(b)'s
saved history statistics (max diff 0.0); the mean column is re-checked
here. References: real history gain (+0.0712 at 1 ns) and the 6-statistic
condition (static + real history columns [0,1,2,3,5,7], K34/K37's
history6 control), plus static-only.

Pre-fixed reading: if the raw series clearly increases the gain over
the 6 statistics, summarization into statistics loses information. If
the gain is unchanged, the 6 statistics suffice.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/k22_cache")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED_MODEL = 20261121
SIX = [0, 1, 2, 3, 5, 7]

assert not (set(CAL) & set(TEST))
print("no overlap: cal", len(CAL), "test", len(TEST), "common", 0, flush=True)

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            saved = {k: z[k] for k in z.files}
        with np.load(SERIES / f"{p}.npz") as z:
            series = {k: z[k] for k in z.files}
        for r in (1, 2, 3):
            rows.append({"static": saved[f"r{r}_static"], "history": saved[f"r{r}_history"],
                         "series": series[f"r{r}_series"], "label": saved[f"r{r}_label"]})
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
S_tr = np.concatenate([r["series"] for r in cal]).astype(np.float64)
S_te = np.concatenate([r["series"] for r in test]).astype(np.float64)
Ytr = np.concatenate([r["label"] for r in cal]); Yte = np.concatenate([r["label"] for r in test])
print("n_cal", len(Ytr), "n_test", len(Yte), flush=True)
# sanity: series mean must equal history[1]
print("series mean vs history[1] max diff:",
      float(np.abs(S_te.mean(axis=1) - Hte[:, 1]).max()),
      float(np.abs(S_tr.mean(axis=1) - Htr[:, 1]).max()), flush=True)
assert np.abs(S_te.mean(axis=1) - Hte[:, 1]).max() < 1e-5
assert np.abs(S_tr.mean(axis=1) - Htr[:, 1]).max() < 1e-5

CONDS = {
    "static": (Str, Ste),
    "raw11": (np.column_stack([Str, S_tr]), np.column_stack([Ste, S_te])),
    "real6": (np.column_stack([Str, Htr[:, SIX]]), np.column_stack([Ste, Hte[:, SIX]])),
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
    for name in ("raw11", "real6"):
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
