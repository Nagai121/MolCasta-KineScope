"""K37: can a correction model trained on generated features be used on
real history? (strong base, hydrogen bonds)

K12b tried a real-trained model on generated features, K34 retrained on
generated and tested on generated. The remaining combination is a
generated-trained model tested on real history: whether generated
features are usable as training data.

Method: K06(b) conditions (frozen T1I2006 + correction model, 8
calibration proteins for training, 100 unseen proteins x 3 replicas for
evaluation, 163,074 candidates; no overlap between the protein sets,
asserted here). The correction model is trained on generated 8 features
(static + gen8) and evaluated on real features in the same 8-feature
space. gen8 cannot be applied to the real history10 directly (K34's
note), so the real side is built from the re-extracted real 11-frame
series (K22 cache, verified against K06(b)'s saved history statistics
with max diff 0.0) with summarize()'s definitions (score_b/gen_sub):
  change = series - series[0]
  [mean, std, min, max of change; frac(distance<=3.5); frac(distance<=4.0);
   change at the last frame; frac(diff(distance)<0)]
Check: the real8 frac<=3.5 column must equal history[7] exactly.
Secondary condition (K12b's (d), re-verified here): train gen6, eval
real6 (real history columns [0,1,2,3,5,7]). Reference: gen8 train->test
re-verified against K06(b), and real history (a10).

Pre-fixed reading: if the generated-trained model recovers most of the
real-history gain on real history, generated features are usable as
training data. If it gains nothing, they are not usable as training
data either.
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

assert not (set(CAL) & set(TEST)), "calibration and test proteins overlap"
print("no overlap: cal", len(CAL), "test", len(TEST), "common", 0, flush=True)

def real8(series):
    """series: (n, 11) rows -> (n, 8), summarize() definitions applied
    to the real series. series[:, 0] is the origin distance."""
    s = series.astype(np.float64)
    ch = s - s[:, [0]]
    return np.column_stack((ch.mean(axis=1), ch.std(axis=1), ch.min(axis=1), ch.max(axis=1),
                            (s <= 3.5).mean(axis=1), (s <= 4.0).mean(axis=1),
                            ch[:, -1], (np.diff(s, axis=1) < 0).mean(axis=1))).astype(np.float32)

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            saved = {k: z[k] for k in z.files}
        with np.load(SERIES / f"{p}.npz") as z:
            series = {k: z[k] for k in z.files}
        for r in (1, 2, 3):
            rows.append({"static": saved[f"r{r}_static"], "history": saved[f"r{r}_history"],
                         "gen8": saved[f"r{r}_gen8"], "gen6": saved[f"r{r}_gen6"],
                         "real8": real8(series[f"r{r}_series"]),
                         "label": saved[f"r{r}_label"]})
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
# verification: real8 frac<=3.5 must equal history[7]
worst = max(float(np.abs(np.concatenate([r["real8"] for r in test])[:, 4]
                    - np.concatenate([r["history"] for r in test])[:, 7]).max()),
            float(np.abs(np.concatenate([r["real8"] for r in cal])[:, 4]
                    - np.concatenate([r["history"] for r in cal])[:, 7]).max()))
print("real8 frac<=3.5 vs history[7] max diff =", worst, flush=True)
assert worst == 0.0
Str = np.concatenate([r["static"] for r in cal]); Ste = np.concatenate([r["static"] for r in test])
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

def stack(rows, key):
    return np.concatenate([np.column_stack([r["static"], r[key]]) for r in rows])

CONDS = {
    # (train features, test features)
    "gen8_train_real8_test": ("gen8", "real8"),
    "gen8_train_gen8_test": ("gen8", "gen8"),
    "gen6_train_real6_test": ("gen6", "real6"),
    "real6_train_real6_test": ("real6", "real6"),
}
def feats(rows, key):
    if key == "real6":
        return np.concatenate([np.column_stack([r["static"], r["history"][:, SIX]]) for r in rows])
    return stack(rows, key)

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())

for name, (ktr, kte) in CONDS.items():
    results.setdefault(name, {})
    Xtr = feats(cal, ktr); Xte = feats(test, kte)
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

# verification of the gen8 re-run against K06(b)
for h in HORIZONS:
    ref = k06b["by_horizon"][str(h)]["gen8"]
    got = results["gen8_train_gen8_test"][str(h)]
    print("verify gen8", h, "auprc diff", round(abs(got["auprc"] - ref["auprc"]), 6),
          "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

summary = {}
for h in HORIZONS:
    s = k06b["by_horizon"][str(h)]["static"]
    row = {"static_auprc": s["auprc"], "static_macro_auprc": s["macro_auprc"],
           "a10_gain_pooled": float(k06b["by_horizon"][str(h)]["history"]["auprc"] - s["auprc"]),
           "a10_gain_macro": float(k06b["by_horizon"][str(h)]["history"]["macro_auprc"] - s["macro_auprc"]),
           "gen8_gain_pooled": float(k06b["by_horizon"][str(h)]["gen8"]["auprc"] - s["auprc"]),
           "gen6_gain_pooled": float(k06b["by_horizon"][str(h)]["gen6"]["auprc"] - s["auprc"]),
           "gen8_gain_macro": float(k06b["by_horizon"][str(h)]["gen8"]["macro_auprc"] - s["macro_auprc"]),
           "gen6_gain_macro": float(k06b["by_horizon"][str(h)]["gen6"]["macro_auprc"] - s["macro_auprc"])}
    for name in CONDS:
        r = results[name][str(h)]
        row[f"{name}_gain_pooled"] = float(r["auprc"] - s["auprc"])
        row[f"{name}_gain_macro"] = float(r["macro_auprc"] - s["macro_auprc"])
        row[f"{name}_recovery_pooled"] = float((r["auprc"] - s["auprc"]) / row["a10_gain_pooled"])
        row[f"{name}_recovery_macro"] = float((r["macro_auprc"] - s["macro_auprc"]) / row["a10_gain_macro"])
    summary[str(h)] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}
results["_no_overlap"] = {"calibration": CAL, "test": TEST, "common": []}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for name in CONDS:
        for h in HORIZONS:
            r = results[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
