"""K75: does the dissociation readout transfer across horizons?
(strong base, hydrogen-bond dissociation)

Formation: the readout transfers across horizons (K56). This experiment
asks the same for dissociation.

Method: K06(b) conditions (frozen T1I2006 + correction model,
HistGradientBoosting max_iter=160, lr .065, max_leaf_nodes 15,
min_samples_leaf 20, L2 .15, seed 20261121), on the dissociation
features extracted by extract_features.py (candidates formed at the
origin, label = distance > 3.5 A at the horizon). Train on the 8
calibration proteins, evaluate on the 100 test proteins. The features
are horizon-independent, so one model trained at horizon h1 is evaluated
against every horizon h2's labels; the full 6x6 matrix is computed.
  - primary: train@1 -> eval@90, and train@90 -> eval@1
  - reference: the matched-horizon condition's gain (train@h -> eval@h)
The gain of a condition is its AUPRC minus the static-only condition's
AUPRC trained at the SAME train horizon and evaluated at the same eval
horizon (pooled over the 100 test proteins, and per-protein macro).
Recovery = cross-horizon gain / matched-horizon gain.
Sanity checks: the frozen-alone dissociation score is 1 - frozen
(the frozen model predicts P(formed at the horizon)); its AUC/AUPRC
against the dissociation label should be above 0.5. The matched gains
should be positive and roughly in the ballpark of K46's weak-base
dissociation gains (+0.1785 at 1 ns, +0.0679 at 90 ns).

Pre-fixed reading: if the gain recovers 80% or more of the reference
gain in BOTH directions (1 -> 90 and 90 -> 1), the dissociation
readout also transfers across horizons. If it does not, the
dissociation readout differs per horizon.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k75_cache")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED_MODEL = 20261121

assert not (set(CAL) & set(TEST))


def load(proteins):
    rows = []
    for p in proteins:
        with np.load(CACHE / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({"protein": p, "static": z[f"r{r}_static"],
                             "history": z[f"r{r}_history"], "label": z[f"r{r}_label"]})
    return rows


def stack(rows):
    S = np.concatenate([r["static"] for r in rows]).astype(np.float64)
    H = np.concatenate([r["history"] for r in rows]).astype(np.float64)
    Y = np.concatenate([r["label"] for r in rows])
    return S, H, Y


def learner(X, y):
    return HistGradientBoostingClassifier(learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED_MODEL, max_iter=160).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def macro_auprc(rows, Y, p, j):
    spans, cur = [], 0
    for r in rows:
        n = len(r["label"])
        spans.append((cur, cur + n))
        cur += n
    vals = [average_precision_score(Y[s:e, j], p[s:e])
            for s, e in spans if 0 < Y[s:e, j].sum() < e - s]
    return float(np.mean(vals))


cal_rows, test_rows = load(CAL), load(TEST)
Str, Htr, Ytr = stack(cal_rows)
Ste, Hte, Yte = stack(test_rows)
print("cal8 n:", len(Ytr), "test100 n:", len(Yte), flush=True)
print("cal8 positives:", [int(Ytr[:, j].sum()) for j in range(6)],
      "test100 positives:", [int(Yte[:, j].sum()) for j in range(6)], flush=True)

# Frozen-alone dissociation score: 1 - frozen (frozen predicts P(formed)).
frozen = {}
for j, h in enumerate(HORIZONS):
    s = 1.0 - Ste[:, 0]
    frozen[str(h)] = dict(auc=float(roc_auc_score(Yte[:, j], s)),
                          auprc=float(average_precision_score(Yte[:, j], s)))
    print("frozen-alone (1-frozen) eval", h, frozen[str(h)], flush=True)

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)

for j1, h1 in enumerate(HORIZONS):
    for feat_name, Xtr, Xte in (("static", Str, Ste),
                                ("history", np.column_stack([Str, Htr]), np.column_stack([Ste, Hte]))):
        key = f"{feat_name}_train[{h1}]"
        results.setdefault(key, {})
        if all(str(h2) in results[key] for h2 in HORIZONS):
            print(key, "already done, skipped", flush=True)
            continue
        m = learner(Xtr, Ytr[:, j1])
        print(key, "n_iter_ =", m.n_iter_, flush=True)
        for j2, h2 in enumerate(HORIZONS):
            if str(h2) in results[key]:
                continue
            p = m.predict_proba(Xte)[:, 1]
            row = metrics(Yte[:, j2], p)
            row["macro_auprc"] = macro_auprc(test_rows, Yte, p, j2)
            results[key][str(h2)] = row
            print(key, "eval", h2, {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in row.items()}, flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))


def gain(h1, h2, macro=False):
    s = results[f"static_train[{h1}]"][str(h2)]
    a = results[f"history_train[{h1}]"][str(h2)]
    key = "macro_auprc" if macro else "auprc"
    return float(a[key] - s[key])


summary = {}
for h2 in HORIZONS:
    row = {}
    for h1 in HORIZONS:
        g = gain(h1, h2)
        g_ref = gain(h2, h2)
        gm = gain(h1, h2, macro=True)
        gm_ref = gain(h2, h2, macro=True)
        row[str(h1)] = {"transfer_gain": g, "matched_gain": g_ref,
                        "recovery_vs_matched": g / g_ref if g_ref else None,
                        "transfer_gain_macro": gm, "matched_gain_macro": gm_ref,
                        "recovery_vs_matched_macro": gm / gm_ref if gm_ref else None}
    summary[str(h2)] = row
print("summary (rows=eval horizon, cols=train horizon):", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_frozen_alone"] = frozen

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train_horizon_ns", "eval_horizon_ns", "n", "positives",
                "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for key, per_h in results.items():
        if key.startswith("_"):
            continue
        feat, h1 = key.split("_train[")
        h1 = h1[:-1]
        for h2 in HORIZONS:
            r = per_h[str(h2)]
            w.writerow([feat, h1, h2, r["n"], r["positives"], r["auc"], r["auprc"],
                        r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
