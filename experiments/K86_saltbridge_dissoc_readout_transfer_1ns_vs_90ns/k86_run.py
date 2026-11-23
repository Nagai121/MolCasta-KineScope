"""K86: does the salt-bridge DISSOCIATION readout transfer across horizons?
(K53 conditions: weak DocD base, K15 expanded split, salt dissociation)

Method: K53's conditions (HistGradientBoosting max_iter=120, lr .075,
max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42), on the K53
cache (candidates formed at the origin, label = min distance > 4.0 A
at the horizon). Train on the 58 train proteins, evaluate on the 30
test proteins. The features are horizon-independent, so one model
trained at horizon h1 is evaluated against every horizon h2's labels;
the full 6x6 matrix is computed.
  - primary: train@1 -> eval@90, and train@90 -> eval@1
  - reference: the matched-horizon condition's gain (train@h -> eval@h)
The gain of a condition is its AUPRC minus the static-only condition's
AUPRC trained at the SAME EVAL horizon and evaluated at the same eval
horizon (K56's definition; the static reference is always the one
trained at the horizon being evaluated, not the one trained at the
learning horizon). Pooled over the 30 test proteins, and per-protein
macro: AUPRC per protein pooled over its 3 replicas, averaged over
proteins with both classes present at that horizon.
Recovery = cross-horizon gain / matched-horizon gain.
Sanity check: the matched gains must reproduce K53's recorded values
(1 ns +0.1788, 90 ns +0.0582).

Pre-fixed reading: if the gain recovers 80% or more of the reference
gain in BOTH directions (1 -> 90 and 90 -> 1), the salt-bridge
dissociation readout also transfers across horizons. If only one
direction is weak, investigate why.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, csv
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
REPO = Path("/home/hatch/workspace/kinescope/repo/experiments")
HERE = REPO / "K86_saltbridge_dissoc_readout_transfer_1ns_vs_90ns"
OUT = HERE
PARTIAL = OUT / "results_partial.json"
_SPLIT = json.loads((REPO / "K15_saltbridge_expansion" / "split.json").read_text())
TRAIN_P = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"])
TEST_P = tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.0
K53_RECORDED_GAIN = {"1": 0.1788, "90": 0.0582}


def load(prots, track_protein=False):
    parts, prots_out = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt_diss.npz")
            parts.append({k: z[k] for k in z.files})
            prots_out.append(np.full(len(z["y"]), p, dtype=object))
    return parts, (np.concatenate(prots_out) if track_protein else None)


def cat(parts, key):
    return np.concatenate([p[key] for p in parts]) if parts else np.empty((0,))


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def fit(X, y):
    if np.unique(y).size < 2:
        return None, float(y.mean())
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                       max_leaf_nodes=31, min_samples_leaf=15,
                                       l2_regularization=.02, random_state=42)
    m.fit(X, y)
    return m, None


def apply(model, const, X):
    if model is None:
        return np.full(len(X), const)
    return model.predict_proba(X)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def macro_auprc(prot_labels, Y, p, j):
    vals, included = [], 0
    for prot in np.unique(prot_labels):
        m = prot_labels == prot
        yj = Y[m, j]
        if 0 < yj.sum() < m.sum():
            vals.append(average_precision_score(yj, p[m]))
            included += 1
    return float(np.mean(vals)) if vals else None, included


train_parts, _ = load(TRAIN_P)
test_parts, test_prot = load(TEST_P, track_protein=True)
Xtr = cat(train_parts, "X"); Xte = cat(test_parts, "X")
Ftr = hist6(cat(train_parts, "hist")); Fte = hist6(cat(test_parts, "hist"))
Ytr = cat(train_parts, "y"); Yte = cat(test_parts, "y")
print("train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)
print("test per-protein n:", {p: int((test_prot == p).sum()) for p in TEST_P}, flush=True)

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed from partial", flush=True)

for j1, h1 in enumerate(HORIZONS):
    for feat_name, Xtr_f, Xte_f in (("static", Xtr, Xte),
                                    ("history", np.concatenate([Xtr, Ftr], axis=1),
                                     np.concatenate([Xte, Fte], axis=1))):
        key = f"{feat_name}_train[{h1}]"
        results.setdefault(key, {})
        if all(str(h2) in results[key] for h2 in HORIZONS):
            print(key, "already done, skipped", flush=True)
            continue
        model, const = fit(Xtr_f, Ytr[:, j1])
        p = apply(model, const, Xte_f)
        for j2, h2 in enumerate(HORIZONS):
            if str(h2) in results[key]:
                continue
            row = metrics(Yte[:, j2], p)
            ma, inc = macro_auprc(test_prot, Yte, p, j2)
            row["macro_auprc"] = ma
            row["macro_n_proteins"] = inc
            results[key][str(h2)] = row
            print(key, "eval", h2, {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in row.items()}, flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))


def gain(h1, h2, macro=False):
    # K56's definition: the static reference is trained at the EVAL horizon.
    s = results[f"static_train[{h2}]"][str(h2)]
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

print("matched gains vs K53 recorded:")
for h in ("1", "90"):
    print(" ", h, "ns: matched gain", round(summary[h][h]["matched_gain"], 4),
          "recorded", K53_RECORDED_GAIN[h],
          "diff", round(summary[h][h]["matched_gain"] - K53_RECORDED_GAIN[h], 6))
print("primary 1->90 recovery (pooled / macro):",
      round(summary["90"]["1"]["recovery_vs_matched"], 3),
      round(summary["90"]["1"]["recovery_vs_matched_macro"], 3))
print("primary 90->1 recovery (pooled / macro):",
      round(summary["1"]["90"]["recovery_vs_matched"], 3),
      round(summary["1"]["90"]["recovery_vs_matched_macro"], 3))
print("summary (rows=eval horizon, cols=train horizon):", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_k53_recorded_gain"] = K53_RECORDED_GAIN

(OUT / "k86_results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "k86_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train_horizon_ns", "eval_horizon_ns", "n", "positives",
                "auc", "auprc", "macro_auprc", "macro_n_proteins", "brier", "logloss"])
    for key, per_h in results.items():
        if key.startswith("_"):
            continue
        feat, h1 = key.split("_train[")
        h1 = h1[:-1]
        for h2 in HORIZONS:
            r = per_h[str(h2)]
            w.writerow([feat, h1, h2, r["n"], r["positives"], r["auc"], r["auprc"],
                        r["macro_auprc"], r["macro_n_proteins"], r["brier"], r["logloss"]])
print("saved to", OUT)
