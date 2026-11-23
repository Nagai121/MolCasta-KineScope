"""G09: salt-bridge dissociation - how far does a short prefix of the real
history restore guidance? (K15 expanded conditions, 1 ns horizon)

Design follows G02/G06: take the real 0-1 ns history (11 frames at 100 ps)
and use only its beginning - 0.1 ns (frames [0,1]) or 0.5 ns (frames [0-5]).
Measures:
  (a) residual std of the prefix mean vs the full 0-1 ns mean (test + train).
  (b) recovery of the 1 ns gain when only the prefix mean is given to the
      DocD correction model. Reference: full-history gain +0.1788 (K53).
      The 0.1 ns prefix with 6 statistics is also measured as a reference.
Pre-fixed reading: 0.1 ns mean recovery >= 0.5 -> dissociation can go the
short-real-MD route for all 3 phenomena; < 0.5 -> salt dissociation needs a
longer prefix or another approach.
Base: K53 cache (~/workspace/kinescope/work/k53_cache/, suffix salt_diss),
split from K15 split.json. Static features 5, history threshold 4.0 A.
Model: HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
min_samples_leaf 15, L2 0.02, seed 42), retrained per condition/horizon.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
HERE = Path(__file__).resolve().parent
_SPLIT = json.loads((HERE.parent / "K15_saltbridge_expansion" / "split.json").read_text())
TRAIN_P = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"])
TEST_P = tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.0
PREFIXES = {"0.1ns": (0, 2), "0.5ns": (0, 6)}  # (start, stop) frame slices


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt_diss.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts]) if parts else np.empty((0,))


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                       max_leaf_nodes=31, min_samples_leaf=15,
                                       l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


train = load(TRAIN_P)
test = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")  # (n, 11 frames)
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)

full_mean_tr, full_mean_te = Htr.mean(1), Hte.mean(1)

# (a) residual std of prefix mean vs full mean
resid = {}
for tag, (a, b) in PREFIXES.items():
    for split, H, fm in (("train", Htr, full_mean_tr), ("test", Hte, full_mean_te)):
        d = H[:, a:b].mean(1) - fm
        resid[f"prefix_{tag}_{split}"] = dict(
            std=float(d.std()), bias=float(d.mean()),
            mae=float(np.abs(d).mean()))
        print(f"(a) prefix {tag} {split}: resid std {d.std():.4f} "
              f"bias {d.mean():+.4f}", flush=True)

# (b) correction-model conditions
conds = {
    "static_only": (None, None),
    "full_hist6": (hist6(Htr), hist6(Hte)),
    "mean_only_full": (full_mean_tr[:, None], full_mean_te[:, None]),
}
for tag, (a, b) in PREFIXES.items():
    pm_tr = Htr[:, a:b].mean(1)[:, None]
    pm_te = Hte[:, a:b].mean(1)[:, None]
    conds[f"prefix_mean_{tag}"] = (pm_tr, pm_te)
    conds[f"prefix_hist6_{tag}"] = (hist6(Htr[:, a:b]), hist6(Hte[:, a:b]))

out = {}
for name, (ftr, fte) in conds.items():
    out[name] = {}
    for hi, h in enumerate(HORIZONS):
        ytr, yte = Ytr[:, hi], Yte[:, hi]
        if ftr is None:
            p = predict(Xtr, ytr, Xte)
        else:
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                        np.concatenate([Xte, fte], axis=1))
        out[name][str(h)] = metrics(yte, p)
        print(name, h, "auprc", round(out[name][str(h)]["auprc"], 4), flush=True)
    (HERE / "results_partial.json").write_text(json.dumps(out, indent=2))

# base check against K53 (1 ns)
s = out["static_only"]["1"]["auprc"]
f = out["full_hist6"]["1"]["auprc"]
print("base check 1ns: static", round(s, 4), "(recorded 0.3734), full",
      round(f, 4), "(recorded 0.5522), gain", round(f - s, 4),
      "(recorded +0.1788)", flush=True)
assert abs(s - 0.3734) < 1e-4 and abs(f - 0.5522) < 1e-4, "base mismatch"

recovery = {}
for name in out:
    recovery[name] = {}
    for h in HORIZONS:
        g = out["full_hist6"][str(h)]["auprc"] - out["static_only"][str(h)]["auprc"]
        recovery[name][str(h)] = float(
            (out[name][str(h)]["auprc"] - out["static_only"][str(h)]["auprc"]) / g)

results = {"conditions": out, "prefix_residual": resid,
           "recovery_vs_full_hist6": recovery,
           "base_gain_1ns": float(f - s)}
(HERE / "results.json").write_text(json.dumps(results, indent=2))

rows = []
for name, by_h in out.items():
    for h, m in by_h.items():
        rows.append(dict(condition=name, horizon_ns=h,
                         recovery=recovery[name][h], **m))
with open(HERE / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", HERE / "results.json", flush=True)
print("recovery @1ns:",
      {n: round(recovery[n]["1"], 3) for n in recovery}, flush=True)
