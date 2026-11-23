#!/usr/bin/env python3
"""G20 bootstrap: resample test candidates (2000x) and recompute the
recovery of the predicted-fraction conditions from fixed test predictions.
Refits the same models as g20_run.py (deterministic, same seeds)."""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import average_precision_score

CACHE = "/home/hatch/workspace/kinescope/work/data_cache"
RICH = "/home/hatch/workspace/k40/out"
TRAIN_P = ["1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A"]
TEST_P = ["1mix_A", "2qt4_A"]
BOUND = 3.5


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(f"{CACHE}/{p}_R{r}.npz")
            w = np.load(f"{RICH}/{p}_R{r}.npz")
            parts.append(dict(X=z["X"], Xrich=w["Xrich"], series=z["hist"].T,
                              y=z["y"][:, 0]))
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def hist6(s):
    return np.stack([s[:, 0], s.mean(1), s.std(1), s.min(1),
                     s[:, -1] - s[:, 0], (s <= BOUND).mean(1)],
                    axis=1).astype(np.float32)


def clf():
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)


def reg():
    return HistGradientBoostingRegressor(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)


train, test = load(TRAIN_P), load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Rtr, Rte = cat(train, "Xrich"), cat(test, "Xrich")
Str, Ste = cat(train, "series"), cat(test, "series")
ytr, yte = cat(train, "y"), cat(test, "y")
ftr, fte = (Str <= BOUND).mean(1), (Ste <= BOUND).mean(1)

m7 = reg().fit(Xtr, ftr)
mr = reg().fit(Rtr, ftr)
conds = {
    "static": (Xtr, Xte),
    "static+pred_frac_docd7": (np.column_stack([Xtr, m7.predict(Xtr)]),
                               np.column_stack([Xte, np.clip(m7.predict(Xte), 0, 1)])),
    "static+pred_frac_rich99": (np.column_stack([Xtr, mr.predict(Rtr)]),
                                np.column_stack([Xte, np.clip(mr.predict(Rte), 0, 1)])),
    "full_history": (np.column_stack([Xtr, hist6(Str)]),
                     np.column_stack([Xte, hist6(Ste)])),
}
P = {}
for name, (a, b) in conds.items():
    P[name] = clf().fit(a, ytr).predict_proba(b)[:, 1]

rng = np.random.default_rng(20261124)
n = len(yte)
B = 2000
rec = {k: np.empty(B) for k in ("docd7", "rich99")}
for i in range(B):
    idx = rng.integers(0, n, n)
    yy = yte[idx]
    if yy.sum() == 0 or yy.sum() == len(yy):
        rec["docd7"][i] = np.nan
        rec["rich99"][i] = np.nan
        continue
    a_static = average_precision_score(yy, P["static"][idx])
    a_full = average_precision_score(yy, P["full_history"][idx])
    rec["docd7"][i] = (average_precision_score(
        yy, P["static+pred_frac_docd7"][idx]) - a_static) / (a_full - a_static)
    rec["rich99"][i] = (average_precision_score(
        yy, P["static+pred_frac_rich99"][idx]) - a_static) / (a_full - a_static)

out = {}
for k, v in rec.items():
    v = v[~np.isnan(v)]
    out[k] = dict(median=float(np.median(v)),
                  ci95=[float(np.percentile(v, 2.5)),
                        float(np.percentile(v, 97.5))],
                  p_gt_0=float((v > 0).mean()),
                  p_ge_0p5=float((v >= 0.5).mean()),
                  p_lt_0p2=float((v < 0.2).mean()))
print(json.dumps(out, indent=2))
with open("/home/hatch/workspace/g20/g20_bootstrap.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE")
