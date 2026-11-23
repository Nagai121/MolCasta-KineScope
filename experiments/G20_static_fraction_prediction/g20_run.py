#!/usr/bin/env python3
"""G20: can the fraction (of history points <= 3.5 A) be predicted from static
information? DocD conditions, HB formation, 1 ns.

- Fraction: fraction of the 11-point backward history (frames 0-10) <= 3.5 A.
- Predictors: HistGradientBoostingRegressor (DocD settings) from (a) DocD-7
  static features, (b) K40's rich 99 static features. Trained on train proteins.
- Replica floor: pairs that are candidates in all 3 replicas of a protein;
  pooled std of pairwise replica differences of the fraction.
- Recovery: DocD HistGB classifier, static7 + predicted fraction (only history
  feature), gain vs static-only, recovery vs full-history gain (+0.1178 ref)
  and vs the real-fraction-only gain.
Inputs: K08 cache (~/workspace/kinescope/work/data_cache) + K40 rich features
(~/workspace/k40/out) - rows verified to be in identical order.
"""
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
ALL_P = TRAIN_P + TEST_P
BOUND = 3.5
H = 0  # 1 ns


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(f"{CACHE}/{p}_R{r}.npz")
            w = np.load(f"{RICH}/{p}_R{r}.npz")
            assert np.array_equal(z["donor_atom"], w["donor_atom"])
            assert np.array_equal(z["acceptor_atom"], w["acceptor_atom"])
            parts.append(dict(
                X=z["X"], Xrich=w["Xrich"], series=z["hist"].T,
                y=z["y"][:, H], donor=z["donor_atom"],
                acceptor=z["acceptor_atom"], protein=p, replica=r))
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def hist6(series):
    return np.stack([series[:, 0], series.mean(1), series.std(1),
                     series.min(1), series[:, -1] - series[:, 0],
                     (series <= BOUND).mean(1)], axis=1).astype(np.float32)


def clf():
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)


def reg():
    return HistGradientBoostingRegressor(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)


def auprc(xtr, ytr, xte, yte):
    m = clf().fit(xtr, ytr)
    return float(average_precision_score(yte, m.predict_proba(xte)[:, 1]))


train, test = load(TRAIN_P), load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Rtr, Rte = cat(train, "Xrich"), cat(test, "Xrich")
Str, Ste = cat(train, "series"), cat(test, "series")
ytr, yte = cat(train, "y"), cat(test, "y")
ftr = (Str <= BOUND).mean(1)
fte = (Ste <= BOUND).mean(1)
print(f"train {len(ytr)} test {len(yte)} test pos {int(yte.sum())}", flush=True)
print(f"fraction: train mean {ftr.mean():.4f} test mean {fte.mean():.4f} "
      f"test std {fte.std():.4f}", flush=True)

# --- Replica floor for the fraction ---
floor = {}
all_parts = load(ALL_P)
for p in ALL_P:
    reps = [q for q in all_parts if q["protein"] == p]
    keys = [set(zip(q["donor"].tolist(), q["acceptor"].tolist())) for q in reps]
    common = keys[0] & keys[1] & keys[2]
    fr = []
    for q in reps:
        d = {(int(a), int(b)): float(f) for a, b, f in
             zip(q["donor"], q["acceptor"], (q["series"] <= BOUND).mean(1))}
        fr.append(np.array([d[k] for k in sorted(common)]))
    F = np.stack(fr, 1)  # (pairs, 3 replicas)
    diffs = np.concatenate([F[:, 0] - F[:, 1], F[:, 0] - F[:, 2],
                            F[:, 1] - F[:, 2]])
    floor[p] = dict(n_pairs=int(len(F)), diff_std=float(diffs.std()),
                    per_pair_std_mean=float(F.std(1).mean()))
    if p == ALL_P[0]:
        all_diffs = [diffs]
    else:
        all_diffs.append(diffs)
all_diffs = np.concatenate(all_diffs)
floor["_pooled"] = dict(
    n_pairs=int(sum(floor[p]["n_pairs"] for p in ALL_P)),
    diff_std=float(all_diffs.std()),
    diff_mae=float(np.abs(all_diffs).mean()))
print("floor:", json.dumps(floor["_pooled"]), flush=True)

# --- Static prediction of the fraction ---
pred = {}
for name, Atr, Ate in (("docd7", Xtr, Xte), ("rich99", Rtr, Rte)):
    m = reg().fit(Atr, ftr)
    p = np.clip(m.predict(Ate), 0.0, 1.0)
    err = p - fte
    pred[name] = dict(pred=p, bias=float(err.mean()),
                      std=float(err.std()), mae=float(np.abs(err).mean()),
                      corr=float(np.corrcoef(p, fte)[0, 1]))
    print(f"pred {name}: bias {err.mean():+.4f} std {err.std():.4f} "
          f"corr {pred[name]['corr']:.4f}", flush=True)

# --- Classification: recovery ---
conds = {
    "static": (Xtr, Xte),
    "static+real_frac": (np.column_stack([Xtr, ftr]),
                         np.column_stack([Xte, fte])),
    "static+pred_frac_docd7": (np.column_stack([Xtr, reg().fit(Xtr, ftr).predict(Xtr)]),
                               np.column_stack([Xte, pred["docd7"]["pred"]])),
    "static+pred_frac_rich99": (np.column_stack([Xtr, reg().fit(Rtr, ftr).predict(Rtr)]),
                                np.column_stack([Xte, pred["rich99"]["pred"]])),
    "full_history": (np.column_stack([Xtr, hist6(Str)]),
                     np.column_stack([Xte, hist6(Ste)])),
}
res = {}
for name, (a, b) in conds.items():
    res[name] = auprc(a, ytr, b, yte)
    print(f"{name}: AUPRC {res[name]:.4f}", flush=True)

gain_full = res["full_history"] - res["static"]
gain_real = res["static+real_frac"] - res["static"]
out = {
    "n_train": int(len(ytr)), "n_test": int(len(yte)),
    "test_positives": int(yte.sum()),
    "fraction_test_mean": float(fte.mean()),
    "fraction_test_std": float(fte.std()),
    "replica_floor": floor,
    "fraction_prediction": {k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                            for k, v in pred.items()},
    "auprc": res,
    "gain_full_history": float(gain_full),
    "gain_real_frac": float(gain_real),
    "recovery_vs_full": {k: float((v - res["static"]) / gain_full)
                         for k, v in res.items()},
    "recovery_vs_real_frac": {
        k: (float((v - res["static"]) / gain_real) if gain_real else None)
        for k, v in res.items()},
}
with open("/home/hatch/workspace/g20/g20_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: out[k] for k in
                  ("auprc", "recovery_vs_full", "recovery_vs_real_frac")},
                 indent=2))
print("DONE")
