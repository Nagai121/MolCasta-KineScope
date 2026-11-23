"""G42: contact formation - does adding a generated mean to the short
prefix mean help at 10 ns? (DocD conditions, contact formation, 10 ns)

G36 measured the same mixture for contact formation at 5 ns: only
proposal 11 rose +0.057 above the 0.1 ns prefix mean (p12 +0.019,
p13 -0.035, p14 -0.007). Here the same question is asked at 10 ns,
where the 0.1 ns prefix mean alone has not been measured yet.

Generated mean: K17's residue-level generated series (proposals 11-14),
branch-mean 9-point series, mean over the 9 points - the same quantity
K17 used as the mean column of its 6 statistics. Real side: the K13 cache
(static 5, 11-frame history, labels), verified array-identical to the real
side stored in K17's files, so the generated means join positionally.
No generator checkpoint is loaded; no new trajectories are generated.

Conditions (10 ns, HistGradientBoosting 120 iters, lr .075, 31 leaves,
min_samples_leaf 15, L2 .02, seed 42, retrained per condition):
  static_only, full_hist (static + real hist6, bound 4.5 A),
  prefix_mean_0.1ns (static + mean of frames [0,1]),
  gen_mean_only_<n> (static + generated mean of proposal n),
  mixed_<n> (static + prefix mean + generated mean of proposal n),
for n in 11..14. Recovery = (AUPRC - static) / (full - static);
the full-history gain at 10 ns is measured in this run (the 1 ns gain
was +0.2271, K16/K13).

Pre-fixed reading: if any proposal's mixed recovery rises >= +0.05
above the prefix-mean-only recovery, mixing helps for contact at
10 ns too; if all are < +0.05, mixing for contact is limited to short
horizons.
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

K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")
K17 = Path("/home/hatch/workspace/k17/out")
HERE = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
H_IDX = 2  # 10 ns
PROPOSALS = (11, 12, 13, 14)


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


def load(prots):
    """Real side from K13 cache; generated means from K17 files, joined
    positionally after verifying the real sides are array-identical."""
    real, gen = [], {n: [] for n in PROPOSALS}
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K13 / f"{p}_R{r}_contact.npz")
            k = np.load(K17 / f"{p}_R{r}.npz")
            assert np.array_equal(k["X"], z["X"]) and np.array_equal(k["hist"], z["hist"]) \
                and np.array_equal(k["y"], z["y"]), f"real-side mismatch {p} R{r}"
            real.append({kk: z[kk] for kk in z.files})
            for n in PROPOSALS:
                # branch-mean 9-point series -> mean over the 9 points
                gen[n].append(k[f"gen{n}"].mean(axis=0).mean(axis=0).astype(np.float64))
    return real, gen


train, gtr = load(TRAIN_P)
test, gte = load(TEST_P)
Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
Htr = np.concatenate([d["hist"] for d in train])
Hte = np.concatenate([d["hist"] for d in test])
Ytr = np.concatenate([d["y"] for d in train])
Yte = np.concatenate([d["y"] for d in test])
print("train", Xtr.shape, "test", Xte.shape, "test pos", Yte[:, H_IDX].sum(), flush=True)

prefix_tr = Htr[:, [0, 1]].mean(1)
prefix_te = Hte[:, [0, 1]].mean(1)
full_mean_tr, full_mean_te = Htr.mean(1), Hte.mean(1)
gen_tr = {n: np.concatenate(gtr[n]) for n in PROPOSALS}
gen_te = {n: np.concatenate(gte[n]) for n in PROPOSALS}

conds = {
    "static_only": (Xtr, Xte),
    "full_hist": (np.column_stack([Xtr, hist6(Htr)]),
                  np.column_stack([Xte, hist6(Hte)])),
    "prefix_mean_0.1ns": (np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32)]),
                          np.column_stack([Xte, prefix_te[:, None].astype(np.float32)])),
}
for n in PROPOSALS:
    conds[f"gen_mean_only_p{n}"] = (
        np.column_stack([Xtr, gen_tr[n][:, None].astype(np.float32)]),
        np.column_stack([Xte, gen_te[n][:, None].astype(np.float32)]))
    conds[f"mixed_p{n}"] = (
        np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32),
                         gen_tr[n][:, None].astype(np.float32)]),
        np.column_stack([Xte, prefix_te[:, None].astype(np.float32),
                         gen_te[n][:, None].astype(np.float32)]))

out = {}
ytr, yte = Ytr[:, H_IDX], Yte[:, H_IDX]
for name, (xtr, xte) in conds.items():
    p = predict(xtr, ytr, xte)
    out[name] = metrics(yte, p)
    print(name, "AUPRC %.4f AUC %.4f" % (out[name]["auprc"], out[name]["auc"]), flush=True)
    (HERE / "results_partial.json").write_text(json.dumps(out, indent=2))

s = out["static_only"]["auprc"]
f = out["full_hist"]["auprc"]
print("base check 10ns: static %.4f, full %.4f, gain %.4f"
      % (s, f, f - s), flush=True)

recovery = {name: float((m["auprc"] - s) / (f - s)) for name, m in out.items()}

# relation stats (same form as G11), test side
rel = {}
for n in PROPOSALS:
    g, pm, fm = gen_te[n], prefix_te, full_mean_te
    rel[f"p{n}"] = dict(
        corr_prefix_gen=float(np.corrcoef(pm, g)[0, 1]),
        gen_minus_prefix_mean=float((g - pm).mean()),
        gen_minus_prefix_std=float((g - pm).std()),
        gen_minus_full_mean=float((g - fm).mean()),
        gen_minus_full_std=float((g - fm).std()),
        corr_gen_full=float(np.corrcoef(g, fm)[0, 1]),
        corr_prefix_full=float(np.corrcoef(pm, fm)[0, 1]))

results = {"conditions": out, "recovery_vs_full_hist": recovery,
           "base_gain_10ns": float(f - s),
           "prefix_recovery_10ns": recovery["prefix_mean_0.1ns"],
           "mixed_delta_vs_prefix": {f"p{n}": recovery[f"mixed_p{n}"] - recovery["prefix_mean_0.1ns"]
                                     for n in PROPOSALS},
           "relations_test": rel}
(HERE / "results.json").write_text(json.dumps(results, indent=2))

rows = [dict(condition=name, recovery=recovery[name], **m) for name, m in out.items()]
with open(HERE / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", HERE / "results.json", flush=True)
print("recovery:", {n: round(v, 3) for n, v in recovery.items()}, flush=True)
print("mixed delta vs prefix:", {n: round(v, 3) for n, v in results["mixed_delta_vs_prefix"].items()}, flush=True)
