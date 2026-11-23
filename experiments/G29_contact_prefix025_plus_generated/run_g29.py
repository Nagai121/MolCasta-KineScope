"""G29: contact formation - where does the mixing benefit disappear?
(DocD conditions, contact formation, 1 ns)

G13 found that for contact formation, adding K17's residue-level
generated mean (proposals 11-14) to the 0.1 ns prefix mean raised
recovery by +0.070 to +0.097 (proposals 11-13; proposal 14 did not
help). G25 found that with the 0.5 ns prefix, no proposal reached
+0.05. This experiment measures the in-between 0.25 ns prefix, whose
mean alone recovers 0.512 (G07), to find where the benefit cuts off.

Generated mean: K17's residue-level generated series (proposals 11-14),
branch-mean 9-point series, mean over the 9 points - the same quantity
K17 used as the mean column of its 6 statistics. Real side: the K13
cache (static 5, 11-frame history, labels), verified array-identical
to the real side stored in K17's files, so the generated means join
positionally. No generator checkpoint is loaded; no new trajectories
are generated.

Conditions (1 ns, HistGradientBoosting 120 iters, lr .075, 31 leaves,
min_samples_leaf 15, L2 .02, seed 42, retrained per condition):
  static_only, full_hist (static + real hist6, bound 4.5 A),
  prefix_mean_0.25ns (static + mean of frames [0,1,2]),
  gen_mean_only_p<n> (static + generated mean of proposal n),
  mixed_p<n> (static + 0.25 ns prefix mean + generated mean of proposal n),
for n in 11..14. Recovery = (AUPRC - static) / (full - static);
the full-history gain is +0.2271 (0.5669 - 0.3398, K16/K13).

Pre-fixed reading: if any proposal's mixed recovery rises >= +0.05
above the prefix-mean-only recovery, the mixing benefit lasts to the
0.25 ns prefix. If all proposals are below +0.05, the benefit cuts off
between the 0.1 ns and 0.25 ns prefixes.
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
H_IDX = 0  # 1 ns
PROPOSALS = (11, 12, 13, 14)
PREFIX_FRAMES = [0, 1, 2]  # 0.25 ns の切り取り（G07 と同じ）


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

prefix_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_te = Hte[:, PREFIX_FRAMES].mean(1)
full_mean_tr, full_mean_te = Htr.mean(1), Hte.mean(1)
gen_tr = {n: np.concatenate(gtr[n]) for n in PROPOSALS}
gen_te = {n: np.concatenate(gte[n]) for n in PROPOSALS}

# relations between the 0.5 ns prefix mean and each generated mean (test side)
relations = {}
for n in PROPOSALS:
    g = gen_te[n]
    relations[f"p{n}"] = {
        "corr_prefix_gen": float(np.corrcoef(prefix_te, g)[0, 1]),
        "gen_minus_prefix_mean": float(np.mean(g - prefix_te)),
        "gen_minus_prefix_std": float(np.std(g - prefix_te)),
        "gen_minus_full_mean": float(np.mean(g - full_mean_te)),
        "gen_minus_full_std": float(np.std(g - full_mean_te)),
        "corr_gen_full": float(np.corrcoef(g, full_mean_te)[0, 1]),
        "corr_prefix_full": float(np.corrcoef(prefix_te, full_mean_te)[0, 1]),
    }
    print(f"p{n} relations (test):", json.dumps(relations[f"p{n}"]), flush=True)

conds = {
    "static_only": (Xtr, Xte),
    "full_hist": (np.column_stack([Xtr, hist6(Htr)]),
                  np.column_stack([Xte, hist6(Hte)])),
    "prefix_mean_0.25ns": (np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32)]),
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

out = {"label": "contact formation 1ns, 0.25ns prefix + K17 generated mean",
       "train_n": int(len(Xtr)), "test_n": int(len(Xte)),
       "test_positives": int(Yte[:, H_IDX].sum()), "relations": relations}
ytr, yte = Ytr[:, H_IDX], Yte[:, H_IDX]
for name, (xtr, xte) in conds.items():
    p = predict(xtr, ytr, xte)
    out[name] = metrics(yte, p)
    print(name, "AUPRC %.4f AUC %.4f" % (out[name]["auprc"], out[name]["auc"]), flush=True)

s = out["static_only"]["auprc"]
f = out["full_hist"]["auprc"]
out["full_gain"] = f - s
print("verify static_only diff %.6f, full_hist diff %.6f" % (abs(s - 0.3398), abs(f - 0.5669)), flush=True)
print("verify prefix_mean_0.25ns diff %.6f (G07: 0.4561)" % abs(out["prefix_mean_0.25ns"]["auprc"] - 0.4561), flush=True)
for name in conds:
    if name in ("static_only", "full_hist"):
        continue
    out[name]["gain"] = out[name]["auprc"] - s
    out[name]["recovery"] = (out[name]["auprc"] - s) / (f - s)
    print(f"{name}: gain {out[name]['gain']:.4f} recovery {out[name]['recovery']:.3f}", flush=True)
for n in PROPOSALS:
    out[f"mixed_p{n}"]["mixed_minus_prefix"] = out[f"mixed_p{n}"]["recovery"] - out["prefix_mean_0.25ns"]["recovery"]
    print(f"mixed_p{n} - prefix recovery: {out[f'mixed_p{n}']['mixed_minus_prefix']:+.3f}", flush=True)

(HERE / "results.json").write_text(json.dumps(out, indent=2))
print("saved to", HERE)
