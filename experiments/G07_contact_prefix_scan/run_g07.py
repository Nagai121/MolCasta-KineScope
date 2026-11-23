"""G07: for contact formation, how far does lengthening the prefix
recover from the 0.1 ns mean-only 0.432? (Doc D, contact formation,
1 ns)

G03 found that the 0.1 ns prefix mean alone recovers only 0.432 of
the real-history gain for contact formation (Doc D, 1 ns), below the
0.5 line, while salt-bridge formation (0.647) and HB formation at
90 ns (0.590) cleared it. This experiment scans longer prefixes:
0.25 ns (frames [0,1,2]) and 0.5 ns (frames [0..5]), each with the
mean only and with all 6 history statistics computed on the prefix
(first, mean, std, min, last-first, frac <= 4.5 A; "last" is the
last frame of the prefix). Same definitions as G02.

Reference gain: the real-history gain +0.2271 (static 0.3398 ->
full 0.5669, K16). Recovery = (condition AUPRC - static AUPRC) /
full gain. All models are HistGradientBoosting (max_iter=120,
lr .075, max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42),
retrained per condition. Distances use the per-frame box, axis-wise
(already applied at extraction; the K13 cache is reused as-is).

Pre-fixed reading: if the 0.5 ns prefix with 6 statistics recovers
0.5 or more, contact works on the short-real-MD route with a longer
prefix. If it is still below 0.5 at 0.5 ns, contact needs the full
1 ns or another approach.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

OUT = Path(__file__).resolve().parent
K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
HORIZON_IDX = 0  # 1 ns

CUTS = {
    "0.25ns": [0, 1, 2],
    "0.5ns": [0, 1, 2, 3, 4, 5],
}


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K13 / f"{p}_R{r}_contact.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


tr = load(TRAIN_P)
te = load(TEST_P)
Xtr = np.concatenate([p["X"] for p in tr]).astype(np.float64)
Xte = np.concatenate([p["X"] for p in te]).astype(np.float64)
Htr = np.concatenate([p["hist"] for p in tr])
Hte = np.concatenate([p["hist"] for p in te])
Ytr = np.concatenate([p["y"] for p in tr])
Yte = np.concatenate([p["y"] for p in te])
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train n:", len(ytr), "test n:", len(yte), "test pos 1ns:", int(yte.sum()), flush=True)

full_mean_tr, full_mean_te = Htr.mean(1), Hte.mean(1)

conds = {
    "static_only": (Xtr, Xte),
    "full_hist": (np.column_stack([Xtr, hist6(Htr)]), np.column_stack([Xte, hist6(Hte)])),
}
for name, idx in CUTS.items():
    pm_tr = Htr[:, idx].mean(1)
    pm_te = Hte[:, idx].mean(1)
    conds[f"prefix_mean_{name}"] = (
        np.column_stack([Xtr, pm_tr[:, None].astype(np.float32)]),
        np.column_stack([Xte, pm_te[:, None].astype(np.float32)]))
    conds[f"prefix_hist6_{name}"] = (
        np.column_stack([Xtr, hist6(Htr[:, idx])]),
        np.column_stack([Xte, hist6(Hte[:, idx])]))

results = {"label": "contact formation 1ns", "train_n": int(len(ytr)), "test_n": int(len(yte)),
           "test_positives": int(yte.sum()), "cuts": {k: v for k, v in CUTS.items()}}
for name, (xtr, xte) in conds.items():
    p = predict(xtr, ytr, xte)
    results[name] = metrics(yte, p)
    print(f"{name}: AUPRC {results[name]['auprc']:.4f} AUC {results[name]['auc']:.4f}", flush=True)

s = results["static_only"]["auprc"]
f = results["full_hist"]["auprc"]
results["full_gain"] = f - s
print(f"verify static_only diff {abs(s - 0.3398):.6f}, full_hist diff {abs(f - 0.5669):.6f}", flush=True)
for name in conds:
    if name in ("static_only", "full_hist"):
        continue
    results[name]["gain"] = results[name]["auprc"] - s
    results[name]["recovery"] = (results[name]["auprc"] - s) / (f - s)
    print(f"{name}: gain {results[name]['gain']:.4f} recovery {results[name]['recovery']:.3f}", flush=True)
for name, idx in CUTS.items():
    pm_te = Hte[:, idx].mean(1)
    results[f"residual_std_test_{name}"] = float(np.std(pm_te - full_mean_te))
    print(f"residual std test {name}: {results[f'residual_std_test_{name}']:.3f} A", flush=True)

(OUT / "results.json").write_text(json.dumps(results, indent=2))
print("saved to", OUT)
