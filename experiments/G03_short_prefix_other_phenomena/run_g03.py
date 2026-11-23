"""G03: does the 0.1 ns short real-MD prefix restore guidance for other
phenomena and longer horizons? (G02's follow-up)

G02 found that the 0.1 ns prefix (frames [0,1]) of the real 0-1 ns
history restores 0.570 of the guidance for hydrogen-bond formation at
1 ns (weak base, Doc D). This experiment asks whether the same holds
for (a) contact formation (Doc D, 1 ns), (b) salt-bridge formation
(K15 conditions, 1 ns), (c) hydrogen-bond formation (Doc D, 90 ns).

Method (same as G02): the prefix mean is the mean of frames [0,1]
of the 11-frame 0-1 ns distance series. The condition is static
features + the prefix mean only (no other history statistics).
Recovery = (prefix-mean-only AUPRC - static-only AUPRC) /
           (full-history AUPRC - static-only AUPRC), where the
reference gain is the real-history gain for that same condition and
horizon. All models are HistGradientBoosting (max_iter=120, lr .075,
max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42), retrained
per condition. Distances use the per-frame box, axis-wise (already
applied at extraction; the caches are reused as-is).

Pre-fixed reading: if all three recoveries are 0.5 or more, the short
real-MD path works across phenomena and horizons. If any one is below
0.5, shortening alone is not enough for that phenomenon/horizon.

Verification: (a) static/full at 1 ns must reproduce K16 (0.3398 /
0.5669); (b) static/full at 1 ns must reproduce K15 (0.2402 /
0.3842); (c) is computed fresh at 90 ns (no prior record to check
against; the 1 ns values from the same code reproduce G02: static
0.3637 / full 0.4815).
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

OUT = Path(__file__).resolve().parent
K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")
K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
DCACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")

ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")


def hist6(series, bound):
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


def load_contact(cache, prots, suffix):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(cache / f"{p}_R{r}_{suffix}.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def load_hb(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(DCACHE / f"{p}_R{r}.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def run_one(label, Xtr, Htr, Ytr, Xte, Hte, Yte, horizon_idx, bound, verify=None):
    """X: static, H: (n,11) series, Y: (n,6) labels. Returns dict of metrics."""
    j = horizon_idx
    ytr, yte = Ytr[:, j], Yte[:, j]
    full_mean_tr, full_mean_te = Htr.mean(1), Hte.mean(1)
    prefix_tr = Htr[:, [0, 1]].mean(1)
    prefix_te = Hte[:, [0, 1]].mean(1)
    conds = {
        "static_only": (Xtr, Xte),
        "full_hist": (np.column_stack([Xtr, hist6(Htr, bound)]),
                      np.column_stack([Xte, hist6(Hte, bound)])),
        "prefix_mean_0.1ns": (np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32)]),
                              np.column_stack([Xte, prefix_te[:, None].astype(np.float32)])),
    }
    out = {"label": label, "horizon_ns": (1, 5, 10, 30, 50, 90)[j],
           "train_n": int(len(ytr)), "test_n": int(len(yte)),
           "test_positives": int(yte.sum()),
           "residual_std_test": float(np.std(prefix_te - full_mean_te)),
           "residual_std_train": float(np.std(prefix_tr - full_mean_tr)),
           "residual_mean_test": float(np.mean(prefix_te - full_mean_te))}
    for name, (xtr, xte) in conds.items():
        p = predict(xtr, ytr, xte)
        out[name] = metrics(yte, p)
        print(f"{label} {name}: AUPRC {out[name]['auprc']:.4f} AUC {out[name]['auc']:.4f}", flush=True)
    s, f, pm = out["static_only"]["auprc"], out["full_hist"]["auprc"], out["prefix_mean_0.1ns"]["auprc"]
    out["full_gain"] = f - s
    out["prefix_gain"] = pm - s
    out["recovery"] = (pm - s) / (f - s) if f != s else None
    print(f"{label} full gain {out['full_gain']:.4f} prefix gain {out['prefix_gain']:.4f} "
          f"recovery {out['recovery']:.3f} residual std (test) {out['residual_std_test']:.3f} A", flush=True)
    if verify:
        for name, ref in verify.items():
            print(f"{label} verify {name}: diff {abs(out[name]['auprc'] - ref):.6f}", flush=True)
    return out


results = {}

# (a) contact formation, Doc D, 1 ns (K13 cache, bound 4.5)
tr = load_contact(K13, ORIG_TRAIN, "contact")
te = load_contact(K13, ORIG_TEST, "contact")
Xtr = np.concatenate([p["X"] for p in tr]).astype(np.float64)
Xte = np.concatenate([p["X"] for p in te]).astype(np.float64)
Htr = np.concatenate([p["hist"] for p in tr])
Hte = np.concatenate([p["hist"] for p in te])
Ytr = np.concatenate([p["y"] for p in tr])
Yte = np.concatenate([p["y"] for p in te])
results["a_contact_formation_1ns"] = run_one(
    "contact formation 1ns", Xtr, Htr, Ytr, Xte, Hte, Yte, 0, 4.5,
    verify={"static_only": 0.3398, "full_hist": 0.5669})

# (b) salt-bridge formation, K15 conditions, 1 ns (K15 cache, bound 4.0)
split = json.loads((K15_DIR / "split.json").read_text())
train_p = ORIG_TRAIN + tuple(split["train_additions"])
test_p = ORIG_TEST + tuple(split["test_additions"])
tr = load_contact(K15, train_p, "salt")
te = load_contact(K15, test_p, "salt")
Xtr = np.concatenate([p["X"] for p in tr]).astype(np.float64)
Xte = np.concatenate([p["X"] for p in te]).astype(np.float64)
Htr = np.concatenate([p["hist"] for p in tr])
Hte = np.concatenate([p["hist"] for p in te])
Ytr = np.concatenate([p["y"] for p in tr])
Yte = np.concatenate([p["y"] for p in te])
print("salt-bridge train n:", len(Ytr), "test n:", len(Yte), flush=True)
results["b_saltbridge_formation_1ns"] = run_one(
    "salt-bridge formation 1ns", Xtr, Htr, Ytr, Xte, Hte, Yte, 0, 4.0,
    verify={"static_only": 0.2402, "full_hist": 0.3842})

# (c) hydrogen-bond formation, Doc D, 90 ns (data_cache, bound 3.5)
tr = load_hb(ORIG_TRAIN)
te = load_hb(ORIG_TEST)
Xtr = np.concatenate([p["X"] for p in tr]).astype(np.float64)
Xte = np.concatenate([p["X"] for p in te]).astype(np.float64)
Htr = np.concatenate([p["hist"].T for p in tr])
Hte = np.concatenate([p["hist"].T for p in te])
Ytr = np.concatenate([p["y"] for p in tr])
Yte = np.concatenate([p["y"] for p in te])
results["c_hb_formation_90ns"] = run_one(
    "HB formation 90ns", Xtr, Htr, Ytr, Xte, Hte, Yte, 5, 3.5)

# sanity: same code at 1 ns must reproduce G02 (static 0.3637 / full 0.4815)
chk = run_one("HB formation 1ns (check vs G02)", Xtr, Htr, Ytr, Xte, Hte, Yte, 0, 3.5,
              verify={"static_only": 0.3637, "full_hist": 0.4815})
results["check_hb_formation_1ns"] = chk

(OUT / "results.json").write_text(json.dumps(results, indent=2))
print("saved to", OUT)
