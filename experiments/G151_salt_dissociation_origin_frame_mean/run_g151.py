"""G151: for salt-bridge dissociation under the K15 expanded conditions,
does including the origin frame change the prefix mean-only recovery?

G142 measured the prefix mean only over frames [0,1,2,3,4,5,6,7,8,9]
(10 frames, covering 0-0.9 ns): recovery 1.052/0.957/1.028/1.240/0.745/0.892
(horizons 1/5/10/30/50/90 ns). G144 (hydrogen-bond formation) found that
including the origin frame changed the recovery a lot at some horizons
(10 ns +0.269, 50 ns -0.158). This experiment measures, for the same six
horizons, the prefix mean only over all 11 frames [0..10] (0-1 ns,
including the origin frame 10) and compares it with the 10-frame mean,
reusing the K53 cache and split as-is.

Method (same as G02/G03/G09/G69/G142): the condition is static features +
the prefix mean only (no other history statistics). Recovery =
(prefix-mean-only AUPRC - static-only AUPRC) /
(full-history AUPRC - static-only AUPRC), where the reference gain is
the real-history gain for that same horizon (K53 record: 1 ns +0.1788,
5 ns +0.1139, 10 ns +0.1781, 30 ns +0.0633, 50 ns +0.1302,
90 ns +0.0582). Models are HistGradientBoosting (max_iter=120,
lr .075, max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42),
retrained per condition. Distances use the per-frame box, axis-wise
(already applied at extraction; the cache is reused as-is).

Why including the origin frame is legitimate: the origin frame (frame 10)
is the structure at the prediction time itself. Its distance is already
part of the static input (the static features include the origin distance),
so averaging it into the prefix mean does not add information from the
future; it only changes how the prediction-time structure is weighted
inside the mean. The 10-frame mean and the 11-frame mean differ only in
whether that one frame is included.

Frame note: the 10-frame "1.0 ns" prefix covers 0-0.9 ns; the 11-frame
mean covers 0-1 ns including the origin frame. Same counting as the
0.25 ns prefix, which actually covers 0.2 ns.

Pre-fixed reading: put the two mean-only recoveries side by side per
horizon and list the horizons where the difference is large. The base
gains are small at some horizons (30 ns +0.0633, 90 ns +0.0582), so the
absolute AUPRC differences are recorded too, and recovery values at
those horizons must be read with that caution.

Verification: static_only / full_hist AUPRC at each horizon must
reproduce the K53 record (1 ns 0.3734/0.5522, 5 ns 0.4641/0.5780,
10 ns 0.3572/0.5353, 30 ns 0.4799/0.5432, 50 ns 0.4863/0.6165,
90 ns 0.4824/0.5405), and the 10-frame mean recovery must reproduce
G142's record (1 ns 1.052, 5 ns 0.957, 10 ns 1.028, 30 ns 1.240,
50 ns 0.745, 90 ns 0.892).
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

OUT = Path(__file__).resolve().parent
K53 = Path("/home/hatch/workspace/kinescope/work/k53_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")

ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
PREFIXES = {  # name -> frame indices of the 11-frame 0-1 ns series
    "prefix_mean_10f": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "prefix_mean_11f": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
# G142's record for the 10-frame mean recovery (verified from its results.json)
G142_MEAN_RECOVERY = {1: 1.0515, 5: 0.9572, 10: 1.0278, 30: 1.2399, 50: 0.7450, 90: 0.8916}
K53_REF = {  # horizon_ns -> (static_only AUPRC, full_hist AUPRC) from K53 record
    1: (0.3734, 0.5522),
    5: (0.4641, 0.5780),
    10: (0.3572, 0.5353),
    30: (0.4799, 0.5432),
    50: (0.4863, 0.6165),
    90: (0.4824, 0.5405),
}


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


def load(cache, prots, suffix):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(cache / f"{p}_R{r}_{suffix}.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


split = json.loads((K15_DIR / "split.json").read_text())
train_p = ORIG_TRAIN + tuple(split["train_additions"])
test_p = ORIG_TEST + tuple(split["test_additions"])
tr = load(K53, train_p, "salt_diss")
te = load(K53, test_p, "salt_diss")
Xtr = np.concatenate([p["X"] for p in tr]).astype(np.float64)
Xte = np.concatenate([p["X"] for p in te]).astype(np.float64)
Htr = np.concatenate([p["hist"] for p in tr])
Hte = np.concatenate([p["hist"] for p in te])
Ytr = np.concatenate([p["y"] for p in tr])
Yte = np.concatenate([p["y"] for p in te])
print("salt-bridge dissociation train n:", len(Ytr), "test n:", len(Yte), flush=True)

results = {"train_n": int(len(Ytr)), "test_n": int(len(Yte)), "horizons": {}}
rows = []
for j, hns in enumerate(HORIZONS):
    ytr, yte = Ytr[:, j], Yte[:, j]
    conds = {
        "static_only": (Xtr, Xte),
        "full_hist": (np.column_stack([Xtr, hist6(Htr, 4.0)]),
                      np.column_stack([Xte, hist6(Hte, 4.0)])),
    }
    for name, idx in PREFIXES.items():
        conds[name] = (np.column_stack([Xtr, Htr[:, idx].mean(1)[:, None].astype(np.float32)]),
                       np.column_stack([Xte, Hte[:, idx].mean(1)[:, None].astype(np.float32)]))
    entry = {"horizon_ns": hns, "train_n": int(len(ytr)), "test_n": int(len(yte)),
             "test_positives": int(yte.sum())}
    for name, (xtr, xte) in conds.items():
        p = predict(xtr, ytr, xte)
        entry[name] = metrics(yte, p)
        print(f"{hns} ns {name}: AUPRC {entry[name]['auprc']:.4f} AUC {entry[name]['auc']:.4f}", flush=True)
    s, f = entry["static_only"]["auprc"], entry["full_hist"]["auprc"]
    entry["full_gain"] = f - s
    for name, idx in PREFIXES.items():
        pm = entry[name]["auprc"]
        entry[name + "_gain"] = pm - s
        entry[name + "_recovery"] = (pm - s) / (f - s) if f != s else None
        entry[name + "_residual_std_test"] = float(np.std(Hte[:, idx].mean(1) - Hte.mean(1)))
        print(f"{hns} ns {name}: gain {pm - s:+.4f} recovery {entry[name + '_recovery']:.3f}", flush=True)
    ref_s, ref_f = K53_REF[hns]
    entry["verify_static_diff"] = abs(s - ref_s)
    entry["verify_full_diff"] = abs(f - ref_f)
    print(f"{hns} ns verify vs K53: static diff {entry['verify_static_diff']:.6f} "
          f"full diff {entry['verify_full_diff']:.6f}", flush=True)
    results["horizons"][str(hns)] = entry
    for name in ["static_only", "full_hist"] + list(PREFIXES):
        m = entry[name]
        row = {"horizon_ns": hns, "condition": name, "n": m["n"], "positives": m["positives"],
               "auc": m["auc"], "auprc": m["auprc"], "brier": m["brier"], "logloss": m["logloss"],
               "gain": "" if name == "static_only" else (entry["full_gain"] if name == "full_hist"
                                                         else entry[name + "_gain"]),
               "recovery": "" if name in ("static_only", "full_hist") else entry[name + "_recovery"]}
        rows.append(row)

# origin-frame comparison (the pre-fixed reading)
comparison = {}
for hns in HORIZONS:
    e = results["horizons"][str(hns)]
    rec10 = e["prefix_mean_10f_recovery"]
    rec11 = e["prefix_mean_11f_recovery"]
    # verify the 10-frame mean reproduces G142
    assert abs(rec10 - G142_MEAN_RECOVERY[hns]) < 5e-3, (hns, rec10)
    comparison[str(hns)] = {
        "recovery_10f": rec10, "recovery_11f": rec11,
        "difference_11f_minus_10f": rec11 - rec10,
        "gain_10f": e["prefix_mean_10f_gain"], "gain_11f": e["prefix_mean_11f_gain"],
        "auprc_difference_11f_minus_10f": e["prefix_mean_11f"]["auprc"] - e["prefix_mean_10f"]["auprc"],
    }
    print(f"{hns} ns: 10f recovery {rec10:.3f} vs 11f {rec11:.3f} "
          f"(diff {rec11 - rec10:+.3f}, AUPRC diff "
          f"{e['prefix_mean_11f']['auprc'] - e['prefix_mean_10f']['auprc']:+.4f})", flush=True)
results["origin_frame_comparison"] = comparison

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with (OUT / "results.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["horizon_ns", "condition", "n", "positives",
                                       "auc", "auprc", "brier", "logloss", "gain", "recovery"])
    w.writeheader()
    w.writerows(rows)
print("saved to", OUT)
