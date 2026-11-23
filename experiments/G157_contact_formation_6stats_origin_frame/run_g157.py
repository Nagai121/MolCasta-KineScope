"""G157: for contact formation under Doc D conditions, does including
the origin frame change the 6-statistic prefix recovery?

G123/G128 measured the prefix hist6 over frames [0,1,2,3,4,5,6,7,8,9]
(10 frames, covering 0-0.9 ns): recovery 1.006/1.015/0.917/1.129/0.862/0.885
(horizons 1/5/10/30/50/90 ns). G155 (hydrogen-bond formation) found that
including the origin frame changed the 6-statistic recovery a lot at
some horizons (30 ns -0.403, 10 ns +0.286). This experiment measures,
for the same six horizons, the prefix hist6 over all 11 frames [0..10]
(0-1 ns, including the origin frame 10) and compares it with the
10-frame hist6, reusing the K13 cache as-is.

Note (same as G155): the 11-frame hist6 is the full-history condition
itself (hist6 over all 11 frames), so its recovery is 1.0 by
construction. The comparison therefore asks how far the 10-frame hist6
sits from 1.0 per horizon, and whether that gap is large. The 11-frame
condition is still computed and verified to be identical to full_hist
(AUPRC difference 0 within rounding), rather than assumed.

Method (same as G02/G03/G07/G75/G123/G128): the condition is static
features + the prefix hist6 (no other history statistics). Recovery =
(prefix-hist6 AUPRC - static-only AUPRC) /
(full-history AUPRC - static-only AUPRC), where the reference gain is
the real-history gain for that same horizon (K16 record: 1 ns
+0.2271, 5 ns +0.2322, 10 ns +0.2030, 30 ns +0.1518, 50 ns +0.2034,
90 ns +0.1850). Models are HistGradientBoosting (max_iter=120,
lr .075, max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42),
retrained per condition. Distances use the per-frame box, axis-wise
(already applied at extraction; the cache is reused as-is).

Why including the origin frame is legitimate: the origin frame (frame 10)
is the structure at the prediction time itself. Its distance is already
part of the static input (the static features include the origin distance),
so averaging it into the prefix statistics does not add information from
the future; it only changes how the prediction-time structure is weighted
inside the statistics. The 10-frame hist6 and the 11-frame hist6 differ
only in whether that one frame is included.

Frame note: the "1.0 ns" prefix is frames [0,1,2,3,4,5,6,7,8,9]
(10 frames, covering 0-0.9 ns). It is actually 0.9 ns of coverage,
not 1.0 ns; the origin frame 10 (1 ns) is not included. Same counting
as the 0.25 ns prefix, which actually covers 0.2 ns.

Pre-fixed reading: put the two hist6 recoveries side by side per
horizon and list the horizons where the difference is large.

Verification: static_only / full_hist AUPRC at each horizon must
reproduce the K16 record (1 ns 0.3398/0.5669, 5 ns 0.2870/0.5192,
10 ns 0.3204/0.5234, 30 ns 0.2868/0.4386, 50 ns 0.2604/0.4638,
90 ns 0.2872/0.4722), and the 10-frame hist6 recovery must reproduce
the G123/G128 record (1 ns 1.006, 5 ns 1.015, 10 ns 0.917,
30 ns 1.129, 50 ns 0.862, 90 ns 0.885).
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

OUT = Path(__file__).resolve().parent
K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")

TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
PREFIXES = {  # name -> frame indices of the 11-frame 0-1 ns series
    "prefix_hist6_10f": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "prefix_hist6_11f": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}
# G123/G128 record for the 10-frame hist6 recovery (verified from their results.json)
G123_G128_HIST6_RECOVERY = {1: 1.0059, 5: 1.0145, 10: 0.9174, 30: 1.1285, 50: 0.8623, 90: 0.8852}
K16_REF = {  # horizon_ns -> (static_only AUPRC, full_hist AUPRC) from K16 record
    1: (0.3398, 0.5669),
    5: (0.2870, 0.5192),
    10: (0.3204, 0.5234),
    30: (0.2868, 0.4386),
    50: (0.2604, 0.4638),
    90: (0.2872, 0.4722),
}


def hist6(series, bound=4.5):
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
print("contact train n:", len(Ytr), "test n:", len(Yte), flush=True)

results = {"train_n": int(len(Ytr)), "test_n": int(len(Yte)), "horizons": {}}
rows = []
for j, hns in enumerate(HORIZONS):
    ytr, yte = Ytr[:, j], Yte[:, j]
    conds = {
        "static_only": (Xtr, Xte),
        "full_hist": (np.column_stack([Xtr, hist6(Htr)]),
                      np.column_stack([Xte, hist6(Hte)])),
    }
    for name, idx in PREFIXES.items():
        conds[name] = (np.column_stack([Xtr, hist6(Htr[:, idx])]),
                       np.column_stack([Xte, hist6(Hte[:, idx])]))
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
    ref_s, ref_f = K16_REF[hns]
    entry["verify_static_diff"] = abs(s - ref_s)
    entry["verify_full_diff"] = abs(f - ref_f)
    print(f"{hns} ns verify vs K16: static diff {entry['verify_static_diff']:.6f} "
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
    rec10 = e["prefix_hist6_10f_recovery"]
    rec11 = e["prefix_hist6_11f_recovery"]
    # verify the 10-frame hist6 reproduces G123/G128
    assert abs(rec10 - G123_G128_HIST6_RECOVERY[hns]) < 5e-3, (hns, rec10)
    # verify the 11-frame hist6 is the full-history condition itself
    assert abs(e["prefix_hist6_11f"]["auprc"] - e["full_hist"]["auprc"]) < 5e-4, (hns,)
    assert abs(rec11 - 1.0) < 5e-4, (hns, rec11)
    comparison[str(hns)] = {
        "recovery_10f": rec10, "recovery_11f": rec11,
        "difference_11f_minus_10f": rec11 - rec10,
        "gain_10f": e["prefix_hist6_10f_gain"], "gain_11f": e["prefix_hist6_11f_gain"],
        "auprc_difference_11f_minus_10f": e["prefix_hist6_11f"]["auprc"] - e["prefix_hist6_10f"]["auprc"],
    }
    print(f"{hns} ns: 10f recovery {rec10:.3f} vs 11f {rec11:.3f} "
          f"(diff {rec11 - rec10:+.3f}, AUPRC diff "
          f"{e['prefix_hist6_11f']['auprc'] - e['prefix_hist6_10f']['auprc']:+.4f})", flush=True)
results["origin_frame_comparison"] = comparison

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with (OUT / "results.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["horizon_ns", "condition", "n", "positives",
                                       "auc", "auprc", "brier", "logloss", "gain", "recovery"])
    w.writeheader()
    w.writerows(rows)
print("saved to", OUT)
