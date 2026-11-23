"""G65: for contact formation under Doc D conditions, does a short
real-MD prefix restore guidance at long horizons too?

G03 measured the 0.1 ns prefix (frames [0,1]) mean only at 1 ns
(recovery 0.432). G07 scanned longer prefixes at 1 ns only (0.25 ns
mean 0.512, 0.5 ns mean 0.773). The long horizons (5/10/30/50/90 ns)
were blank. This experiment measures, for each horizon
5/10/30/50/90 ns, the recovery of the prefix mean only for prefixes
0.1 ns (frames [0,1]), 0.25 ns (frames [0,1,2]) and 0.5 ns
(frames [0..5]), reusing the K13 cache as-is.

Method (same as G02/G03/G07/G60): the condition is static features
+ the prefix mean only (no other history statistics). Recovery =
(prefix-mean-only AUPRC - static-only AUPRC) /
(full-history AUPRC - static-only AUPRC), where the reference gain
is the real-history gain for that same horizon (K16 record: 1 ns
+0.2271, 5 ns +0.2322, 10 ns +0.2030, 30 ns +0.1518, 50 ns +0.2034,
90 ns +0.1850). Models are HistGradientBoosting (max_iter=120,
lr .075, max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42),
retrained per condition. Distances use the per-frame box, axis-wise
(already applied at extraction; the cache is reused as-is).

Pre-fixed reading (same as G60): for each horizon, report the
smallest prefix whose recovery exceeds 0.5. If none of the three
prefixes exceeds 0.5, the horizon is recorded as "not reached even
at 0.5 ns".

Verification: static_only / full_hist AUPRC at each horizon must
reproduce the K16 record (1 ns 0.3398/0.5669, 5 ns 0.2870/0.5192,
10 ns 0.3204/0.5234, 30 ns 0.2868/0.4386, 50 ns 0.2604/0.4638,
90 ns 0.2872/0.4722), and the 1 ns prefix recoveries must reproduce
G03 (0.1 ns mean 0.432) and G07 (0.25 ns mean 0.512, 0.5 ns mean
0.773).
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
    "prefix_mean_0.1ns": [0, 1],
    "prefix_mean_0.25ns": [0, 1, 2],
    "prefix_mean_0.5ns": [0, 1, 2, 3, 4, 5],
}
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

# smallest prefix exceeding 0.5 per horizon (the pre-fixed reading)
summary = {}
for hns in (5, 10, 30, 50, 90):
    e = results["horizons"][str(hns)]
    smallest = None
    for name in ("prefix_mean_0.1ns", "prefix_mean_0.25ns", "prefix_mean_0.5ns"):
        if e[name + "_recovery"] > 0.5:
            smallest = name.replace("prefix_mean_", "")
            break
    summary[str(hns)] = smallest if smallest else "not reached even at 0.5 ns"
    print(f"{hns} ns: smallest prefix over 0.5 -> {summary[str(hns)]}", flush=True)
results["smallest_prefix_over_0.5"] = summary

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with (OUT / "results.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["horizon_ns", "condition", "n", "positives",
                                       "auc", "auprc", "brier", "logloss", "gain", "recovery"])
    w.writeheader()
    w.writerows(rows)
print("saved to", OUT)
