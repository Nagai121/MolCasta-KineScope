"""G140: for contact dissociation under Doc D conditions, the mean-only
recovery of the 1.0 ns prefix (frames [0..9]) at all horizons.

G71 measured the mean-only prefix recovery for 0.1 ns (frames [0,1]),
0.25 ns (frames [0,1,2]) and 0.5 ns (frames [0..5]) at every horizon.
The 1.0 ns prefix (frames [0..9], actually spanning 0.9 ns) was not
measured yet. This experiment measures it at horizons 1/5/10/30/50/90 ns,
reusing the K50 cache as-is.

Method (same as G02/G03/G06/G60/G65/G71): the condition is static
features + the prefix mean only (no other history statistics).
Recovery = (prefix-mean-only AUPRC - static-only AUPRC) /
(full-history AUPRC - static-only AUPRC), where the reference gain
is the real-history gain for that same horizon (K50 record: 1 ns
+0.2142, 5 ns +0.2506, 10 ns +0.2022, 30 ns +0.2368, 50 ns +0.1606,
90 ns +0.1864). Models are HistGradientBoosting (max_iter=120,
lr .075, max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42),
retrained per condition. Distances use the per-frame box, axis-wise
(already applied at extraction; the cache is reused as-is). The prefix
mean is cast to float32 before fitting, exactly as in G71 (G115 showed
that with float64 the 30 ns 0.5 ns prefix recovery is 0.784 instead of
G71's recorded 0.827).

Pre-fixed reading (same as G60/G65/G71): for each horizon, report the
smallest prefix whose recovery exceeds 0.5. G71 already showed that
0.1 ns clears 0.5 at every horizon, so the smallest prefix is already
known to be 0.1 ns; the new content here is the 1.0 ns recovery value
itself at each horizon.

Verification: static_only / full_hist AUPRC at each horizon must
reproduce the K50 record (1 ns 0.3105/0.5246, 5 ns 0.3022/0.5528,
10 ns 0.2772/0.4794, 30 ns 0.2756/0.5124, 50 ns 0.2927/0.4533,
90 ns 0.2886/0.4750). The 1.0 ns prefix recovery must also reproduce
the values already measured in the hybrid verify steps: 1 ns 0.915
(G131), 30 ns 0.840 (G131), 90 ns 0.917 (G134).
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

OUT = Path(__file__).resolve().parent
K50 = Path("/home/hatch/workspace/kinescope/work/k50_cache")

TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
PREFIXES = {  # name -> frame indices of the 11-frame 0-1 ns series
    # 1.0 ns の切り取り = フレーム [0..9]（実際は 0.9 ns の範囲を覆います）
    "prefix_mean_1.0ns": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}
# 切り取り 1.0 ns の平均だけの回復率として既に測られている値
# （G131・G134 の全体の候補での検証、float32 で計算）
PRIOR_1NS_RECOVERY = {1: 0.915, 30: 0.840, 90: 0.917}
K50_REF = {  # horizon_ns -> (static_only AUPRC, full_hist AUPRC) from K50 record
    1: (0.3105, 0.5246),
    5: (0.3022, 0.5528),
    10: (0.2772, 0.4794),
    30: (0.2756, 0.5124),
    50: (0.2927, 0.4533),
    90: (0.2886, 0.4750),
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
            z = np.load(K50 / f"{p}_R{r}_contact_diss.npz")
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
print("contact dissociation train n:", len(Ytr), "test n:", len(Yte), flush=True)

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
    ref_s, ref_f = K50_REF[hns]
    entry["verify_static_diff"] = abs(s - ref_s)
    entry["verify_full_diff"] = abs(f - ref_f)
    assert entry["verify_static_diff"] < 1e-4 and entry["verify_full_diff"] < 1e-4, \
        (hns, s, ref_s, f, ref_f)
    if hns in PRIOR_1NS_RECOVERY:
        prior = PRIOR_1NS_RECOVERY[hns]
        diff = abs(entry["prefix_mean_1.0ns_recovery"] - prior)
        print(f"{hns} ns verify vs G131/G134: 1.0ns recovery "
              f"{entry['prefix_mean_1.0ns_recovery']:.3f} (prior {prior}) diff {diff:.4f}",
              flush=True)
        assert diff < 0.005, (hns, entry["prefix_mean_1.0ns_recovery"], prior)
    print(f"{hns} ns verify vs K50: static diff {entry['verify_static_diff']:.6f} "
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

# 1.0 ns の回復率が 0.5 を超えるか（G71 で 0.1 ns が全予測幅で超えることは
# 既に分かっているため、最小の切り取りは 0.1 ns で変わらない）
summary = {}
for hns in (1, 5, 10, 30, 50, 90):
    e = results["horizons"][str(hns)]
    summary[str(hns)] = bool(e["prefix_mean_1.0ns_recovery"] > 0.5)
    print(f"{hns} ns: 1.0ns recovery {e['prefix_mean_1.0ns_recovery']:.3f} "
          f"over 0.5 -> {summary[str(hns)]}", flush=True)
results["prefix_1.0ns_over_0.5"] = summary

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with (OUT / "results.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["horizon_ns", "condition", "n", "positives",
                                       "auc", "auprc", "brier", "logloss", "gain", "recovery"])
    w.writeheader()
    w.writerows(rows)
print("saved to", OUT)
