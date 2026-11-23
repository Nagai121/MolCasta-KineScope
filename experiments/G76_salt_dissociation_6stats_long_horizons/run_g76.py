"""G76: for salt-bridge dissociation under the K15 expanded conditions,
does the 6-statistic prefix restore guidance at long horizons too?

G69 measured the prefix MEAN only at 5/10/30/50/90 ns (0.1 ns already
exceeds 0.5 at every horizon). This experiment measures the prefix
6 STATISTICS (first, mean, std, min, last-first, fraction <= 4.0 A,
computed on the prefix frames only) for prefixes 0.1 ns (frames
[0,1]), 0.25 ns (frames [0,1,2]) and 0.5 ns (frames [0..5]), reusing
the K53 cache and split as-is.

Note: G09 recorded the 0.1 ns and 0.5 ns 6-statistic recoveries at
all horizons as reference values (results.json, recovery_vs_full_hist6).
This run re-measures them (they must reproduce) and adds the 0.25 ns
prefix, per-protein recovery, and the pre-fixed reading.

Method: the condition is static features + the prefix 6 statistics.
Recovery = (prefix-6stats AUPRC - static-only AUPRC) /
(full-history AUPRC - static-only AUPRC), where the reference gain is
the real-history gain for that same horizon (K53 record: 1 ns +0.1788,
5 ns +0.1139, 10 ns +0.1781, 30 ns +0.0633, 50 ns +0.1302,
90 ns +0.0582). Models are HistGradientBoosting (max_iter=120,
lr .075, max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42),
retrained per condition. Distances use the per-frame box, axis-wise
(already applied at extraction; the cache is reused as-is).

Per-protein recovery: for each test protein, the AUPRC of each
condition is computed on that protein's candidates only (pooled-model
predictions, sliced per protein), and the recovery is formed with
that protein's own static/full AUPRCs. It is null when the protein
has no positives at that horizon or the protein's full gain is zero.
With ~12 candidates per protein the per-protein numbers are noisy;
they are recorded as a distribution, not a per-protein verdict.

Pre-fixed reading (same as G60/G69): for each horizon, report the
smallest prefix whose recovery exceeds 0.5. If none of the three
prefixes exceeds 0.5, the horizon is recorded as "not reached even
at 0.5 ns".

Verification: static_only / full_hist AUPRC at each horizon must
reproduce the K53 record (1 ns 0.3734/0.5522, 5 ns 0.4641/0.5780,
10 ns 0.3572/0.5353, 30 ns 0.4799/0.5432, 50 ns 0.4863/0.6165,
90 ns 0.4824/0.5405), and the 1 ns 6-statistic recoveries must
reproduce G09 (0.1 ns 0.566, 0.5 ns 0.985).
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
    "prefix_hist6_0.1ns": [0, 1],
    "prefix_hist6_0.25ns": [0, 1, 2],
    "prefix_hist6_0.5ns": [0, 1, 2, 3, 4, 5],
}
K53_REF = {  # horizon_ns -> (static_only AUPRC, full_hist AUPRC) from K53 record
    1: (0.3734, 0.5522),
    5: (0.4641, 0.5780),
    10: (0.3572, 0.5353),
    30: (0.4799, 0.5432),
    50: (0.4863, 0.6165),
    90: (0.4824, 0.5405),
}
G09_REF = {  # 1 ns 6-statistic recoveries from G09 (verification)
    "prefix_hist6_0.1ns": 0.566,
    "prefix_hist6_0.5ns": 0.985,
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


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K53 / f"{p}_R{r}_salt_diss.npz")
            parts.append({k: z[k] for k in z.files})
            proteins.append(np.full(len(z["res_i"]), p))
    return parts, np.concatenate(proteins)


split = json.loads((K15_DIR / "split.json").read_text())
train_p = ORIG_TRAIN + tuple(split["train_additions"])
test_p = ORIG_TEST + tuple(split["test_additions"])
tr, _ = load(train_p)
te, te_prot = load(test_p)
Xtr = np.concatenate([p["X"] for p in tr]).astype(np.float64)
Xte = np.concatenate([p["X"] for p in te]).astype(np.float64)
Htr = np.concatenate([p["hist"] for p in tr])
Hte = np.concatenate([p["hist"] for p in te])
Ytr = np.concatenate([p["y"] for p in tr])
Yte = np.concatenate([p["y"] for p in te])
print("salt-bridge dissociation train n:", len(Ytr), "test n:", len(Yte), flush=True)

results = {"train_n": int(len(Ytr)), "test_n": int(len(Yte)), "horizons": {}, "per_protein": {}}
rows = []
per_protein_rows = []
for j, hns in enumerate(HORIZONS):
    ytr, yte = Ytr[:, j], Yte[:, j]
    conds = {
        "static_only": (Xtr, Xte),
        "full_hist": (np.column_stack([Xtr, hist6(Htr, 4.0)]),
                      np.column_stack([Xte, hist6(Hte, 4.0)])),
    }
    for name, idx in PREFIXES.items():
        conds[name] = (np.column_stack([Xtr, hist6(Htr[:, idx], 4.0)]),
                       np.column_stack([Xte, hist6(Hte[:, idx], 4.0)]))
    entry = {"horizon_ns": hns, "train_n": int(len(ytr)), "test_n": int(len(yte)),
             "test_positives": int(yte.sum())}
    preds = {}
    for name, (xtr, xte) in conds.items():
        p = predict(xtr, ytr, xte)
        preds[name] = p
        entry[name] = metrics(yte, p)
        print(f"{hns} ns {name}: AUPRC {entry[name]['auprc']:.4f} AUC {entry[name]['auc']:.4f}", flush=True)
    s, f = entry["static_only"]["auprc"], entry["full_hist"]["auprc"]
    entry["full_gain"] = f - s
    for name in PREFIXES:
        pm = entry[name]["auprc"]
        entry[name + "_gain"] = pm - s
        entry[name + "_recovery"] = (pm - s) / (f - s) if f != s else None
        print(f"{hns} ns {name}: gain {pm - s:+.4f} recovery {entry[name + '_recovery']:.3f}", flush=True)
    ref_s, ref_f = K53_REF[hns]
    entry["verify_static_diff"] = abs(s - ref_s)
    entry["verify_full_diff"] = abs(f - ref_f)
    print(f"{hns} ns verify vs K53: static diff {entry['verify_static_diff']:.6f} "
          f"full diff {entry['verify_full_diff']:.6f}", flush=True)
    results["horizons"][str(hns)] = entry
    for name in ["static_only", "full_hist"] + list(PREFIXES):
        m = entry[name]
        rows.append({"horizon_ns": hns, "condition": name, "n": m["n"], "positives": m["positives"],
                     "auc": m["auc"], "auprc": m["auprc"], "brier": m["brier"], "logloss": m["logloss"],
                     "gain": "" if name == "static_only" else (entry["full_gain"] if name == "full_hist"
                                                               else entry[name + "_gain"]),
                     "recovery": "" if name in ("static_only", "full_hist") else entry[name + "_recovery"]})
    # per-protein recovery on the test proteins (pooled-model predictions, sliced)
    for prot in sorted(set(te_prot.tolist())):
        sel = te_prot == prot
        yy = yte[sel]
        if int(yy.sum()) == 0:
            rec = {name: None for name in PREFIXES}
            aucs = {name: None for name in conds}
        else:
            aucs = {}
            for name in conds:
                pp = np.clip(preds[name][sel], 1e-8, 1 - 1e-8)
                aucs[name] = float(average_precision_score(yy, pp))
            fg = aucs["full_hist"] - aucs["static_only"]
            rec = {name: ((aucs[name] - aucs["static_only"]) / fg if fg != 0 else None)
                   for name in PREFIXES}
        results["per_protein"].setdefault(prot, {})[str(hns)] = {
            "n": int(sel.sum()), "positives": int(yy.sum()),
            "auprc": {name: aucs[name] for name in conds},
            "recovery": rec}
        per_protein_rows.append({"protein": prot, "horizon_ns": hns, "n": int(sel.sum()),
                                 "positives": int(yy.sum()),
                                 **{f"auprc_{name}": aucs[name] for name in conds},
                                 **{f"recovery_{name}": rec[name] for name in PREFIXES}})

# 1 ns check against G09's 6-statistic recoveries
e1 = results["horizons"]["1"]
for name, ref in G09_REF.items():
    print(f"1 ns verify vs G09 {name}: {e1[name + '_recovery']:.3f} (recorded {ref})", flush=True)
    assert abs(e1[name + "_recovery"] - ref) < 0.005

# smallest prefix exceeding 0.5 per horizon (the pre-fixed reading)
summary = {}
for hns in (5, 10, 30, 50, 90):
    e = results["horizons"][str(hns)]
    smallest = None
    for name in ("prefix_hist6_0.1ns", "prefix_hist6_0.25ns", "prefix_hist6_0.5ns"):
        if e[name + "_recovery"] > 0.5:
            smallest = name.replace("prefix_hist6_", "")
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
with (OUT / "per_protein.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(per_protein_rows[0].keys()))
    w.writeheader()
    w.writerows(per_protein_rows)
print("saved to", OUT)
