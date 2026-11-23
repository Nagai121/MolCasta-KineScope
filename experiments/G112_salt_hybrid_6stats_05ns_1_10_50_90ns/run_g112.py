"""G112: salt-bridge formation mixing with the 0.5 ns prefix as 6
statistics, at the remaining horizons 1/10/50/90 ns (K15 expanded
conditions). G108 measured 5 and 30 ns only.

The 0.5 ns prefix 6 statistics (bound 4.0 A) plus G19's rebuilt
proposals 11-14 generated means, as separate features. Conditions
(static features + ...), retrained per condition and horizon:

- static_only
- full_hist6 (static + real 6 statistics of the 0-1 ns history)
- prefix_hist6_0.5ns (static + 0.5 ns prefix 6 statistics)
- gen_mean_only_p{11..14} (static + generated mean only)
- mixed_p{11..14} (static + 0.1 ns prefix 6 statistics + generated
  mean, as separate features = 7 extra columns)

Generated mean: G19's gen/<protein>_R<replica>.npz, proposals 11-14
(4 branches x 9 points), mean over branches and points, same
definition as G19/G82. Residue-pair alignment with the K15 cache is
asserted (res_i/res_j must match exactly).

Recovery = (condition AUPRC - static-only AUPRC) / (real-history
gain). Base gains from K15 (G60 record): 1 ns +0.1440, 10 ns +0.0594,
50 ns +0.0274, 90 ns +0.0601. Verification: static/full AUPRC at each
horizon must reproduce K15 (1 ns 0.2402/0.3842, 10 ns 0.2034/0.2627,
50 ns 0.2295/0.2569, 90 ns 0.2136/0.2736), and the 0.5 ns prefix
6-statistic recovery must reproduce G66 at 10/50/90 ns (1.298, 1.736,
1.209). G66 did not measure 1 ns, so the 1 ns 0.5 ns prefix recovery
is a new measurement with no prior record to check against.

Pre-fixed reading (same as G82): mixed recovery minus the 0.1 ns
prefix-6-statistics-only recovery > +0.05 means the generated mean
helps on top of the 6-statistic prefix. The base gains are small,
so the absolute AUPRC difference (mixed - prefix-only) is recorded
too.

Model: HistGradientBoosting (max_iter=120, lr .075, max_leaf_nodes 31,
min_samples_leaf 15, L2 .02, seed 42). Distances use the per-frame
box, axis-wise (already applied at extraction; the cache is reused
as-is).
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).resolve().parent
K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
GEN_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/G19_salt_prefix_plus_generated/gen")
BOUND = 4.0
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
HORIZONS = {1: 0, 10: 2, 50: 4, 90: 5}  # horizon_ns -> y column index
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5]  # 0.5 ns (covers 0.5 ns: frames 0..5 at 0.1 ns spacing)
PROPOSALS = ["p11", "p12", "p13", "p14"]
# horizon_ns -> (static AUPRC, full AUPRC) from K15/G60 record
K15_REF = {1: (0.2402, 0.3842), 10: (0.2034, 0.2627),
          50: (0.2295, 0.2569), 90: (0.2136, 0.2736)}
# horizon_ns -> 0.1 ns prefix 6-statistic recovery from G66
G66_REF = {10: 1.298, 50: 1.736, 90: 1.209}  # 0.5 ns prefix 6-stat recovery (G66; no 1 ns record exists)


def hist6(series, bound=BOUND):
    return np.stack(
        [series[:, 0], series.mean(1), series.std(1), series.min(1),
         series[:, -1] - series[:, 0], (series <= bound).mean(1)],
        axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def load(prots):
    real, gen = [], {tag: [] for tag in PROPOSALS}
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15 / f"{p}_R{r}_salt.npz")
            d = {k: z[k] for k in z.files}
            g = np.load(GEN_DIR / f"{p}_R{r}.npz")
            assert np.array_equal(g["res_i"], d["res_i"]) and np.array_equal(g["res_j"], d["res_j"]), \
                f"residue pair mismatch {p} R{r}"
            real.append(d)
            for j, tag in enumerate(PROPOSALS):
                gen[tag].append(g[f"gen{11 + j}"].mean(axis=0).mean(axis=0).astype(np.float64))
    return real, gen


def main():
    split = json.loads((K15_DIR / "split.json").read_text())
    train_p = ORIG_TRAIN + tuple(split["train_additions"])
    test_p = ORIG_TEST + tuple(split["test_additions"])
    train, gtr = load(train_p)
    test, gte = load(test_p)
    Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
    Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
    Htr = np.concatenate([d["hist"] for d in train]).astype(np.float64)
    Hte = np.concatenate([d["hist"] for d in test]).astype(np.float64)
    Ytr = np.concatenate([d["y"] for d in train])
    Yte = np.concatenate([d["y"] for d in test])
    GMtr = {tag: np.concatenate(gtr[tag]) for tag in PROPOSALS}
    GMte = {tag: np.concatenate(gte[tag]) for tag in PROPOSALS}
    print("train", Xtr.shape, "test", Xte.shape, flush=True)

    prefix_mean_tr = Htr[:, PREFIX_FRAMES].mean(1)
    prefix_mean_te = Hte[:, PREFIX_FRAMES].mean(1)
    full_mean_tr = Htr.mean(1)
    full_mean_te = Hte.mean(1)

    results = {"horizons": {}}
    rows = []
    for hns, hi in HORIZONS.items():
        ytr, yte = Ytr[:, hi], Yte[:, hi]
        relations = {}
        for tag in PROPOSALS:
            gm_tr, gm_te = GMtr[tag], GMte[tag]
            relations[tag] = dict(
                corr_prefix_gen=float(np.corrcoef(prefix_mean_te, gm_te)[0, 1]),
                corr_gen_full=float(np.corrcoef(gm_te, full_mean_te)[0, 1]),
                corr_prefix_full=float(np.corrcoef(prefix_mean_te, full_mean_te)[0, 1]),
                gen_minus_prefix_mean=float(np.mean(gm_te - prefix_mean_te)),
                gen_minus_prefix_std=float(np.std(gm_te - prefix_mean_te)),
                gen_minus_full_mean=float(np.mean(gm_te - full_mean_te)),
                gen_minus_full_std=float(np.std(gm_te - full_mean_te)))
        conds = {
            "static_only": (Xtr, Xte),
            "full_hist6": (np.column_stack([Xtr, hist6(Htr)]),
                           np.column_stack([Xte, hist6(Hte)])),
            "prefix_hist6_0.5ns": (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
                                   np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])])),
        }
        for tag in PROPOSALS:
            conds[f"gen_mean_only_{tag}"] = (
                np.column_stack([Xtr, GMtr[tag][:, None].astype(np.float32)]),
                np.column_stack([Xte, GMte[tag][:, None].astype(np.float32)]))
            conds[f"mixed_{tag}"] = (
                np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES]),
                                 GMtr[tag][:, None].astype(np.float32)]),
                np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES]),
                                 GMte[tag][:, None].astype(np.float32)]))
        entry = {"horizon_ns": hns, "metrics": {}}
        for cname, (xtr, xte) in conds.items():
            p = predict(xtr, ytr, xte)
            entry["metrics"][cname] = metrics(yte, p)
            print(f"{hns} ns {cname} auprc {entry['metrics'][cname]['auprc']:.4f}", flush=True)
            rows.append({"horizon_ns": hns, "condition": cname, **entry["metrics"][cname]})
        ref_s, ref_f = K15_REF[hns]
        assert abs(entry["metrics"]["static_only"]["auprc"] - ref_s) < 5e-4
        assert abs(entry["metrics"]["full_hist6"]["auprc"] - ref_f) < 5e-4
        gain = entry["metrics"]["full_hist6"]["auprc"] - entry["metrics"]["static_only"]["auprc"]
        entry["base_gain"] = float(gain)
        recovery = {n: float((m["auprc"] - entry["metrics"]["static_only"]["auprc"]) / gain)
                    for n, m in entry["metrics"].items() if n != "static_only"}
        entry["recovery_vs_real_history_gain"] = recovery
        prefix_rec = recovery["prefix_hist6_0.5ns"]
        if hns in G66_REF:
            print(f"{hns} ns verify vs G66 prefix_hist6_0.5ns recovery: {prefix_rec:.3f} "
                  f"(recorded {G66_REF[hns]})", flush=True)
            assert abs(prefix_rec - G66_REF[hns]) < 0.005
        else:
            print(f"{hns} ns prefix_hist6_0.5ns recovery {prefix_rec:.3f} "
                  "(no prior G66 record; new measurement)", flush=True)
        entry["mixed_minus_prefix_hist6"] = {
            tag: float(recovery[f"mixed_{tag}"] - prefix_rec) for tag in PROPOSALS}
        entry["mixed_minus_prefix_hist6_auprc"] = {
            tag: float(entry["metrics"][f"mixed_{tag}"]["auprc"]
                       - entry["metrics"]["prefix_hist6_0.5ns"]["auprc"]) for tag in PROPOSALS}
        entry["relations_test"] = relations
        print(f"{hns} ns base gain {gain:.4f}, prefix_hist6 recovery {prefix_rec:.3f}, "
              f"mixed deltas { {t: round(v, 3) for t, v in entry['mixed_minus_prefix_hist6'].items()} }, "
              f"mixed AUPRC deltas { {t: round(v, 4) for t, v in entry['mixed_minus_prefix_hist6_auprc'].items()} }",
              flush=True)
        results["horizons"][str(hns)] = entry

    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    with (HERE / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["horizon_ns", "condition", "n", "positives",
                                           "auc", "auprc", "brier", "logloss"])
        w.writeheader()
        w.writerows(rows)
    print("saved to", HERE)


if __name__ == "__main__":
    main()
