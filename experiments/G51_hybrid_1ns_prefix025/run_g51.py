"""G51: at 1 ns, does mixing help with the in-between 0.25 ns prefix?
(Doc D, hydrogen-bond formation, 1 ns)

G11 found at 1 ns that mixing the 0.1 ns prefix mean with the
proposal-22 generated mean adds +0.186, and G24 found that the
0.5 ns prefix mixture adds +0.245. This experiment measures
the in-between 0.25 ns prefix at the same horizon to see whether
the 1 ns benefit survives at an intermediate prefix length.

Conditions (per horizon, all retrained per condition):
  - static_only: 7 static features only
  - full_hist: static + real history 6 statistics (0-1 ns, 11 frames; reference)
  - prefix_mean_0.25ns: static + prefix mean only (frames [0,1,2]; G02)
  - gen_mean_only: static + generated mean only (proposal 22, K05 gen5 mean)
  - mixed: static + prefix mean + generated mean (2 columns)

Recovery = (condition AUPRC - static-only AUPRC) / (real-history
gain at that horizon). Model: HistGradientBoosting (max_iter=120,
lr .075, max_leaf_nodes 31, min_samples_leaf 15, L2 .02, seed 42).

Pre-fixed reading: if the mixed recovery rises +0.05 or more above
the prefix-mean-only recovery, mixing helps at 1 ns with the 0.25 ns
prefix. If below +0.05, mixing does not help at 1 ns with the
0.25 ns prefix.

Generated mean source: K05's generated_series.npz (proposal-22
5-point series). The 5 points are [origin distance d0 (frame 10),
+0.1, +0.3, +0.5, +1.0 ns] forward from the origin. Alignment key is
(tid, donor atom, acceptor atom), matching all 14,582 weak-base
candidates 100% (same as G11/G14/K09). The generator checkpoint .pt
file is not loaded. Note (same caveat as G11/G14/K09): the generated
series runs forward from the origin while the real history runs
backward to the origin; they share d0 but run in opposite time
directions. The mean does not depend on direction, so this does not
affect the mean column used here.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = {0: 1}  # horizon index -> ns
PREFIX_FRAMES = [0, 1, 2]  # 0.25 ns の切り取り（G02 と同じ）


def load_gen5():
    z = np.load(GEN)
    tids, pair, gen = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
    assert gen5.shape == (14582, 5), gen5.shape
    return {(str(t), int(pair[i, 0]), int(pair[i, 1])): gen5[i]
            for i, t in enumerate(tids)}


GEN5 = load_gen5()


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            g = np.stack([GEN5[(tid, int(da), int(aa))]
                          for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["gen5"] = g
            parts.append(d)
    return parts


def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)


def hist6(series, bound=3.5):
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
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


train = load(TRAIN_P)
test = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Gtr, Gte = cat(train, "gen5"), cat(test, "gen5")
Ytr, Yte = cat(train, "y"), cat(test, "y")

# alignment check (same as G11/G14): gen5[0] vs origin distance (last history frame)
print("alignment max |gen5[0] - hist[:,-1]| train",
      float(np.abs(Gtr[:, 0] - Htr[:, -1]).max()),
      "test", float(np.abs(Gte[:, 0] - Hte[:, -1]).max()), flush=True)

prefix_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_te = Hte[:, PREFIX_FRAMES].mean(1)
gen_mean_tr = Gtr.mean(1)
gen_mean_te = Gte.mean(1)

results = {"label": "HB formation 1ns, 0.25ns prefix + generated mean",
           "train_n": int(len(Xtr)), "test_n": int(len(Xte)),
           "horizons": {}}
for hidx, hns in HORIZONS.items():
    ytr, yte = Ytr[:, hidx], Yte[:, hidx]
    conds = {
        "static_only": (Xtr, Xte),
        "full_hist": (np.column_stack([Xtr, hist6(Htr)]), np.column_stack([Xte, hist6(Hte)])),
        "prefix_mean_0.25ns": (np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32)]),
                              np.column_stack([Xte, prefix_te[:, None].astype(np.float32)])),
        "gen_mean_only": (np.column_stack([Xtr, gen_mean_tr[:, None].astype(np.float32)]),
                          np.column_stack([Xte, gen_mean_te[:, None].astype(np.float32)])),
        "mixed": (np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32),
                                   gen_mean_tr[:, None].astype(np.float32)]),
                  np.column_stack([Xte, prefix_te[:, None].astype(np.float32),
                                   gen_mean_te[:, None].astype(np.float32)])),
    }
    hres = {"test_positives": int(yte.sum())}
    for name, (xtr, xte) in conds.items():
        p = predict(xtr, ytr, xte)
        hres[name] = metrics(yte, p)
        print(f"{hns}ns {name}: AUPRC {hres[name]['auprc']:.4f} AUC {hres[name]['auc']:.4f}", flush=True)
    s = hres["static_only"]["auprc"]
    f = hres["full_hist"]["auprc"]
    hres["full_gain"] = f - s
    for name in conds:
        if name in ("static_only", "full_hist"):
            continue
        hres[name]["gain"] = hres[name]["auprc"] - s
        hres[name]["recovery"] = (hres[name]["auprc"] - s) / (f - s)
        print(f"{hns}ns {name}: gain {hres[name]['gain']:.4f} recovery {hres[name]['recovery']:.3f}", flush=True)
    hres["mixed_minus_prefix"] = hres["mixed"]["recovery"] - hres["prefix_mean_0.25ns"]["recovery"]
    print(f"{hns}ns mixed - prefix recovery: {hres['mixed_minus_prefix']:+.3f}", flush=True)
    results["horizons"][str(hns)] = hres

(OUT / "results.json").write_text(json.dumps(results, indent=2))
print("saved to", OUT)
