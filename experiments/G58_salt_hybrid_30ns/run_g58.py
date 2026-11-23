"""G58: for salt-bridge formation, does mixing the generated mean into
the 0.1 ns prefix mean help at 30 ns? (K15 expanded conditions, 30 ns)

G19 found at 1 ns that all four proposals clear the +0.05 line
(mixed - prefix: +0.190, +0.212, +0.108, +0.231), and G53 found at
5 ns that three proposals clear it (+0.215, +0.253, -0.073, +0.057).
This experiment measures the same conditions at 30 ns. The generated series
(proposals 11-14) are the ones rebuilt for G19 (g19_build.py in
experiments/G19_salt_prefix_plus_generated, K17's protocol,
salt-bridge candidates and charged-atom distances); the generated
mean of a proposal is the branch-mean 9-point series averaged
over the 9 points (the same reduction as G13/G19).

Real side: the K15 cache (static 5, 11-frame history, labels), joined
to the generated series per trajectory file after asserting the
residue pairs match. Conditions (1 ns, HistGradientBoosting 120 iters,
lr .075, 31 leaves, min_samples_leaf 15, L2 .02, seed 42, retrained
per condition):
  static_only, full_hist (static + real hist6, bound 4.0 A),
  prefix_mean_only (static + 0.1 ns prefix mean, frames [0,1]),
  gen_mean_only_<n> (static + generated mean of proposal n),
  mixed_<n> (static + prefix mean + generated mean of proposal n),
for n in 11..14. Recovery = (AUPRC - static) / (full - static); the
full-history gain at 30 ns is +0.088 (0.2466 - 0.1582, K15) - it is
re-measured here on the same base.

Pre-fixed reading: if some proposal's mixed recovery rises >= +0.05
above the prefix-mean-only recovery at 30 ns, mixing helps for
salt-bridge formation at 30 ns too; if all proposals are < +0.05,
mixing for salt-bridge formation is limited to short horizons.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K15_CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
GEN_DIR = Path(__file__).resolve().parent.parent / "G19_salt_prefix_plus_generated" / "gen"  # rebuilt for G19
OUT = Path(__file__).resolve().parent
SPLIT = json.loads((Path(__file__).resolve().parent.parent
                    / "K15_saltbridge_expansion" / "split.json").read_text())
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A") + tuple(SPLIT["train_additions"])
TEST_P = ("1mix_A", "2qt4_A") + tuple(SPLIT["test_additions"])
BOUND = 4.0
PROPOSALS = (11, 12, 13, 14)


def hist6(series, bound):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02,
                                       random_state=42)
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
    real, gen = [], {n: [] for n in PROPOSALS}
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15_CACHE / f"{p}_R{r}_salt.npz")
            d = {k: z[k] for k in z.files}
            g = np.load(GEN_DIR / f"{p}_R{r}.npz")
            assert np.array_equal(g["res_i"], d["res_i"]) and np.array_equal(g["res_j"], d["res_j"]), \
                f"residue pair mismatch {p} R{r}"
            real.append(d)
            for n in PROPOSALS:
                # branch-mean 9-point series -> mean over the 9 points (G13)
                gen[n].append(g[f"gen{n}"].mean(axis=0).mean(axis=0).astype(np.float64))
    return real, gen


train, gtr = load(TRAIN_P)
test, gte = load(TEST_P)


def cat(parts, key):
    return np.concatenate([d[key] for d in parts])


Xtr = cat(train, "X").astype(np.float64); Xte = cat(test, "X").astype(np.float64)
Htr = cat(train, "hist").astype(np.float64); Hte = cat(test, "hist").astype(np.float64)
ytr = cat(train, "y")[:, 3]; yte = cat(test, "y")[:, 3]  # 30 ns = horizon index 3
GMtr = {n: np.concatenate(gtr[n]) for n in PROPOSALS}
GMte = {n: np.concatenate(gte[n]) for n in PROPOSALS}
print("train", Xtr.shape, "test", Xte.shape, "test pos 30ns", int(yte.sum()), flush=True)

prefix_tr = Htr[:, [0, 1]].mean(1); prefix_te = Hte[:, [0, 1]].mean(1)
full_mean_tr = Htr.mean(1); full_mean_te = Hte.mean(1)

relations = {}
for n in PROPOSALS:
    gm = GMte[n]
    relations[str(n)] = dict(
        corr_prefix_gen=float(np.corrcoef(prefix_te, gm)[0, 1]),
        corr_gen_full=float(np.corrcoef(gm, full_mean_te)[0, 1]),
        corr_prefix_full=float(np.corrcoef(prefix_te, full_mean_te)[0, 1]),
        gen_minus_prefix_mean=float(np.mean(gm - prefix_te)),
        gen_minus_prefix_std=float(np.std(gm - prefix_te)),
        gen_minus_full_mean=float(np.mean(gm - full_mean_te)),
        gen_minus_full_std=float(np.std(gm - full_mean_te)),
    )
    print(f"relations p{n} (test):", {k: round(v, 4) for k, v in relations[str(n)].items()},
          flush=True)

conds = {
    "static_only": (Xtr, Xte),
    "full_hist": (np.column_stack([Xtr, hist6(Htr, BOUND)]),
                  np.column_stack([Xte, hist6(Hte, BOUND)])),
    "prefix_mean_only": (np.column_stack([Xtr, prefix_tr[:, None]]),
                         np.column_stack([Xte, prefix_te[:, None]])),
}
for n in PROPOSALS:
    conds[f"gen_mean_only_{n}"] = (np.column_stack([Xtr, GMtr[n][:, None]]),
                                   np.column_stack([Xte, GMte[n][:, None]]))
    conds[f"mixed_{n}"] = (np.column_stack([Xtr, prefix_tr[:, None], GMtr[n][:, None]]),
                           np.column_stack([Xte, prefix_te[:, None], GMte[n][:, None]]))

res = {}
for name, (xtr, xte) in conds.items():
    res[name] = metrics(yte, predict(xtr, ytr, xte))
    print(f"{name}: AUPRC {res[name]['auprc']:.4f} AUC {res[name]['auc']:.4f}", flush=True)

s = res["static_only"]["auprc"]; f = res["full_hist"]["auprc"]
for name in conds:
    if name in ("static_only", "full_hist"):
        continue
    res[name]["gain"] = res[name]["auprc"] - s
    res[name]["recovery"] = (res[name]["auprc"] - s) / (f - s)
    print(f"{name}: gain {res[name]['gain']:.4f} recovery {res[name]['recovery']:.3f}", flush=True)
mixed_diff = {str(n): res[f"mixed_{n}"]["recovery"] - res["prefix_mean_only"]["recovery"]
              for n in PROPOSALS}
print("mixed - prefix recovery:", {k: round(v, 3) for k, v in mixed_diff.items()}, flush=True)

# base check against the recorded values (K15, 30 ns)
print("base check: static %.4f (recorded 0.1582), full %.4f (recorded 0.2466), "
      "gain %.4f (recorded +0.088)" % (s, f, f - s), flush=True)
assert abs(s - 0.1582) < 1e-4 and abs(f - 0.2466) < 1e-4

results = {"label": "salt formation 30ns, prefix + generated mean (proposals 11-14)",
           "train_n": int(len(Xtr)), "test_n": int(len(Xte)),
           "full_gain": f - s, "conditions": res,
           "relations_test": relations, "mixed_minus_prefix": mixed_diff}
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print("saved to", OUT)
