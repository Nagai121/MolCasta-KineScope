"""K72: where is the generator's mean error large, for salt-bridge
DISSOCIATION?

Salt-bridge version of K67 (hydrogen-bond dissociation) / K70
(contact dissociation), with K38's (salt-bridge formation) form.
Per candidate, the error is generated mean - real mean, broken down
by the same splits:
  (a) origin-distance band: near the 4.0 A boundary (3.0-4.0) vs
      elsewhere (<= 3.0), plus finer 0.5 A bands below 4.0 A.
  (b) size of the real mean's deviation from the overall mean of
      real means: |real_mean - center| bands (<0.25, 0.25-0.5,
      0.5-1.0, >=1.0; same cutoffs as K30/K35/K38/K67/K70).
  (c) residue separation |res_i - res_j| bands (1, 2-5, 6-10,
      11-20, 21+; same as K30/K35/K38/K67/K70).
Also the label relation at 1 ns (K29 (a) / K38): corr(error,
label), mean(error | y=1) - mean(error | y=0), AUC of the error
predicting the label, with the real mean's own label signal as
context and a shrinkage+noise synthetic error (K35/K38's
comparison) for reference.

Definitions (same conventions as K30/K35/K38/K67/K70, adapted to
salt-bridge dissociation):
- Candidates: all 1,095 of K53's salt-bridge dissociation
  candidates (K15 expanded set; origin minimum charged heavy-atom
  distance <= 4.0 A).
- Real mean m_r = mean of the 11-frame minimum-distance history
  (K53 cache `hist`, frames 0..10), the same quantity whose mean is
  the salt-bridge history statistic in K53.
- Generated mean m_g = mean over the branches x (origin + 8 steps)
  block of K65's residue-level series (ideas 11-14), the same
  definition as K26/K30/K38/K67/K70.
- Alignment: K65's series are matched to cache rows by
  (res_i, res_j) per trajectory; every candidate must match.
- Label at 1 ns: minimum distance > 4.0 A at frame 20 (y=1 means
  the salt bridge dissociated).
- Control (definitional gap): the real forward 5-point minimum
  series (`fwd5`, origin + 0.1/0.3/0.5/1.0 ns) mean vs the 11-frame
  history mean -- how much of the generated error is just
  forward-vs-history (K38's control).

Pre-fixed reading (from the assignment): if the error does not grow
near the boundary and is large for pairs whose real mean is far
from the overall mean, salt-bridge dissociation has the same shape
as hydrogen-bond dissociation (K67); if there is a gradient growing
toward the boundary, it has the same shape as contact dissociation
(K70); if neither, salt-bridge dissociation is its own shape.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
GEN_DIR = Path("/home/hatch/workspace/kinescope/work/k65_gen_residue")
OUT = Path(__file__).resolve().parent

_SPLIT = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json").read_text())
PROTS = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"]) + tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])

hist, y, res_i, res_j, fwd5 = [], [], [], [], []
gen = {n: [] for n in (11, 12, 13, 14)}
for p in PROTS:
    for r in (1, 2, 3):
        d = np.load(CACHE / f"{p}_R{r}_salt_diss.npz")
        if len(d["hist"]) == 0:
            continue
        hist.append(d["hist"].astype(np.float64))  # (N, 11), no transpose
        y.append(d["y"].astype(np.float64))
        res_i.append(d["res_i"]); res_j.append(d["res_j"])
        gz = np.load(GEN_DIR / f"{p}_R{r}.npz")
        key = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(gz["res_i"], gz["res_j"]))}
        idx = np.array([key[(int(u), int(v))] for u, v in zip(d["res_i"], d["res_j"])])
        assert len(idx) == len(d["res_i"]), (p, r)
        fwd5.append(gz["fwd5"].astype(np.float64))
        for n in (11, 12, 13, 14):
            gen[n].append(gz[f"gen{n}"][:, :, idx].astype(np.float64))
hist = np.concatenate(hist); y = np.concatenate(y)
res_i = np.concatenate(res_i); res_j = np.concatenate(res_j)
fwd5 = np.concatenate(fwd5)

m_r = hist.mean(axis=1)
d0 = hist[:, -1]
sep = np.abs(res_i.astype(int) - res_j.astype(int))
y1 = y[:, 0].astype(np.float64)  # 1 ns label
m_g = {n: np.concatenate(gen[n], axis=2).mean(axis=(0, 1)) for n in (11, 12, 13, 14)}
m_fwd5 = fwd5.mean(axis=1)

center = float(m_r.mean())
e = {f"idea{n}": m_g[n] - m_r for n in (11, 12, 13, 14)}
e["control_fwd5"] = m_fwd5 - m_r


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def group_stats(err, rm, mask):
    if mask.sum() < 10:
        return dict(n=int(mask.sum()))
    ee, r = err[mask], rm[mask]
    return dict(n=int(mask.sum()), bias=float(ee.mean()), std=float(ee.std()),
                mae=float(np.abs(ee).mean()), rmse=float(np.sqrt((ee ** 2).mean())),
                real_mean_std=float(r.std()),
                corr=float(np.corrcoef(ee, r)[0, 1]) if r.std() > 0 else None)


groups = {
    "a_origin_band": {
        "3.0-4.0 (near boundary)": (d0 > 3.0) & (d0 <= 4.0),
        "<=3.0 (elsewhere)": d0 <= 3.0,
    },
    "a_origin_fine": {
        "2.5-3.0": (d0 > 2.5) & (d0 <= 3.0),
        "3.0-3.5": (d0 > 3.0) & (d0 <= 3.5),
        "3.5-4.0": (d0 > 3.5) & (d0 <= 4.0),
    },
    "b_real_mean_deviation": {
        "<0.25": np.abs(m_r - center) < 0.25,
        "0.25-0.5": (np.abs(m_r - center) >= 0.25) & (np.abs(m_r - center) < 0.5),
        "0.5-1.0": (np.abs(m_r - center) >= 0.5) & (np.abs(m_r - center) < 1.0),
        ">=1.0": np.abs(m_r - center) >= 1.0,
    },
    "c_separation": {
        "1 (adjacent residues)": sep == 1,
        "2-5": (sep >= 2) & (sep <= 5),
        "6-10": (sep >= 6) & (sep <= 10),
        "11-20": (sep >= 11) & (sep <= 20),
        "21+": sep >= 21,
    },
}

res = {"n_candidates": int(len(m_r)), "overall_real_mean": center,
       "real_mean_std": float(m_r.std()), "origin_mean": float(d0.mean()),
       "label_1ns_positive": int(y1.sum()), "methods": {}}
for name, err in e.items():
    res["methods"][name] = {
        "overall": group_stats(err, m_r, np.ones(len(m_r), dtype=bool)),
        **{gname: {g: group_stats(err, m_r, m) for g, m in gdefs.items()}
           for gname, gdefs in groups.items()}}
    print(name, "overall bias %.3f std %.3f mae %.3f" % (
        res["methods"][name]["overall"]["bias"], res["methods"][name]["overall"]["std"],
        res["methods"][name]["overall"]["mae"]), flush=True)

# label relation at 1 ns (K29 (a) / K38) for each idea + synthetic comparison
label = {}
for name in (f"idea{n}" for n in (11, 12, 13, 14)):
    err = e[name]
    label[name] = dict(corr_with_label=corr(err, y1),
                       mean_error_pos_minus_neg=float(err[y1 == 1].mean() - err[y1 == 0].mean()),
                       auc_error_predicts_label=float(roc_auc_score(y1, err)))
label["context"] = dict(corr_real_mean_with_label=corr(m_r, y1),
                        corr_gen14_mean_with_label=corr(m_g[14], y1),
                        real_mean_pos_minus_neg=float(m_r[y1 == 1].mean() - m_r[y1 == 0].mean()),
                        gen14_mean_pos_minus_neg=float(m_g[14][y1 == 1].mean() - m_g[14][y1 == 0].mean()))
b_reg, a_reg = np.polyfit(m_r, m_g[14], 1)
sigma_reg = float((m_g[14] - (a_reg + b_reg * m_r)).std())
syn = {}
for seed in (101, 202, 303):
    rng = np.random.default_rng(seed)
    e_syn = a_reg + (b_reg - 1.0) * m_r + rng.normal(0.0, sigma_reg, len(m_r))
    syn[f"seed{seed}"] = dict(corr_with_label=corr(e_syn, y1),
                              mean_error_pos_minus_neg=float(e_syn[y1 == 1].mean() - e_syn[y1 == 0].mean()),
                              auc_error_predicts_label=float(roc_auc_score(y1, e_syn)))
label["synthetic_shrinkage_plus_noise"] = {
    "regression": dict(a=float(a_reg), b=float(b_reg), residual_sd=sigma_reg),
    "seeds": syn,
    "mean_across_seeds": {k: float(np.mean([syn[s][k] for s in syn])) for k in syn["seed101"]}}
res["label_1ns"] = label

for gname, gdefs in groups.items():
    print(gname)
    for g, m in gdefs.items():
        s = res["methods"]["idea14"][gname][g]
        print("  %-24s n=%6d bias=%+.3f std=%.3f mae=%.3f real_std=%.3f" % (
            g, s["n"], s.get("bias", 0), s.get("std", 0), s.get("mae", 0), s.get("real_mean_std", 0)))
print("label 1ns:", json.dumps(label, indent=1))
(OUT / "k72_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k72_results.json")
