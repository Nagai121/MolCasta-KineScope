"""K38: where is the generator's mean error large, for salt bridges
(K15 expanded set)?

Salt-bridge version of K30 (hydrogen bonds) / K35 (contact). Per
candidate, the error is generated mean - real mean, broken down by the
same splits:
  (a) origin-distance band: near the 4.0 A boundary (4.0-5.0) vs
      elsewhere (> 5.0), plus finer bands.
  (b) size of the real mean's deviation from the overall mean of real
      means: |real_mean - center| bands (<0.25, 0.25-0.5, 0.5-1.0, >=1.0).
  (c) residue separation |res_i - res_j| bands (supplementary, as K30/K35).
Also the label relation at 1 ns (K29 (a)): corr(error, label),
mean(error | y=1) - mean(error | y=0), AUC of the error predicting the
label, with the real mean's own label signal as context and a
shrinkage+noise synthetic error (K35's comparison) for reference.

Definitions (same conventions as K30/K35, adapted to salt bridges):
- Candidates: all 3,727 salt-bridge candidates of K15's expanded set
  (train 2,557 + test 1,170; K20's files, ~/workspace/k20/out/).
- Real mean m_r = mean of the 11-frame minimum-distance history
  (K15 cache `hist`, frames 0..10), the same quantity whose mean is the
  salt-bridge history statistic in K13/K15/K19/K31.
- Generated mean m_g = mean of the branch-mean 9-point generated series
  (origin + 8 steps, K20 `gen11`..`gen14`), i.e. the mean over branches
  first, then over the 9 points including the origin -- the same
  convention as K30's idea22 mean (origin + steps).
- Verification: the generated series' origin equals the real forward
  series' origin exactly (max diff 0.0), and the real history/labels/
  residue indices match the K15 cache exactly (checked on sample files).
- Control (definitional gap): the real forward 5-point minimum series
  (`fwd5`, origin + 0.1/0.3/0.5/1.0 ns) mean vs the 11-frame history
  mean -- how much of the generated error is just forward-vs-history.
- The salt-bridge generated series is already the minimum over charged
  atom pairs at each time point (K20), so unlike K35 there is no
  representative-pair-vs-minimum mismatch to control for.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

K20 = Path("/home/hatch/workspace/k20/out")
OUT = Path(__file__).resolve().parent

hist, y, res_i, res_j, fwd5 = [], [], [], [], []
gen = {n: [] for n in (11, 12, 13, 14)}
for f in sorted(K20.glob("*.npz")):
    z = np.load(f)
    if len(z["hist"]) == 0:
        continue
    hist.append(z["hist"].astype(np.float64))
    y.append(z["y"].astype(np.float64))
    res_i.append(z["res_i"]); res_j.append(z["res_j"])
    fwd5.append(z["fwd5"].astype(np.float64))
    for n in gen:
        gen[n].append(z[f"gen{n}"].astype(np.float64).mean(axis=0))  # (9, N) branch-mean
hist = np.concatenate(hist); y = np.concatenate(y)
res_i = np.concatenate(res_i); res_j = np.concatenate(res_j)
fwd5 = np.concatenate(fwd5)
gen = {n: np.concatenate(v, axis=1) for n, v in gen.items()}  # (9, N)

m_r = hist.mean(axis=1)                       # 11-frame history mean
d0 = hist[:, -1]                              # origin minimum distance (frame 10)
sep = np.abs(res_i.astype(int) - res_j.astype(int))
y1 = y[:, 0].astype(np.float64)               # 1 ns label
m_g = {n: gen[n].mean(axis=0) for n in gen}   # mean over the 9 points
m_fwd5 = fwd5.mean(axis=1)

center = float(m_r.mean())
e = {f"idea{n}": m_g[n] - m_r for n in gen}
e["control_fwd5"] = m_fwd5 - m_r


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def group_stats(err, rm, mask):
    if mask.sum() < 10:
        return dict(n=int(mask.sum()))
    ee, r = err[mask], rm[mask]
    return dict(n=int(mask.sum()), bias=float(ee.mean()), std=float(ee.std()),
                mae=float(np.abs(ee).mean()), rmse=float(np.sqrt((ee ** 2).mean())),
                real_mean_std=float(r.std()), corr=float(np.corrcoef(ee, r)[0, 1]) if r.std() > 0 else None)


groups = {
    "a_origin_band": {
        "4.0-5.0 (near boundary)": (d0 > 4.0) & (d0 <= 5.0),
        ">5.0 (elsewhere)": d0 > 5.0,
    },
    "a_origin_fine": {
        "4.0-4.5": (d0 > 4.0) & (d0 <= 4.5),
        "4.5-5.0": (d0 > 4.5) & (d0 <= 5.0),
        "5.0-6.0": (d0 > 5.0) & (d0 <= 6.0),
        "6.0-8.0": (d0 > 6.0) & (d0 <= 8.0),
        "8.0+": d0 > 8.0,
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

# label relation at 1 ns (K29 (a)) for each idea + synthetic comparison
# (regression of the idea14 mean on the real mean; idea14 is closest to real)
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
(OUT / "k38_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k38_results.json")
