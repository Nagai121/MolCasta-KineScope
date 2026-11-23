"""K88: where is idea 22's mean error large, for salt-bridge
FORMATION?

Salt-bridge formation version of K84 (contact formation, idea 22)
and K76/K81 (dissociation, idea 22). Per candidate, the error is
generated mean - real mean, broken down by the same splits as K38:
  (a) origin-distance band: near the 4.0 A boundary (4.0-5.0) vs
      elsewhere (> 5.0), plus finer bands.
  (b) size of the real mean's deviation from the overall mean of
      real means: |real_mean - center| bands (<0.25, 0.25-0.5,
      0.5-1.0, >=1.0; same cutoffs as K30/K35/K38/K67/K70/K72/K76).
  (c) residue separation |res_i - res_j| bands (1, 2-5, 6-10,
      11-20, 21+; same as K30/K35/K38/K67/K70/K72/K76).
Also the label relation at 1 ns (K29 (a) / K38): corr(error,
label), mean(error | y=1) - mean(error | y=0), AUC of the error
predicting the label, with the real mean's own label signal as
context and a shrinkage+noise synthetic error (K35/K38's
comparison) for reference.

Definitions (same conventions as K30/K35/K38, adapted to salt
bridges + idea 22):
- Candidates: K14's 367 salt-bridge formation candidates (the
  original 6 proteins of Doc D; K15's expanded set has 3,727, but
  K14's probe covered only the original 6). All 367 are feedable:
  the representative pair (minimum at the origin) is always a
  (donor, acceptor) pair for salt bridges, since the minimum is
  over charged (N, O) pairs. So there is no feedable-subset
  restriction within the original 6, unlike K35/K84 (contact).
  All methods are measured on these same 367 candidates.
- Real mean m_r = mean of the 11-frame minimum-distance history
  (K20's `hist` for the original 6 proteins, frames 0..10), the
  same quantity whose mean is the salt-bridge history statistic
  in K13/K15/K19/K31.
- Generated mean for idea 22: mean of the 5-point generated series
  (origin + 0.1/0.3/0.5/1.0 ns, K14 `gen`); elastic-only likewise
  from K14 `coarse`. The generated series is the representative
  pair's distance, not the minimum over atom pairs.
- Generated mean for ideas 11-14: mean of the branch-mean 9-point
  generated series (origin + 8 steps, K20 `gen11`..`gen14`), the
  same definition as K38.
- Alignment: K14's salt series correspond 1:1, in order, to the
  K20 files of the original 6 proteins loaded in PROTEINS order
  (1bgf_A, 1ifg_A, 1j8r_A, 2b0a_A, 1mix_A, 2qt4_A). Verified: K14's
  `min` vs K20's `fwd5` agree to 9.0e-07 (float32 rounding) in
  that order. (Loading K20's files in sorted-glob order instead
  swaps 1mix_A and 2b0a_A and breaks the correspondence -- the
  same order trap as K84's K14/K13 alignment.)
- Controls (K38's): the real forward 5-point minimum series
  (`fwd5`) mean vs the 11-frame history mean; and the real
  representative-pair series (`rep`, 5 points) mean vs the history
  mean (K35's control, added here because idea 22 tracks a
  representative pair, not the minimum).
- Label at 1 ns: minimum distance <= 4.0 A at frame 20 (y=1 means
  the salt bridge formed).

Pre-fixed reading (from the assignment): if idea 22's error too
does not grow near the boundary and is large for pairs whose real
mean is far from the overall mean, the shape is the same
regardless of the generation method for salt-bridge formation; if
the shape differs, the error shape depends on the generation
method. K84 split only in the deviation dimension, so watch
whether salt-bridge formation splits the same way.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

K20 = Path("/home/hatch/workspace/k20/out")
K14 = Path("/home/hatch/workspace/kinescope/repo/experiments/K14_other_phenomena_generated/k14_salt.npz")
OUT = Path(__file__).resolve().parent

PROTS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")

# --- K20 original-6 rows, in PROTEINS order (K14's order) ---
hist, y, res_i, res_j, fwd5 = [], [], [], [], []
gen = {n: [] for n in (11, 12, 13, 14)}
for p in PROTS:
    for r in (1, 2, 3):
        z = np.load(K20 / f"{p}_R{r}.npz")
        if len(z["hist"]) == 0:
            continue
        hist.append(z["hist"].astype(np.float64))
        y.append(z["y"].astype(np.float64))
        res_i.append(z["res_i"]); res_j.append(z["res_j"])
        fwd5.append(z["fwd5"].astype(np.float64))
        for n in gen:
            gen[n].append(z[f"gen{n}"].astype(np.float64).mean(axis=0))  # (9, N)
hist = np.concatenate(hist); y = np.concatenate(y)
res_i = np.concatenate(res_i); res_j = np.concatenate(res_j)
fwd5 = np.concatenate(fwd5)
gen = {n: np.concatenate(v, axis=1) for n, v in gen.items()}  # (9, N)

with np.load(K14) as z:
    gen22 = z["gen"].astype(np.float64)
    coarse = z["coarse"].astype(np.float64)
    rep = z["rep"].astype(np.float64)
    mn = z["min"].astype(np.float64)

assert len(hist) == len(gen22) == 367, (len(hist), len(gen22))
assert np.abs(mn - fwd5).max() < 1e-6, np.abs(mn - fwd5).max()
assert np.abs(rep[:, 0] - gen22[:, 0]).max() < 1e-4
print("alignment verified: 367 original-6 rows, K14 min vs K20 fwd5 max diff %.2e"
      % np.abs(mn - fwd5).max(), flush=True)

m_r = hist.mean(axis=1)                       # 11-frame history mean
d0 = hist[:, -1]                              # origin minimum distance (frame 10)
sep = np.abs(res_i.astype(int) - res_j.astype(int))
y1 = y[:, 0].astype(np.float64)               # 1 ns label
m_g22 = gen22.mean(axis=1)
m_c = coarse.mean(axis=1)
m_rep = rep.mean(axis=1)
m_min5 = mn.mean(axis=1)
m_g = {n: gen[n].mean(axis=0) for n in gen}   # ideas 11-14, branch-mean 9-point
m_fwd5 = fwd5.mean(axis=1)

center = float(m_r.mean())
methods = {"idea22": m_g22, "elastic_only": m_c,
           "idea14": m_g[14], "idea11": m_g[11]}


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def group_stats(err, rm, mask):
    if mask.sum() < 10:
        return dict(n=int(mask.sum()))
    e, r = err[mask], rm[mask]
    return dict(n=int(mask.sum()), bias=float(e.mean()), std=float(e.std()),
                mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                real_mean_std=float(r.std()),
                corr=float(np.corrcoef(e, r)[0, 1]) if r.std() > 0 else None)


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

errors = {name: gm - m_r for name, gm in methods.items()}
errors["control_rep_pair"] = m_rep - m_r
errors["control_min_5pt"] = m_min5 - m_r
errors["control_fwd5"] = m_fwd5 - m_r
res = {"n_candidates": int(len(m_r)), "n_candidates_all": 3727,
       "overall_real_mean": center, "real_mean_std": float(m_r.std()),
       "origin_mean": float(d0.mean()), "label_1ns_positive": int(y1.sum()),
       "methods": {}}
for name, err in errors.items():
    res["methods"][name] = {
        "overall": group_stats(err, m_r, np.ones(len(m_r), dtype=bool)),
        **{gname: {g: group_stats(err, m_r, m) for g, m in gdefs.items()}
           for gname, gdefs in groups.items()}}
    print(name, "overall bias %.3f std %.3f mae %.3f" % (
        res["methods"][name]["overall"]["bias"], res["methods"][name]["overall"]["std"],
        res["methods"][name]["overall"]["mae"]), flush=True)

# label relation at 1 ns (K29 (a) / K38)
label = {}
for name in ("idea22", "idea14"):
    err = errors[name]
    label[name] = dict(corr_with_label=corr(err, y1),
                       mean_error_pos_minus_neg=float(err[y1 == 1].mean() - err[y1 == 0].mean()),
                       auc_error_predicts_label=float(roc_auc_score(y1, err)))
label["context"] = dict(corr_real_mean_with_label=corr(m_r, y1),
                        corr_gen22_mean_with_label=corr(m_g22, y1),
                        corr_gen14_mean_with_label=corr(m_g[14], y1),
                        real_mean_pos_minus_neg=float(m_r[y1 == 1].mean() - m_r[y1 == 0].mean()),
                        gen22_mean_pos_minus_neg=float(m_g22[y1 == 1].mean() - m_g22[y1 == 0].mean()),
                        gen14_mean_pos_minus_neg=float(m_g[14][y1 == 1].mean() - m_g[14][y1 == 0].mean()))
b_reg, a_reg = np.polyfit(m_r, m_g22, 1)
sigma_reg = float((m_g22 - (a_reg + b_reg * m_r)).std())
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
        s = res["methods"]["idea22"][gname][g]
        print("  %-24s n=%6d bias=%+.3f std=%.3f mae=%.3f real_std=%.3f" % (
            g, s["n"], s.get("bias", 0), s.get("std", 0), s.get("mae", 0), s.get("real_mean_std", 0)))
print("label 1ns:", json.dumps(label, indent=1))
(OUT / "k88_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k88_results.json")
