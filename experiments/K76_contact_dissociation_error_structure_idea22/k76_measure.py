"""K76: where is idea 22's mean error large, for contact DISSOCIATION?

Contact version of K35 (contact formation, idea 22) / K70 (contact
dissociation, ideas 11-14). Per candidate, the error is generated
mean - real mean, broken down by the same three splits:
  (a) origin-distance band: near the 4.5 A boundary (3.5-4.5) vs
      elsewhere (<= 3.5), plus finer 0.5 A bands below 4.5 A.
  (b) size of the real mean's deviation from the overall mean of
      real means: |real_mean - center| bands (<0.25, 0.25-0.5,
      0.5-1.0, >=1.0; same cutoffs as K30/K35/K67/K70).
  (c) residue separation |res_i - res_j| bands (1, 2-5, 6-10,
      11-20, 21+; same as K30/K35/K67/K70).
Also the label relation at 1 ns (K29 (a) / K35/K70): corr(error,
label), mean(error | y=1) - mean(error | y=0), AUC of the error
predicting the label, with the real mean's own label signal as
context and a shrinkage+noise synthetic error (K35/K70's
comparison) for reference.

Definitions (same conventions as K30/K35/K67/K70, adapted to
contact dissociation + idea 22):
- Candidates: the feedable subset of K50's contact dissociation
  candidates -- 3,164 of 9,298 -- whose representative heavy-atom
  pair (the pair attaining the minimum distance at the origin) is a
  (donor, acceptor) pair in either orientation. Idea 22's
  descriptors are defined only for such pairs (the same restriction
  as K14/K35). Results apply to this subset, not to contact
  dissociation as a whole. The deviation-band centre is the overall
  real mean on this subset.
- Real mean m_r = mean of the 11-frame minimum-distance history
  (K50 cache `hist`, frames 0..10), the same quantity whose mean is
  the contact history statistic in K50.
- Generated mean m_g = mean of the 5-point idea-22 series (origin +
  0.1/0.3/0.5/1.0 ns), the same definition as K30/K35's idea-22
  mean (origin + steps, one branch).
- Control (definitional gap): the real forward 5-point minimum
  series (`min5`, same 5 points) mean vs the 11-frame history mean.
- Same-subset comparison: K58's residue-level ideas 14 and 11,
  restricted to the same 3,164 feedable candidates, measured with
  the same splits, so the idea-22 vs residue-level comparison is on
  identical candidates.
- Label at 1 ns: minimum distance > 4.5 A at frame 20 (y=1 means
  the contact dissociated).

Pre-fixed reading (from the assignment): if idea 22's error too
grows toward the boundary and is large for pairs whose real mean
is far from the overall mean, the shape is the same regardless of
the generation method; if the shape differs, the error shape
depends on the generation method.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
GEN22 = Path("/home/hatch/workspace/kinescope/work/k76_gen22")
GEN58 = Path("/home/hatch/workspace/kinescope/work/k58_gen_residue")
OUT = Path(__file__).resolve().parent

PROTS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")

hist, y, res_i, res_j = [], [], [], []
gen22, min5 = [], []
g58_14, g58_11 = [], []
for p in PROTS:
    for r in (1, 2, 3):
        d = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
        g22 = np.load(GEN22 / f"atlas-{p}-analysis-R{r}.npz")
        g58 = np.load(GEN58 / f"{p}_R{r}.npz")
        key22 = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(g22["res_i"], g22["res_j"]))}
        key58 = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(g58["res_i"], g58["res_j"]))}
        idx22, idx58 = [], []
        for u, v in zip(d["res_i"], d["res_j"]):
            k = key22.get((int(u), int(v)))
            if k is None:
                continue  # not feedable
            idx22.append(k); idx58.append(key58[(int(u), int(v))])
        idx22 = np.array(idx22, dtype=int); idx58 = np.array(idx58, dtype=int)
        sel = np.array([key22.get((int(u), int(v))) is not None
                        for u, v in zip(d["res_i"], d["res_j"])])
        hist.append(d["hist"][sel].astype(np.float64))
        y.append(d["y"][sel].astype(np.float64))
        res_i.append(d["res_i"][sel]); res_j.append(d["res_j"][sel])
        gen22.append(g22["gen22"][idx22].astype(np.float64))
        min5.append(g22["min5"][idx22].astype(np.float64))
        g58_14.append(g58["gen14"][:, :, idx58].astype(np.float64))
        g58_11.append(g58["gen11"][:, :, idx58].astype(np.float64))
hist = np.concatenate(hist); y = np.concatenate(y)
res_i = np.concatenate(res_i); res_j = np.concatenate(res_j)
gen22 = np.concatenate(gen22); min5 = np.concatenate(min5)

m_r = hist.mean(axis=1)
d0 = hist[:, -1]
sep = np.abs(res_i.astype(int) - res_j.astype(int))
y1 = y[:, 0].astype(np.float64)  # 1 ns label
center = float(m_r.mean())
methods = {
    "idea22": gen22.mean(axis=1),
    "control_fwd5": min5.mean(axis=1),
    "idea14_same_subset": np.concatenate(g58_14, axis=2).mean(axis=(0, 1)),
    "idea11_same_subset": np.concatenate(g58_11, axis=2).mean(axis=(0, 1)),
}


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
        "3.5-4.5 (near boundary)": (d0 > 3.5) & (d0 <= 4.5),
        "<=3.5 (elsewhere)": d0 <= 3.5,
    },
    "a_origin_fine": {
        "<=2.5": d0 <= 2.5,
        "2.5-3.0": (d0 > 2.5) & (d0 <= 3.0),
        "3.0-3.5": (d0 > 3.0) & (d0 <= 3.5),
        "3.5-4.0": (d0 > 3.5) & (d0 <= 4.0),
        "4.0-4.5": (d0 > 4.0) & (d0 <= 4.5),
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

res = {"n_candidates": int(len(m_r)), "n_candidates_all": 9298,
       "overall_real_mean": center, "real_mean_std": float(m_r.std()),
       "origin_mean": float(d0.mean()), "label_1ns_positive": int(y1.sum()),
       "methods": {}}
for name, gm in methods.items():
    err = gm - m_r
    res["methods"][name] = {
        "overall": group_stats(err, m_r, np.ones(len(m_r), dtype=bool)),
        **{gname: {g: group_stats(err, m_r, m) for g, m in gdefs.items()}
           for gname, gdefs in groups.items()}}
    print(name, "overall bias %.3f std %.3f mae %.3f" % (
        res["methods"][name]["overall"]["bias"], res["methods"][name]["overall"]["std"],
        res["methods"][name]["overall"]["mae"]), flush=True)

# label relation at 1 ns (K29 (a) / K35/K70)
label = {}
for name in ("idea22", "idea14_same_subset"):
    err = methods[name] - m_r
    label[name] = dict(corr_with_label=corr(err, y1),
                       mean_error_pos_minus_neg=float(err[y1 == 1].mean() - err[y1 == 0].mean()),
                       auc_error_predicts_label=float(roc_auc_score(y1, err)))
label["context"] = dict(corr_real_mean_with_label=corr(m_r, y1),
                        corr_gen22_mean_with_label=corr(methods["idea22"], y1),
                        real_mean_pos_minus_neg=float(m_r[y1 == 1].mean() - m_r[y1 == 0].mean()),
                        gen22_mean_pos_minus_neg=float(methods["idea22"][y1 == 1].mean() - methods["idea22"][y1 == 0].mean()))
b_reg, a_reg = np.polyfit(m_r, methods["idea22"], 1)
sigma_reg = float((methods["idea22"] - (a_reg + b_reg * m_r)).std())
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
(OUT / "k76_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k76_results.json")
