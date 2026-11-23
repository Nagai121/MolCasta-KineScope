"""K35: where is the generator's mean error large, for contact (Doc D)?

Contact version of K30 (hydrogen bonds). Per candidate, the error is
generated mean - real mean, broken down by the same three splits:
  (a) origin-distance band: near the 4.5 A boundary (4.5-5.5) vs
      elsewhere (> 5.5), plus finer bands.
  (b) size of the real mean's deviation from the overall mean of real
      means: |real_mean - center| bands (<0.25, 0.25-0.5, 0.5-1.0, >=1.0).
  (c) residue separation |res_i - res_j| bands (1, 2-5, 6-10, 11-20, 21+).
Also the label relation at 1 ns (K29 (a)): corr(error, label),
mean(error | y=1) - mean(error | y=0), AUC of the error predicting the
label, with the real mean's own label signal as context.

Definitions (same conventions as K30, adapted to contact):
- Candidates: K14's 4,968 feedable contact candidates (of K13's 23,310).
  Feedable means the representative atom pair (minimum at the origin)
  is a donor-acceptor pair, so the subset is biased toward
  donor-acceptor-like pairs; results apply to this subset, not to
  contact as a whole.
- Real mean m_r = mean of the 11-frame minimum-distance history
  (K13 cache `hist`, frames 0..10), the same quantity whose mean is the
  contact history statistic in K13/K16/K27.
- Generated mean m_g = mean of the 5-point generated series
  (origin + 0.1/0.3/0.5/1.0 ns, K14 `gen`); elastic-only m_c likewise
  from K14 `coarse`. The generated series is the representative pair's
  distance, not the minimum over atom pairs.
- Alignment: K14's `min` (5-point minimum series, feedable subset) is a
  subsequence of `min_all` (all candidates), whose origin column matches
  the K13 cache's hist[-1] exactly (max diff 0.0) in the same
  protein/replica order. Each generated row is matched to its cache row
  by the full 5-value minimum series (all 4,968 rows matched uniquely;
  nonfinite = 0, so no row was dropped).
- Control (definitional gap): the real representative-pair series
  (`rep`, 5 points) vs the real minimum series (`min`, 5 points) vs
  the 11-frame history mean. The generated mean's error against the
  history mean mixes the generator's own error with the
  representative-pair-vs-minimum difference; the controls separate them.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
K14 = Path("/home/hatch/workspace/kinescope/repo/experiments/K14_other_phenomena_generated/k14_contact.npz")
ALIGN = Path("/tmp/k35_align.npy")
OUT = Path(__file__).resolve().parent

PROTS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")

# --- load cache in K14's protein/replica order ---
hist, y, res_i, res_j = [], [], [], []
for p in PROTS:
    for r in (1, 2, 3):
        d = np.load(CACHE / f"{p}_R{r}_contact.npz")
        hist.append(d["hist"].astype(np.float64))
        y.append(d["y"].astype(np.float64))
        res_i.append(d["res_i"]); res_j.append(d["res_j"])
hist = np.concatenate(hist); y = np.concatenate(y)
res_i = np.concatenate(res_i); res_j = np.concatenate(res_j)

with np.load(K14) as z:
    gen = z["gen"].astype(np.float64)
    coarse = z["coarse"].astype(np.float64)
    rep = z["rep"].astype(np.float64)
    mn = z["min"].astype(np.float64)

idx = np.load(ALIGN)
assert len(idx) == len(gen) == 4968 and len(set(idx.tolist())) == 4968
assert np.abs(rep[:, 0] - gen[:, 0]).max() < 1e-4

m_r = hist[idx].mean(axis=1)          # 11-frame history mean (minimum series)
d0 = hist[idx, -1]                    # origin minimum distance (frame 10)
sep = np.abs(res_i[idx].astype(int) - res_j[idx].astype(int))
y1 = y[idx, 0].astype(np.float64)     # 1 ns label
m_g = gen.mean(axis=1)
m_c = coarse.mean(axis=1)
m_rep = rep.mean(axis=1)              # real representative-pair mean (5 points)
m_min5 = mn.mean(axis=1)              # real minimum mean (5 points)

center = float(m_r.mean())
e_gen = m_g - m_r
e_coarse = m_c - m_r
e_rep = m_rep - m_r                   # control: rep-pair vs history mean
e_min5 = m_min5 - m_r                 # control: forward-min vs history mean


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def group_stats(err, rm, mask):
    if mask.sum() < 10:
        return dict(n=int(mask.sum()))
    e, r = err[mask], rm[mask]
    return dict(n=int(mask.sum()), bias=float(e.mean()), std=float(e.std()),
                mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                real_mean_std=float(r.std()), corr=float(np.corrcoef(e, r)[0, 1]) if r.std() > 0 else None)


groups = {
    "a_origin_band": {
        "4.5-5.5 (near boundary)": (d0 > 4.5) & (d0 <= 5.5),
        ">5.5 (elsewhere)": d0 > 5.5,
    },
    "a_origin_fine": {
        "4.5-5.0": (d0 > 4.5) & (d0 <= 5.0),
        "5.0-5.5": (d0 > 5.0) & (d0 <= 5.5),
        "5.5-6.0": (d0 > 5.5) & (d0 <= 6.0),
        "6.0-7.0": (d0 > 6.0) & (d0 <= 7.0),
        "7.0+": d0 > 7.0,
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

errors = {"idea22": e_gen, "elastic_only": e_coarse,
          "control_rep_pair": e_rep, "control_min_5pt": e_min5}
res = {"n_candidates": int(len(m_r)), "overall_real_mean": center,
       "real_mean_std": float(m_r.std()), "origin_mean": float(d0.mean()),
       "label_1ns_positive": int(y1.sum()), "methods": {}}
for name, err in errors.items():
    res["methods"][name] = {
        "overall": group_stats(err, m_r, np.ones(len(m_r), dtype=bool)),
        **{gname: {g: group_stats(err, m_r, m) for g, m in gdefs.items()}
           for gname, gdefs in groups.items()}}
    print(name, "overall bias %.3f std %.3f mae %.3f" % (
        res["methods"][name]["overall"]["bias"], res["methods"][name]["overall"]["std"],
        res["methods"][name]["overall"]["mae"]), flush=True)

# label relation at 1 ns (K29 (a)), for the two generated errors
label = {}
for name, err in (("idea22", e_gen), ("elastic_only", e_coarse)):
    label[name] = dict(corr_with_label=corr(err, y1),
                       mean_error_pos_minus_neg=float(err[y1 == 1].mean() - err[y1 == 0].mean()),
                       auc_error_predicts_label=float(roc_auc_score(y1, err)))
label["context"] = dict(corr_real_mean_with_label=corr(m_r, y1),
                        corr_gen_mean_with_label=corr(m_g, y1),
                        corr_coarse_mean_with_label=corr(m_c, y1),
                        real_mean_pos_minus_neg=float(m_r[y1 == 1].mean() - m_r[y1 == 0].mean()),
                        gen_mean_pos_minus_neg=float(m_g[y1 == 1].mean() - m_g[y1 == 0].mean()))
# Synthetic shrinkage+noise error (K29's comparison): fit m_g = a + b*m_r,
# then e_syn = a + (b-1)*m_r + noise with the regression residual SD.
# The generated error inherits a label relation through the shrinkage term,
# so the synthetic shows how much of the label relation shrinkage alone
# would produce.
b_reg, a_reg = np.polyfit(m_r, m_g, 1)
sigma_reg = float((m_g - (a_reg + b_reg * m_r)).std())
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
(OUT / "k35_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k35_results.json")
