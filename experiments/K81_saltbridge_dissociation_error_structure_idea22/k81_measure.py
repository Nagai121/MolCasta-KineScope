"""K81: where is idea 22's mean error large, for salt-bridge
DISSOCIATION?

Salt-bridge version of K76 (contact dissociation, idea 22) / K72
(salt-bridge dissociation, ideas 11-14). Per candidate, the error
is generated mean - real mean, broken down by the same three
splits:
  (a) origin-distance band: near the 4.0 A boundary (3.0-4.0) vs
      elsewhere (<= 3.0), plus finer 0.5 A bands below 4.0 A
      (K72's form).
  (b) size of the real mean's deviation from the overall mean of
      real means: |real_mean - center| bands (<0.25, 0.25-0.5,
      0.5-1.0, >=1.0; same cutoffs as K30/K35/K38/K67/K70/K72/K76).
  (c) residue separation |res_i - res_j| bands (1, 2-5, 6-10,
      11-20, 21+; same as K30/K35/K38/K67/K70/K72/K76).
Also the label relation at 1 ns (K29 (a) / K38/K72): corr(error,
label), mean(error | y=1) - mean(error | y=0), AUC of the error
predicting the label, with the real mean's own label signal as
context and a shrinkage+noise synthetic error (K35/K38's
comparison) for reference.

Definitions (same conventions as K30/K35/K38/K67/K70/K72/K76,
adapted to salt-bridge dissociation + idea 22):
- Candidates: all 1,095 of K53's salt-bridge dissociation
  candidates (K15 expanded set; origin minimum charged heavy-atom
  distance <= 4.0 A). Unlike K76, there is no feedable-subset
  restriction: every candidate's representative pair is a
  (donor, acceptor) pair (measured by check_feedable.py: 1,095 of
  1,095, 100%).
- Real mean m_r = mean of the 11-frame minimum-distance history
  (K53 cache `hist`, frames 0..10), the same quantity whose mean is
  the salt-bridge history statistic in K53.
- Generated mean m_g = mean of the 5-point idea-22 series (origin +
  0.1/0.3/0.5/1.0 ns), the same definition as K30/K35/K76's
  idea-22 mean (origin + steps, one branch).
- Control (definitional gap): the real forward 5-point minimum
  series (`min5`, same 5 points) mean vs the 11-frame history mean.
- Same-candidate comparison: K65's residue-level ideas 14 and 11,
  measured with the same splits on the same 1,095 candidates, so
  the idea-22 vs residue-level comparison is on identical
  candidates (as in K76).
- Label at 1 ns: minimum distance > 4.0 A at frame 20 (y=1 means
  the salt bridge dissociated).

Pre-fixed reading (from the assignment): if idea 22's error too
does not grow near the boundary and is large for pairs whose real
mean is far from the overall mean, the shape is the same
regardless of the generation method; if the shape differs, the
error shape depends on the generation method.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
GEN22 = Path("/home/hatch/workspace/kinescope/work/k81_gen22")
GEN65 = Path("/home/hatch/workspace/kinescope/work/k65_gen_residue")
OUT = Path(__file__).resolve().parent

_SPLIT = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json").read_text())
PROTS = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"]) + tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])

hist, y, res_i, res_j = [], [], [], []
gen22, min5 = [], []
g65_14, g65_11 = [], []
for p in PROTS:
    for r in (1, 2, 3):
        d = np.load(CACHE / f"{p}_R{r}_salt_diss.npz")
        if len(d["hist"]) == 0:
            continue
        g22 = np.load(GEN22 / f"atlas-{p}-analysis-R{r}.npz")
        g65 = np.load(GEN65 / f"{p}_R{r}.npz")
        key22 = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(g22["res_i"], g22["res_j"]))}
        key65 = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(g65["res_i"], g65["res_j"]))}
        idx22 = np.array([key22[(int(u), int(v))] for u, v in zip(d["res_i"], d["res_j"])])
        idx65 = np.array([key65[(int(u), int(v))] for u, v in zip(d["res_i"], d["res_j"])])
        assert len(idx22) == len(d["res_i"]) and len(idx65) == len(d["res_i"]), (p, r)
        hist.append(d["hist"].astype(np.float64))
        y.append(d["y"].astype(np.float64))
        res_i.append(d["res_i"]); res_j.append(d["res_j"])
        gen22.append(g22["gen22"][idx22].astype(np.float64))
        min5.append(g22["min5"][idx22].astype(np.float64))
        g65_14.append(g65["gen14"][:, :, idx65].astype(np.float64))
        g65_11.append(g65["gen11"][:, :, idx65].astype(np.float64))
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
    "idea14_same_candidates": np.concatenate(g65_14, axis=2).mean(axis=(0, 1)),
    "idea11_same_candidates": np.concatenate(g65_11, axis=2).mean(axis=(0, 1)),
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
for name, gm in methods.items():
    err = gm - m_r
    res["methods"][name] = {
        "overall": group_stats(err, m_r, np.ones(len(m_r), dtype=bool)),
        **{gname: {g: group_stats(err, m_r, m) for g, m in gdefs.items()}
           for gname, gdefs in groups.items()}}
    print(name, "overall bias %.3f std %.3f mae %.3f" % (
        res["methods"][name]["overall"]["bias"], res["methods"][name]["overall"]["std"],
        res["methods"][name]["overall"]["mae"]), flush=True)

# label relation at 1 ns (K29 (a) / K38/K72)
label = {}
for name in ("idea22", "idea14_same_candidates"):
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
(OUT / "k81_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k81_results.json")
