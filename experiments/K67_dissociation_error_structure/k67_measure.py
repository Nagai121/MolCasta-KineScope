"""K67: where is the generator's mean error large for
hydrogen-bond DISSOCIATION?

K30's method, applied to K46's dissociation candidates (6,131,
Doc D conditions) with K54's redo generated series (ideas 11-14,
residue-level). Per-candidate error = generated mean - real mean
(real mean = 11-point history mean; generated mean = mean over the
branches x (origin + 8 steps) block, the same definition as
K26/K30). Broken down by:
  (a) origin-distance band: near the 3.5 A boundary (3.0-3.5) vs
      elsewhere (<= 3.0), plus finer 0.5 A bands below 3.5 A.
  (b) size of the real mean's deviation from the overall mean of
      real means: |real_mean - mean(real_mean)| bands (same cutoffs
      as K30).
  (c) residue separation |donor_res - acceptor_res| bands (same as
      K30).
Per group: n, error bias/std/MAE/RMSE, real-mean std and
correlation, so a group's error can be read against how much its
real means spread in the first place. Idea 22 is not measured: it
is formation-only by construction (K54).

Pre-fixed reading (from the assignment): if the error does not grow
near the boundary and is large for pairs whose real mean is far
from the overall mean, dissociation has the same shape as
formation (K30); if the error is large near the boundary,
dissociation differs.
"""
import json
from pathlib import Path
import numpy as np

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
GEN_DIR = Path("/home/hatch/workspace/kinescope/work/k54_gen_residue")
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")

real_mean, sep, d0 = [], [], []
gen = {n: [] for n in (11, 12, 13, 14)}
for p in PROTEINS:
    for r in (1, 2, 3):
        with np.load(CACHE / f"{p}_R{r}.npz") as zz:
            hist = zz["hist"].astype(np.float64)  # (11, N)
            da, aa = zz["donor_atom"], zz["acceptor_atom"]
            dr, ar = zz["donor_res"], zz["acceptor_res"]
        with np.load(GEN_DIR / f"{p}_R{r}.npz") as gz:
            # align by (donor_atom, acceptor_atom); the K54 build
            # generated in cache order, but verify by key anyway
            key = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(gz["donor_atom"], gz["acceptor_atom"]))}
            idx = np.array([key[(int(u), int(v))] for u, v in zip(da, aa)])
            assert len(idx) == len(da)
            for n in (11, 12, 13, 14):
                gen[n].append(gz[f"gen{n}"][:, :, idx].astype(np.float64))
        real_mean.append(hist.mean(axis=0))
        sep.append(np.abs(dr.astype(np.int64) - ar.astype(np.int64)))
        d0.append(hist[-1])
real_mean = np.concatenate(real_mean)
sep = np.concatenate(sep)
d0 = np.concatenate(d0)
N = len(real_mean)
methods = {f"idea{n}": np.concatenate(gen[n], axis=2).mean(axis=(0, 1)) for n in (11, 12, 13, 14)}

overall_real_mean = float(real_mean.mean())

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
        "3.0-3.5 (near boundary)": (d0 > 3.0) & (d0 <= 3.5),
        "<=3.0 (elsewhere)": d0 <= 3.0,
    },
    "a_origin_fine": {
        "2.0-2.5": (d0 > 2.0) & (d0 <= 2.5),
        "2.5-3.0": (d0 > 2.5) & (d0 <= 3.0),
        "3.0-3.5": (d0 > 3.0) & (d0 <= 3.5),
    },
    "b_real_mean_deviation": {
        "<0.25": np.abs(real_mean - overall_real_mean) < 0.25,
        "0.25-0.5": (np.abs(real_mean - overall_real_mean) >= 0.25) & (np.abs(real_mean - overall_real_mean) < 0.5),
        "0.5-1.0": (np.abs(real_mean - overall_real_mean) >= 0.5) & (np.abs(real_mean - overall_real_mean) < 1.0),
        ">=1.0": np.abs(real_mean - overall_real_mean) >= 1.0,
    },
    "c_separation": {
        "1 (adjacent residues)": sep == 1,
        "2-5": (sep >= 2) & (sep <= 5),
        "6-10": (sep >= 6) & (sep <= 10),
        "11-20": (sep >= 11) & (sep <= 20),
        "21+": sep >= 21,
    },
}
res = {"n_candidates": int(N), "overall_real_mean": overall_real_mean,
       "real_mean_std": float(real_mean.std()), "methods": {}}
for name, gm in methods.items():
    err = gm - real_mean
    res["methods"][name] = {
        "overall": group_stats(err, real_mean, np.ones(N, dtype=bool)),
        **{gname: {g: group_stats(err, real_mean, m) for g, m in gdefs.items()}
           for gname, gdefs in groups.items()}}
    print(name, "overall bias", round(res["methods"][name]["overall"]["bias"], 4),
          "std", round(res["methods"][name]["overall"]["std"], 4), flush=True)
for gname, gdefs in groups.items():
    print(gname)
    for g, m in gdefs.items():
        s = res["methods"]["idea14"][gname][g]
        print(f"  {g:24s} n={s['n']:6d} bias={s['bias']:+.3f} std={s['std']:.3f} mae={s['mae']:.3f} real_std={s['real_mean_std']:.3f}")
Path(__file__).with_name("k67_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k67_results.json")
