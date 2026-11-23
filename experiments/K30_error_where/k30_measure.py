"""K30: where is the generator's mean error large?

Same candidates and definitions as K23/K26 (14,582 HB candidates,
DocD conditions). Per-candidate error = generated mean - real mean
(11-point history mean). Broken down by:
  (a) origin-distance band: near the 3.5 A boundary (3.5-4.5) vs
      elsewhere (4.5-6.0), plus finer 0.5 A bands.
  (b) size of the real mean's deviation from the overall mean of
      real means: |real_mean - mean(real_mean)| bands.
  (c) residue separation |donor_res - acceptor_res| bands.
Methods: idea22 (K05) and ideas 11-14 (K11 series), with the same
generated-mean definition as K26 (mean over the branches x
(origin + steps) block). Per group: n, error bias/std/MAE, real-mean
std and correlation, so a group's error can be read against how much
its real means spread in the first place."""
import json
from pathlib import Path
import numpy as np

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
K05_NPZ = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
K11_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K11_method_generality/series")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")

with np.load(K05_NPZ) as z:
    tids, pair = z["tid"].copy(), z["pair"].copy()
    gen22 = z["gen"].copy()
real_mean_by_key, sep_by_key = {}, {}
for p in ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A"):
    for r in (1, 2, 3):
        with np.load(CACHE / f"{p}_R{r}.npz") as zz:
            hist = zz["hist"]; da = zz["donor_atom"]; aa = zz["acceptor_atom"]
            dr, ar = zz["donor_res"], zz["acceptor_res"]
        tid = f"atlas-{p}-analysis-R{r}"
        rm = hist.mean(axis=0)
        for i in range(len(rm)):
            k = (tid, int(da[i]), int(aa[i]))
            real_mean_by_key[k] = float(rm[i])
            sep_by_key[k] = int(abs(int(dr[i]) - int(ar[i])))
keys = [(str(t), int(pair[i, 0]), int(pair[i, 1])) for i, t in enumerate(tids)]
real_mean = np.array([real_mean_by_key[k] for k in keys])
sep = np.array([sep_by_key[k] for k in keys])
d0 = pair[:, 2].astype(np.float64)
N = len(pair)

def with_origin(arr, d0):
    out = np.empty(arr.shape[:2] + (arr.shape[2] + 1,), dtype=np.float64)
    out[:, :, 0] = d0[:, None]
    out[:, :, 1:] = arr
    return out

methods = {"idea22": with_origin(gen22[:, None, :].astype(np.float64), d0).reshape(N, -1).mean(axis=1)}
store = {}
for tid in sorted(set(tids.tolist())):
    idx = np.flatnonzero(tids == tid)
    with np.load(K11_DIR / f"{tid}.npz") as z:
        assert np.array_equal(z["pair"], pair[tids == tid]), tid
        for n in (11, 12, 13, 14):
            store.setdefault(n, np.empty((N, 4, 8))).__setitem__(idx, z[f"elas{n}"].astype(np.float64))
for n in (11, 12, 13, 14):
    methods[f"idea{n}"] = with_origin(store[n], d0).reshape(N, -1).mean(axis=1)

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
        "3.5-4.5 (near boundary)": (d0 > 3.5) & (d0 <= 4.5),
        "4.5-6.0 (elsewhere)": (d0 > 4.5) & (d0 <= 6.0),
    },
    "a_origin_fine": {
        "3.5-4.0": (d0 > 3.5) & (d0 <= 4.0),
        "4.0-4.5": (d0 > 4.0) & (d0 <= 4.5),
        "4.5-5.0": (d0 > 4.5) & (d0 <= 5.0),
        "5.0-5.5": (d0 > 5.0) & (d0 <= 5.5),
        "5.5-6.0": (d0 > 5.5) & (d0 <= 6.0),
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
    print(name, "overall std", round(res["methods"][name]["overall"]["std"], 4), flush=True)
for gname, gdefs in groups.items():
    print(gname)
    for g, m in gdefs.items():
        s = res["methods"]["idea22"][gname][g]
        print(f"  {g:24s} n={s['n']:6d} bias={s['bias']:+.3f} std={s['std']:.3f} mae={s['mae']:.3f} real_std={s['real_mean_std']:.3f}")
Path("/home/hatch/workspace/k30/k30_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k30_results.json")
