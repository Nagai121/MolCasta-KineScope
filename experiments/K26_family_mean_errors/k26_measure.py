"""K26: per-method mean errors on the K23 yardstick.

Same candidates as K05/K08/K23 (14,582). Real reference is K23's:
mean of the 11-point real history series (0-1 ns), from the base cache.
Generated means follow K11's convention: for each method, the series
block is (branches x (origin + steps)) and the generated mean is the
mean over the whole block.
  idea02 heavy : K11 series 'heavy'  (3 replicas x 8 steps)
  idea11-14    : K11 series 'elas11'..'elas14' (4 branches x 8 steps)
  idea08 rot   : K11's rotamer assembly (donor-side + acceptor-side
                 paths as branches; candidates without any path get a
                 constant series at the origin distance, as in K11)
  idea22       : K05 generated_series.npz 'gen' (single path, 4 steps);
                 'coarse' (elastic only) included as a reference row.
Errors: gen mean - real mean per candidate; report bias, std, MAE and
correlation with the real mean, pooled and by train/test split.
Yardsticks from K23: replica pairwise-diff std 0.249, static-feature
prediction residual std 0.492."""
import json
from pathlib import Path
import numpy as np

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
K05_NPZ = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
K11_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K11_method_generality/series")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")

# real means in K05 row order, keyed like K23
with np.load(K05_NPZ) as z:
    tids, pair = z["tid"].copy(), z["pair"].copy()
    gen22, coarse = z["gen"].copy(), z["coarse"].copy()
real_mean_by_key = {}
split_by_key = {}
for p in ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A"):
    for r in (1, 2, 3):
        with np.load(CACHE / f"{p}_R{r}.npz") as zz:
            hist = zz["hist"]; da = zz["donor_atom"]; aa = zz["acceptor_atom"]
        tid = f"atlas-{p}-analysis-R{r}"
        rm = hist.mean(axis=0)
        for i in range(len(rm)):
            real_mean_by_key[(tid, int(da[i]), int(aa[i]))] = float(rm[i])
            split_by_key[(tid, int(da[i]), int(aa[i]))] = "train" if p in TRAIN_P else "test"
keys = [(str(t), int(pair[i, 0]), int(pair[i, 1])) for i, t in enumerate(tids)]
real_mean = np.array([real_mean_by_key[k] for k in keys])
split = np.array([split_by_key[k] for k in keys])
d0 = pair[:, 2].astype(np.float64)
N = len(pair)

def with_origin(arr, d0):
    """arr: (N, B, S) -> (N, B, S+1) with origin prepended per branch."""
    out = np.empty(arr.shape[:2] + (arr.shape[2] + 1,), dtype=np.float64)
    out[:, :, 0] = d0[:, None]
    out[:, :, 1:] = arr
    return out

def block_mean(arr_with_origin):
    return arr_with_origin.reshape(len(arr_with_origin), -1).mean(axis=1)

methods = {}
methods["idea22"] = block_mean(with_origin(gen22[:, None, :].astype(np.float64), d0))
methods["idea22_elastic_only"] = block_mean(with_origin(coarse[:, None, :].astype(np.float64), d0))

store = {k: np.empty((N,) + s, dtype=np.float64) for k, s in
         (("heavy", (3, 8)), ("elas11", (4, 8)), ("elas12", (4, 8)), ("elas13", (4, 8)), ("elas14", (4, 8)))}
rot_series = np.full((N, 18, 9), np.nan)
any_path = np.zeros(N, dtype=bool)
for tid in sorted(set(tids.tolist())):
    idx = np.flatnonzero(tids == tid)
    with np.load(K11_DIR / f"{tid}.npz") as z:
        assert np.array_equal(z["pair"], pair[tids == tid]), tid
        for k in store:
            store[k][idx] = z[k].astype(np.float64)
        d_ab, d_ba = z["rot_ab"].astype(np.float64), z["rot_ba"].astype(np.float64)
        count_a, count_b = z["count_a"], z["count_b"]
    for j, gi in enumerate(idx):
        branches = [d_ab[j, p] for p in range(int(count_a[j]))] + \
                   [d_ba[j, p] for p in range(int(count_b[j]))]
        if branches:
            any_path[gi] = True
            arr = np.stack(branches)
        else:
            arr = np.repeat(d0[gi], 8)[None, :]
        full = np.empty((arr.shape[0], 9))
        full[:, 0] = d0[gi]
        full[:, 1:] = arr
        rot_series[gi, :arr.shape[0]] = full

methods["idea02_heavy"] = block_mean(with_origin(store["heavy"], d0))
for n in (11, 12, 13, 14):
    methods[f"idea{n}"] = block_mean(with_origin(store[f"elas{n}"], d0))
methods["idea08_rotamer"] = np.nanmean(rot_series.reshape(N, -1), axis=1)

def err_stats(gm, mask=None):
    e = (gm - real_mean) if mask is None else (gm[mask] - real_mean[mask])
    g = gm if mask is None else gm[mask]
    r = real_mean if mask is None else real_mean[mask]
    return dict(n=int(len(e)), bias=float(e.mean()), std=float(e.std()),
                mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                corr=float(np.corrcoef(g, r)[0, 1]))

res = {"n_candidates": int(N),
       "yardsticks_from_K23": {"replica_pairwise_diff_std": 0.2486,
                               "static_prediction_resid_std_test": 0.4921,
                               "idea22_gen_error_std": 0.4830},
       "methods": {}}
for name, gm in methods.items():
    entry = {"all": err_stats(gm), "train": err_stats(gm, split == "train"),
             "test": err_stats(gm, split == "test")}
    if name == "idea08_rotamer":
        entry["frac_candidates_with_any_path"] = float(any_path.mean())
        entry["with_paths_only"] = err_stats(gm, any_path)
        entry["without_paths_only"] = err_stats(gm, ~any_path)
    res["methods"][name] = entry
    print(name, "std(all)", round(entry["all"]["std"], 4), "bias", round(entry["all"]["bias"], 4),
          "corr", round(entry["all"]["corr"], 3), flush=True)
Path("/home/hatch/workspace/k26/k26_results.json").write_text(json.dumps(res, indent=1))
print("WROTE k26_results.json")
