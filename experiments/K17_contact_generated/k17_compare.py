"""K17 shape comparison: K05's series_stats (copied unchanged from
experiments/K05_generated_quality/measure_real_baseline.py) with the
contact boundary 4.5 A. Generated series are origin + 8 steps, one
series per branch (4 branches pooled as separate series); the real
series is the forward 5-point grid (origin + 0.1/0.3/0.5/1.0 ns).
Point counts differ, so crossings are also read per interval."""
import json
from pathlib import Path
import numpy as np

def series_stats(series, bound=3.5):
    T, N = series.shape
    below = series <= bound
    x = series - series.mean(axis=0, keepdims=True)
    var = (x * x).sum(axis=0)
    ok = var > 1e-12
    lag1 = np.full(N, np.nan)
    if ok.any():
        lag1[ok] = (x[:-1, ok] * x[1:, ok]).sum(axis=0) / var[ok]
    crossings = (np.diff(below.astype(np.int8), axis=0) != 0).sum(axis=0)
    dwell = []
    for j in range(N):
        col = below[:, j]
        if not col.any():
            continue
        runs, cur = [], 0
        for v in col:
            if v: cur += 1
            elif cur: runs.append(cur); cur = 0
        if cur: runs.append(cur)
        dwell.append(np.mean(runs))
    incr = np.diff(series, axis=0)
    return {
        "n_pairs": int(N), "n_frames": int(T),
        "dist_mean": float(series.mean()), "dist_std": float(series.std()),
        "dist_q10_q50_q90": [float(v) for v in np.quantile(series, [0.1, 0.5, 0.9])],
        "per_pair_std_mean": float(series.std(axis=0).mean()),
        "per_pair_range_mean": float((series.max(axis=0) - series.min(axis=0)).mean()),
        "lag1_autocorr_mean": float(np.nanmean(lag1)) if ok.any() else None,
        "lag1_autocorr_median": float(np.nanmedian(lag1)) if ok.any() else None,
        "increment_std": float(incr.std()), "increment_abs_mean": float(np.abs(incr).mean()),
        "frac_frames_below": float(below.mean()),
        "frac_pairs_ever_below": float(below.any(axis=0).mean()),
        "crossings_per_pair_mean": float(crossings.mean()),
        "frac_pairs_with_crossing": float((crossings > 0).mean()),
        "dwell_below_frames_mean": float(np.mean(dwell)) if dwell else None,
    }

PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
OUT = Path("/home/hatch/workspace/k17/out")
BOUND = 4.5
real, gen = [], {n: [] for n in (11, 12, 13, 14)}
gen_mean_series = {n: [] for n in (11, 12, 13, 14)}
for p in PROTEINS:
    for r in (1, 2, 3):
        with np.load(OUT / f"{p}_R{r}.npz") as z:
            real.append(z["fwd5"].astype(np.float64))
            for n in (11, 12, 13, 14):
                g = z[f"gen{n}"].astype(np.float64)      # (4, 9, N)
                gen[n].append(g.transpose(1, 0, 2).reshape(9, -1))
                gen_mean_series[n].append(g.mean(axis=0))  # (9, N)
R = np.concatenate(real, axis=0).T                        # (5, N)
res = {"bound": BOUND, "n_candidates": int(R.shape[1]),
       "real_5pt": series_stats(R, BOUND), "generated": {}, "generated_branch_mean": {}}
for n in (11, 12, 13, 14):
    G = np.concatenate(gen[n], axis=1)
    res["generated"][str(n)] = series_stats(G, BOUND)
    GM = np.concatenate(gen_mean_series[n], axis=1)
    res["generated_branch_mean"][str(n)] = series_stats(GM, BOUND)
Path("/home/hatch/workspace/k17/k17_comparison.json").write_text(json.dumps(res, indent=1))
def show(name, s):
    print(f"{name:22s} mean {s['dist_mean']:.3f} | pp-std {s['per_pair_std_mean']:.3f} range {s['per_pair_range_mean']:.3f} | "
          f"incr {s['increment_std']:.3f} | lag1 {s['lag1_autocorr_mean']:.3f} | below {s['frac_frames_below']:.3f} "
          f"ever {s['frac_pairs_ever_below']:.3f} cross {s['crossings_per_pair_mean']:.3f} dwell {s['dwell_below_frames_mean']}")
show("real 5pt", res["real_5pt"])
for n in (11, 12, 13, 14):
    show(f"gen{n} (per branch)", res["generated"][str(n)])
    show(f"gen{n} (branch mean)", res["generated_branch_mean"][str(n)])
print("WROTE k17_comparison.json")
