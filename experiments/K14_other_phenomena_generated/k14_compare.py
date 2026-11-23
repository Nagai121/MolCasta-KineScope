"""K14 comparison: K05's series_stats (copied unchanged from
experiments/K05_generated_quality/measure_real_baseline.py) applied with
each phenomenon's boundary (salt 4.0 A, contact 4.5 A) to the generated,
elastic-only, real representative-pair and real minimum-distance series,
all on the 5-point grid (origin + 0.1/0.3/0.5/1.0 ns)."""
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

OUT = Path("/home/hatch/workspace/k14/out")
meta = json.loads((OUT / "k14_meta.json").read_text())
result = {"meta": meta, "phenomena": {}}
for phen, bound in (("salt", 4.0), ("contact", 4.5)):
    with np.load(OUT / f"k14_{phen}.npz") as z:
        d = {kk: z[kk].astype(np.float64) for kk in z.files}
    entry = {"bound": bound,
             "origin_min_distance": {
                 "mean": float(d["min_all"][:, 0].mean()),
                 "frac_above_6A_outside_hb_training_range": float((d["min_all"][:, 0] > 6.0).mean()),
                 "q10_q50_q90": [float(v) for v in np.quantile(d["min_all"][:, 0], [0.1, 0.5, 0.9])]},
             "feedable_subset": {
                 "generated": series_stats(d["gen"].T, bound),
                 "elastic_only": series_stats(d["coarse"].T, bound),
                 "real_representative_pair": series_stats(d["rep"].T, bound),
                 "real_minimum": series_stats(d["min"].T, bound)},
             "all_candidates": {"real_minimum": series_stats(d["min_all"].T, bound)},
             "at_1ns_frac_below_bound": {
                 "generated": float((d["gen"][:, -1] <= bound).mean()),
                 "elastic_only": float((d["coarse"][:, -1] <= bound).mean()),
                 "real_representative_pair": float((d["rep"][:, -1] <= bound).mean()),
                 "real_minimum": float((d["min"][:, -1] <= bound).mean()),
                 "real_minimum_all": float((d["min_all"][:, -1] <= bound).mean())}}
    result["phenomena"][phen] = entry
    print(f"== {phen} (bound {bound}) candidates {meta[phen]['candidates']} feedable {meta[phen]['feedable']} "
          f"nonfinite {meta[phen]['nonfinite']} clipfrac {meta[phen]['clipfrac_mean']}")
    for group in ("feedable_subset",):
        for name, s in entry[group].items():
            print(f"  {name:26s} mean {s['dist_mean']:.3f} std {s['dist_std']:.3f} | pp-std {s['per_pair_std_mean']:.3f} "
                  f"range {s['per_pair_range_mean']:.3f} | incr {s['increment_std']:.3f} | lag1 {s['lag1_autocorr_mean']:.3f} | "
                  f"below {s['frac_frames_below']:.3f} ever {s['frac_pairs_ever_below']:.3f} cross {s['crossings_per_pair_mean']:.3f} "
                  f"dwell {s['dwell_below_frames_mean']}")
    s = entry["all_candidates"]["real_minimum"]
    print(f"  {'real_minimum (ALL)':26s} mean {s['dist_mean']:.3f} std {s['dist_std']:.3f} | pp-std {s['per_pair_std_mean']:.3f} "
          f"range {s['per_pair_range_mean']:.3f} | incr {s['increment_std']:.3f} | lag1 {s['lag1_autocorr_mean']:.3f} | "
          f"below {s['frac_frames_below']:.3f} ever {s['frac_pairs_ever_below']:.3f} cross {s['crossings_per_pair_mean']:.3f} dwell {s['dwell_below_frames_mean']}")
    print("  at 1ns frac below:", {k: round(v, 4) for k, v in entry["at_1ns_frac_below_bound"].items()})
    print("  origin min dist:", entry["origin_min_distance"])
(OUT / "k14_comparison.json").write_text(json.dumps(result, indent=1))
print("WROTE k14_comparison.json")
