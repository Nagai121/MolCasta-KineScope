"""K11 comparison: distance-series statistics for each regenerated method
vs the real trajectory, using the same statistic definitions as K05
(measure_real_baseline.series_stats), generalised to several branch
series per candidate pair (replicas / mode branches / rotamer paths).

Series layout: (N, R, T) with the origin distance prepended, so T = 9 for
the K11 methods (8 generated points) and T = 5 for the K05 reference
series (4 generated/real points). NaN-padded branches (rotamer pairs
with fewer paths) are ignored statistic by statistic.

Reported quantities (same set as K05): overall mean / std of distance,
per-pair std and range, increment std, lag-1 autocorrelation, fraction of
points at or below 3.5 A, fraction of pairs ever below, boundary
crossings per series, dwell length below 3.5 A in points, and the
per-pair mean-distance difference against the real 5-point series.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K05_NPZ = HERE.parent / "K05_generated_quality" / "generated_series.npz"
GEN_DIR = Path("/tmp/k11_out")
BOUND = 3.5


def series_stats(series):
    """series: (N, R, T) float array, NaN allowed for missing branches."""
    N, R, T = series.shape
    flat = series.reshape(N, R * T)
    with np.errstate(all="ignore"):
        per_pair_std = np.nanstd(flat, axis=1)
        per_pair_range = np.nanmax(flat, axis=1) - np.nanmin(flat, axis=1)
        per_pair_mean = np.nanmean(flat, axis=1)
    incr = np.diff(series, axis=2)
    # lag-1 autocorrelation per branch series, K05 definition
    x = series - np.nanmean(series, axis=2, keepdims=True)
    var = np.nansum(x * x, axis=2)
    ok = var > 1e-12
    lag1 = np.full((N, R), np.nan)
    if ok.any():
        num = np.nansum(x[:, :, :-1] * x[:, :, 1:], axis=2)
        lag1[ok] = num[ok] / var[ok]
    below = series <= BOUND
    valid = np.isfinite(series)
    crossings = (np.diff(below.astype(np.int8), axis=2) != 0).sum(axis=2)
    valid_branch = np.isfinite(series).any(axis=2)
    crossings = np.where(valid_branch, crossings, np.nan)
    dwell_vals = []
    for i in range(N):
        for r in range(R):
            col = below[i, r]
            if not np.isfinite(series[i, r]).all() or not col.any():
                continue
            runs, cur = [], 0
            for v in col:
                if v:
                    cur += 1
                elif cur:
                    runs.append(cur)
                    cur = 0
            if cur:
                runs.append(cur)
            dwell_vals.append(np.mean(runs))
    return {
        "n_pairs": int(N), "n_branches": int(R), "n_points": int(T),
        "dist_mean": float(np.nanmean(series)), "dist_std": float(np.nanstd(series)),
        "per_pair_std_mean": float(np.nanmean(per_pair_std)),
        "per_pair_range_mean": float(np.nanmean(per_pair_range)),
        "increment_std": float(np.nanstd(incr)),
        "lag1_autocorr_mean": float(np.nanmean(lag1)) if ok.any() else None,
        "frac_frames_below": float(np.nansum(below & valid) / np.sum(valid)),
        "frac_pairs_ever_below": float(np.nanmean(np.any(below & valid, axis=(1, 2)))),
        "crossings_per_series_mean": float(np.nanmean(crossings)),
        "dwell_below_points_mean": float(np.mean(dwell_vals)) if dwell_vals else None,
        "_per_pair_mean": per_pair_mean,
    }


def public(stats):
    return {k: v for k, v in stats.items() if not k.startswith("_")}


def with_origin(future, d0):
    """future: (N, R, T-1) -> (N, R, T) with origin prepended."""
    N, R, Tm1 = future.shape
    out = np.empty((N, R, Tm1 + 1), dtype=np.float64)
    out[:, :, 0] = d0[:, None]
    out[:, :, 1:] = future
    return out


def main():
    with np.load(K05_NPZ) as z:
        tids, pair = z["tid"].copy(), z["pair"].copy()
        gen, coarse, real = z["gen"].copy(), z["coarse"].copy(), z["real"].copy()
    d0 = pair[:, 2].astype(np.float64)
    N = len(pair)

    real5 = with_origin(real[:, None, :], d0)
    real_stats = series_stats(real5)
    real_pair_mean = real_stats["_per_pair_mean"]

    methods = {}
    methods["real_5pt"] = public(real_stats)
    methods["gen22_5pt"] = public(series_stats(with_origin(gen[:, None, :], d0)))
    methods["gen22_elastic_only_5pt"] = public(series_stats(with_origin(coarse[:, None, :], d0)))

    # Assemble K11 arrays in K05 row order, verifying pair identity per tid.
    store = {k: [] for k in ("heavy", "elas11", "elas12", "elas13", "elas14")}
    rot_series = np.full((N, 18, 9), np.nan)
    any_path = np.zeros(N, dtype=bool)
    offset = {}
    for tid in sorted(set(tids.tolist())):
        mask = tids == tid
        idx = np.flatnonzero(mask)
        with np.load(GEN_DIR / f"{tid}.npz") as z:
            assert np.array_equal(z["pair"], pair[mask]), tid
            for k in store:
                store[k].append((idx, z[k].copy()))
            d_ab, d_ba = z["rot_ab"].astype(np.float64), z["rot_ba"].astype(np.float64)
            count_a, count_b = z["count_a"], z["count_b"]
        for j, gi in enumerate(idx):
            branches = []
            for p in range(int(count_a[j])):
                branches.append(d_ab[j, p])
            for p in range(int(count_b[j])):
                branches.append(d_ba[j, p])
            if branches:
                any_path[gi] = True
                arr = np.stack(branches)
            else:
                arr = np.repeat(d0[gi], 8)[None, :]
            full = np.empty((arr.shape[0], 9))
            full[:, 0] = d0[gi]
            full[:, 1:] = arr
            rot_series[gi, :arr.shape[0]] = full

    def assemble(key):
        out = np.empty((N,) + store[key][0][1].shape[1:], dtype=np.float64)
        for idx, arr in store[key]:
            out[idx] = arr
        return out

    for key, label in (("heavy", "idea02_heavy_langevin"), ("elas11", "idea11_elastic_modes"),
                       ("elas12", "idea12_elastic_focused"), ("elas13", "idea13_local_fluctuation"),
                       ("elas14", "idea14_elastic_plus_fast")):
        s = series_stats(with_origin(assemble(key), d0))
        s["mean_diff_vs_real_per_pair"] = float(np.mean(s["_per_pair_mean"] - real_pair_mean))
        s["mean_diff_vs_real_per_pair_std"] = float(np.std(s["_per_pair_mean"] - real_pair_mean))
        methods[label] = public(s)

    # mean-difference for the K05 references as well
    for label, arr in (("gen22", gen), ("coarse", coarse)):
        s = series_stats(with_origin(arr[:, None, :], d0))
        key = "gen22_5pt" if label == "gen22" else "gen22_elastic_only_5pt"
        methods[key]["mean_diff_vs_real_per_pair"] = float(np.mean(s["_per_pair_mean"] - real_pair_mean))
        methods[key]["mean_diff_vs_real_per_pair_std"] = float(np.std(s["_per_pair_mean"] - real_pair_mean))

    rot = series_stats(rot_series)
    rot["mean_diff_vs_real_per_pair"] = float(np.mean(rot["_per_pair_mean"] - real_pair_mean))
    rot["mean_diff_vs_real_per_pair_std"] = float(np.std(rot["_per_pair_mean"] - real_pair_mean))
    methods["idea08_rotamer_all_pairs"] = public(rot)
    sub = series_stats(rot_series[any_path])
    sub_real = real_pair_mean[any_path]
    sub["mean_diff_vs_real_per_pair"] = float(np.mean(sub["_per_pair_mean"] - sub_real))
    sub["mean_diff_vs_real_per_pair_std"] = float(np.std(sub["_per_pair_mean"] - sub_real))
    methods["idea08_rotamer_pairs_with_path"] = public(sub)
    methods["idea08_rotamer_pairs_with_path"]["frac_of_all_pairs"] = float(any_path.mean())

    out = {"n_pairs": int(N), "bound_A": BOUND, "methods": methods}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    for name, s in methods.items():
        print("%-34s mean %.3f (diff %+.3f) | per-pair std %.3f | incr %.3f | lag1 %s | "
              "below %.3f ever %.3f cross %.3f dwell %s" % (
                  name, s["dist_mean"], s.get("mean_diff_vs_real_per_pair", 0.0),
                  s["per_pair_std_mean"], s["increment_std"],
                  None if s["lag1_autocorr_mean"] is None else round(s["lag1_autocorr_mean"], 3),
                  s["frac_frames_below"], s["frac_pairs_ever_below"],
                  s["crossings_per_series_mean"],
                  None if s["dwell_below_points_mean"] is None else round(s["dwell_below_points_mean"], 2)))


if __name__ == "__main__":
    main()
