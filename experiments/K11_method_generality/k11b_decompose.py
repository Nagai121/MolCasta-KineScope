"""K11b: decompose the error of the mean and below-3.5-A fraction statistics.

For each candidate, the mean statistic (per-pair mean distance over the
series) and the below-3.5-A fraction are computed for the generated series
and for the real series, and the error (generated - real) is split into a
systematic shift (mean of the error) and scatter (standard deviation of
the error), with RMSE^2 = bias^2 + scatter^2.

Definitions (fixed before running, same statistic shapes as K09/K11):
- Series are taken as in K11: origin distance prepended. Real and proposal
  22 are 5-point series (origin + 0.1/0.3/0.5/1.0 ns forward continuation).
  Idea 02 is origin + 8 recorded points x 3 replicas, ideas 11-14 are
  origin + 8 points x 4 branches, idea 08 pools every rotamer path of
  either endpoint atom (origin + 8 steps per path; pairs with no path keep
  the origin distance, as in the original implementation).
- "Real" here is the FORWARD continuation, as in K05/K11. Note this differs
  from K09, where the real 5 points were the BACKWARD history
  (frames [0,1,3,5,10]); K09's mean statistic is therefore a history mean,
  while K11b's is a forward-series mean. The two are not the same quantity.
- The drift check uses future points only: per candidate, (mean of future
  distances) - d0, for real and for each generated method.
- Grid caveat: generated series have 9 points (origin weight 1/9) against
  real 5 points (origin weight 1/5). Since the origin distance is shared,
  this mechanically pulls generated means toward d0 by about +0.01 A and
  scales fractions by 8/9 vs 4/5. Future-only versions of both statistics
  are computed alongside to show this does not drive the result.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K05_NPZ = HERE.parent / "K05_generated_quality" / "generated_series.npz"
SERIES_DIR = HERE / "series"
BOUND = 3.5


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std()))


def stats_1d(future):
    """future: (N, R, T-1) distances, or (N, R, T) with origin prepended.

    Returns per-candidate mean and below-3.5 fraction over all points.
    future must already include the origin as its first point."""
    flat = future.reshape(len(future), -1)
    mean = np.nanmean(flat, axis=1)
    frac = np.nanmean((flat <= BOUND).astype(np.float64), axis=1)
    return mean, frac


def with_origin(future, d0):
    N, R, Tm1 = future.shape
    out = np.empty((N, R, Tm1 + 1), dtype=np.float64)
    out[:, :, 0] = d0[:, None]
    out[:, :, 1:] = future
    return out


def decompose(err):
    bias = float(err.mean())
    scatter = float(err.std())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"bias": bias, "scatter_std": scatter, "rmse": rmse,
            "bias_share_of_mse": float(bias ** 2 / rmse ** 2) if rmse > 0 else 0.0}


def main():
    with np.load(K05_NPZ) as z:
        tids, pair = z["tid"].copy(), z["pair"].copy()
        gen, coarse, real = z["gen"].copy(), z["coarse"].copy(), z["real"].copy()
    d0 = pair[:, 2].astype(np.float64)
    N = len(pair)

    real5 = with_origin(real[:, None, :], d0)
    real_mean, real_frac = stats_1d(real5)
    real_fut_mean = np.mean(real, axis=1)
    real_drift = real_fut_mean - d0

    methods = {}
    methods["real_5pt"] = {"mean_statistic": float(real_mean.mean()),
                           "frac_below": float(real_frac.mean()),
                           "future_mean": float(real_fut_mean.mean()),
                           "drift_mean": float(real_drift.mean()),
                           "drift_std": float(real_drift.std())}

    # Assemble K11 future arrays in K05 row order.
    store = {k: [] for k in ("heavy", "elas11", "elas12", "elas13", "elas14")}
    rot_series = np.full((N, 18, 9), np.nan)
    any_path = np.zeros(N, dtype=bool)
    for tid in sorted(set(tids.tolist())):
        mask = tids == tid
        idx = np.flatnonzero(mask)
        with np.load(SERIES_DIR / f"{tid}.npz") as z:
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

    series = {"gen22": with_origin(gen[:, None, :], d0),
              "idea02_heavy_langevin": with_origin(assemble("heavy"), d0),
              "idea11_elastic_modes": with_origin(assemble("elas11"), d0),
              "idea12_elastic_focused": with_origin(assemble("elas12"), d0),
              "idea13_local_fluctuation": with_origin(assemble("elas13"), d0),
              "idea14_elastic_plus_fast": with_origin(assemble("elas14"), d0),
              "idea08_rotamer_all_pairs": rot_series}
    for label, s in series.items():
        mean, frac = stats_1d(s)
        fut_mean = np.nanmean(s[:, :, 1:], axis=(1, 2))
        drift = fut_mean - d0
        out = {
            "mean_statistic": float(mean.mean()),
            "frac_below": float(frac.mean()),
            "mean_error": decompose(mean - real_mean),
            "frac_error_pp": decompose((frac - real_frac) * 100.0),
            "corr_mean_with_real": corr(mean, real_mean),
            "corr_frac_with_real": corr(frac, real_frac),
            "drift_mean": float(drift.mean()),
            "drift_std": float(drift.std()),
            "corr_drift_with_real": corr(drift, real_drift),
            "future_only": {
                "mean_error": decompose(fut_mean - real_fut_mean),
                "frac_error_pp": decompose(
                    (np.nanmean((s[:, :, 1:] <= BOUND).astype(float), axis=(1, 2)) - np.mean(real <= BOUND, axis=1)) * 100.0),
            },
        }
        methods[label] = out

    # Rotamer, pairs with a path only (same quantities, subset rows).
    sub = rot_series[any_path]
    mean, frac = stats_1d(sub)
    fut_mean = np.nanmean(sub[:, :, 1:], axis=(1, 2))
    drift = fut_mean - d0[any_path]
    methods["idea08_rotamer_pairs_with_path"] = {
        "frac_of_all_pairs": float(any_path.mean()),
        "mean_statistic": float(mean.mean()),
        "frac_below": float(frac.mean()),
        "mean_error": decompose(mean - real_mean[any_path]),
        "frac_error_pp": decompose((frac - real_frac[any_path]) * 100.0),
        "corr_mean_with_real": corr(mean, real_mean[any_path]),
        "corr_frac_with_real": corr(frac, real_frac[any_path]),
        "drift_mean": float(drift.mean()),
        "drift_std": float(drift.std()),
        "corr_drift_with_real": corr(drift, real_drift[any_path]),
        "future_only": {
            "mean_error": decompose(fut_mean - real_fut_mean[any_path]),
            "frac_error_pp": decompose(
                (np.nanmean((sub[:, :, 1:] <= BOUND).astype(float), axis=(1, 2)) - np.mean(real[any_path] <= BOUND, axis=1)) * 100.0),
        },
    }

    out = {"n_pairs": int(N), "bound_A": BOUND, "methods": methods,
           "reading_rule": "If the error is mostly scatter, the generator fails to capture per-pair differences. If it is mostly bias, the generator output is shifted as a whole."}
    (HERE / "k11b_results.json").write_text(json.dumps(out, indent=1))
    for name, m in methods.items():
        if "mean_error" not in m:
            continue
        e = m["mean_error"]
        f = m["frac_error_pp"]
        print("%-34s mean err bias %+.3f scatter %.3f (bias %.0f%%) r=%.3f | frac err bias %+.2fpp scatter %.2fpp (bias %.0f%%) r=%.3f | drift %+.3f (real +0.136) r_drift %.3f" % (
            name, e["bias"], e["scatter_std"], 100 * e["bias_share_of_mse"], m["corr_mean_with_real"],
            f["bias"], f["scatter_std"], 100 * f["bias_share_of_mse"], m["corr_frac_with_real"],
            m["drift_mean"], m["corr_drift_with_real"]))


if __name__ == "__main__":
    main()
