"""K60: which protein drives the spread of band-B mean-only recovery?
(K55/K57 follow-up; HB formation, D conditions).

Inputs (no retraining): K57's results.json (per-protein gains/recoveries,
recomputed from K55's training) + the D-condition cache
~/workspace/kinescope/work/data_cache/ + the converted trajectories
~/workspace/gpt-model-check/data/trajectories/ for protein properties.

Judgment rule (fixed BEFORE computing the protein properties below;
the per-protein recovery values themselves are already published in K57):
- "A specific protein drives the spread" if, in band B, the two proteins'
  recovery range is clearly larger than in band A in most horizons
  (>= 4 of 6) AND the same protein is the higher (or lower) one in most
  horizons (>= 4 of 6).
- Otherwise (range comparable to band A, or the higher/lower protein flips
  across horizons), the spread is not attributable to one protein: the
  mean carrier is not stable in band B.
Note: the gain sign itself is also compared (a protein whose gain is
negative while the other's is positive is a protein-level difference in
guidance, distinct from the recovery-ratio spread).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import json
import numpy as np
import pandas as pd
import zarr
from pathlib import Path

import minitopol  # noqa: F401  (registers the minitopol_agx3 zarr codec)

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
TRAJ = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
K57_RESULTS = Path(__file__).parent.parent / "K57_bandB_gain_size" / "results.json"
OUT = Path(__file__).parent
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BANDS = {"A_3.5-4.5": (3.5, 4.5), "B_4.5-6.0": (4.5, 6.0)}


def mic_step_disp(pos, box):
    """Mean per-step C-alpha displacement (A) over frames 0..10, MIC-aware.

    pos: (11, n_atoms, 3), box: (11, 3, 3). Consecutive-frame displacement
    vectors are wrapped with the per-frame box (diagonal here, as in K53).
    """
    d = pos[1:] - pos[:-1]  # (10, n, 3)
    lengths = np.diagonal(box[:-1], axis1=1, axis2=2)  # (10, 3)
    d = d - lengths[:, None, :] * np.rint(d / lengths[:, None, :])
    return float(np.linalg.norm(d, axis=-1).mean())


def load_protein_cache(p):
    parts = []
    for r in (1, 2, 3):
        z = np.load(CACHE / f"{p}_R{r}.npz")
        parts.append({k: z[k] for k in z.files})
    X = np.concatenate([q["X"] for q in parts])
    H = np.concatenate([q["hist"].T for q in parts])        # (n, 11) pair dist
    Hca = np.concatenate([q["hist_ca"].T for q in parts])   # (n, 11) CA dist
    Y = np.concatenate([q["y"] for q in parts])
    return X, H, Hca, Y


def ca_step_disp(p):
    """Mean per-step C-alpha displacement over frames 0..10, all 3 replicas."""
    vals = []
    df = pd.read_parquet(TRAJ / f"atlas-{p}-analysis-R1" / "atoms.parquet")
    ca_idx = np.flatnonzero(df["name"].values == "CA")
    n_res = int(df["residue_index"].nunique())
    for r in (1, 2, 3):
        z = zarr.open_group(str(TRAJ / f"atlas-{p}-analysis-R{r}" / "A.zarr"), mode="r")
        pos = z["positions"][0:11].astype(np.float64)[:, ca_idx, :]
        box = z["box"][0:11].astype(np.float64)
        vals.append(mic_step_disp(pos, box))
    return float(np.mean(vals)), n_res


def main():
    k57 = json.loads(K57_RESULTS.read_text())
    results = {"bands": {}, "protein_properties": {}, "judgment": {}}

    for p in TEST_P:
        ca_disp, n_res = ca_step_disp(p)
        results["protein_properties"][p] = {
            "n_residues": n_res,
            "ca_per_step_displacement_A_frames0_10": round(ca_disp, 4),
        }
        print(p, "residues", n_res, "CA per-step disp", round(ca_disp, 4), flush=True)

    for band, (lo, hi) in BANDS.items():
        results["bands"][band] = {}
        for p in TEST_P:
            X, H, Hca, Y = load_protein_cache(p)
            m = (X[:, 0] > lo) & (X[:, 0] <= hi)
            Xb, Hb, Hcab, Yb = X[m], H[m], Hca[m], Y[m]
            d0 = Xb[:, 0].astype(np.float64)
            rec = k57["bands"][band]["recovery"]  # keyed by horizon str
            props = {
                "n_candidates": int(m.sum()),
                "positives_per_horizon": {str(h): int(Yb[:, i].sum())
                                          for i, h in enumerate(HORIZONS)},
                "origin_distance": {
                    "mean": round(float(d0.mean()), 4),
                    "median": round(float(np.median(d0)), 4),
                    "q1": round(float(np.percentile(d0, 25)), 4),
                    "q3": round(float(np.percentile(d0, 75)), 4),
                    "min": round(float(d0.min()), 4),
                    "max": round(float(d0.max()), 4),
                },
                # history fluctuation (11-frame series, per candidate)
                "hist_pair_std_mean": round(float(Hb.std(axis=1).mean()), 4),
                "hist_pair_range_mean": round(float((Hb.max(axis=1) - Hb.min(axis=1)).mean()), 4),
                "hist_pair_abs_last_minus_first_mean": round(float(np.abs(Hb[:, -1] - Hb[:, 0]).mean()), 4),
                "hist_ca_std_mean": round(float(Hcab.std(axis=1).mean()), 4),
                "hist_ca_range_mean": round(float((Hcab.max(axis=1) - Hcab.min(axis=1)).mean()), 4),
                "gain_per_protein": {str(h): rec[str(h)]["per_protein"][p]["gain"] for h in HORIZONS},
                "recovery_per_protein": {str(h): rec[str(h)]["per_protein"][p]["recovery"] for h in HORIZONS},
                "auprc_static": {str(h): k57["bands"][band]["static_only"]["per_protein"][p][str(h)]["auprc"] for h in HORIZONS},
                "auprc_full": {str(h): k57["bands"][band]["full_hist6"]["per_protein"][p][str(h)]["auprc"] for h in HORIZONS},
                "auprc_mean": {str(h): k57["bands"][band]["mean_only"]["per_protein"][p][str(h)]["auprc"] for h in HORIZONS},
            }
            props["ca_per_step_displacement_A_frames0_10"] = results["protein_properties"][p]["ca_per_step_displacement_A_frames0_10"]
            props["n_residues"] = results["protein_properties"][p]["n_residues"]
            results["bands"][band][p] = props
            print(band, p, "n", props["n_candidates"],
                  "d0 mean", props["origin_distance"]["mean"],
                  "hist std", props["hist_pair_std_mean"],
                  "CA disp", props["ca_per_step_displacement_A_frames0_10"], flush=True)

    # highest / lowest protein per horizon, and range comparison B vs A
    for band in BANDS:
        rec = k57["bands"][band]["recovery"]  # keyed by horizon str
        hi_lo = {}
        for h in HORIZONS:
            r1 = rec[str(h)]["per_protein"]["1mix_A"]["recovery"]
            r2 = rec[str(h)]["per_protein"]["2qt4_A"]["recovery"]
            hi_lo[str(h)] = {
                "higher": "1mix_A" if r1 > r2 else "2qt4_A",
                "lower": "2qt4_A" if r1 > r2 else "1mix_A",
                "recovery_1mix_A": r1, "recovery_2qt4_A": r2,
                "range": abs(r1 - r2),
            }
        results["bands"][band]["higher_lower_per_horizon"] = hi_lo

    ranges = {b: [results["bands"][b]["higher_lower_per_horizon"][str(h)]["range"]
                  for h in HORIZONS] for b in BANDS}
    n_b_larger = sum(1 for a, b in zip(ranges["A_3.5-4.5"], ranges["B_4.5-6.0"]) if b > a)
    higher_b = [results["bands"]["B_4.5-6.0"]["higher_lower_per_horizon"][str(h)]["higher"]
                for h in HORIZONS]
    from collections import Counter
    cnt = Counter(higher_b)
    results["judgment"] = {
        "range_median_A": round(float(np.median(ranges["A_3.5-4.5"])), 4),
        "range_median_B": round(float(np.median(ranges["B_4.5-6.0"])), 4),
        "n_horizons_B_range_larger_than_A": int(n_b_larger),
        "higher_protein_counts_bandB": dict(cnt),
        "same_protein_higher_in_at_least_4_of_6": bool(max(cnt.values()) >= 4),
        "range_larger_in_at_least_4_of_6": bool(n_b_larger >= 4),
    }
    print("judgment:", results["judgment"])

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    rows = []
    for band in BANDS:
        for h in HORIZONS:
            hl = results["bands"][band]["higher_lower_per_horizon"][str(h)]
            rows.append({
                "band": band, "horizon_ns": h,
                "recovery_1mix_A": round(hl["recovery_1mix_A"], 4),
                "recovery_2qt4_A": round(hl["recovery_2qt4_A"], 4),
                "range": round(hl["range"], 4),
                "higher": hl["higher"], "lower": hl["lower"],
                "gain_1mix_A": round(results["bands"][band]["1mix_A"]["gain_per_protein"][str(h)], 4),
                "gain_2qt4_A": round(results["bands"][band]["2qt4_A"]["gain_per_protein"][str(h)], 4),
            })
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    print("saved", OUT / "results.json", OUT / "results.csv")


if __name__ == "__main__":
    main()
