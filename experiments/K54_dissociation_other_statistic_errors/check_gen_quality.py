"""K54 generation quality check (K05-style) for the DISSOCIATION
candidates only.

The network was trained on formation pairs, so the generated series
for dissociation candidates (origin <= 3.5 A) are an extrapolation.
Before the swap evaluation in run_k54.py, check whether they are
sane against the real forward continuation at the same 4 times
(0.1/0.3/0.5/1.0 ns), using the same quantities as K05:
per-pair std, mean distance, fraction of frames <= 3.5 A, fraction of
pairs ever <= 3.5 A, boundary crossings per pair, and drift
(mean of last point minus origin).
"""
import json
from pathlib import Path

import numpy as np

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
GEN_DIR = Path("/home/hatch/workspace/kinescope/work/k54_gen22")
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")


def stats(series):
    """series: (N, 5) with origin in column 0."""
    per_std = series.std(axis=1)
    per_range = series.max(axis=1) - series.min(axis=1)
    frac = (series <= 3.5).mean()
    ever = (series <= 3.5).any(axis=1).mean()
    below = series <= 3.5
    crossings = (np.diff(below.astype(int), axis=1) != 0).sum(axis=1).mean()
    drift = (series[:, -1] - series[:, 0]).mean()
    return dict(n=int(len(series)), mean=float(series.mean()), std=float(series.std()),
                per_pair_std=float(per_std.mean()), per_pair_range=float(per_range.mean()),
                frac_le35=float(frac), ever_le35=float(ever),
                crossings_per_pair=float(crossings), drift=float(drift))


def main():
    gen_all, real_all = [], []
    for p in PROTEINS:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            g = np.load(GEN_DIR / f"atlas-{p}-analysis-R{r}.npz")
            key = {(int(pp[0]), int(pp[1])): k for k, pp in enumerate(g["pair"])}
            idx = np.array([key[(int(u), int(v))] for u, v in zip(z["donor_atom"], z["acceptor_atom"])])
            assert len(idx) == len(z["donor_atom"])
            origin = z["hist"][-1]
            gen_all.append(np.column_stack([origin, g["gen_dist"][idx]]))
            real_all.append(np.column_stack([origin, g["real_sub"][idx]]))
    gen = np.concatenate(gen_all)
    real = np.concatenate(real_all)
    out = {"dissociation_candidates": int(len(gen)),
           "generated": stats(gen), "real_forward": stats(real)}
    print(json.dumps(out, indent=2))
    Path(__file__).with_name("gen_quality.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
