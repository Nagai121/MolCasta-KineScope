"""K84 quality check (K20/K58-style series stats, bound 4.5): is the
residue-level generated series for the contact formation candidates
sane against the real forward continuation? Series are the 9-point
series (origin + 8 steps) from gen_k84.py: gen11-gen14, branch-mean
used (K48 convention), against the real forward 5-point minimum
series fwd5. Candidates: all 23,310 K13 formation candidates.
"""
import json
from pathlib import Path
import numpy as np

GEN_DIR = Path("/home/hatch/workspace/kinescope/work/k84_gen_residue")
OUT = Path(__file__).resolve().parent
PROTS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")


def series_stats(s, bound=4.5):
    """s: (N, T) series. Returns the K20-style summary."""
    s = np.asarray(s, dtype=np.float64)
    d = np.diff(s, axis=1)
    lag1 = np.corrcoef(s[:, :-1].ravel(), s[:, 1:].ravel())[0, 1] if s.shape[0] > 1 else None
    below = s <= bound
    crossings = (below[:, 1:] != below[:, :-1]).sum(axis=1)
    runs = []
    for row in below:
        cur, n = row[0], 1
        for v in row[1:]:
            if v == cur:
                n += 1
            else:
                runs.append(n); cur, n = v, 1
        runs.append(n)
    return dict(n=int(s.shape[0]), mean=float(s.mean()),
                per_pair_std=float(s.std(axis=1).mean()),
                range=float((s.max(axis=1) - s.min(axis=1)).mean()),
                increment_std=float(d.std()),
                lag1=float(lag1) if lag1 is not None else None,
                frac_below=float(below.mean()),
                ever_below=float(below.any(axis=1).mean()),
                crossings_per_pair=float(crossings.mean()),
                dwell=float(np.mean(runs)))


def main():
    fwd5, g11, g12, g13, g14 = [], [], [], [], []
    per_branch = {11: [], 12: [], 13: [], 14: []}
    for p in PROTS:
        for r in (1, 2, 3):
            z = np.load(GEN_DIR / f"{p}_R{r}.npz")
            if len(z["res_i"]) == 0:
                continue
            fwd5.append(z["fwd5"].astype(np.float64))
            for num, lst in ((11, g11), (12, g12), (13, g13), (14, g14)):
                per_branch[num].append(z[f"gen{num}"].astype(np.float64))
                lst.append(z[f"gen{num}"].mean(axis=0).T.astype(np.float64))
    res = {
        "n_candidates": int(sum(len(a) for a in fwd5)),
        "real_fwd5": series_stats(np.concatenate(fwd5)),
        "gen11_branch_mean": series_stats(np.concatenate(g11)),
        "gen12_branch_mean": series_stats(np.concatenate(g12)),
        "gen13_branch_mean": series_stats(np.concatenate(g13)),
        "gen14_branch_mean": series_stats(np.concatenate(g14)),
        "gen11_per_branch": series_stats(np.concatenate([a.reshape(-1, 9) for a in per_branch[11]])),
        "gen12_per_branch": series_stats(np.concatenate([a.reshape(-1, 9) for a in per_branch[12]])),
        "gen13_per_branch": series_stats(np.concatenate([a.reshape(-1, 9) for a in per_branch[13]])),
        "gen14_per_branch": series_stats(np.concatenate([a.reshape(-1, 9) for a in per_branch[14]])),
    }
    for name, s in res.items():
        if isinstance(s, dict):
            print("%-18s n=%5d mean=%.3f pp_std=%.3f range=%.3f incr_std=%.3f lag1=%.3f "
                  "below=%.3f ever=%.3f crossings=%.3f dwell=%.2f" % (
                      name, s["n"], s["mean"], s["per_pair_std"], s["range"],
                      s["increment_std"], s["lag1"], s["frac_below"], s["ever_below"],
                      s["crossings_per_pair"], s["dwell"]))
    (OUT / "gen_quality.json").write_text(json.dumps(res, indent=1))
    print("WROTE gen_quality.json")


if __name__ == "__main__":
    main()
