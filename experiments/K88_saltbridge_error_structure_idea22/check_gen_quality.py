"""K88 quality check (K20/K76-style series stats, bound 4.0): is the
idea-22 generated series for the salt-bridge formation candidates
sane against the real forward continuation? Series are the 5-point
series (origin + 0.1/0.3/0.5/1.0 ns) from K14's k14_salt.npz:
gen22, coarse, rep5 (real representative pair), min5 (real minimum).
Candidates: all 367 original-6 formation candidates (all feedable).
"""
import json
from pathlib import Path
import numpy as np

K14 = Path("/home/hatch/workspace/kinescope/repo/experiments/K14_other_phenomena_generated/k14_salt.npz")
OUT = Path(__file__).resolve().parent


def series_stats(s, bound=4.0):
    """s: (N, 5) series. Returns the K20-style summary."""
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
    z = np.load(K14)
    res = {
        "n_candidates": int(len(z["gen"])),
        "real_min5": series_stats(z["min"].astype(np.float64)),
        "real_rep5": series_stats(z["rep"].astype(np.float64)),
        "gen22": series_stats(z["gen"].astype(np.float64)),
        "coarse": series_stats(z["coarse"].astype(np.float64)),
    }
    for name, s in res.items():
        if isinstance(s, dict):
            print("%-12s n=%5d mean=%.3f pp_std=%.3f range=%.3f incr_std=%.3f lag1=%.3f "
                  "below=%.3f ever=%.3f crossings=%.3f dwell=%.2f" % (
                      name, s["n"], s["mean"], s["per_pair_std"], s["range"],
                      s["increment_std"], s["lag1"], s["frac_below"], s["ever_below"],
                      s["crossings_per_pair"], s["dwell"]))
    (OUT / "gen_quality.json").write_text(json.dumps(res, indent=1))
    print("WROTE gen_quality.json")


if __name__ == "__main__":
    main()
