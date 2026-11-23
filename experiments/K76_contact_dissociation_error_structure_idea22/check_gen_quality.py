"""K76 quality check (K20/K58-style series stats, bound 4.5): is the
idea-22 generated series for the feedable contact-dissociation
candidates sane against the real forward continuation?
Series are the 5-point series (origin + 0.1/0.3/0.5/1.0 ns) from
gen_k76.py's output: gen22, coarse, rep5 (real representative pair),
min5 (real minimum). Candidates: the feedable subset only (3,164).
"""
import json
from pathlib import Path
import numpy as np

GEN_DIR = Path("/home/hatch/workspace/kinescope/work/k76_gen22")
OUT = Path(__file__).resolve().parent
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")


def series_stats(s, bound=4.5):
    """s: (N, 5) series. Returns the K20-style summary."""
    s = np.asarray(s, dtype=np.float64)
    d = np.diff(s, axis=1)
    lag1 = np.corrcoef(s[:, :-1].ravel(), s[:, 1:].ravel())[0, 1] if s.shape[0] > 1 else None
    below = s <= bound
    crossings = (below[:, 1:] != below[:, :-1]).sum(axis=1)
    # dwell: mean run length of the below-bound state
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
    gen22, coarse, rep5, min5 = [], [], [], []
    for p in PROTEINS:
        for r in (1, 2, 3):
            z = np.load(GEN_DIR / f"atlas-{p}-analysis-R{r}.npz")
            if len(z["res_i"]) == 0:
                continue
            gen22.append(z["gen22"].astype(np.float64))
            coarse.append(z["coarse"].astype(np.float64))
            rep5.append(z["rep5"].astype(np.float64))
            min5.append(z["min5"].astype(np.float64))
    res = {
        "n_candidates": int(sum(len(a) for a in gen22)),
        "real_min5": series_stats(np.concatenate(min5)),
        "real_rep5": series_stats(np.concatenate(rep5)),
        "gen22": series_stats(np.concatenate(gen22)),
        "coarse": series_stats(np.concatenate(coarse)),
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
