"""Recompute the real-mean spread cited in the K51 README (teacher vs
unseen DocD-6) and record it in k51_results.json under 'real_mean_spread'.

Real mean = mean of the 11-point backward history (frames 0-10), K08/K23
definition. Teacher side: K51 npz real_hist; unseen side: K08 cache hist.
This is the script the README numbers come from."""
import json
from pathlib import Path
import numpy as np

K51OUT = Path("/home/hatch/workspace/k51/out")
TEACHER = ("1u84_A", "4u5h_G", "5hub_A", "1x6j_A", "2y1y_A", "1krn_A", "2y4x_B", "4xqm_A")
DOCD = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")

def spread(values):
    v = np.concatenate(values)
    return dict(n=int(len(v)), mean=float(v.mean()), std=float(v.std()))

def main():
    rms_t = []
    for p in TEACHER:
        for r in (1, 2, 3):
            with np.load(K51OUT / f"atlas-{p}-analysis-R{r}.npz") as z:
                rms_t.append(z["real_hist"].mean(axis=0).astype(np.float64))
    rms_u = []
    for p in DOCD:
        for r in (1, 2, 3):
            with np.load(CACHE / f"{p}_R{r}.npz") as c:
                rms_u.append(c["hist"].mean(axis=0).astype(np.float64))
    res = json.loads(Path("/home/hatch/workspace/k51/k51_results.json").read_text())
    res["real_mean_spread"] = {
        "teacher": spread(rms_t),
        "unseen_docd": spread(rms_u),
        "definition": "std of the real mean (mean of the 11-point backward history, frames 0-10) over all candidates; teacher from K51 npz real_hist, unseen from K08 cache hist",
    }
    print("teacher:", res["real_mean_spread"]["teacher"])
    print("unseen :", res["real_mean_spread"]["unseen_docd"])
    Path("/home/hatch/workspace/k51/k51_results.json").write_text(json.dumps(res, indent=1))
    print("WROTE k51_results.json")

if __name__ == "__main__":
    main()
