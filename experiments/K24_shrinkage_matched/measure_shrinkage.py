"""K24 step 1 (measurement): is the generated per-pair mean shrunk toward
the overall mean?

Same candidates and same mean definitions as K05/K08 (weak base, data D):
  generated mean = mean of the K05 5-point generated series
                   (origin distance + generated points at +0.1/+0.3/+0.5/+1.0 ns)
  real mean      = mean of the 11-frame real history (0 to 1 ns)
Alignment by (trajectory id, donor atom, acceptor atom), as in K08.

Outputs, pooled over all candidates and split by train/test proteins:
  SD of generated means, SD of real means, their ratio (alpha),
  Pearson correlation between the two means, and the K08 cross-check
  (mean and SD of generated - real).
"""
import numpy as np, json
from pathlib import Path

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")

z = np.load(GEN)
tids, pair, gen = z["tid"], z["pair"], z["gen"]
gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
GEN_MEAN = {(str(t), int(pair[i, 0]), int(pair[i, 1])): float(gen5[i].mean())
            for i, t in enumerate(tids)}

def collect(prots):
    gm, rm = [], []
    for p in prots:
        for r in (1, 2, 3):
            d = np.load(CACHE / f"{p}_R{r}.npz")
            tid = f"atlas-{p}-analysis-R{r}"
            gm.append(np.array([GEN_MEAN[(tid, int(da), int(aa))]
                                for da, aa in zip(d["donor_atom"], d["acceptor_atom"])]))
            rm.append(d["hist"].mean(axis=0).astype(np.float64))  # hist: (11, N)
    return np.concatenate(gm), np.concatenate(rm)

def stats(gm, rm):
    slope = float(np.polyfit(rm, gm, 1)[0])
    return {"n": int(len(gm)),
            "gen_mean_sd": float(gm.std()), "real_mean_sd": float(rm.std()),
            "sd_ratio_alpha": float(gm.std() / rm.std()),
            "correlation": float(np.corrcoef(gm, rm)[0, 1]),
            "regression_slope_gen_on_real": slope,
            "gen_minus_real_mean": float((gm - rm).mean()),
            "gen_minus_real_sd": float((gm - rm).std()),
            "gen_mean_overall": float(gm.mean()), "real_mean_overall": float(rm.mean())}

out = {}
for name, prots in (("train", TRAIN_P), ("test", TEST_P), ("all", TRAIN_P + TEST_P)):
    gm, rm = collect(prots)
    out[name] = stats(gm, rm)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
(OUT / "measure_results.json").write_text(json.dumps(out, indent=2))
print("alpha (all) =", out["all"]["sd_ratio_alpha"])
