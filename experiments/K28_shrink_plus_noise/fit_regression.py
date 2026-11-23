"""K28 preparation: fit the generated per-pair mean as a linear function
of the real per-pair mean, on the K05/K08 candidates (weak base, data D)
where both exist. Same mean definitions as K24/K08:
  generated mean = mean of the K05 5-point generated series
  real mean      = mean of the 11-frame real history
Fit: generated mean = a + b * real mean + residual. b is the regression
slope measured in K24 (0.697); a and the residual SD are measured here.
The fitted equation is then applied to the strong-base candidates in
run_experiments.py.
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
            rm.append(d["hist"].mean(axis=0).astype(np.float64))
    return np.concatenate(gm), np.concatenate(rm)

out = {}
for name, prots in (("train", TRAIN_P), ("test", TEST_P), ("all", TRAIN_P + TEST_P)):
    gm, rm = collect(prots)
    b, a = np.polyfit(rm, gm, 1)
    resid = gm - (a + b * rm)
    out[name] = {"n": int(len(gm)), "a": float(a), "b": float(b),
                 "residual_sd": float(resid.std()), "residual_mean": float(resid.mean()),
                 "gen_minus_real_mean": float((gm - rm).mean()),
                 "gen_minus_real_sd": float((gm - rm).std())}
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
(OUT / "fit_results.json").write_text(json.dumps(out, indent=2))
