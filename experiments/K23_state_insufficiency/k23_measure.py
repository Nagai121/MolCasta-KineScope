"""K23: is the generator's per-pair mean error close to the floor set by
replica-to-replica variability (state insufficiency), or does the static
structure contain the mean and the generator fails to extract it?

Same candidates as K05/K08 (DocD conditions, HB formation, 6 proteins x
3 replicas, 14,582 candidates). Definitions aligned with K08:
  real mean  = mean of the 11-point real history series (0-1 ns)
  gen mean   = mean of the 5-point generated series (origin + 4 points)
Measurements:
  1. replica spread: same (donor, acceptor) pair in all 3 replicas of a
     protein -> pairwise differences of the real means, and the
     across-replica std per pair.
  2. generator error: gen mean - real mean per candidate (as in K08).
  3. static prediction: HistGradientBoostingRegressor (DocD settings)
     trained on the 4 train proteins to predict the real mean from the
     7 static features (which include the origin distance); residual std
     on the 2 test proteins. Ridge regression as a reference."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
ALL_P = TRAIN_P + TEST_P

z = np.load(GEN)
gen_means = {}
gen5 = np.column_stack([z["pair"][:, 2].astype(np.float64), z["gen"].astype(np.float64)])
for i, t in enumerate(z["tid"]):
    gen_means[(str(t), int(z["pair"][i, 0]), int(z["pair"][i, 1]))] = float(gen5[i].mean())

rows = []          # per trajectory dicts
for p in ALL_P:
    for r in (1, 2, 3):
        with np.load(CACHE / f"{p}_R{r}.npz") as zz:
            d = {k: zz[k] for k in zz.files}
        tid = f"atlas-{p}-analysis-R{r}"
        real_mean = d["hist"].mean(axis=0)                       # (N,)
        gm = np.array([gen_means[(tid, int(da), int(aa))]
                       for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
        rows.append(dict(protein=p, replica=r, X=d["X"], real_mean=real_mean,
                         gen_mean=gm, donor=d["donor_atom"], acceptor=d["acceptor_atom"],
                         split="train" if p in TRAIN_P else "test"))

# ---- measurement 2: generator error (K08 alignment check included) ----
res = {}
for split in ("all", "train", "test"):
    sel = [r for r in rows if split == "all" or r["split"] == split]
    err = np.concatenate([r["gen_mean"] - r["real_mean"] for r in sel])
    gm = np.concatenate([r["gen_mean"] for r in sel])
    rm = np.concatenate([r["real_mean"] for r in sel])
    res[f"gen_error_{split}"] = dict(n=int(len(err)), mean=float(err.mean()),
                                     std=float(err.std()), mae=float(np.abs(err).mean()),
                                     corr=float(np.corrcoef(gm, rm)[0, 1]),
                                     gen_mean_std=float(gm.std()), real_mean_std=float(rm.std()))

# ---- measurement 1: replica spread of the real mean ----
pairwise_diffs, across_stds, counts = [], [], {}
for p in ALL_P:
    by_pair = {}
    for r in rows:
        if r["protein"] != p:
            continue
        for da, aa, m in zip(r["donor"], r["acceptor"], r["real_mean"]):
            by_pair.setdefault((int(da), int(aa)), {})[r["replica"]] = float(m)
    full = [v for v in by_pair.values() if len(v) == 3]
    ge2 = [v for v in by_pair.values() if len(v) >= 2]
    counts[p] = dict(pairs_in_any_replica=len(by_pair), pairs_all_3=len(full), pairs_ge2=len(ge2))
    for v in full:
        ms = np.array([v[1], v[2], v[3]])
        across_stds.append(ms.std())
        pairwise_diffs += [ms[0] - ms[1], ms[0] - ms[2], ms[1] - ms[2]]
pd_arr, as_arr = np.array(pairwise_diffs), np.array(across_stds)
res["replica_spread"] = dict(
    n_pairs_all_3=int(len(as_arr)), counts=counts,
    pairwise_diff_mean=float(pd_arr.mean()), pairwise_diff_std=float(pd_arr.std()),
    pairwise_diff_mae=float(np.abs(pd_arr).mean()),
    across_replica_std_mean=float(as_arr.mean()), across_replica_std_median=float(np.median(as_arr)))

# ---- measurement 3: static-feature prediction of the real mean ----
Xtr = np.concatenate([r["X"] for r in rows if r["split"] == "train"])
ytr = np.concatenate([r["real_mean"] for r in rows if r["split"] == "train"])
Xte = np.concatenate([r["X"] for r in rows if r["split"] == "test"])
yte = np.concatenate([r["real_mean"] for r in rows if r["split"] == "test"])
gbm = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.075, max_leaf_nodes=31,
                                    min_samples_leaf=15, l2_regularization=0.02, random_state=42)
gbm.fit(Xtr, ytr)
pred = gbm.predict(Xte)
resid = yte - pred
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(Xtr, ytr)
resid_r = yte - ridge.predict(Xte)
resid_origin = yte - Xte[:, 0].astype(np.float64)   # identity on origin distance
res["static_prediction"] = dict(
    n_train=int(len(ytr)), n_test=int(len(yte)),
    gbm_resid_mean=float(resid.mean()), gbm_resid_std=float(resid.std()),
    gbm_resid_mae=float(np.abs(resid).mean()),
    gbm_r2=float(1 - resid.var() / yte.var()),
    ridge_resid_std=float(resid_r.std()),
    origin_identity_resid_std=float(resid_origin.std()),
    test_real_mean_std=float(yte.std()))

Path("/home/hatch/workspace/k23/k23_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
