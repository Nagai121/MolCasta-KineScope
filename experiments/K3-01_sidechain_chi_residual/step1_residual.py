"""K3-01 step 1: reproduce the residual (K39 method, strong base).
Residual = real mean (history[:,1]) - prediction from the 5 static features,
models fitted on the 8 calibration proteins only (OLS + HistGB, K39 settings).
Compare against K39 measure_results.json."""
import numpy as np, json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
K39 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K39_mean_signal_redundancy/measure_results.json").read_text())

def load(proteins, keys):
    out = {k: [] for k in keys}
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                for k in keys:
                    out[k].append(z[f"r{r}_{k}"])
    return {k: np.concatenate(v) for k, v in out.items()}

cal = load(CAL, ["static"])
test = load(TEST, ["static", "history"])
Str = cal["static"].astype(np.float64)
Ste = test["static"].astype(np.float64)
mr_cal = None
cal_h = load(CAL, ["history"])
mr_cal = cal_h["history"][:, 1].astype(np.float64)
mr_te = test["history"][:, 1].astype(np.float64)
print("n_cal", len(mr_cal), "n_test", len(mr_te), flush=True)

res = {}
for method in ("ols", "gbm"):
    if method == "ols":
        model = LinearRegression().fit(Str, mr_cal)
    else:
        model = HistGradientBoostingRegressor(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                              min_samples_leaf=20, l2_regularization=.15,
                                              random_state=20261121).fit(Str, mr_cal)
    pred = model.predict(Ste)
    resid = mr_te - pred
    ref = K39[f"real_mean_{method}"]
    res[method] = dict(r2=float(r2_score(mr_te, pred)), resid_sd=float(resid.std()),
                       resid_mean=float(resid.mean()),
                       ref_r2=ref["r2_on_test"], ref_resid_sd=ref["residual_sd"],
                       match_r2=abs(r2_score(mr_te, pred) - ref["r2_on_test"]) < 5e-4,
                       match_sd=abs(float(resid.std()) - ref["residual_sd"]) < 5e-4)
    print(method, "R2", round(res[method]["r2"], 4), "ref", round(ref["r2_on_test"], 4),
          "| resid_sd", round(res[method]["resid_sd"], 4), "ref", round(ref["residual_sd"], 4),
          "| match", res[method]["match_r2"] and res[method]["match_sd"], flush=True)
    np.save(f"/home/hatch/workspace/kinescope/work/k3_01/resid_{method}.npy", resid.astype(np.float64))
    np.save(f"/home/hatch/workspace/kinescope/work/k3_01/pred_{method}.npy", pred.astype(np.float64))
np.save("/home/hatch/workspace/kinescope/work/k3_01/real_mean_test.npy", mr_te)
Path("/home/hatch/workspace/kinescope/work/k3_01/step1_result.json").write_text(json.dumps(res, indent=2))
print("done")
