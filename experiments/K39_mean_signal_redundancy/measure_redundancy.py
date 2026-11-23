"""K39: is the generated mean's signal redundant with the static features?
(strong base, hydrogen bonds)

K29 found the generated mean itself carries nearly as much label signal
as the real mean, yet K34 found retraining on generated features
recovers almost no gain. One way both can hold: the generated mean's
signal overlaps with the static features, while the real mean's signal
does not.

Method: K06(b) candidates (frozen T1I2006 + correction setup, 8
calibration proteins / 13,808 candidates for fitting, 100 unseen
proteins x 3 replicas / 163,074 candidates for evaluation). Static
features are exactly the static correction's 5 (frozen probability,
origin distance, 3 base descriptors). For the real mean (history[1])
and the generated mean (gen6[1]) separately, predict the mean from the
static features (OLS and a HistGB regressor, both fitted on cal), take
the residual on test, and measure its relation with the label at each
horizon: correlation, mean(residual|y=1) - mean(residual|y=0), and AUC
of the sign-flipped residual (the residual is expected to relate
negatively: a mean below the static prediction means the pair drifted
closer, so formation is more likely). Context: raw corr(mean, label),
R^2 of the mean~statics fit, residual SD, corr(mean, origin distance).

Pre-fixed reading: if the real mean's residual relates to the label
while the generated mean's residual does not, the generated mean's
signal overlaps with the static features and the real mean's does not.
If both residuals relate to the label, this explanation fails.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score

SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
HORIZONS = (1, 5, 10, 30, 50, 90)

assert not (set(CAL) & set(TEST))
print("no overlap: cal", len(CAL), "test", len(TEST), "common", 0, flush=True)

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({"static": z[f"r{r}_static"], "history": z[f"r{r}_history"],
                             "gen6": z[f"r{r}_gen6"], "label": z[f"r{r}_label"]})
    return rows

cal = load(CAL); test = load(TEST)
Str = np.concatenate([r["static"] for r in cal]).astype(np.float64)
Ste = np.concatenate([r["static"] for r in test]).astype(np.float64)
mr_cal = np.concatenate([r["history"] for r in cal])[:, 1].astype(np.float64)
mr_te = np.concatenate([r["history"] for r in test])[:, 1].astype(np.float64)
mg_cal = np.concatenate([r["gen6"] for r in cal])[:, 1].astype(np.float64)
mg_te = np.concatenate([r["gen6"] for r in test])[:, 1].astype(np.float64)
Yte = np.concatenate([r["label"] for r in test])
print("n_cal", len(mr_cal), "n_test", len(mr_te), flush=True)

def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])

results = {"n_cal": int(len(mr_cal)), "n_test": int(len(mr_te)), "no_overlap": True}
for tag, y_cal, y_te in (("real_mean", mr_cal, mr_te), ("gen_mean", mg_cal, mg_te)):
    for method in ("ols", "gbm"):
        if method == "ols":
            model = LinearRegression().fit(Str, y_cal)
        else:
            model = HistGradientBoostingRegressor(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                                  min_samples_leaf=20, l2_regularization=.15,
                                                  random_state=20261121).fit(Str, y_cal)
        pred = model.predict(Ste)
        resid = y_te - pred
        row = {"r2_on_test": float(r2_score(y_te, pred)),
               "residual_sd": float(resid.std()),
               "corr_mean_with_origin": corr(y_te, Ste[:, 1]),
               "corr_residual_with_origin": corr(resid, Ste[:, 1]),
               "corr_mean_with_frozen": corr(y_te, Ste[:, 0]),
               "per_horizon": {}}
        for j, h in enumerate(HORIZONS):
            y = Yte[:, j]
            row["per_horizon"][str(h)] = {
                "corr_mean_with_label": corr(y_te, y),
                "corr_residual_with_label": corr(resid, y),
                "mean_residual_pos_minus_neg": float(resid[y == 1].mean() - resid[y == 0].mean()),
                "auc_neg_residual": float(roc_auc_score(y, -resid)),
                "auc_residual": float(roc_auc_score(y, resid))}
        results[f"{tag}_{method}"] = row
        print(tag, method, "R2", round(row["r2_on_test"], 4), "resid_sd", round(row["residual_sd"], 4),
              "corr(resid,y) 1ns", round(row["per_horizon"]["1"]["corr_residual_with_label"], 4),
              "auc(-resid) 1ns", round(row["per_horizon"]["1"]["auc_neg_residual"], 4), flush=True)

(OUT / "measure_results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "measure_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mean", "method", "horizon_ns", "r2_on_test", "residual_sd", "corr_mean_with_label",
                "corr_residual_with_label", "mean_residual_pos_minus_neg", "auc_neg_residual"])
    for tag in ("real_mean", "gen_mean"):
        for method in ("ols", "gbm"):
            r = results[f"{tag}_{method}"]
            for h in HORIZONS:
                p = r["per_horizon"][str(h)]
                w.writerow([tag, method, h, r["r2_on_test"], r["residual_sd"], p["corr_mean_with_label"],
                            p["corr_residual_with_label"], p["mean_residual_pos_minus_neg"], p["auc_neg_residual"]])
print("saved to", OUT)
