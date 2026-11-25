"""K3-01 step 3: relation between sidechain chi state and the residual.

Residual (ずれ): K39 method, reproduced in step 1 (GBM primary, OLS reference).
Chi: frame 10 (origin frame) primary, frame 0 secondary; slots
D (donor residue), A (acceptor residue), Dm/Dp (donor neighbours), Am/Ap.
Analyses:
 (1) correlations of angle / sin / cos with the residual, and rotamer-class
     mean differences (classes: g+ (0,120], t (120,180]&(-180,-120], g- (-120,0]).
 (2) variance reduction: HistGB regressor (K39 settings) predicting the real
     mean from static5 (+ chi feature sets), fitted on the 8 calibration
     proteins, evaluated on the 100 test proteins.
 (3) shuffle controls: global and within-protein permutation of the chi rows.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

W = Path("/home/hatch/workspace/kinescope/work/k3_01")
SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
SLOTS = ["D", "A", "Dm", "Dp", "Am", "Ap"]
GBM = dict(max_iter=160, learning_rate=.065, max_leaf_nodes=15, min_samples_leaf=20,
           l2_regularization=.15, random_state=20261121)

def load_base(proteins):
    S, H, P = [], [], []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                S.append(z[f"r{r}_static"]); H.append(z[f"r{r}_history"]); P.append(z[f"r{r}_pair"])
    return (np.concatenate(S).astype(np.float64), np.concatenate(H), np.concatenate(P))

def load_chi(proteins, npairs_expected):
    rows, prot, rep = [], [], []
    for p in proteins:
        with np.load(W / "chi" / f"{p}.npz", allow_pickle=True) as z:
            with np.load(SAVED / f"{p}.npz") as zz:
                for r in (1, 2, 3):
                    c = z[f"r{r}_chi"]
                    assert len(c) == len(zz[f"r{r}_pair"]), (p, r)
                    rows.append(c); prot += [p] * len(c); rep += [r] * len(c)
    chi = np.concatenate(rows)
    assert len(chi) == npairs_expected, (len(chi), npairs_expected)
    return chi, np.array(prot), np.array(rep)

Ste, Hte, Pte = load_base(TEST)
Scal, Hcal, _ = load_base(CAL)
mr_te = Hte[:, 1].astype(np.float64); mr_cal = Hcal[:, 1].astype(np.float64)
resid_gbm = np.load(W / "resid_gbm.npy"); resid_ols = np.load(W / "resid_ols.npy")
chi_te, prot_te, rep_te = load_chi(TEST, len(mr_te))
chi_cal, prot_cal, rep_cal = load_chi(CAL, len(mr_cal))
print("test", chi_te.shape, "cal", chi_cal.shape, flush=True)

def rotamer(a):
    """a in degrees (-180,180] -> 0:g+, 1:t, 2:g- ; NaN -> -1"""
    out = np.full(a.shape, -1, dtype=np.int8)
    out[(a > 0) & (a <= 120)] = 0
    out[(a > 120) | (a <= -120)] = 1
    out[(a > -120) & (a <= 0)] = 2
    out[np.isnan(a)] = -1
    return out

def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan, 0
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())

# ---------- (1) correlations ----------
corr_rows = []
for frame, fname in ((1, "frame10"),):
    for s in range(6):
        for k in range(4):
            ang = chi_te[:, s, k, frame].astype(np.float64)
            s_, c_ = np.sin(np.radians(ang)), np.cos(np.radians(ang))
            r_ang, n = pearson(ang, resid_gbm)
            r_sin, _ = pearson(s_, resid_gbm); r_cos, _ = pearson(c_, resid_gbm)
            r_sin_o, _ = pearson(s_, resid_ols); r_cos_o, _ = pearson(c_, resid_ols)
            # joint circular-linear R^2 for resid ~ sin + cos (GBM residual)
            m = np.isfinite(ang)
            if m.sum() > 100:
                X = np.column_stack([s_[m], c_[m], np.ones(m.sum())])
                beta, *_ = np.linalg.lstsq(X, resid_gbm[m], rcond=None)
                fit = X @ beta
                ss = 1 - np.sum((resid_gbm[m] - fit) ** 2) / np.sum((resid_gbm[m] - resid_gbm[m].mean()) ** 2)
                jointR = float(np.sqrt(max(ss, 0)))
            else:
                jointR = np.nan
            cls = rotamer(ang)
            means, counts = {}, {}
            for ci, nm in ((0, "g+"), (1, "t"), (2, "g-")):
                mm = cls == ci
                counts[nm] = int(mm.sum())
                means[nm] = float(resid_gbm[mm].mean()) if mm.sum() else np.nan
            vals = [v for v in means.values() if np.isfinite(v)]
            maxdiff = float(max(vals) - min(vals)) if len(vals) == 3 else np.nan
            corr_rows.append(dict(frame=fname, slot=SLOTS[s], chi=k + 1, n=n,
                                  r_angle=r_ang, r_sin=r_sin, r_cos=r_cos,
                                  r_sin_ols=r_sin_o, r_cos_ols=r_cos_o, joint_R=jointR,
                                  mean_gplus=means["g+"], mean_t=means["t"], mean_gminus=means["g-"],
                                  n_gplus=counts["g+"], n_t=counts["t"], n_gminus=counts["g-"],
                                  max_class_diff=maxdiff))
corr_rows.sort(key=lambda d: -max(abs(d["r_sin"] or 0), abs(d["r_cos"] or 0), abs(d["joint_R"] or 0)))
print("top correlations (frame10):", flush=True)
for d in corr_rows:
    if d["frame"] == "frame10":
        print(" ", d["slot"], "chi", d["chi"], "n", d["n"], "r_sin", round(d["r_sin"], 3),
              "r_cos", round(d["r_cos"], 3), "jointR", round(d["joint_R"], 3),
              "classdiff", round(d["max_class_diff"], 3), flush=True)
    # print only first 12 frame10 rows effectively via ordering

# ---------- (2) variance reduction ----------
def chi_feats(chi, frame, slots):
    cols = []
    for s in slots:
        for k in range(4):
            ang = chi[:, s, k, frame].astype(np.float64)
            cols.append(np.sin(np.radians(ang))); cols.append(np.cos(np.radians(ang)))
    return np.column_stack(cols)

def fit_eval(Xcal, Xte, tag):
    m = HistGradientBoostingRegressor(**GBM).fit(Xcal, mr_cal)
    pred = m.predict(Xte)
    sd = float((mr_te - pred).std())
    print(tag, "resid_sd", round(sd, 4), "R2", round(float(r2_score(mr_te, pred)), 4), flush=True)
    return sd, pred

vr = {}
vr["static_only"] = fit_eval(Scal, Ste, "static5")[0]
sets = {
    "static+chi_f10_all": (1, [0, 1, 2, 3, 4, 5]),
    "static+chi_f10_DA": (1, [0, 1]),
    "static+chi_f10_neighbours": (1, [2, 3, 4, 5]),
    "static+chi_f10_D_only": (1, [0]),
    "static+chi_f10_A_only": (1, [1]),
}
preds = {}
for tag, (fr, slots) in sets.items():
    Xc = np.column_stack([Scal, chi_feats(chi_cal, fr, slots)])
    Xt = np.column_stack([Ste, chi_feats(chi_te, fr, slots)])
    vr[tag], preds[tag] = fit_eval(Xc, Xt, tag)

# ---------- (3) shuffle controls ----------
rng = np.random.default_rng(123)
fr, slots = 1, [0, 1, 2, 3, 4, 5]
Xc = np.column_stack([Scal, chi_feats(chi_cal, fr, slots)])
model = HistGradientBoostingRegressor(**GBM).fit(Xc, mr_cal)
Fte = chi_feats(chi_te, fr, slots)
perm_g = rng.permutation(len(Fte))
sd_shuf_g = float((mr_te - model.predict(np.column_stack([Ste, Fte[perm_g]]))).std())
# within-protein shuffle
Fte_w = Fte.copy()
for p in np.unique(prot_te):
    idx = np.where(prot_te == p)[0]
    Fte_w[idx] = Fte_w[idx][rng.permutation(len(idx))]
sd_shuf_w = float((mr_te - model.predict(np.column_stack([Ste, Fte_w]))).std())
# refit with shuffled calibration chi
Fcal = chi_feats(chi_cal, fr, slots)
perm_c = rng.permutation(len(Fcal))
model_s = HistGradientBoostingRegressor(**GBM).fit(np.column_stack([Scal, Fcal[perm_c]]), mr_cal)
sd_refit_shuf = float((mr_te - model_s.predict(np.column_stack([Ste, Fte]))).std())
print("shuffle eval-global sd", round(sd_shuf_g, 4), "| within-protein sd", round(sd_shuf_w, 4),
      "| refit-shuffled sd", round(sd_refit_shuf, 4), flush=True)

# correlation shuffle control: max |r_sin|,|r_cos| over frame10 quantities after global shuffle
ang_all = chi_te[:, :, :, 1].astype(np.float64)
resid_shuf = resid_gbm[perm_g]
mx = 0.0
for s in range(6):
    for k in range(4):
        ang = ang_all[:, s, k]
        for v in (np.sin(np.radians(ang)), np.cos(np.radians(ang))):
            r, _ = pearson(v, resid_shuf)
            if np.isfinite(r):
                mx = max(mx, abs(r))
print("shuffle correlation control: max |r| =", round(mx, 4), flush=True)

# missingness
miss = {}
for s in range(6):
    for k in range(4):
        miss[f"{SLOTS[s]}_chi{k+1}"] = float(np.isnan(chi_te[:, s, k, 1]).mean())

out = dict(n_test=int(len(mr_te)), n_cal=int(len(mr_cal)),
           resid_sd_gbm=float(resid_gbm.std()), resid_sd_ols=float(resid_ols.std()),
           correlations=corr_rows, variance_reduction=vr,
           shuffle=dict(eval_global_sd=sd_shuf_g, eval_within_protein_sd=sd_shuf_w,
                        refit_shuffled_sd=sd_refit_shuf, corr_max_abs_after_shuffle=mx),
           missing_fraction_frame10=miss)
Path(W / "analysis_result.json").write_text(json.dumps(out, indent=2))
with open(W / "correlations.csv", "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys())); wcsv.writeheader(); wcsv.writerows(corr_rows)
np.savez_compressed(W / "test_pairs.npz", protein=prot_te, replica=rep_te,
                    resid_gbm=resid_gbm, resid_ols=resid_ols, real_mean=mr_te,
                    chi=chi_te, static=Ste.astype(np.float32))
print("saved analysis_result.json, correlations.csv, test_pairs.npz")
