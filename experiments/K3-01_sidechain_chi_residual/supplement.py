"""K3-01 supplement: (a) per-protein correlations for the top quantities,
(b) rotamer one-hot model (D+A) as an alternative encoding."""
import numpy as np, json
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

W = Path("/home/hatch/workspace/kinescope/work/k3_01")
SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
GBM = dict(max_iter=160, learning_rate=.065, max_leaf_nodes=15, min_samples_leaf=20,
           l2_regularization=.15, random_state=20261121)

def load_base(proteins):
    S, H = [], []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                S.append(z[f"r{r}_static"]); H.append(z[f"r{r}_history"])
    return np.concatenate(S).astype(np.float64), np.concatenate(H)

def load_chi(proteins):
    rows, prot = [], []
    for p in proteins:
        with np.load(W / "chi" / f"{p}.npz", allow_pickle=True) as z:
            for r in (1, 2, 3):
                c = z[f"r{r}_chi"]
                rows.append(c); prot += [p] * len(c)
    return np.concatenate(rows), np.array(prot)

Ste, Hte = load_base(TEST); Scal, Hcal = load_base(CAL)
mr_te = Hte[:, 1].astype(np.float64); mr_cal = Hcal[:, 1].astype(np.float64)
resid_gbm = np.load(W / "resid_gbm.npy")
chi_te, prot_te = load_chi(TEST); chi_cal, _ = load_chi(CAL)

# (a) per-protein correlations for top quantities: (slot, chi, component)
tops = [("A", 1, 2, "cos"), ("D", 1, 2, "sin"), ("A", 0, 1, "sin")]
SLOTS = ["D", "A", "Dm", "Dp", "Am", "Ap"]
out = {}
for name, s, k, comp in tops:
    ang = chi_te[:, s, k, 1].astype(np.float64)
    v = np.sin(np.radians(ang)) if comp == "sin" else np.cos(np.radians(ang))
    rs = []
    for p in np.unique(prot_te):
        m = (prot_te == p) & np.isfinite(v)
        if m.sum() > 50:
            rs.append(float(np.corrcoef(v[m], resid_gbm[m])[0, 1]))
    rs = np.array(rs)
    out[f"{name}_chi{k+1}_{comp}"] = dict(n_proteins=int(len(rs)), median=float(np.median(rs)),
        q1=float(np.quantile(rs, .25)), q3=float(np.quantile(rs, .75)),
        frac_pos=float((rs > 0).mean()), frac_abs_ge_0_3=float((np.abs(rs) >= 0.3).mean()))
    print(name, "chi", k + 1, comp, "median", round(out[f"{name}_chi{k+1}_{comp}"]["median"], 3),
          "IQR", (round(out[f"{name}_chi{k+1}_{comp}"]["q1"], 3), round(out[f"{name}_chi{k+1}_{comp}"]["q3"], 3)),
          "frac>=0.3:", round(out[f"{name}_chi{k+1}_{comp}"]["frac_abs_ge_0_3"], 3), flush=True)

# (b) rotamer one-hot model, D+A slots, frame 10
def rotamer_onehot(chi, slots):
    cols = []
    for s in slots:
        for k in range(4):
            a = chi[:, s, k, 1].astype(np.float64)
            cols.append(((a > 0) & (a <= 120)).astype(float))
            cols.append(((a > 120) | (a <= -120)).astype(float))
            cols.append(((a > -120) & (a <= 0)).astype(float))
    return np.column_stack(cols)

Xc = np.column_stack([Scal, rotamer_onehot(chi_cal, [0, 1])])
Xt = np.column_stack([Ste, rotamer_onehot(chi_te, [0, 1])])
m = HistGradientBoostingRegressor(**GBM).fit(Xc, mr_cal)
pred = m.predict(Xt)
sd = float((mr_te - pred).std())
print("static+rotamer_onehot_DA resid_sd", round(sd, 4), "R2", round(float(r2_score(mr_te, pred)), 4), flush=True)
out["rotamer_onehot_DA_sd"] = sd
Path(W / "supplement.json").write_text(json.dumps(out, indent=2))
print("saved supplement.json")
