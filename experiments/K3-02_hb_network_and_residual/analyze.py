"""K3-02 analysis: residual (K39/K42 construction) vs frame-0 H-bond
network counts. Correlations, added-value refits, permutation control.
sklearn threads pinned to 1 by the caller's environment.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats as st

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
COUNTS = Path("/home/hatch/workspace/k3_02/counts")
WORK = Path("/home/hatch/workspace/k3_02")
CAL = ["2vh3_A", "3ga3_A", "3wx4_A", "3ol3_A", "2es9_A", "2in8_A", "2iay_A", "1lsl_A"]
TEST = json.loads((Path("/home/hatch/workspace/k06b/results_100.json")).read_text())["test_proteins"]
QCOLS = ["n_out_donor_res", "n_in_donor_res", "n_out_acceptor_res", "n_in_acceptor_res",
         "n_hb_total", "n_partner_backbone", "n_partner_sidechain", "n_partner_water",
         "n_competing", "n_hb_donor_res_total", "n_hb_acceptor_res_total"]
SEED = 20261121

def load_side(proteins):
    S, M, Q, groups = [], [], [], []
    for p in proteins:
        z = np.load(K06B / f"{p}.npz")
        c = np.load(COUNTS / f"{p}.npz")
        cols = list(c["columns"])
        qi = [cols.index(q) for q in QCOLS]
        for r in (1, 2, 3):
            S.append(z[f"r{r}_static"].astype(np.float64))
            M.append(z[f"r{r}_history"][:, 1].astype(np.float64))
            Q.append(c[f"r{r}_counts"][:, qi].astype(np.float64))
            groups += [(p, r)] * len(S[-1])
    return np.vstack(S), np.concatenate(M), np.vstack(Q), groups

def gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=160, learning_rate=0.065,
                                         max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=0.15, random_state=SEED)

def fit_eval(model_fn, Xc, yc, Xt, yt, Qc=None, Qt=None):
    Xc2 = Xc if Qc is None else np.hstack([Xc, Qc])
    Xt2 = Xt if Qt is None else np.hstack([Xt, Qt])
    m = model_fn(); m.fit(Xc2, yc)
    pred = m.predict(Xt2)
    resid = yt - pred
    ss = 1.0 - float(np.sum(resid ** 2) / np.sum((yt - yt.mean()) ** 2))
    return float(resid.std()), ss, resid

def shuffle_within_groups(Q, groups, seed):
    rng = np.random.default_rng(seed)
    Qs = Q.copy()
    g = np.array([f"{p}|{r}" for p, r in groups])
    for key in np.unique(g):
        idx = np.where(g == key)[0]
        Qs[idx] = Qs[idx][rng.permutation(len(idx))]
    return Qs

def main():
    Sc, Mc, Qc, gc = load_side(CAL)
    St, Mt, Qt, gt = load_side(TEST)
    out = {"n_cal": int(len(Mc)), "n_test": int(len(Mt))}
    # baseline residuals: GBM (primary) and linear (check), K39/K42 settings
    base = {}
    for name, fn in [("gbm", gbm), ("linear", None)]:
        if fn is None:
            from sklearn.linear_model import LinearRegression
            fn = LinearRegression
        std, r2, resid = fit_eval(fn, Sc, Mc, St, Mt)
        base[name] = {"resid_std_test": std, "r2_test": r2}
        if name == "gbm":
            resid_gbm = resid
        else:
            resid_lin = resid
    out["baseline"] = base
    # premise vs K39: GBM R2 0.635 / resid std 0.60 ; linear R2 0.626 / 0.608
    # correlations of each quantity with the GBM residual (primary) and linear residual
    corr = {}
    for j, q in enumerate(QCOLS):
        row = {}
        for rname, resid in [("gbm", resid_gbm), ("linear", resid_lin)]:
            row[rname] = {"pearson": float(np.corrcoef(Qt[:, j], resid)[0, 1]),
                          "spearman": float(st.spearmanr(Qt[:, j], resid).statistic)}
        corr[q] = row
    out["correlations"] = corr
    # quantity distributions on test
    out["quantity_stats_test"] = {q: {"mean": float(Qt[:, j].mean()), "std": float(Qt[:, j].std()),
                                      "min": float(Qt[:, j].min()), "max": float(Qt[:, j].max()),
                                      "frac_zero": float((Qt[:, j] == 0).mean())}
                                  for j, q in enumerate(QCOLS)}
    # added value: one quantity at a time, then all together (GBM)
    added = {}
    for j, q in enumerate(QCOLS):
        std, r2, _ = fit_eval(gbm, Sc, Mc, St, Mt, Qc[:, [j]], Qt[:, [j]])
        added[q] = {"resid_std_test": std, "r2_test": r2}
    std_all, r2_all, _ = fit_eval(gbm, Sc, Mc, St, Mt, Qc, Qt)
    added["ALL"] = {"resid_std_test": std_all, "r2_test": r2_all}
    out["added_value_gbm"] = added
    # permutation control: shuffle all quantities jointly (same permutation
    # across columns) within protein x replica groups, cal and test
    Qc_s = shuffle_within_groups(Qc, gc, SEED)
    Qt_s = shuffle_within_groups(Qt, gt, SEED + 1)
    std_s, r2_s, _ = fit_eval(gbm, Sc, Mc, St, Mt, Qc_s, Qt_s)
    out["permutation_control_all"] = {"resid_std_test": std_s, "r2_test": r2_s,
                                      "note": "quantities shuffled within protein x replica groups"}
    # per-quantity permutation for any quantity with |pearson| >= 0.3 (GBM residual)
    singles = {}
    for j, q in enumerate(QCOLS):
        if abs(corr[q]["gbm"]["pearson"]) >= 0.3 or abs(corr[q]["gbm"]["spearman"]) >= 0.3:
            qc1 = shuffle_within_groups(Qc[:, [j]], gc, SEED + 10 + j)
            qt1 = shuffle_within_groups(Qt[:, [j]], gt, SEED + 100 + j)
            s1, r21, _ = fit_eval(gbm, Sc, Mc, St, Mt, qc1, qt1)
            singles[q] = {"resid_std_test": s1, "r2_test": r21}
    out["permutation_control_singles"] = singles
    # save residuals + quantities for the record
    np.savez_compressed(WORK / "analysis_arrays.npz", residual_gbm=resid_gbm,
                        residual_linear=resid_lin, quantities_test=Qt,
                        quantity_names=np.array(QCOLS),
                        groups=np.array([f"{p}|{r}" for p, r in gt]))
    (WORK / "analysis_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
