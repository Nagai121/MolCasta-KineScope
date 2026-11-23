"""K40 eval: predict the real mean (11-point history mean) from static
features. Train = 4 proteins (1bgf_A, 1ifg_A, 1j8r_A, 2b0a_A), test =
1mix_A + 2qt4_A (same split as K23). Yardsticks from K23: DocD-7
residual std 0.492, replica floor 0.249 (both test)."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

OUT = Path("/home/hatch/workspace/k40/out")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")

def load(prots):
    Xs, X7s, ys = [], [], []
    fnames = None
    for p in prots:
        for r in (1, 2, 3):
            with np.load(OUT / f"{p}_R{r}.npz") as z:
                Xs.append(z["Xrich"]); X7s.append(z["X7"]); ys.append(z["real_mean"])
                fnames = [str(f) for f in z["feature_names"]]
    return np.concatenate(Xs), np.concatenate(X7s), np.concatenate(ys), fnames

def resid_stats(y, p):
    e = p - y
    return dict(n=int(len(y)), mean=float(e.mean()), std=float(e.std()),
                mae=float(np.abs(e).mean()),
                r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                corr=float(np.corrcoef(p, y)[0, 1]))

def hgb(**kw):
    d = dict(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
             min_samples_leaf=15, l2_regularization=.02, random_state=42)
    d.update(kw); return HistGradientBoostingRegressor(**d)

def main():
    Xtr, X7tr, ytr, fnames = load(TRAIN_P)
    Xte, X7te, yte, _ = load(TEST_P)
    print(f"train n={len(ytr)} test n={len(yte)} F={Xtr.shape[1]}")
    print("test target std:", round(float(yte.std()), 4))
    res = {}
    # identity baseline: origin distance as the prediction
    res["origin_identity"] = {"train": resid_stats(ytr, X7tr[:, 0]),
                              "test": resid_stats(yte, X7te[:, 0])}
    # (a) DocD-7, DocD settings -- K23 reproduction
    m = hgb().fit(X7tr, ytr)
    res["docd7"] = {"train": resid_stats(ytr, m.predict(X7tr)),
                    "test": resid_stats(yte, m.predict(X7te))}
    # (b) rich, same settings
    m = hgb().fit(Xtr, ytr)
    res["rich_same"] = {"train": resid_stats(ytr, m.predict(Xtr)),
                        "test": resid_stats(yte, m.predict(Xte))}
    # (c) rich, larger capacity
    m = hgb(max_iter=400, learning_rate=.05, max_leaf_nodes=63,
            min_samples_leaf=10).fit(Xtr, ytr)
    res["rich_big"] = {"train": resid_stats(ytr, m.predict(Xtr)),
                       "test": resid_stats(yte, m.predict(Xte))}
    # (d) ridge on rich (median-imputed, standardized)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=10.0, random_state=42))
    pipe.fit(Xtr, ytr)
    res["ridge_rich"] = {"train": resid_stats(ytr, pipe.predict(Xtr)),
                         "test": resid_stats(yte, pipe.predict(Xte))}
    for k, v in res.items():
        print(f"{k}: test std {v['test']['std']:.4f} R2 {v['test']['r2']:.3f} "
              f"corr {v['test']['corr']:.3f} (train std {v['train']['std']:.4f})")
    # ablation: drop one group at a time (rich, DocD settings)
    groups = {
        "residue_onehot": lambda f: f.startswith(("donor_res_", "acceptor_res_")),
        "atomname_onehot": lambda f: f.startswith(("donor_atom_", "acceptor_atom_")),
        "density_packing": lambda f: f in ("donor_heavy_5A", "donor_heavy_8A", "acceptor_heavy_5A",
                                           "acceptor_heavy_8A", "mid_heavy_5A", "donor_competing_O_4A",
                                           "acceptor_competing_N_4A", "donor_nearest_other_O",
                                           "acceptor_nearest_other_N", "donor_min_other_heavy",
                                           "acceptor_min_other_heavy"),
        "geometry": lambda f: f in ("cb_dist", "donor_ca_i2", "donor_ca_im2", "acceptor_ca_i2",
                                    "acceptor_ca_im2", "cos_da_vs_caca", "cos_da_vs_donor_chain",
                                    "cos_da_vs_acceptor_chain", "donor_atom_to_ca", "acceptor_atom_to_ca",
                                    "donor_ca_to_acceptor", "acceptor_ca_to_donor"),
        "neighbourhood": lambda f: f in ("donor_res_neighbours_8A", "acceptor_res_neighbours_8A",
                                         "donor_heavy_count", "acceptor_heavy_count"),
        "docd7": lambda f: f.startswith("docd_"),
    }
    abl = {}
    for g, pred in groups.items():
        keep = np.array([not pred(f) for f in fnames])
        m = hgb().fit(Xtr[:, keep], ytr)
        abl[g] = {"kept_F": int(keep.sum()),
                  "test": resid_stats(yte, m.predict(Xte[:, keep]))}
        print(f"ablation drop {g}: test std {abl[g]['test']['std']:.4f} (kept {keep.sum()} feats)")
    # extras only (no DocD 7)
    keep = np.array([not f.startswith("docd_") for f in fnames])
    m = hgb().fit(Xtr[:, keep], ytr)
    abl["extras_only"] = {"kept_F": int(keep.sum()),
                          "test": resid_stats(yte, m.predict(Xte[:, keep]))}
    print(f"ablation extras_only: test std {abl['extras_only']['test']['std']:.4f}")
    res["ablation"] = abl
    res["feature_names"] = fnames
    res["yardsticks"] = {"k23_docd7_test_std": 0.4921, "k23_replica_floor_std": 0.2486,
                         "k23_ridge_test_std": 0.4869, "k23_origin_identity_test_std": 0.4891}
    Path("/home/hatch/workspace/k40/k40_results.json").write_text(json.dumps(res, indent=1))
    print("WROTE k40_results.json")

if __name__ == "__main__":
    main()
