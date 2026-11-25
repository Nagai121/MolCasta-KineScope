"""E02 Part 1 (Zero-download local audit on 100 unseen ATLAS proteins, 163,074 pairs):
1. Verifies exact row alignment across K3-01, K3-03, K3-04, K3-06 saved .npz files.
2. Evaluates 5-fold protein-grouped CV across all 100 unseen proteins (80 train / 20 test per fold),
   fixing K3's 8-protein calibration bottleneck.
3. Evaluates BOTH:
   - Public static baseline (4 features: origin distance + 3 base descriptors, STRIPPING frozen T1I2006)
   - Original strong-base static5 (with T1I2006 included)
4. Tests adding each K3 physical family individually AND ALL 4 K3 physical families simultaneously
   (sidechain chi 48d + backbone phi/psi 8d + E-field 2d + CHARMM36m frame-10 energies 10d = 68 features)
   with within-(protein x replica) shuffle controls.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

KS = Path(r"d:\KineScope\MolCasta-KineScope\experiments")
OUT = Path(r"d:\KineScope\experiments\E02_public_k3_and_replica_floor")

GBM_KW = dict(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=0.2,
    random_state=20261126,
)


def chi_to_sincos(chi_f10: np.ndarray) -> np.ndarray:
    """chi_f10: (N, 6, 4) degrees with NaN -> (N, 48) sin/cos."""
    rad = np.radians(chi_f10.astype(np.float64))
    s = np.sin(rad).reshape(len(chi_f10), -1)
    c = np.cos(rad).reshape(len(chi_f10), -1)
    return np.column_stack([s, c])


def eval_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str):
    gkf = GroupKFold(n_splits=5)
    oof_gbm = np.empty_like(y, dtype=np.float64)
    oof_lin = np.empty_like(y, dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(**GBM_KW)
        m.fit(X[tr], y[tr])
        oof_gbm[te] = m.predict(X[te])

        # Simple median imputation for linear Ridge reference
        Xtr = X[tr].copy()
        Xte = X[te].copy()
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds_tr = np.where(~np.isfinite(Xtr))
        Xtr[inds_tr] = med[inds_tr[1]]
        inds_te = np.where(~np.isfinite(Xte))
        Xte[inds_te] = med[inds_te[1]]
        # Standardize for Ridge stability
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        lin = Ridge(alpha=10.0)
        lin.fit((Xtr - mu) / sd, y[tr])
        oof_lin[te] = lin.predict((Xte - mu) / sd)

    res_gbm = y - oof_gbm
    res_lin = y - oof_lin
    out = {
        "name": name,
        "n_features": int(X.shape[1]),
        "gbm_r2": float(r2_score(y, oof_gbm)),
        "gbm_resid_sd": float(res_gbm.std()),
        "gbm_rmse": float(np.sqrt(np.mean(res_gbm ** 2))),
        "ridge_r2": float(r2_score(y, oof_lin)),
        "ridge_resid_sd": float(res_lin.std()),
    }
    print(f"[{name:36s}] d={X.shape[1]:2d} | GBM R2={out['gbm_r2']:.4f} SD={out['gbm_resid_sd']:.4f} A | Ridge R2={out['ridge_r2']:.4f} SD={out['ridge_resid_sd']:.4f} A", flush=True)
    return out, res_gbm, res_lin


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z1 = np.load(KS / "K3-01_sidechain_chi_residual" / "test_pairs.npz", allow_pickle=True)
    z3 = np.load(KS / "K3-03_electric_field_residual" / "analysis_data.npz", allow_pickle=True)
    z4 = np.load(KS / "K3-04_backbone_angles_residual" / "analysis_data.npz", allow_pickle=True)
    z6 = np.load(KS / "K3-06_interaction_energy_residual" / "candidates.npz", allow_pickle=True)

    print("K3-01 keys:", list(z1.keys()))
    print("K3-03 keys:", list(z3.keys()))
    print("K3-04 keys:", list(z4.keys()))
    print("K3-06 keys:", list(z6.keys()))

    prot = z1["protein"]
    rep = z1["replica"]
    real_mean = z1["real_mean"].astype(np.float64)
    static5 = z1["static"].astype(np.float64)
    n = len(real_mean)
    assert n == 163074, n

    # Verify exact alignment across all 4 archives
    assert np.all(z6["protein"] == prot), "K3-06 protein mismatch"
    assert np.all(z6["replica"] == rep), "K3-06 replica mismatch"
    assert np.allclose(z6["real_mean"], real_mean, atol=1e-5), "K3-06 real_mean mismatch"
    assert np.allclose(z3["residual_gbm"], z1["resid_gbm"], atol=1e-4), "K3-03 residual_gbm mismatch"
    assert np.allclose(z4["residual_gbm"], z1["resid_gbm"], atol=1e-4), "K3-04 residual_gbm mismatch"
    print("Alignment verified across K3-01, K3-03, K3-04, K3-06 (N = 163,074, 100 proteins).", flush=True)

    # Extract feature blocks at frame 10 (origin frame)
    # K3-01 chi shape: (N, 6, 4, 2) where last dim 1 = frame10
    chi_f10 = chi_to_sincos(z1["chi"][:, :, :, 1])
    chi_da_f10 = chi_to_sincos(z1["chi"][:, :2, :, 1])

    ef_f10 = z3["quantities"].astype(np.float64).reshape(n, -1)
    bb_f10 = z4["quantities"].astype(np.float64).reshape(n, -1)

    # K3-06 frame10 quantities (10 cols) & frame0 quantities (10 cols)
    en_f10 = z6["quant10"].astype(np.float64) if "quant10" in z6 else z6["quant_frame10"].astype(np.float64)
    en_f0 = z6["quant"].astype(np.float64) if "quant" in z6 else z6["quant_frame0"].astype(np.float64)
    print(f"Feature block shapes: chi={chi_f10.shape}, ef={ef_f10.shape}, bb={bb_f10.shape}, en_f10={en_f10.shape}", flush=True)

    # Public static4 (strips static5[:, 0] = frozen T1I2006 probability)
    origin_dist = static5[:, 1:2]
    public_static4 = static5[:, 1:5]

    all_phys_f10 = np.column_stack([chi_f10, bb_f10, ef_f10, en_f10])

    # Within-(protein x replica) shuffle control for all_phys_f10
    rng = np.random.default_rng(20261126)
    all_phys_shuf = all_phys_f10.copy()
    pr_keys = np.array([f"{p}_R{r}" for p, r in zip(prot, rep)])
    for g in np.unique(pr_keys):
        idx = np.where(pr_keys == g)[0]
        all_phys_shuf[idx] = all_phys_shuf[idx][rng.permutation(len(idx))]

    results = {"n_candidates": n, "n_proteins": int(len(np.unique(prot))), "runs": []}

    # --- Track A: T1I2006-FREE (Public Baseline) ---
    print("\n=== Track A: T1I2006-FREE (Public Static Baseline, 5-fold protein CV) ===", flush=True)
    r_d, res_d, _ = eval_cv(origin_dist, real_mean, prot, "A0_origin_dist_only")
    r_s4, res_s4, res_s4_lin = eval_cv(public_static4, real_mean, prot, "A1_public_static4")
    r_s4_chi, _, _ = eval_cv(np.column_stack([public_static4, chi_f10]), real_mean, prot, "A2_static4+chi_all_f10")
    r_s4_bb, _, _ = eval_cv(np.column_stack([public_static4, bb_f10]), real_mean, prot, "A3_static4+backbone_phipsi_f10")
    r_s4_ef, _, _ = eval_cv(np.column_stack([public_static4, ef_f10]), real_mean, prot, "A4_static4+efield_f10")
    r_s4_en, _, _ = eval_cv(np.column_stack([public_static4, en_f10]), real_mean, prot, "A5_static4+charmm36m_energy_f10")
    r_s4_all, _, _ = eval_cv(np.column_stack([public_static4, all_phys_f10]), real_mean, prot, "A6_static4+ALL_4_K3_families_f10")
    r_s4_shuf, _, _ = eval_cv(np.column_stack([public_static4, all_phys_shuf]), real_mean, prot, "A7_static4+ALL_shuffled_control")
    r_s4_enf0, _, _ = eval_cv(np.column_stack([public_static4, en_f0]), real_mean, prot, "A8_static4+energy_f0_ref")

    # --- Track B: With T1I2006 included (5-fold CV across 100 proteins, fixing 8-protein calibration limit) ---
    print("\n=== Track B: With T1I2006 included (5-fold protein CV across 100 proteins) ===", flush=True)
    r_s5, res_s5, _ = eval_cv(static5, real_mean, prot, "B1_static5_with_T1I2006")
    r_s5_all, _, _ = eval_cv(np.column_stack([static5, all_phys_f10]), real_mean, prot, "B2_static5+ALL_4_K3_families_f10")
    r_s5_shuf, _, _ = eval_cv(np.column_stack([static5, all_phys_shuf]), real_mean, prot, "B3_static5+ALL_shuffled_control")

    results["runs"] = [
        r_d, r_s4, r_s4_chi, r_s4_bb, r_s4_ef, r_s4_en, r_s4_all, r_s4_shuf, r_s4_enf0,
        r_s5, r_s5_all, r_s5_shuf,
    ]

    # Univariate correlations against T1I2006-FREE residual (res_s4)
    def max_corr(mat, target):
        best_p, best_s = 0.0, 0.0
        for j in range(mat.shape[1]):
            v = mat[:, j]
            m = np.isfinite(v)
            if m.sum() > 100 and v[m].std() > 0:
                rp = abs(float(pearsonr(v[m], target[m])[0]))
                rs = abs(float(spearmanr(v[m], target[m])[0]))
                best_p = max(best_p, rp)
                best_s = max(best_s, rs)
        return {"max_abs_pearson": best_p, "max_abs_spearman": best_s}

    results["max_univariate_corr_vs_public_static4_resid"] = {
        "chi_f10": max_corr(chi_f10, res_s4),
        "backbone_phipsi_f10": max_corr(bb_f10, res_s4),
        "efield_f10": max_corr(ef_f10, res_s4),
        "charmm36m_energy_f10": max_corr(en_f10, res_s4),
        "charmm36m_energy_f0": max_corr(en_f0, res_s4),
    }
    print("\nMax univariate |r| vs T1I2006-FREE static4 residual:")
    for k, v in results["max_univariate_corr_vs_public_static4_resid"].items():
        print(f"  {k:24s}: max |Pearson|={v['max_abs_pearson']:.4f}, max |Spearman|={v['max_abs_spearman']:.4f}")

    (OUT / "e02_part1_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved e02_part1_results.json", flush=True)


if __name__ == "__main__":
    main()
