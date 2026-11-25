"""E02 Part 2: 100-protein Public Replica Floor, Autocorrelation Time, Residual Half-Correlation, and K3-02 (H-bond Network).

Uses shared_data/extracted_100/*.npz (100 unseen ATLAS proteins x 3 replicas, 100% public, ZERO T1I2006).
Computes:
  1. 100-protein Replica-to-Replica Floor (exact atom-pair / residue-pair matching across R1, R2, R3):
     - Pairwise replica difference SD (K23 exact definition)
     - Per-pair across-replica SD (mean & median)
     - Inter-replica Pearson correlation of the static-unpredictable offset
     For ALL 4 phenomena: HB formation, HB dissociation, Contact formation, Salt-bridge formation.
  2. 100-protein Distance Autocorrelation Time (lags 1..5 = 100..500 ps; standard & variogram estimators).
  3. 100-protein Residual Half-Correlation (first half [0..4] vs second half [6..10], 5-fold protein CV).
  4. K3-02 (Pre-existing H-bond Network & Competing H-bonds at origin frame 10) on 100 unseen proteins:
     - Univariate Pearson & Spearman vs public 20-feature static residual
     - 5-fold protein CV residual SD reduction (with within-protein-replica shuffle control)
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

CACHE = Path(r"d:\KineScope\experiments\shared_data\extracted_100")
OUT = Path(r"d:\KineScope\experiments\E02_public_k3_and_replica_floor")

GBM_KW = dict(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=0.2,
    random_state=20261126,
)

K302_NAMES = [
    "hb_dres_out", "hb_dres_in", "hb_dres_tot",
    "hb_ares_out", "hb_ares_in", "hb_ares_tot",
    "hb_bb_partners", "hb_sc_partners",
    "hb_competing_d_atom", "hb_competing_a_atom",
]


def oof_reg(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(**GBM_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def compute_autocorr(S: np.ndarray):
    """S: (N, 11) distance series -> lags 1..5 standard z-score and variogram estimates."""
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    valid = sd[:, 0] > 1e-8
    Sv = S[valid].astype(np.float64)
    Z = (Sv - mu[valid]) / sd[valid]
    var_unbiased = Sv.var(axis=1) * (11.0 / 10.0)
    mean_var = float(var_unbiased.mean())
    out = {}
    for lag in range(1, 6):
        z_ac = float(np.mean(Z[:, :-lag] * Z[:, lag:]))
        msd = float(np.mean((Sv[:, lag:] - Sv[:, :-lag]) ** 2))
        vario_ac = float(1.0 - msd / (2.0 * mean_var))
        out[f"lag{lag}_{lag*100}ps"] = {"zscore": z_ac, "variogram": vario_ac}
    return out


def analyze_phenomenon(files, mode: str, n_base_cols: int):
    Xs, Ss, Ps, Rs, Pairs = [], [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            prot = str(z["protein"])
            for r in (1, 2, 3):
                X = z[f"r{r}_{mode}_X"]
                if len(X) == 0:
                    continue
                Xs.append(X)
                Ss.append(z[f"r{r}_{mode}_series"])
                Ps.append(np.full(len(X), prot, dtype=object))
                Rs.append(np.full(len(X), r, dtype=np.int32))
                Pairs.append(z[f"r{r}_{mode}_pair"])

    X_all = np.concatenate(Xs).astype(np.float64)
    X_base = X_all[:, :n_base_cols]
    S = np.concatenate(Ss).astype(np.float64)
    P = np.concatenate(Ps)
    R = np.concatenate(Rs)
    Pair = np.concatenate(Pairs)

    mean_all = S.mean(axis=1)
    mean_h1 = S[:, 0:5].mean(axis=1)
    mean_h2 = S[:, 6:11].mean(axis=1)

    pred_all = oof_reg(X_base, mean_all, P)
    pred_h1 = oof_reg(X_base, mean_h1, P)
    pred_h2 = oof_reg(X_base, mean_h2, P)

    res_all = mean_all - pred_all
    res_h1 = mean_h1 - pred_h1
    res_h2 = mean_h2 - pred_h2

    # Half-correlation
    r_half_p, _ = pearsonr(res_h1, res_h2)
    r_half_s, _ = spearmanr(res_h1, res_h2)
    r_raw_p, _ = pearsonr(mean_h1, mean_h2)

    # Autocorrelation
    autocorr = compute_autocorr(S)

    # Replica-to-replica floor (matching exact (protein, pair0, pair1) across replicas)
    by_key = {}
    for i in range(len(mean_all)):
        key = (P[i], int(Pair[i, 0]), int(Pair[i, 1]))
        by_key.setdefault(key, []).append((int(R[i]), float(mean_all[i]), float(res_all[i])))

    diffs_3rep, diffs_2plus = [], []
    per_pair_sd_3rep = []
    res_pairs_a, res_pairs_b = [], []
    mean_pairs_a, mean_pairs_b = [], []
    n_3rep = n_2plus = 0
    for key, vals in by_key.items():
        if len(vals) >= 2:
            n_2plus += 1
            for a_i in range(len(vals)):
                for b_i in range(a_i + 1, len(vals)):
                    diffs_2plus.append(vals[a_i][1] - vals[b_i][1])
                    mean_pairs_a.append(vals[a_i][1])
                    mean_pairs_b.append(vals[b_i][1])
                    res_pairs_a.append(vals[a_i][2])
                    res_pairs_b.append(vals[b_i][2])
        if len(vals) == 3:
            n_3rep += 1
            ms = [v[1] for v in vals]
            per_pair_sd_3rep.append(float(np.std(ms, ddof=0)))
            diffs_3rep.extend([ms[0] - ms[1], ms[1] - ms[2], ms[0] - ms[2]])

    diffs_3rep = np.asarray(diffs_3rep)
    diffs_2plus = np.asarray(diffs_2plus)
    # Note: K23 measures SD of pairwise replica differences (R_a - R_b), AND single-replica noise SD = diff_SD / sqrt(2)
    rep_res_corr = float(pearsonr(res_pairs_a, res_pairs_b)[0]) if len(res_pairs_a) > 10 else float("nan")
    rep_mean_corr = float(pearsonr(mean_pairs_a, mean_pairs_b)[0]) if len(mean_pairs_a) > 10 else float("nan")

    rec = {
        "mode": mode,
        "n_candidates": int(len(mean_all)),
        "n_proteins": int(len(np.unique(P))),
        "real_mean_sd": float(mean_all.std()),
        "public_static_r2": float(r2_score(mean_all, pred_all)),
        "public_static_resid_sd": float(res_all.std()),
        "replica_floor": {
            "n_distinct_pairs": int(len(by_key)),
            "n_pairs_in_all_3_replicas": int(n_3rep),
            "n_pairs_in_2plus_replicas": int(n_2plus),
            "diff_sd_3rep_K23_def": float(diffs_3rep.std()) if len(diffs_3rep) else float("nan"),
            "single_rep_noise_sd_3rep": float(diffs_3rep.std() / np.sqrt(2.0)) if len(diffs_3rep) else float("nan"),
            "diff_sd_2plus_rep": float(diffs_2plus.std()) if len(diffs_2plus) else float("nan"),
            "per_pair_sd_3rep_mean": float(np.mean(per_pair_sd_3rep)) if per_pair_sd_3rep else float("nan"),
            "per_pair_sd_3rep_median": float(np.median(per_pair_sd_3rep)) if per_pair_sd_3rep else float("nan"),
            "inter_replica_mean_pearson": rep_mean_corr,
            "inter_replica_resid_pearson": rep_res_corr,
        },
        "residual_half_correlation": {
            "pearson_resid": float(r_half_p),
            "spearman_resid": float(r_half_s),
            "pearson_raw_mean": float(r_raw_p),
        },
        "autocorrelation": autocorr,
    }

    print(
        f"[{mode:8s}] N={len(mean_all):7,d} | pub_static R2={rec['public_static_r2']:.4f} resid_SD={rec['public_static_resid_sd']:.4f} A | "
        f"replica_diff_SD(3rep)={rec['replica_floor']['diff_sd_3rep_K23_def']:.4f} A (noise={rec['replica_floor']['single_rep_noise_sd_3rep']:.4f} A, rep_resid_r={rep_res_corr:.3f}) | "
        f"half_r={r_half_p:.3f} | lag1_vario={autocorr['lag1_100ps']['variogram']:.3f}",
        flush=True,
    )

    # K3-02 H-bond network test (for hb_form and hb_diss where X_all has 30 cols = 20 base + 10 K3-02)
    if X_all.shape[1] >= 30:
        K302 = X_all[:, 20:30]
        pred_plus = oof_reg(X_all[:, :30], mean_all, P)
        res_plus = mean_all - pred_plus
        # Within-(protein x replica) shuffle control
        rng = np.random.default_rng(20261126)
        K302_sh = K302.copy()
        pr_keys = np.array([f"{p}_R{r}" for p, r in zip(P, R)])
        for g in np.unique(pr_keys):
            idx = np.where(pr_keys == g)[0]
            K302_sh[idx] = K302_sh[idx][rng.permutation(len(idx))]
        pred_sh = oof_reg(np.column_stack([X_base, K302_sh]), mean_all, P)
        res_sh = mean_all - pred_sh

        corrs = {}
        for j, nm in enumerate(K302_NAMES):
            v = K302[:, j]
            rp = float(pearsonr(v, res_all)[0]) if v.std() > 0 else 0.0
            rs = float(spearmanr(v, res_all)[0]) if v.std() > 0 else 0.0
            corrs[nm] = {"pearson": rp, "spearman": rs, "mean": float(v.mean()), "sd": float(v.std())}
        max_p = max(abs(c["pearson"]) for c in corrs.values())
        max_s = max(abs(c["spearman"]) for c in corrs.values())
        rec["k3_02_hb_network"] = {
            "base20_resid_sd": float(res_all.std()),
            "base20_plus_k302_resid_sd": float(res_plus.std()),
            "shuffle_control_resid_sd": float(res_sh.std()),
            "sd_delta": float(res_plus.std() - res_all.std()),
            "max_abs_pearson": max_p,
            "max_abs_spearman": max_s,
            "per_feature": corrs,
        }
        print(
            f"  -> K3-02 ({mode}): max|r|={max_p:.4f} (Spearman {max_s:.4f}) | "
            f"SD: {res_all.std():.4f} -> {res_plus.std():.4f} A (shuffle: {res_sh.std():.4f} A)",
            flush=True,
        )
    return rec


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(CACHE.glob("*.npz"))
    assert len(files) >= 50, f"Too few extracted files: {len(files)}"
    print(f"Running E02 Part 2 on {len(files)} unseen ATLAS proteins...", flush=True)

    out = {
        "n_proteins": len(files),
        "phenomena": {
            "hb_form": analyze_phenomenon(files, "hb_form", n_base_cols=20),
            "hb_diss": analyze_phenomenon(files, "hb_diss", n_base_cols=20),
            "ct": analyze_phenomenon(files, "ct", n_base_cols=5),
            "sb": analyze_phenomenon(files, "sb", n_base_cols=5),
        },
        "elapsed_sec": time.time() - t0,
    }
    (OUT / "e02_part2_atlas100_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved e02_part2_atlas100_results.json ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
