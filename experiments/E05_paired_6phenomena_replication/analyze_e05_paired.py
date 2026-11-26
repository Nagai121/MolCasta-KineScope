"""E05 Step 5: Paired 6-phenomena analysis on the EXACT SAME strictly unseen atlas_test proteins
where real ATLAS R1, OpenMM GBn2 (4x100ps), and MDGen (100ns rollout) all succeeded.

Evaluates all 6 phenomena x 6 horizons (1, 5, 10, 30, 50, 90 ns) under identical 5-fold protein CV
with paired protein-level bootstrap 95% CIs (2,000 resamples):
  - Static explainability R^2 (real vs GBn2 vs MDGen short vs MDGen full)
  - Residual alignment r (GBn2 vs real, MDGen vs real, AND GBn2 vs MDGen)
  - Real-mean residual SD: static -> +GBn2_mean -> +MDGen_full -> +Hybrid(GBn2+MDGen) -> shuffle
  - Downstream AUPRC & recovery:
      static, real_1ns_6stats, gbn2_mean, gbn2_all7, mdgen_short, mdgen_full,
      hybrid_mean (gbn2_mean + mdgen_full), hybrid_all (gbn2_all7 + mdgen_full), shuffle
  - Paired AUPRC delta CIs: (GBn2 - MDGen), (Hybrid - GBn2), (Hybrid - MDGen)
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, r"d:\KineScope\experiments\E05_paired_6phenomena_replication")
import fetch_e05_atlas

OUT = Path(r"d:\KineScope\experiments\E05_paired_6phenomena_replication")
REAL_CACHE = OUT / "cache_real"
GBN2_CACHE = OUT / "cache_gbn2"
MDGEN_CACHE = OUT / "cache_mdgen"

MODES = ("hb_form", "hb_diss", "ct_form", "ct_diss", "sb_form", "sb_diss")
THRESH = {
    "hb_form": 3.5, "hb_diss": 3.5,
    "ct_form": 4.5, "ct_diss": 4.5,
    "sb_form": 4.0, "sb_diss": 4.0,
}
HORIZON_NS = [1, 5, 10, 30, 50, 90]

REG_KW = dict(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    min_samples_leaf=25, l2_regularization=0.2, random_state=20261126,
)
CLF_KW = dict(
    max_iter=160, learning_rate=0.06, max_leaf_nodes=31,
    min_samples_leaf=25, l2_regularization=0.15, random_state=20261126,
)


def oof_reg(X, y, groups, splits):
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in splits:
        m = HistGradientBoostingRegressor(**REG_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def oof_clf(X, y, groups, splits):
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in splits:
        m = HistGradientBoostingClassifier(**CLF_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def bootstrap_paired(y, preds_dict, groups, n_boot=2000, seed=20261126):
    """Protein-cluster bootstrap 95% CIs for recovery ratios and paired AUPRC deltas."""
    uniq = np.unique(groups)
    idx_by_p = [np.where(groups == p)[0] for p in uniq]
    n_p = len(uniq)
    rng = np.random.default_rng(seed)

    rec_keys = ["gbn2_mean", "gbn2_all7", "mdgen_full", "hybrid_all"]
    recs = {k: [] for k in rec_keys}
    d_gb_md, d_hy_gb, d_hy_md = [], [], []

    for _ in range(n_boot):
        draw = rng.integers(0, n_p, size=n_p)
        idx = np.concatenate([idx_by_p[i] for i in draw])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ap_s = average_precision_score(yb, preds_dict["static"][idx])
        ap_r = average_precision_score(yb, preds_dict["real_1ns"][idx])
        g_r = ap_r - ap_s
        ap_gm = average_precision_score(yb, preds_dict["gbn2_mean"][idx])
        ap_g7 = average_precision_score(yb, preds_dict["gbn2_all7"][idx])
        ap_mf = average_precision_score(yb, preds_dict["mdgen_full"][idx])
        ap_ha = average_precision_score(yb, preds_dict["hybrid_all"][idx])
        if g_r > 1e-4:
            recs["gbn2_mean"].append((ap_gm - ap_s) / g_r)
            recs["gbn2_all7"].append((ap_g7 - ap_s) / g_r)
            recs["mdgen_full"].append((ap_mf - ap_s) / g_r)
            recs["hybrid_all"].append((ap_ha - ap_s) / g_r)
        d_gb_md.append(ap_g7 - ap_mf)
        d_hy_gb.append(ap_ha - ap_g7)
        d_hy_md.append(ap_ha - ap_mf)

    def ci(arr):
        if len(arr) < 20:
            return [float("nan"), float("nan")]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "rec_ci95": {k: ci(recs[k]) for k in rec_keys},
        "delta_ap_gbn2_minus_mdgen_ci95": ci(d_gb_md),
        "delta_ap_hybrid_minus_gbn2_ci95": ci(d_hy_gb),
        "delta_ap_hybrid_minus_mdgen_ci95": ci(d_hy_md),
        "n_valid_boot": int(len(recs["gbn2_all7"])),
    }


def analyze_mode(prots, mode):
    Xs, Ss, Ys, Gs, Ms, M10s, Mfs, Ps = [], [], [], [], [], [], [], []
    for p in prots:
        zr = np.load(REAL_CACHE / f"{p}.npz", allow_pickle=True)
        zg = np.load(GBN2_CACHE / f"{p}.npz", allow_pickle=True)
        zm = np.load(MDGEN_CACHE / f"{p}.npz", allow_pickle=True)
        n = len(zr[f"{mode}_y"])
        if n == 0:
            continue
        vm = zg[f"{mode}_valid"].astype(bool) & zm[f"{mode}_valid"].astype(bool)
        if not vm.any():
            continue
        Xs.append(zr[f"{mode}_X"][vm])
        Ss.append(zr[f"{mode}_series"][vm])
        Ys.append(zr[f"{mode}_y"][vm])
        Gs.append(zg[f"{mode}_stats7"][vm])
        Ms.append(zm[f"{mode}_short"][vm])
        M10s.append(zm[f"{mode}_10ns"][vm])
        Mfs.append(zm[f"{mode}_full"][vm])
        Ps.append(np.full(int(vm.sum()), p, dtype=object))

    X = np.concatenate(Xs).astype(np.float64)
    S = np.concatenate(Ss).astype(np.float64)
    Y = np.concatenate(Ys).astype(np.uint8)
    G = np.concatenate(Gs).astype(np.float64)
    M_short = np.concatenate(Ms).astype(np.float64)
    M_10ns = np.concatenate(M10s).astype(np.float64)
    M_full = np.concatenate(Mfs).astype(np.float64)
    P = np.concatenate(Ps)
    n_prots_mode = int(len(np.unique(P)))
    thresh = THRESH[mode]

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, Y[:, 0], P))

    real_mean = S.mean(axis=1)
    real_h6 = np.column_stack([S[:, 0], S.mean(1), S.std(1), S.min(1), S[:, -1] - S[:, 0], (S <= thresh).mean(1)])
    g_mean = G[:, 0]
    ms_mean = M_short[:, 1]
    mf_mean = M_full[:, 1]

    # Within-protein shuffle control
    rng = np.random.default_rng(20261126)
    G_sh = G.copy()
    Mf_sh = M_full.copy()
    for pr in np.unique(P):
        idx = np.where(P == pr)[0]
        perm = rng.permutation(len(idx))
        G_sh[idx] = G_sh[idx][perm]
        Mf_sh[idx] = Mf_sh[idx][perm]

    # 1. Static explainability & residual alignment
    pred_real = oof_reg(X, real_mean, P, splits)
    pred_gbn2 = oof_reg(X, g_mean, P, splits)
    pred_ms = oof_reg(X, ms_mean, P, splits)
    pred_mf = oof_reg(X, mf_mean, P, splits)

    res_real = real_mean - pred_real
    res_gbn2 = g_mean - pred_gbn2
    res_ms = ms_mean - pred_ms
    res_mf = mf_mean - pred_mf

    pred_r_gbn2_m = oof_reg(np.column_stack([X, g_mean]), real_mean, P, splits)
    pred_r_gbn2_7 = oof_reg(np.column_stack([X, G]), real_mean, P, splits)
    pred_r_mdgen = oof_reg(np.column_stack([X, M_full]), real_mean, P, splits)
    pred_r_hybrid = oof_reg(np.column_stack([X, G, M_full]), real_mean, P, splits)
    pred_r_shuf = oof_reg(np.column_stack([X, G_sh, Mf_sh]), real_mean, P, splits)

    print(f"\n================ {mode.upper()} (N={len(real_mean):,} pairs, {n_prots_mode} proteins) ================", flush=True)
    print(
        f"Static R2: real={r2_score(real_mean, pred_real):.4f} (SD={res_real.std():.4f}A) | "
        f"GBn2={r2_score(g_mean, pred_gbn2):.4f} | MDGen_short={r2_score(ms_mean, pred_ms):.4f} | MDGen_full={r2_score(mf_mean, pred_mf):.4f}",
        flush=True,
    )
    print(
        f"Resid r vs real: GBn2={pearsonr(res_gbn2, res_real)[0]:+.4f} (rho={spearmanr(res_gbn2, res_real)[0]:+.4f}) | "
        f"MDGen_short={pearsonr(res_ms, res_real)[0]:+.4f} | MDGen_full={pearsonr(res_mf, res_real)[0]:+.4f} | "
        f"r(GBn2, MDGen_full)={pearsonr(res_gbn2, res_mf)[0]:+.4f}",
        flush=True,
    )
    print(
        f"Real-mean resid SD: static={res_real.std():.4f}A -> +GBn2_m={(real_mean-pred_r_gbn2_m).std():.4f}A "
        f"| +GBn2_7={(real_mean-pred_r_gbn2_7).std():.4f}A | +MDGen={(real_mean-pred_r_mdgen).std():.4f}A "
        f"| +Hybrid={(real_mean-pred_r_hybrid).std():.4f}A | shuf={(real_mean-pred_r_shuf).std():.4f}A",
        flush=True,
    )

    # 2. Downstream AUPRC across 6 horizons
    downstream = {}
    for hi, h_ns in enumerate(HORIZON_NS):
        y = Y[:, hi]
        pos_rate = float(y.mean())
        p_s = oof_clf(X, y, P, splits)
        p_r = oof_clf(np.column_stack([X, real_h6]), y, P, splits)
        p_gm = oof_clf(np.column_stack([X, g_mean]), y, P, splits)
        p_g7 = oof_clf(np.column_stack([X, G]), y, P, splits)
        p_ms = oof_clf(np.column_stack([X, M_short]), y, P, splits)
        p_mf = oof_clf(np.column_stack([X, M_full]), y, P, splits)
        p_hm = oof_clf(np.column_stack([X, g_mean, M_full]), y, P, splits)
        p_ha = oof_clf(np.column_stack([X, G, M_full]), y, P, splits)
        p_sh = oof_clf(np.column_stack([X, G_sh, Mf_sh]), y, P, splits)

        ap = {
            "static": float(average_precision_score(y, p_s)),
            "real_1ns": float(average_precision_score(y, p_r)),
            "gbn2_mean": float(average_precision_score(y, p_gm)),
            "gbn2_all7": float(average_precision_score(y, p_g7)),
            "mdgen_short": float(average_precision_score(y, p_ms)),
            "mdgen_full": float(average_precision_score(y, p_mf)),
            "hybrid_mean": float(average_precision_score(y, p_hm)),
            "hybrid_all": float(average_precision_score(y, p_ha)),
            "shuffle": float(average_precision_score(y, p_sh)),
        }
        g_r = ap["real_1ns"] - ap["static"]
        rec = {
            k: (float((ap[k] - ap["static"]) / g_r) if g_r > 1e-6 else float("nan"))
            for k in ("gbn2_mean", "gbn2_all7", "mdgen_short", "mdgen_full", "hybrid_mean", "hybrid_all", "shuffle")
        }
        boot = bootstrap_paired(
            y,
            {"static": p_s, "real_1ns": p_r, "gbn2_mean": p_gm, "gbn2_all7": p_g7, "mdgen_full": p_mf, "hybrid_all": p_ha},
            P,
            n_boot=2000,
        )
        downstream[str(h_ns)] = {
            "pos_rate": pos_rate,
            "n_pos": int(y.sum()),
            "ap": ap,
            "gain_real": g_r,
            "recovery": rec,
            "bootstrap": boot,
            "delta_ap_gbn2_minus_mdgen": ap["gbn2_all7"] - ap["mdgen_full"],
            "delta_ap_hybrid_minus_gbn2": ap["hybrid_all"] - ap["gbn2_all7"],
            "delta_ap_hybrid_minus_mdgen": ap["hybrid_all"] - ap["mdgen_full"],
        }
        print(
            f"  H={h_ns:2d}ns (pos={int(y.sum()):5d}) | st={ap['static']:.4f} real={ap['real_1ns']:.4f} (+{g_r:.4f}) | "
            f"GBn2_m={rec['gbn2_mean']:+.3f} GBn2_7={rec['gbn2_all7']:+.3f} | "
            f"MDG_s={rec['mdgen_short']:+.3f} MDG_f={rec['mdgen_full']:+.3f} | "
            f"HYB={rec['hybrid_all']:+.3f} | shuf={rec['shuffle']:+.3f}",
            flush=True,
        )

    return {
        "mode": mode,
        "n_candidates": int(len(real_mean)),
        "n_proteins": n_prots_mode,
        "static_r2": {
            "real_1ns": float(r2_score(real_mean, pred_real)),
            "gbn2": float(r2_score(g_mean, pred_gbn2)),
            "mdgen_short": float(r2_score(ms_mean, pred_ms)),
            "mdgen_full": float(r2_score(mf_mean, pred_mf)),
        },
        "residual_alignment": {
            "gbn2_vs_real_pearson": float(pearsonr(res_gbn2, res_real)[0]),
            "gbn2_vs_real_spearman": float(spearmanr(res_gbn2, res_real)[0]),
            "mdgen_short_vs_real_pearson": float(pearsonr(res_ms, res_real)[0]),
            "mdgen_full_vs_real_pearson": float(pearsonr(res_mf, res_real)[0]),
            "mdgen_full_vs_real_spearman": float(spearmanr(res_mf, res_real)[0]),
            "gbn2_vs_mdgen_full_pearson": float(pearsonr(res_gbn2, res_mf)[0]),
        },
        "real_mean_resid_sd": {
            "static": float(res_real.std()),
            "plus_gbn2_mean": float((real_mean - pred_r_gbn2_m).std()),
            "plus_gbn2_all7": float((real_mean - pred_r_gbn2_7).std()),
            "plus_mdgen_full": float((real_mean - pred_r_mdgen).std()),
            "plus_hybrid": float((real_mean - pred_r_hybrid).std()),
            "plus_shuffle": float((real_mean - pred_r_shuf).std()),
        },
        "downstream": downstream,
    }


def main():
    t0 = time.time()
    chosen = [p for p, _ in fetch_e05_atlas.select_e05_proteins(250)]
    paired = [
        p for p in chosen
        if (REAL_CACHE / f"{p}.npz").exists()
        and (GBN2_CACHE / f"{p}.npz").exists()
        and (MDGEN_CACHE / f"{p}.npz").exists()
    ]
    print(f"E05 Paired Analysis on {len(paired)}/{len(chosen)} strictly unseen atlas_test proteins (ALL 3 caches present)", flush=True)
    assert len(paired) >= 20, f"Too few paired proteins: {len(paired)}"

    gbn2_times = [float(np.load(GBN2_CACHE / f"{p}.npz", allow_pickle=True)["elapsed_s"]) for p in paired]
    mdgen_times = [float(np.load(MDGEN_CACHE / f"{p}.npz", allow_pickle=True)["elapsed_s"]) for p in paired]
    gbn2_errs = [float(np.load(GBN2_CACHE / f"{p}.npz", allow_pickle=True)["max_d0_err"]) for p in paired]
    mdgen_errs = [float(np.load(MDGEN_CACHE / f"{p}.npz", allow_pickle=True)["max_d0_err"]) for p in paired]

    res = {
        "n_paired_proteins": len(paired),
        "proteins": paired,
        "mean_gpu_sec_gbn2": float(np.mean(gbn2_times)),
        "mean_gpu_sec_mdgen": float(np.mean(mdgen_times)),
        "max_d0_err_gbn2": float(np.max(gbn2_errs)),
        "max_d0_err_mdgen": float(np.max(mdgen_errs)),
        "phenomena": {},
    }
    for mode in MODES:
        res["phenomena"][mode] = analyze_mode(paired, mode)

    res["analysis_elapsed_s"] = float(time.time() - t0)
    (OUT / "e05_paired_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved e05_paired_results.json in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
