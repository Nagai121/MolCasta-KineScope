"""K3-03: does the assigned physical quantity track the residual?

Strong base (K06(b)): the residual ("zure") is the part of the real 0-1 ns
mean donor-acceptor distance that a static-only prediction cannot explain,
built exactly as in K39/K42 (linear and GBM regression from the 5 static
features, fitted on the 8 calibration proteins only). Test = 100 unseen
proteins x 3 replicas.

Readings fixed in advance:
  - |Pearson or Spearman correlation| >= 0.3 with the residual -> "related"
    (same line as K2-03).
  - Adding the quantity to the static prediction: residual std <= 0.30 A ->
    "explains most of the residual". The floor is 0.249 A.
  - Swap control: the quantity permuted across pairs within each
    protein-replica must lose its effect.
Quantities come from work/k3_extract.py (frame 10 of the history).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

OUT = Path(__file__).resolve().parent
K06B_OUT = Path("/home/hatch/workspace/k06b/run/k06b_out")
K3_DATA = Path("/home/hatch/workspace/kinescope/work/k3_data")
SPLIT = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL = SPLIT["calibration_proteins"]
TEST = SPLIT["test_proteins"]
SEED = 20261121

RAW_KEYS = ["e_par", "e_mag"]
SENS_RAW_KEYS = ["e_par_hsd", "e_mag_hsd"]


def build_features(Qraw):
    return Qraw, ["e_parallel_V_per_A", "e_magnitude_V_per_A"]


def gbm():
    return HistGradientBoostingRegressor(
        max_iter=160, learning_rate=0.065, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=0.15, random_state=SEED)


def load(proteins, raw_keys):
    S, M, Q, meta = [], [], [], []
    for p in proteins:
        k = np.load(K06B_OUT / f"{p}.npz")
        z = np.load(K3_DATA / f"{p}.npz")
        for rep in (1, 2, 3):
            S.append(k[f"r{rep}_static"].astype(np.float64))
            M.append(k[f"r{rep}_history"][:, 1].astype(np.float64))
            Q.append(np.column_stack(
                [z[f"r{rep}_{key}"].astype(np.float64) for key in raw_keys]))
            vd = float(z[f"r{rep}_verify_d10_maxdiff"][0])
            vm = float(z[f"r{rep}_verify_dmean_maxdiff"][0])
            assert vd < 1e-4 and vm < 1e-4, (p, rep, vd, vm)
            meta.extend([(p, rep)] * len(k[f"r{rep}_static"]))
    Qf, labels = build_features(np.vstack(Q))
    return np.vstack(S), np.concatenate(M), Qf, meta, labels


def corr_block(resid, Q, meta, labels):
    out = {}
    prot_idx = {}
    for i, (p, _) in enumerate(meta):
        prot_idx.setdefault(p, []).append(i)
    for j, lab in enumerate(labels):
        q = Q[:, j]
        ok = np.isfinite(q) & np.isfinite(resid)
        pr = pearsonr(q[ok], resid[ok])[0]
        sr = spearmanr(q[ok], resid[ok])[0]
        per_protein = []
        for p, rows in prot_idx.items():
            rows = np.array(rows)
            qq, rr = Q[rows, j], resid[rows]
            okp = np.isfinite(qq) & np.isfinite(rr)
            if okp.sum() > 10 and np.std(qq[okp]) > 0 and np.std(rr[okp]) > 0:
                per_protein.append(pearsonr(qq[okp], rr[okp])[0])
        per_protein = np.array(per_protein)
        out[lab] = {
            "pearson_pooled": float(pr), "spearman_pooled": float(sr),
            "n": int(ok.sum()),
            "per_protein_pearson_median": float(np.median(per_protein)),
            "per_protein_pearson_q1": float(np.percentile(per_protein, 25)),
            "per_protein_pearson_q3": float(np.percentile(per_protein, 75)),
            "per_protein_frac_abs_ge_0.3": float((np.abs(per_protein) >= 0.3).mean()),
            "related_by_fixed_line": bool(max(abs(pr), abs(sr)) >= 0.3),
        }
    return out


def add_model_std(cal_S, cal_M, cal_Q, te_S, te_M, te_Q, kind):
    Xc = np.hstack([cal_S, cal_Q])
    Xt = np.hstack([te_S, te_Q])
    if kind == "gbm":
        m = gbm().fit(Xc, cal_M)
    else:
        m = LinearRegression().fit(np.nan_to_num(Xc), cal_M)
        Xt = np.nan_to_num(Xt)
    return float(np.std(te_M - m.predict(Xt)))


def main():
    cal_S, cal_M, cal_Q, _, labels = load(CAL, RAW_KEYS)
    te_S, te_M, te_Q, te_meta, _ = load(TEST, RAW_KEYS)

    # residual, K39/K42 construction
    base = {}
    resid = {}
    for kind in ("gbm", "linear"):
        if kind == "gbm":
            m = gbm().fit(cal_S, cal_M)
        else:
            m = LinearRegression().fit(cal_S, cal_M)
        pred_cal = m.predict(cal_S)
        pred_te = m.predict(te_S)
        base[kind] = {
            "residual_std_calibration": float(np.std(cal_M - pred_cal)),
            "residual_std_test": float(np.std(te_M - pred_te)),
        }
        resid[kind] = te_M - pred_te
    check = {
        "gbm_test_std_expected_from_K42": 0.60,
        "linear_test_std_expected_from_K42": 0.61,
        "gbm_ok": abs(base["gbm"]["residual_std_test"] - 0.60) < 0.02,
        "linear_ok": abs(base["linear"]["residual_std_test"] - 0.61) < 0.02,
    }
    print("base residual std:", base)
    print("premise check vs K42:", check)
    assert check["gbm_ok"] and check["linear_ok"]

    correlations = {
        "vs_gbm_residual": corr_block(resid["gbm"], te_Q, te_meta, labels),
        "vs_linear_residual": corr_block(resid["linear"], te_Q, te_meta, labels),
    }

    added = {}
    for kind in ("gbm", "linear"):
        added[kind] = {
            "residual_std_test_with_quantity": add_model_std(
                cal_S, cal_M, cal_Q, te_S, te_M, te_Q, kind),
        }
        added[kind]["baseline_residual_std_test"] = base[kind]["residual_std_test"]
        added[kind]["explains_most_by_fixed_line"] = bool(
            added[kind]["residual_std_test_with_quantity"] <= 0.30)

    # swap control: permute quantity rows within each protein-replica
    rng = np.random.default_rng(42)
    groups = {}
    for i, key in enumerate(te_meta):
        groups.setdefault(key, []).append(i)
    perm = np.arange(len(te_M))
    for rows in groups.values():
        rows = np.array(rows)
        perm[rows] = rng.permutation(rows)
    te_Q_swapped = te_Q[perm]
    cal_groups = {}
    swap_added_std = add_model_std(cal_S, cal_M, cal_Q, te_S, te_M, te_Q_swapped, "gbm")
    swap_corr = corr_block(resid["gbm"], te_Q_swapped, te_meta, labels)

    sensitivity = None
    if SENS_RAW_KEYS is not None:
        _, _, cal_Qs, _, slabels = load(CAL, SENS_RAW_KEYS)
        _, _, te_Qs, _, _ = load(TEST, SENS_RAW_KEYS)
        sensitivity = {
            "correlations_vs_gbm_residual": corr_block(resid["gbm"], te_Qs, te_meta, slabels),
            "residual_std_test_with_quantity_gbm": add_model_std(
                cal_S, cal_M, cal_Qs, te_S, te_M, te_Qs, "gbm"),
        }

    results = {
        "experiment": "K3-03",
        "phenomenon": "hydrogen_bond_formation",
        "base": "strong (K06(b)): correction model on frozen T1I2006, calibration 8 proteins, test 100 unseen proteins x 3 replicas",
        "residual_definition": "real mean distance over history frames 0-10 minus static-only prediction (5 static features), regressors fitted on calibration candidates only (K39/K42 construction)",
        "n_calibration": int(len(cal_M)), "n_test": int(len(te_M)),
        "quantity_keys": RAW_KEYS, "quantity_labels": labels,
        "quantity_source": "electric field at the pair midpoint from CHARMM36 (Jul2022 GROMACS port) partial charges of heavy atoms at the origin frame (frame 10); donor and acceptor atoms excluded; HIS = HSE primary, HSD sensitivity; termini patched (NH3+ / COO-)",
        "base_residual_std": base,
        "premise_check_vs_K42": check,
        "correlations": correlations,
        "added_to_static_prediction": added,
        "sensitivity": sensitivity,
        "swap_control": {
            "residual_std_test_with_swapped_quantity_gbm": swap_added_std,
            "correlations_vs_gbm_residual": swap_corr,
        },
        "fixed_lines": {"correlation_abs": 0.3, "residual_std_explains_most_A": 0.30,
                        "residual_std_floor_A": 0.249},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    rows = ["quantity,pearson_vs_gbm_residual,spearman_vs_gbm_residual,pearson_vs_linear_residual,spearman_vs_linear_residual,related_by_fixed_line"]
    for lab in labels:
        a = correlations["vs_gbm_residual"][lab]
        b = correlations["vs_linear_residual"][lab]
        rows.append(f"{lab},{a['pearson_pooled']:.4f},{a['spearman_pooled']:.4f},"
                    f"{b['pearson_pooled']:.4f},{b['spearman_pooled']:.4f},{a['related_by_fixed_line']}")
    (OUT / "results.csv").write_text("\n".join(rows) + "\n")

    np.savez_compressed(
        OUT / "analysis_data.npz",
        residual_gbm=resid["gbm"].astype(np.float32),
        residual_linear=resid["linear"].astype(np.float32),
        quantities=te_Q.astype(np.float32),
        meta=np.array([f"{p}_R{r}" for p, r in te_meta]))
    print(json.dumps({k: correlations["vs_gbm_residual"][k] for k in labels}, indent=1))
    print("added:", added)
    print("swap std (gbm):", swap_added_std)


if __name__ == "__main__":
    main()
