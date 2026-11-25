"""E01: Rigorous re-verification of Finding 1 (Short Real MD Guidance) on 100 unseen ATLAS proteins.

Eliminates BOTH flaws of KineScope G02-G180 / L01-L74:
  1. Eliminates the 1.0 ns time-gap trap (frames [0,1] paired with static origin frame 10).
  2. Eliminates proprietary T1I2006 and 8-protein calibration degradation via a 100% public
     20-feature structural baseline evaluated with strict 5-fold protein-grouped CV across 100 proteins.

Protocols compared side-by-side (horizons: 1, 5, 10, 30, 50, 90 ns):
  - Protocol A (TRUE backward/adjacent window ending at origin frame 10):
      L = 0.1 ns -> [9, 10], 0.2 ns -> [8..10], 0.3 ns -> [7..10], 0.5 ns -> [5..10], 1.0 ns -> [0..10]
      Evaluated across ALL 4 phenomena on all 100 proteins:
        HB formation, HB dissociation, Contact formation, Salt-bridge formation.
  - Protocol B (TRUE forward prefix from t=0 with moving origin t=L):
      Origin k in {0, 1, 2, 5, 10} (L = 0.0, 0.1, 0.2, 0.5, 1.0 ns), labels at k + horizon.
      Evaluated on HB formation (100 unseen proteins).
  - Control C (Original G/L FLAWED prefix [0..k] with fixed origin frame 10):
      k in {0, 1, 2, 5, 10} on the exact same public baseline & 100 proteins to quantify the gap artifact.
"""
import json
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

CACHE = Path(r"d:\KineScope\experiments\shared_data\extracted_100")
OUT = Path(r"d:\KineScope\experiments\E01_true_short_md_guidance")
HORIZON_NS = [1, 5, 10, 30, 50, 90]

CLF_KW = dict(
    max_iter=160,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=25,
    l2_regularization=0.15,
    random_state=20261126,
)


def hist_stats(ser: np.ndarray, thresh: float) -> np.ndarray:
    """ser: (N, T) -> 6 stats: first, mean, std, min, last-first, frac<=thresh."""
    return np.column_stack([
        ser[:, 0],
        ser.mean(axis=1),
        ser.std(axis=1),
        ser.min(axis=1),
        ser[:, -1] - ser[:, 0],
        (ser <= thresh).mean(axis=1),
    ]).astype(np.float32)


def oof_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(**CLF_KW)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def eval_predictions(y: np.ndarray, p_static: np.ndarray, p_full: np.ndarray,
                     p_cond: np.ndarray, prots: np.ndarray, n_boot: int = 1000):
    """Compute pooled & protein-macro AUPRC, gain, recovery, and protein-bootstrap 95% CI."""
    ap_s = float(average_precision_score(y, p_static))
    ap_f = float(average_precision_score(y, p_full))
    ap_c = float(average_precision_score(y, p_cond))
    gain_f = ap_f - ap_s
    gain_c = ap_c - ap_s
    rec_pooled = float(gain_c / gain_f) if gain_f > 1e-6 else float("nan")

    # Per-protein macro AUPRC (only proteins with both classes present)
    uniq = np.unique(prots)
    pp_s, pp_f, pp_c = [], [], []
    for p in uniq:
        m = prots == p
        yp = y[m]
        if yp.sum() == 0 or yp.sum() == len(yp):
            continue
        pp_s.append(average_precision_score(yp, p_static[m]))
        pp_f.append(average_precision_score(yp, p_full[m]))
        pp_c.append(average_precision_score(yp, p_cond[m]))
    pp_s = np.asarray(pp_s)
    pp_f = np.asarray(pp_f)
    pp_c = np.asarray(pp_c)
    macro_s = float(pp_s.mean())
    macro_f = float(pp_f.mean())
    macro_c = float(pp_c.mean())
    macro_gain_f = macro_f - macro_s
    macro_gain_c = macro_c - macro_s
    rec_macro = float(macro_gain_c / macro_gain_f) if macro_gain_f > 1e-6 else float("nan")

    # Protein-level bootstrap 95% CI on macro recovery
    rng = np.random.default_rng(42)
    n_p = len(pp_s)
    idx = rng.integers(0, n_p, size=(n_boot, n_p))
    gf_b = (pp_f[idx] - pp_s[idx]).mean(axis=1)
    gc_b = (pp_c[idx] - pp_s[idx]).mean(axis=1)
    valid = gf_b > 1e-6
    if valid.sum() > 50:
        rb = gc_b[valid] / gf_b[valid]
        ci_lo, ci_hi = float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    return {
        "ap_static": ap_s,
        "ap_full": ap_f,
        "ap_cond": ap_c,
        "gain_full_pooled": gain_f,
        "gain_cond_pooled": gain_c,
        "recovery_pooled": rec_pooled,
        "macro_static": macro_s,
        "macro_full": macro_f,
        "macro_cond": macro_c,
        "macro_gain_full": macro_gain_f,
        "macro_gain_cond": macro_gain_c,
        "recovery_macro": rec_macro,
        "recovery_macro_ci95": [ci_lo, ci_hi],
        "n_valid_proteins": int(n_p),
        "auc_static": float(roc_auc_score(y, p_static)),
        "auc_full": float(roc_auc_score(y, p_full)),
        "auc_cond": float(roc_auc_score(y, p_cond)),
    }


def load_fixed_origin(files, mode: str, n_static_cols: int = None):
    Xs, Ss, Ys, Ps = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            prot = str(z["protein"])
            for r in (1, 2, 3):
                X = z[f"r{r}_{mode}_X"]
                if len(X) == 0:
                    continue
                if n_static_cols is not None:
                    X = X[:, :n_static_cols]
                Xs.append(X)
                Ss.append(z[f"r{r}_{mode}_series"])
                Ys.append(z[f"r{r}_{mode}_y"])
                Ps.append(np.full(len(X), prot, dtype=object))
    return (
        np.concatenate(Xs).astype(np.float32),
        np.concatenate(Ss).astype(np.float32),
        np.concatenate(Ys).astype(np.uint8),
        np.concatenate(Ps),
    )


def run_protocol_a_and_c(files, mode: str, thresh: float, n_static_cols: int = None, run_control_c: bool = True):
    print(f"\n================ {mode.upper()} (100 unseen ATLAS proteins) ================", flush=True)
    X, S, Y, P = load_fixed_origin(files, mode, n_static_cols)
    print(f"Candidates: {len(X):,} | Proteins: {len(np.unique(P))} | Positives by horizon: {Y.sum(axis=0).tolist()}", flush=True)

    # Full 1.0 ns history = 6 stats on frames [0..10]
    H_full = hist_stats(S, thresh)
    X_full = np.column_stack([X, H_full])

    # Protocol A: TRUE adjacent windows ending at origin frame 10
    proto_a_windows = {
        "A_true_0.1ns_mean": lambda s: s[:, 9:11].mean(axis=1, keepdims=True),
        "A_true_0.2ns_mean": lambda s: s[:, 8:11].mean(axis=1, keepdims=True),
        "A_true_0.3ns_mean": lambda s: s[:, 7:11].mean(axis=1, keepdims=True),
        "A_true_0.5ns_mean": lambda s: s[:, 5:11].mean(axis=1, keepdims=True),
        "A_true_1.0ns_mean": lambda s: s[:, 0:11].mean(axis=1, keepdims=True),
        "A_true_0.1ns_6stats": lambda s: hist_stats(s[:, 9:11], thresh),
        "A_true_0.2ns_6stats": lambda s: hist_stats(s[:, 8:11], thresh),
        "A_true_0.5ns_6stats": lambda s: hist_stats(s[:, 5:11], thresh),
    }

    # Control C: Original flawed G/L prefix [0..k] paired with origin frame 10
    control_c_windows = {
        "C_flawed_0.0ns_f0": lambda s: s[:, 0:1],
        "C_flawed_0.1ns_f01_mean": lambda s: s[:, 0:2].mean(axis=1, keepdims=True),
        "C_flawed_0.2ns_f02_mean": lambda s: s[:, 0:3].mean(axis=1, keepdims=True),
        "C_flawed_0.5ns_f05_mean": lambda s: s[:, 0:6].mean(axis=1, keepdims=True),
    } if run_control_c else {}

    out = {"mode": mode, "n_candidates": int(len(X)), "n_proteins": int(len(np.unique(P))), "by_horizon": {}}
    all_windows = {**proto_a_windows, **control_c_windows}

    for hi, h_ns in enumerate(HORIZON_NS):
        y = Y[:, hi]
        p_s = oof_predict(X, y, P)
        p_f = oof_predict(X_full, y, P)
        h_res = {}
        for cname, fn in all_windows.items():
            Xc = np.column_stack([X, fn(S)])
            p_c = oof_predict(Xc, y, P)
            h_res[cname] = eval_predictions(y, p_s, p_f, p_c, P)
        out["by_horizon"][str(h_ns)] = h_res

        # Print concise summary line
        a01 = h_res["A_true_0.1ns_mean"]
        a05 = h_res["A_true_0.5ns_mean"]
        a10 = h_res["A_true_1.0ns_mean"]
        line = (
            f"  H={h_ns:2d}ns | static={a01['ap_static']:.4f} full={a01['ap_full']:.4f} (gain=+{a01['gain_full_pooled']:.4f}) | "
            f"TRUE 0.1ns={a01['recovery_pooled']:.3f} (macro {a01['recovery_macro']:.3f}) | "
            f"TRUE 0.5ns={a05['recovery_pooled']:.3f} | 1.0ns_mean={a10['recovery_pooled']:.3f}"
        )
        if run_control_c:
            c00 = h_res["C_flawed_0.0ns_f0"]
            c01 = h_res["C_flawed_0.1ns_f01_mean"]
            line += f" || FLAWED f0={c00['recovery_pooled']:.3f}, f01={c01['recovery_pooled']:.3f}"
        print(line, flush=True)
    return out


def run_protocol_b(files):
    """Protocol B: TRUE forward prefix from t=0 with moving origin t=L (HB formation, 100 proteins)."""
    print("\n================ PROTOCOL B: TRUE FORWARD PREFIX (moving origin t=L, HB formation) ================", flush=True)
    out = {"by_origin_ns": {}}
    # k=0,1,2,5 from pb_k{k}; k=10 from hb_form
    origins = [(0, "0.0"), (1, "0.1"), (2, "0.2"), (5, "0.5"), (10, "1.0")]
    for k, label in origins:
        Xs, Hs, Ys, Ps = [], [], [], []
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                prot = str(z["protein"])
                for r in (1, 2, 3):
                    if k < 10:
                        X = z[f"r{r}_pb_k{k}_X"]
                        H = z[f"r{r}_pb_k{k}_h6"]
                        Y = z[f"r{r}_pb_k{k}_y"]
                    else:
                        X = z[f"r{r}_hb_form_X"][:, :20]
                        H = hist_stats(z[f"r{r}_hb_form_series"], 3.5)
                        Y = z[f"r{r}_hb_form_y"]
                    Xs.append(X)
                    Hs.append(H)
                    Ys.append(Y)
                    Ps.append(np.full(len(X), prot, dtype=object))
        X = np.concatenate(Xs).astype(np.float32)
        H = np.concatenate(Hs).astype(np.float32)
        Y = np.concatenate(Ys).astype(np.uint8)
        P = np.concatenate(Ps)
        res_k = {"origin_frame": k, "n_candidates": int(len(X)), "by_horizon": {}}
        for hi, h_ns in enumerate(HORIZON_NS):
            y = Y[:, hi]
            p_s = oof_predict(X, y, P)
            ap_s = float(average_precision_score(y, p_s))
            if k == 0:
                res_k["by_horizon"][str(h_ns)] = {
                    "ap_static": ap_s, "ap_mean": ap_s, "ap_6stats": ap_s,
                    "gain_mean": 0.0, "gain_6stats": 0.0,
                }
            else:
                p_m = oof_predict(np.column_stack([X, H[:, 1:2]]), y, P)
                p_6 = oof_predict(np.column_stack([X, H]), y, P)
                ap_m = float(average_precision_score(y, p_m))
                ap_6 = float(average_precision_score(y, p_6))
                res_k["by_horizon"][str(h_ns)] = {
                    "ap_static": ap_s, "ap_mean": ap_m, "ap_6stats": ap_6,
                    "gain_mean": ap_m - ap_s, "gain_6stats": ap_6 - ap_s,
                }
        out["by_origin_ns"][label] = res_k
        g1 = res_k["by_horizon"]["1"]["gain_6stats"]
        g10 = res_k["by_horizon"]["10"]["gain_6stats"]
        g90 = res_k["by_horizon"]["90"]["gain_6stats"]
        gm1 = res_k["by_horizon"]["1"]["gain_mean"]
        gm90 = res_k["by_horizon"]["90"]["gain_mean"]
        print(f"  Origin L={label}ns (N={len(X):,}) | 6stats gain: 1ns=+{g1:.4f}, 10ns=+{g10:.4f}, 90ns=+{g90:.4f} | mean gain: 1ns=+{gm1:.4f}, 90ns=+{gm90:.4f}", flush=True)

    # Compute recovery relative to L=1.0ns 6stats gain
    out["recovery_vs_1ns_6stats"] = {}
    for h_ns in HORIZON_NS:
        hs = str(h_ns)
        ref = out["by_origin_ns"]["1.0"]["by_horizon"][hs]["gain_6stats"]
        out["recovery_vs_1ns_6stats"][hs] = {
            lbl: {
                "mean_rec": float(out["by_origin_ns"][lbl]["by_horizon"][hs]["gain_mean"] / ref) if ref > 1e-6 else float("nan"),
                "stats6_rec": float(out["by_origin_ns"][lbl]["by_horizon"][hs]["gain_6stats"] / ref) if ref > 1e-6 else float("nan"),
            }
            for _, lbl in origins[1:]
        }
    return out


def main():
    t0 = time.time()
    files = sorted(CACHE.glob("*.npz"))
    assert len(files) >= 50, f"Too few extracted files: {len(files)}"
    print(f"Loaded {len(files)} extracted protein bundles.", flush=True)

    results = {
        "n_proteins": len(files),
        "protocol_a_and_c": {},
    }
    # Use 20 base public features (excluding K3-02 HB network cols 20..29 so static baseline is clean geometry/packing)
    results["protocol_a_and_c"]["hb_form"] = run_protocol_a_and_c(files, "hb_form", 3.5, n_static_cols=20, run_control_c=True)
    results["protocol_a_and_c"]["hb_diss"] = run_protocol_a_and_c(files, "hb_diss", 3.5, n_static_cols=20, run_control_c=True)
    results["protocol_a_and_c"]["ct"] = run_protocol_a_and_c(files, "ct", 4.5, n_static_cols=None, run_control_c=True)
    results["protocol_a_and_c"]["sb"] = run_protocol_a_and_c(files, "sb", 4.0, n_static_cols=None, run_control_c=True)
    results["protocol_b_hb_form"] = run_protocol_b(files)
    results["elapsed_sec"] = time.time() - t0

    (OUT / "e01_short_md_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nE01 short-MD verification finished in {time.time() - t0:.1f}s -> e01_short_md_results.json", flush=True)


if __name__ == "__main__":
    main()
