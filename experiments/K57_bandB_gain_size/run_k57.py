"""K57: re-analysis of K55 — is band B's mean-only recovery spread just
because its gain is small? (HB formation, D conditions).

K55 saved only pooled (1mix_A + 2qt4_A) metrics, no per-protein
probabilities. This script re-runs K55's training exactly (same cache,
same HistGradientBoosting settings, same seed 42, same band-restricted
train/test split) and evaluates per test protein separately, so the
pooled values can be checked against K55's results.json.

Bands: A 3.5 < d0 <= 4.5, B 4.5 < d0 <= 6.0 (origin distance, frame 10).
Conditions per band: static only, static + full 6 history stats,
static + mean only. Horizons 1/5/10/30/50/90 ns.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).parent
K55_RESULTS = Path(__file__).parent.parent / "K55_mean_guidance_by_distance_band" / "results.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BANDS = {"A_3.5-4.5": (3.5, 4.5), "B_4.5-6.0": (4.5, 6.0)}


def load_protein(p):
    parts = []
    for r in (1, 2, 3):
        z = np.load(CACHE / f"{p}_R{r}.npz")
        parts.append({k: z[k] for k in z.files})
    X = np.concatenate([q["X"] for q in parts])
    H = np.concatenate([q["hist"].T for q in parts])  # (n, 11)
    Y = np.concatenate([q["y"] for q in parts])
    return X, H, Y


def hist6(series, bound=3.5):
    return np.stack([series[:, 0], series.mean(1), series.std(1),
                     series.min(1), series[:, -1] - series[:, 0],
                     (series <= bound).mean(1)], axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def auprc(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(average_precision_score(y, p))


def main():
    Xtr = {p: load_protein(p) for p in TRAIN_P}
    Xte = {p: load_protein(p) for p in TEST_P}
    # concatenate train across proteins, keep test per protein
    Xtr_all = np.concatenate([Xtr[p][0] for p in TRAIN_P])
    Htr_all = np.concatenate([Xtr[p][1] for p in TRAIN_P])
    Ytr_all = np.concatenate([Xtr[p][2] for p in TRAIN_P])
    Ftr_all = hist6(Htr_all)
    k55 = json.loads(K55_RESULTS.read_text())
    results = {"bands": {}, "gain_vs_deviation": []}
    for band, (lo, hi) in BANDS.items():
        mtr = (Xtr_all[:, 0] > lo) & (Xtr_all[:, 0] <= hi)
        results["bands"][band] = {}
        for name, ftr_all, fte_fn in (
                ("static_only", None, None),
                ("full_hist6", Ftr_all, lambda H: hist6(H)),
                ("mean_only", Ftr_all[:, [1]], lambda H: hist6(H)[:, [1]])):
            results["bands"][band][name] = {"pooled": {}, "per_protein": {}}
            for h in HORIZONS:
                hi_idx = HORIZONS.index(h)
                ytr = Ytr_all[mtr, hi_idx]
                # pooled test
                preds = {}
                for p in TEST_P:
                    Xp, Hp, Yp = Xte[p]
                    mte = (Xp[:, 0] > lo) & (Xp[:, 0] <= hi)
                    if ftr_all is None:
                        pr = predict(Xtr_all[mtr], ytr, Xp[mte])
                    else:
                        fte = fte_fn(Hp)
                        pr = predict(np.concatenate([Xtr_all[mtr], ftr_all[mtr]], axis=1), ytr,
                                     np.concatenate([Xp[mte], fte[mte]], axis=1))
                    preds[p] = (Yp[mte, hi_idx], pr, int(mte.sum()))
                y_pool = np.concatenate([preds[p][0] for p in TEST_P])
                p_pool = np.concatenate([preds[p][1] for p in TEST_P])
                results["bands"][band][name]["pooled"][str(h)] = {
                    "n": int(len(y_pool)), "positives": int(y_pool.sum()),
                    "auprc": auprc(y_pool, p_pool)}
                for p in TEST_P:
                    y, pr, n = preds[p]
                    results["bands"][band][name]["per_protein"].setdefault(p, {})[str(h)] = {
                        "n": n, "positives": int(y.sum()), "auprc": auprc(y, pr)}
                print(band, name, h, "pooled auprc",
                      round(results["bands"][band][name]["pooled"][str(h)]["auprc"], 4),
                      "K55", round(k55[band][name][str(h)]["auprc"], 4), flush=True)
        # gain and recovery: pooled and per protein
        rec = results["bands"][band].setdefault("recovery", {})
        for h in HORIZONS:
            s = results["bands"][band]["static_only"]["pooled"][str(h)]["auprc"]
            f = results["bands"][band]["full_hist6"]["pooled"][str(h)]["auprc"]
            m = results["bands"][band]["mean_only"]["pooled"][str(h)]["auprc"]
            gain = f - s
            rec_pool = (m - s) / gain if gain else None
            per = {}
            for p in TEST_P:
                sp = results["bands"][band]["static_only"]["per_protein"][p][str(h)]["auprc"]
                fp = results["bands"][band]["full_hist6"]["per_protein"][p][str(h)]["auprc"]
                mp = results["bands"][band]["mean_only"]["per_protein"][p][str(h)]["auprc"]
                gp = fp - sp
                per[p] = {"gain": gp,
                          "recovery": (mp - sp) / gp if gp else None,
                          "auprc_static": sp, "auprc_full": fp, "auprc_mean": mp}
            vals = [per[p]["recovery"] for p in TEST_P if per[p]["recovery"] is not None]
            rec[str(h)] = {
                "gain_pooled": gain, "recovery_pooled": rec_pool,
                "per_protein": per,
                "recovery_macro_mean": float(np.mean(vals)) if vals else None,
                "recovery_range": float(max(vals) - min(vals)) if len(vals) == 2 else None,
                "gain_per_protein": {p: per[p]["gain"] for p in TEST_P},
            }
            results["gain_vs_deviation"].append({
                "band": band, "horizon": h, "gain": gain,
                "recovery_pooled": rec_pool,
                "deviation_from_1": abs(rec_pool - 1) if rec_pool is not None else None})
            print(band, h, f"gain {gain:+.4f} rec_pool {rec_pool:.3f}",
                  "per-protein rec",
                  {p: (round(per[p]['recovery'], 3) if per[p]['recovery'] is not None else None)
                   for p in TEST_P}, flush=True)
    # Spearman correlation between gain and |recovery-1| over the 12 cells
    # (rank-based, no scipy dependency)
    def _ranks(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind="stable")
        r = np.empty(len(a))
        r[order] = np.arange(1, len(a) + 1)
        return r
    g = [c["gain"] for c in results["gain_vs_deviation"]]
    d = [c["deviation_from_1"] for c in results["gain_vs_deviation"]]
    rho = float(np.corrcoef(_ranks(g), _ranks(d))[0, 1])
    results["gain_vs_deviation_summary"] = {
        "spearman_rho_gain_vs_abs_recovery_minus_1": float(rho),
        "n_cells": len(g)}
    print("Spearman gain vs |rec-1|:", round(float(rho), 3))
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
