"""K62: does band B's mean-only recovery stabilise without 2qt4_A?
(K55/K57/K60 follow-up, HB formation, D conditions).

K57 re-ran K55's training exactly (same cache, same HistGradientBoosting
settings, same seed 42, same band-restricted train/test split) and
evaluated per test protein. This script does the same training and
evaluates on 1mix_A ONLY (2qt4_A dropped from the test set). The
training side is unchanged: K55's band-restricted training set
(4 train proteins, band-restricted). Only the evaluation set changes.

Bands: A 3.5 < d0 <= 4.5, B 4.5 < d0 <= 6.0 (origin distance, frame 10).
Conditions per band: static only, static + full 6 history stats,
static + mean only. Horizons 1/5/10/30/50/90 ns.

Fixed reading (decided before seeing the results):
- If dropping 2qt4_A makes the recovery >= 0.8 at every horizon, the
  spread was down to 2qt4_A alone.
- Otherwise (a horizon below 0.8, or a horizon at/above 1.5), band B's
  mean carrier is not stable in the first place.
- With 2qt4_A dropped the test set is 1mix_A only, so "pooled" and
  "per-protein mean" are the same number (one protein). The 1mix_A-only
  recovery must match K60's per-protein 1mix_A values; a mismatch is
  investigated, not papered over.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).parent
K55_RESULTS = Path(__file__).parent.parent / "K55_mean_guidance_by_distance_band" / "results.json"
K57_RESULTS = Path(__file__).parent.parent / "K57_bandB_gain_size" / "results.json"
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
    Xtr_all = np.concatenate([Xtr[p][0] for p in TRAIN_P])
    Htr_all = np.concatenate([Xtr[p][1] for p in TRAIN_P])
    Ytr_all = np.concatenate([Xtr[p][2] for p in TRAIN_P])
    Ftr_all = hist6(Htr_all)
    k55 = json.loads(K55_RESULTS.read_text())
    k57 = json.loads(K57_RESULTS.read_text())
    results = {"bands": {}, "checks": {}}
    max_pooled_diff = 0.0
    max_1mix_diff = 0.0
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
                ap_pool = auprc(y_pool, p_pool)
                results["bands"][band][name]["pooled"][str(h)] = {
                    "n": int(len(y_pool)), "positives": int(y_pool.sum()), "auprc": ap_pool}
                max_pooled_diff = max(max_pooled_diff,
                                      abs(ap_pool - k55[band][name][str(h)]["auprc"]))
                for p in TEST_P:
                    y, pr, n = preds[p]
                    ap = auprc(y, pr)
                    results["bands"][band][name]["per_protein"].setdefault(p, {})[str(h)] = {
                        "n": n, "positives": int(y.sum()), "auprc": ap}
                # check per-protein 1mix_A recovery against K57 later
                ap_1mix = auprc(preds["1mix_A"][0], preds["1mix_A"][1])
                print(band, name, h, "pooled", round(ap_pool, 4),
                      "K55", round(k55[band][name][str(h)]["auprc"], 4),
                      "1mix", round(ap_1mix, 4) if ap_1mix is not None else None, flush=True)
        # recovery, 1mix_A only (= pooled after dropping 2qt4_A)
        rec = results["bands"][band].setdefault("recovery_1mix_only", {})
        for h in HORIZONS:
            s = results["bands"][band]["static_only"]["per_protein"]["1mix_A"][str(h)]["auprc"]
            f = results["bands"][band]["full_hist6"]["per_protein"]["1mix_A"][str(h)]["auprc"]
            m = results["bands"][band]["mean_only"]["per_protein"]["1mix_A"][str(h)]["auprc"]
            gain = f - s
            r = (m - s) / gain if gain else None
            k57_r = k57["bands"][band]["recovery"][str(h)]["per_protein"]["1mix_A"]["recovery"]
            if r is not None and k57_r is not None:
                max_1mix_diff = max(max_1mix_diff, abs(r - k57_r))
            rec[str(h)] = {
                "gain_1mix": gain, "recovery_1mix": r,
                "auprc_static": s, "auprc_full": f, "auprc_mean": m,
                "n": results["bands"][band]["static_only"]["per_protein"]["1mix_A"][str(h)]["n"],
                "positives": results["bands"][band]["static_only"]["per_protein"]["1mix_A"][str(h)]["positives"],
                "k57_recovery_1mix": k57_r,
                "k55_recovery_pooled": k55[band].get("recovery", {}).get(str(h)) if isinstance(k55[band].get("recovery"), dict) else None,
            }
            print(band, h, f"gain_1mix {gain:+.4f} rec_1mix {r:.3f} K57 {k57_r:.3f}", flush=True)
    results["checks"] = {
        "max_abs_diff_pooled_auprc_vs_K55": float(max_pooled_diff),
        "max_abs_diff_recovery_1mix_vs_K57": float(max_1mix_diff),
        "training_unchanged": "band-restricted training set identical to K55/K57; only the evaluation set drops 2qt4_A",
        "pooled_equals_per_protein_mean_after_drop": "test set is 1mix_A only, so pooled == per-protein (one protein)",
    }
    print("max pooled diff vs K55:", max_pooled_diff)
    print("max 1mix recovery diff vs K57:", max_1mix_diff)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
