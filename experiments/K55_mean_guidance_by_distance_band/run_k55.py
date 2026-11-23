"""K55: mean-only guidance by origin-distance band (HB formation, D conditions).

Bands by origin distance d0 = X[:, 0] (frame 10, 1 ns):
  A: 3.5 < d0 <= 4.5 (near the 3.5 A boundary)
  B: 4.5 < d0 <= 6.0 (farther from the boundary)
The base cache has d0 in (3.5, 6.0] only, so A + B cover all candidates.
For each band, BOTH train and test are restricted to that band and the
model is retrained (not train-on-all / evaluate-per-band).
Conditions per band: static only, static + full 6 history stats,
static + mean only (K21/K27 split: the mean column of the 6 stats only).
Model: HistGradientBoosting(120, lr .075, leaves 31, min_samples_leaf 15,
L2 .02, seed 42), refit per condition. Horizons 1/5/10/30/50/90 ns.
Metrics: AUC, AUPRC, Brier, Log Loss. Checkpointed per band x horizon.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss)

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).parent
PARTIAL = Path("/home/hatch/workspace/kinescope/work/k55/results_k55_partial.json")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BANDS = {"A_3.5-4.5": (3.5, 4.5), "B_4.5-6.0": (4.5, 6.0), "ALL": (3.5, 6.0)}


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            parts.append({k: z[k] for k in z.files})
    X = np.concatenate([p["X"] for p in parts])
    H = np.concatenate([p["hist"].T for p in parts])  # (n, 11)
    Y = np.concatenate([p["y"] for p in parts])
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


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def main():
    Xtr, Htr, Ytr = load(TRAIN_P)
    Xte, Hte, Yte = load(TEST_P)
    print("train", Xtr.shape, "test", Xte.shape, flush=True)
    print("origin distance: train",
          float(Xtr[:, 0].min()), float(Xtr[:, 0].max()),
          "test", float(Xte[:, 0].min()), float(Xte[:, 0].max()), flush=True)
    Ftr, Fte = hist6(Htr), hist6(Hte)  # col 1 = mean
    results = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    for band, (lo, hi) in BANDS.items():
        mtr = (Xtr[:, 0] > lo) & (Xtr[:, 0] <= hi)
        mte = (Xte[:, 0] > lo) & (Xte[:, 0] <= hi)
        print(band, "train n", int(mtr.sum()), "test n", int(mte.sum()),
              "test pos", Yte[mte].sum(0), flush=True)
        results.setdefault(band, {})
        for name, ftr, fte in (
                ("static_only", None, None),
                ("full_hist6", Ftr, Fte),
                ("mean_only", Ftr[:, [1]], Fte[:, [1]])):
            results[band].setdefault(name, {})
            for h in HORIZONS:
                if str(h) in results[band][name]:
                    continue
                ytr, yte = Ytr[mtr, HORIZONS.index(h)], Yte[mte, HORIZONS.index(h)]
                if ftr is None:
                    p = predict(Xtr[mtr], ytr, Xte[mte])
                else:
                    p = predict(np.concatenate([Xtr[mtr], ftr[mtr]], axis=1), ytr,
                                np.concatenate([Xte[mte], fte[mte]], axis=1))
                results[band][name][str(h)] = metrics(yte, p)
                print(band, name, h,
                      {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in results[band][name][str(h)].items()}, flush=True)
                PARTIAL.write_text(json.dumps(results, indent=2))
    # summary: gain and mean-only recovery per band
    for band in BANDS:
        print("---", band)
        for h in HORIZONS:
            s = results[band]["static_only"][str(h)]["auprc"]
            f = results[band]["full_hist6"][str(h)]["auprc"]
            m = results[band]["mean_only"][str(h)]["auprc"]
            gain = f - s
            rec = (m - s) / gain if gain else float("nan")
            print(f"  {h} ns: gain {gain:+.4f}  mean-only recovery {rec:.3f}",
                  flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
