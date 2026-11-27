"""L2-23 premise investigation, Test A: replicate E01's evaluation procedure exactly
(alphabetical file order, float32 throughout, E01's own oof_predict / hist_stats)
on my re-extracted cache. HB formation, horizon 1 ns only.
Conditions: static, full (6 stats on [0..10]), mean@0.1 ([9:11]), mean@1.0 ([0:11]).
"""
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

CACHE = Path("/home/hatch/workspace/kinescope/work/e01_cache")
CLF_KW = dict(max_iter=160, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=25, l2_regularization=0.15, random_state=20261126)


def hist_stats(ser, thresh):
    return np.column_stack([
        ser[:, 0], ser.mean(axis=1), ser.std(axis=1), ser.min(axis=1),
        ser[:, -1] - ser[:, 0], (ser <= thresh).mean(axis=1),
    ]).astype(np.float32)


def oof_predict(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(**CLF_KW)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def load(mode):
    files = sorted(CACHE.glob("*.npz"))  # E01: sorted(CACHE.glob("*.npz"))
    Xs, Ss, Ys, Ps = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            prot = f.stem
            for r in (1, 2, 3):
                X = z[f"r{r}_{mode}_X"]
                if len(X) == 0:
                    continue
                Xs.append(X[:, :20])
                Ss.append(z[f"r{r}_{mode}_series"])
                Ys.append(z[f"r{r}_{mode}_y"])
                Ps.append(np.full(len(X), prot, dtype=object))
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(Ss).astype(np.float32),
            np.concatenate(Ys).astype(np.uint8),
            np.concatenate(Ps))


def main():
    t0 = time.time()
    X, S, Y, P = load("hb_form")
    print(f"n={len(X)} proteins={len(np.unique(P))}", flush=True)
    y = Y[:, 0]
    H_full = hist_stats(S, 3.5)
    conds = {
        "static": X,
        "full": np.column_stack([X, H_full]),
        "mean_0.1": np.column_stack([X, S[:, 9:11].mean(axis=1, keepdims=True)]),
        "mean_1.0": np.column_stack([X, S[:, 0:11].mean(axis=1, keepdims=True)]),
    }
    aps = {}
    for name, Xc in conds.items():
        p = oof_predict(Xc, y, P)
        aps[name] = float(average_precision_score(y, p))
        print(f"{name}: AP {aps[name]:.6f} ({time.time()-t0:.0f}s)", flush=True)
    gain = aps["full"] - aps["static"]
    for k in ("mean_0.1", "mean_1.0"):
        rec = (aps[k] - aps["static"]) / gain
        print(f"recovery {k}: {rec:.6f}", flush=True)
    print("E01 refs: static 0.448478 full 0.542879 mean_0.1 AP 0.490004 rec 0.439895 "
          "mean_1.0 AP 0.537422 rec 0.942195", flush=True)


if __name__ == "__main__":
    main()
