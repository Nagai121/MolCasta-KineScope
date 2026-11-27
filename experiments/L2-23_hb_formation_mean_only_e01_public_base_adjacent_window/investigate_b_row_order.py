"""L2-23 premise investigation, Test B: sensitivity to row order within each
protein x replica block. Same E01-exact procedure as Test A (alphabetical,
float32), but rows inside each block are shuffled with a fixed seed.
Usage: inv_l223_b.py <seed>
Conditions: static, full, mean@1.0 (HB formation, horizon 1 ns).
"""
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

CACHE = Path("/home/hatch/workspace/kinescope/work/e01_cache")
CLF_KW = dict(max_iter=160, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=25, l2_regularization=0.15, random_state=20261126)
SEED = int(sys.argv[1])


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


def load(mode, seed):
    files = sorted(CACHE.glob("*.npz"))
    rng = np.random.default_rng(seed)
    Xs, Ss, Ys, Ps = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            prot = f.stem
            for r in (1, 2, 3):
                X = z[f"r{r}_{mode}_X"]
                if len(X) == 0:
                    continue
                perm = rng.permutation(len(X))
                Xs.append(X[perm][:, :20])
                Ss.append(z[f"r{r}_{mode}_series"][perm])
                Ys.append(z[f"r{r}_{mode}_y"][perm])
                Ps.append(np.full(len(X), prot, dtype=object))
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(Ss).astype(np.float32),
            np.concatenate(Ys).astype(np.uint8),
            np.concatenate(Ps))


def main():
    t0 = time.time()
    X, S, Y, P = load("hb_form", SEED)
    y = Y[:, 0]
    conds = {
        "static": X,
        "full": np.column_stack([X, hist_stats(S, 3.5)]),
        "mean_1.0": np.column_stack([X, S[:, 0:11].mean(axis=1, keepdims=True)]),
    }
    aps = {}
    for name, Xc in conds.items():
        p = oof_predict(Xc, y, P)
        aps[name] = float(average_precision_score(y, p))
        print(f"seed {SEED} {name}: AP {aps[name]:.6f} ({time.time()-t0:.0f}s)", flush=True)
    rec = (aps["mean_1.0"] - aps["static"]) / (aps["full"] - aps["static"])
    print(f"seed {SEED} recovery mean_1.0: {rec:.6f}", flush=True)


if __name__ == "__main__":
    main()
