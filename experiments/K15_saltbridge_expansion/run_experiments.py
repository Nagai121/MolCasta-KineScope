"""K15: salt-bridge formation on the expanded protein set.
Conditions: static_only, base_hist_D6, thin6, thin3, shuffle, reverse (as K01/K13).
Also reports the original-6-proteins subset (K13 split) for the base conditions.
OMP_NUM_THREADS should be set to 2 by the caller.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).parent
CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
P108 = [l.strip() for l in open(HERE / "proteins108.txt") if l.strip()]
TRAIN_P = ORIG_TRAIN + tuple(P108[:54])
TEST_P = ORIG_TEST + tuple(P108[54:])
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.0


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts]) if parts else np.empty((0,))


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def shuffled(series, seed):
    r = np.random.default_rng(seed); out = series.copy()
    for i in range(len(out)):
        out[i] = out[i][r.permutation(out.shape[1])]
    return out


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr); return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    out = dict(n=int(len(y)), positives=int(y.sum()),
               prevalence=float(y.mean()) if len(y) else None,
               brier=float(brier_score_loss(y, p)),
               logloss=float(log_loss(y, p, labels=[0, 1])))
    if 0 < y.sum() < len(y):
        out["auprc"] = float(average_precision_score(y, p)); out["auc"] = float(roc_auc_score(y, p))
    else:
        out["auprc"] = None; out["auc"] = None
    return out


def run_split(train_prots, test_prots, label):
    train, test = load(train_prots), load(test_prots)
    Xtr, Xte = cat(train, "X"), cat(test, "X")
    Htr, Hte = cat(train, "hist"), cat(test, "hist")
    Ytr, Yte = cat(train, "y"), cat(test, "y")
    print(label, "train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)
    CONDITIONS = {
        "static_only": None,
        "base_hist_D6": (hist6(Htr), hist6(Hte)),
        "K15_thin6": (hist6(Htr[:, [0, 2, 4, 6, 8, 10]]), hist6(Hte[:, [0, 2, 4, 6, 8, 10]])),
        "K15_thin3": (hist6(Htr[:, [0, 5, 10]]), hist6(Hte[:, [0, 5, 10]])),
        "K15_shuffle": (hist6(shuffled(Htr, 7)), hist6(shuffled(Hte, 7))),
        "K15_reverse": (hist6(Htr[:, ::-1].copy()), hist6(Hte[:, ::-1].copy())),
    }
    if label != "full":
        CONDITIONS = {k: v for k, v in CONDITIONS.items() if k in ("static_only", "base_hist_D6")}
    res = {}
    for name, feats in CONDITIONS.items():
        res[name] = {}
        for hi, h in enumerate(HORIZONS):
            ytr, yte = Ytr[:, hi], Yte[:, hi]
            if feats is None:
                p = predict(Xtr, ytr, Xte)
            else:
                ftr, fte = feats
                p = predict(np.concatenate([Xtr, ftr], axis=1), ytr, np.concatenate([Xte, fte], axis=1))
            res[name][str(h)] = metrics(yte, p)
        print(label, name, "AUPRC", {h: res[name][str(h)]["auprc"] for h in HORIZONS}, flush=True)
    ret = {}
    for name in CONDITIONS:
        if name in ("static_only", "base_hist_D6"):
            continue
        ret[name] = {}
        for h in HORIZONS:
            s = res["static_only"][str(h)]["auprc"]; b = res["base_hist_D6"][str(h)]["auprc"]; c = res[name][str(h)]["auprc"]
            ret[name][str(h)] = None if (s is None or b is None or c is None or abs(b - s) < 1e-9) else float((c - s) / (b - s))
    res["retention_vs_base"] = ret
    res["_split"] = dict(train_proteins=list(train_prots), test_proteins=list(test_prots),
                         n_train=int(len(Xtr)), n_test=int(len(Xte)),
                         test_positives=[int(v) for v in Yte.sum(0)], train_positives=[int(v) for v in Ytr.sum(0)])
    return res


results = {"full": run_split(TRAIN_P, TEST_P, "full"),
           "original6_subset": run_split(ORIG_TRAIN, ORIG_TEST, "original6_subset")}
(HERE / "results.json").write_text(json.dumps(results, indent=2))
rows = []
for split, res in results.items():
    for name, by_h in res.items():
        if name in ("retention_vs_base", "_split"):
            continue
        for h, m in by_h.items():
            rows.append(dict(split=split, condition=name, horizon_ns=h, **m))
with open(HERE / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved")
