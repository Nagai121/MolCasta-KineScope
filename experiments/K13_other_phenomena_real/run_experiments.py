import numpy as np, json
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A"); TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
THRESH = {"salt": 4.0, "contact": 4.5}


def load(prots, ph):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_{ph}.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound):
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
        out["auprc"] = float(average_precision_score(y, p))
        out["auc"] = float(roc_auc_score(y, p))
    else:
        out["auprc"] = None; out["auc"] = None
    return out


results = {}
for ph in ("salt", "contact"):
    bound = THRESH[ph]
    train, test = load(TRAIN_P, ph), load(TEST_P, ph)
    Xtr, Xte = cat(train, "X"), cat(test, "X")
    Htr, Hte = cat(train, "hist"), cat(test, "hist")
    Ytr, Yte = cat(train, "y"), cat(test, "y")
    print(ph, "train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)
    CONDITIONS = {
        "static_only": None,
        "base_hist_D6": (hist6(Htr, bound), hist6(Hte, bound)),
        "K13_thin6": (hist6(Htr[:, [0, 2, 4, 6, 8, 10]], bound), hist6(Hte[:, [0, 2, 4, 6, 8, 10]], bound)),
        "K13_thin3": (hist6(Htr[:, [0, 5, 10]], bound), hist6(Hte[:, [0, 5, 10]], bound)),
        "K13_shuffle": (hist6(shuffled(Htr, 7), bound), hist6(shuffled(Hte, 7), bound)),
        "K13_reverse": (hist6(Htr[:, ::-1].copy(), bound), hist6(Hte[:, ::-1].copy(), bound)),
    }
    results[ph] = {}
    for name, feats in CONDITIONS.items():
        results[ph][name] = {}
        for hi, h in enumerate(HORIZONS):
            ytr, yte = Ytr[:, hi], Yte[:, hi]
            if feats is None:
                p = predict(Xtr, ytr, Xte)
            else:
                ftr, fte = feats
                p = predict(np.concatenate([Xtr, ftr], axis=1), ytr, np.concatenate([Xte, fte], axis=1))
            results[ph][name][str(h)] = metrics(yte, p)
        print(ph, name, "AUPRC", {h: results[ph][name][str(h)]["auprc"] for h in HORIZONS}, flush=True)
    # retention of guidance: (cond - static) / (base - static) per horizon
    ret = {}
    for name in ("K13_thin6", "K13_thin3", "K13_shuffle", "K13_reverse"):
        ret[name] = {}
        for h in HORIZONS:
            s = results[ph]["static_only"][str(h)]["auprc"]
            b = results[ph]["base_hist_D6"][str(h)]["auprc"]
            c = results[ph][name][str(h)]["auprc"]
            ret[name][str(h)] = None if (s is None or b is None or c is None or abs(b - s) < 1e-9) else float((c - s) / (b - s))
    results[ph]["retention_vs_base"] = ret

Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=2))
# CSV
import csv
rows = []
for ph in results:
    for name, by_h in results[ph].items():
        if name == "retention_vs_base":
            continue
        for h, m in by_h.items():
            rows.append(dict(phenomenon=ph, condition=name, horizon_ns=h, **m))
with open(Path(__file__).with_name("results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved")
