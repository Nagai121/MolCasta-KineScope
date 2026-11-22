"""K07: 実履歴の揺らぎの縮小（αによる距離系列の縮小）

土台（実験D再現、experiments/base_reproduction/）と同一条件。
各候補ペアの11フレーム距離系列 d(t) を d'(t) = mean + α (d(t) - mean) に
変換してから履歴6統計量を作り直す。α = 1.0, 0.5, 0.25, 0.18, 0.0。
学習・テスト両方に同じ変換を適用し、条件ごとに学習し直す。
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
ALPHAS = (1.0, 0.5, 0.25, 0.18, 0.0)

def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            parts.append({k: z[k] for k in z.files})
    return parts

def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)

def hist6(series, bound=3.5):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)

def scale(series, alpha):
    m = series.mean(axis=1, keepdims=True)
    return m + alpha * (series - m)

def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]

def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

train = load(TRAIN_P); test = load(TEST_P)
Xtr = cat(train, "X"); Xte = cat(test, "X")
Htr = cat(train, "hist"); Hte = cat(test, "hist")
Ytr = cat(train, "y"); Yte = cat(test, "y")
print("train", Xtr.shape, Htr.shape, "test", Xte.shape, Hte.shape, "test pos", Yte.sum(0), flush=True)
print("pair std: train mean %.4f test mean %.4f" % (Htr.std(1).mean(), Hte.std(1).mean()), flush=True)

CONDITIONS = {"static_only": None}
for a in ALPHAS:
    CONDITIONS[f"alpha_{a}"] = (hist6(scale(Htr, a)), hist6(scale(Hte, a)))

results = {}
for name, feats in CONDITIONS.items():
    results[name] = {}
    for hi, h in enumerate(HORIZONS):
        ytr = Ytr[:, hi]; yte = Yte[:, hi]
        if feats is None:
            p = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = feats
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr, np.concatenate([Xte, fte], axis=1))
        results[name][str(h)] = metrics(yte, p)
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[name][str(h)].items()}, flush=True)

# retention of the AUPRC gain relative to alpha=1.0
retention = {}
for a in ALPHAS:
    name = f"alpha_{a}"
    retention[name] = {}
    for h in HORIZONS:
        gain = results["alpha_1.0"][str(h)]["auprc"] - results["static_only"][str(h)]["auprc"]
        retention[name][str(h)] = float((results[name][str(h)]["auprc"] - results["static_only"][str(h)]["auprc"]) / gain)
    print("retention", name, {h: round(retention[name][str(h)], 4) for h in HORIZONS}, flush=True)

payload = {"conditions": results, "auprc_gain_retention_vs_alpha1": retention,
           "alphas": list(ALPHAS), "horizons": list(HORIZONS)}
(OUT / "results.json").write_text(json.dumps(payload, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss", "auprc_gain_retention_vs_alpha1"])
    for name in CONDITIONS:
        for h in HORIZONS:
            r = results[name][str(h)]
            ret = "" if name == "static_only" else retention[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["brier"], r["logloss"], ret])
print("saved to", OUT)
