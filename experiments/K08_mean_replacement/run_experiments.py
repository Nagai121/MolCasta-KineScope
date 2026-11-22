"""K08: 履歴の平均を生成側の平均に差し替える

土台（実験D再現、experiments/base_reproduction/、K07と同一）と同一条件。
変えるのは履歴系列の平均だけ:
  主条件 mean_replaced: d'(t) = gen_mean + (d(t) - real_mean)
      gen_mean は K05 で作り直した第22案の経路の5点系列（起点 + 0.1/0.3/0.5/1.0 ns）
      の平均。対応付けは (軌道ID, ドナー原子, アクセプター原子) で1対1。
  副条件 shift_minus_0.15: d'(t) = d(t) - 0.15（一律シフト、参考条件）
変換後の系列から履歴6統計量を作り直し、学習・テスト両方に同じ変換を適用し、
条件ごとに学習し直す。
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)

def load_gen_means():
    """生成側の平均: K05 generated_series.npz の5点系列（起点距離 + gen 4点）の平均。
    キーは (tid, donor_atom, acceptor_atom)。"""
    z = np.load(GEN)
    tids, pair, gen = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
    means = {}
    for i, t in enumerate(tids):
        means[(str(t), int(pair[i, 0]), int(pair[i, 1]))] = float(gen5[i].mean())
    return means

GEN_MEANS = load_gen_means()

def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            gm = np.array([GEN_MEANS[(tid, int(da), int(aa))]
                           for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["gen_mean"] = gm
            parts.append(d)
    return parts

def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)

def hist6(series, bound=3.5):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)

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
Gtr = cat(train, "gen_mean"); Gte = cat(test, "gen_mean")
print("train", Xtr.shape, Htr.shape, "test", Xte.shape, Hte.shape, "test pos", Yte.sum(0), flush=True)

delta_tr = Gtr - Htr.mean(1); delta_te = Gte - Hte.mean(1)
diag = {"gen_mean_minus_real_mean_train": {"mean": float(delta_tr.mean()), "std": float(delta_tr.std())},
        "gen_mean_minus_real_mean_test": {"mean": float(delta_te.mean()), "std": float(delta_te.std())},
        "coverage": "14582/14582 (train 9610/9610, test 4972/4972)"}
print("gen-real mean delta: train %.4f+-%.4f test %.4f+-%.4f" % (delta_tr.mean(), delta_tr.std(), delta_te.mean(), delta_te.std()), flush=True)

def mean_replaced(series, gen_mean):
    return gen_mean[:, None] + (series - series.mean(1, keepdims=True))

CONDITIONS = {
    "static_only": None,
    "real_history": (hist6(Htr), hist6(Hte)),
    "mean_replaced": (hist6(mean_replaced(Htr, Gtr)), hist6(mean_replaced(Hte, Gte))),
    "shift_minus_0.15": (hist6(Htr - 0.15), hist6(Hte - 0.15)),
}

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

retention = {}
for name in ("mean_replaced", "shift_minus_0.15"):
    retention[name] = {}
    for h in HORIZONS:
        gain = results["real_history"][str(h)]["auprc"] - results["static_only"][str(h)]["auprc"]
        retention[name][str(h)] = float((results[name][str(h)]["auprc"] - results["static_only"][str(h)]["auprc"]) / gain)
    print("retention", name, {h: round(retention[name][str(h)], 4) for h in HORIZONS}, flush=True)

payload = {"conditions": results, "auprc_gain_retention_vs_real": retention,
           "gen_mean_diagnostics": diag, "horizons": list(HORIZONS)}
(OUT / "results.json").write_text(json.dumps(payload, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss", "auprc_gain_retention_vs_real"])
    for name in CONDITIONS:
        for h in HORIZONS:
            r = results[name][str(h)]
            ret = "" if name in ("static_only", "real_history") else retention[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["brier"], r["logloss"], ret])
print("saved to", OUT)
