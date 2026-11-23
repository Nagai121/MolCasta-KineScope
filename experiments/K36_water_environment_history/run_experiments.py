"""K36: 実履歴に水の環境の履歴を足すと指導力が増えるか。

条件:
  static_only       静的特徴7個のみ（土台）
  base_hist_D6      静的＋ペア自身の履歴6統計量（土台の履歴条件）
  hist_plus_waterA  静的＋ペア自身の履歴6統計量＋水の環境（案A、16特徴量）
  hist_plus_waterB  静的＋ペア自身の履歴6統計量＋水の環境（案B、16特徴量）
  hist_plus_waterAB 静的＋ペア自身の履歴6統計量＋水の環境（案A＋案B、32特徴量）
  waterAB_only      静的＋水の環境（案A＋案B）のみ（参考条件）
モデル・指標は土台（base_reproduction / K25）と同一。
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).parent
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
WATER = Path("/home/hatch/workspace/kinescope/work/k36_water")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            w = np.load(WATER / f"{p}_R{r}.npz")
            parts.append(dict(X=z["X"], hist=z["hist"].T, y=z["y"],
                              waterA=w["waterA"], waterB=w["waterB"]))
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


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


train = load(TRAIN_P)
test = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = hist6(cat(train, "hist")), hist6(cat(test, "hist"))
WAtr, WAte = cat(train, "waterA"), cat(test, "waterA")
WBtr, WBte = cat(train, "waterB"), cat(test, "waterB")
WABtr, WABte = np.concatenate([WAtr, WBtr], 1), np.concatenate([WAte, WBte], 1)
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape,
      "waterA nan", int(np.isnan(WAtr).sum()), int(np.isnan(WAte).sum()),
      "waterB nan", int(np.isnan(WBtr).sum()), int(np.isnan(WBte).sum()), flush=True)

CONDITIONS = {
    "static_only": (Xtr, Xte),
    "base_hist_D6": (np.concatenate([Xtr, Htr], 1), np.concatenate([Xte, Hte], 1)),
    "hist_plus_waterA": (np.concatenate([Xtr, Htr, WAtr], 1), np.concatenate([Xte, Hte, WAte], 1)),
    "hist_plus_waterB": (np.concatenate([Xtr, Htr, WBtr], 1), np.concatenate([Xte, Hte, WBte], 1)),
    "hist_plus_waterAB": (np.concatenate([Xtr, Htr, WABtr], 1), np.concatenate([Xte, Hte, WABte], 1)),
    "waterAB_only": (np.concatenate([Xtr, WABtr], 1), np.concatenate([Xte, WABte], 1)),
}

results = {}
for name, (xtr, xte) in CONDITIONS.items():
    results[name] = {}
    for hi, h in enumerate(HORIZONS):
        p = predict(xtr, Ytr[:, hi], xte)
        results[name][str(h)] = metrics(Yte[:, hi], p)
        print(name, h, {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in results[name][str(h)].items()}, flush=True)

# 土台の再現確認（base_reproduction の記録と小数第4位まで一致するか）
base = json.loads((HERE.parent / "base_reproduction" / "results.json").read_text())
repro = {}
for cond in ("static_only", "base_hist_D6"):
    repro[cond] = {}
    for h in map(str, HORIZONS):
        repro[cond][h] = {m: abs(results[cond][h][m] - base[cond][h][m])
                          for m in ("auprc", "auc", "brier", "logloss")}
maxdiff = max(v[m] for c in repro.values() for v in c.values() for m in v)
print("reproduction max abs diff vs base_reproduction:", maxdiff)

gain = {h: results["base_hist_D6"][h]["auprc"] - results["static_only"][h]["auprc"] for h in map(str, HORIZONS)}
incr = {}
for cond in ("hist_plus_waterA", "hist_plus_waterB", "hist_plus_waterAB"):
    incr[cond] = {h: results[cond][h]["auprc"] - results["base_hist_D6"][h]["auprc"] for h in map(str, HORIZONS)}
water_gain = {h: results["waterAB_only"][h]["auprc"] - results["static_only"][h]["auprc"] for h in map(str, HORIZONS)}
print("base gain:", {k: round(v, 4) for k, v in gain.items()})
for cond in incr:
    print(cond, "increment over base:", {k: round(v, 4) for k, v in incr[cond].items()})
print("waterAB only gain:", {k: round(v, 4) for k, v in water_gain.items()})

out = dict(conditions=results, base_gain=gain, water_increment=incr,
           waterAB_only_gain=water_gain, reproduction_max_abs_diff=maxdiff)
(HERE / "results.json").write_text(json.dumps(out, indent=2))
with open(HERE / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auprc", "auc", "brier", "logloss"])
    for name, by_h in results.items():
        for h in map(str, HORIZONS):
            m = by_h[h]
            w.writerow([name, h, m["n"], m["positives"], m["auprc"], m["auc"], m["brier"], m["logloss"]])
print("saved")
