"""G02: 短い実MDで、ペアの平均距離と指導力はどこまで戻るか（資料Dの条件、水素結合の形成）.

実の 0〜1 ns の履歴を、始めの短い部分だけに切って使う。
切る長さは 0.1 ns、0.25 ns、0.5 ns の3通り。

切り取りの定義（結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 切る長さ L に対して、時刻が L 以下のフレームを使う。
  - 0.1 ns -> フレーム 0, 1（2フレーム、覆う長さ 0.1 ns）
  - 0.25 ns -> フレーム 0, 1, 2（3フレーム、覆う長さは 0.2 ns。0.25 ns は 100 ps 刻みに合わない）
  - 0.5 ns -> フレーム 0〜5（6フレーム、覆う長さ 0.5 ns）
- 起点フレーム10（1 ns）は変えない。ラベルは 1 ns先（フレーム20）。

(a) 切った履歴の平均と 0〜1 ns の平均との残りのばらつき
    = (切った平均 - 0〜1 ns の平均) の候補ごとの差の標準偏差。
(b) 切った平均だけを履歴の特徴として資料Dの補正モデルに入れたときの
    1 ns先の利得の回復率。条件は「静的特徴＋切った平均だけ」
    （K21 の「平均だけ」と同じ形。他の統計量は入れない）。
    回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得 +0.1178)。

参照: 実履歴の利得 +0.1178、複製の下限 0.249 Å、生成器の平均の誤差 0.483 Å。
土台の再現: 1 ns先で静的のみ AUPRC 0.3637、実履歴 0.4815。
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 0  # 1 ns先

# 切り取り: 名前 -> (表示する長さ, フレームの列, 実際に覆う長さ)
CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
}


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
    return np.stack(
        [series[:, 0], series.mean(1), series.std(1), series.min(1),
         series[:, -1] - series[:, 0], (series <= bound).mean(1)],
        axis=1).astype(np.float32)


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


train = load(TRAIN_P)
test = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)

full_mean_tr = Htr.mean(1)
full_mean_te = Hte.mean(1)

# (a) 残りのばらつき
residual = {}
for name, (label, idx, covered) in CUTS.items():
    pm_tr = Htr[:, idx].mean(1)
    pm_te = Hte[:, idx].mean(1)
    residual[name] = dict(
        cut_label=label, frames=idx, covered_length=covered,
        train_n=int(len(pm_tr)), test_n=int(len(pm_te)),
        train_residual_std=float(np.std(pm_tr - full_mean_tr)),
        test_residual_std=float(np.std(pm_te - full_mean_te)),
        train_residual_mean=float(np.mean(pm_tr - full_mean_tr)),
        test_residual_mean=float(np.mean(pm_te - full_mean_te)),
    )
    print(name, "residual std test",
          round(residual[name]["test_residual_std"], 4), "train",
          round(residual[name]["train_residual_std"], 4), flush=True)

# (b) 1 ns先の利得の回復率
conditions = {}
conditions["static_only"] = None
conditions["base_hist_D6"] = (hist6(Htr), hist6(Hte))
conditions["mean_only_full"] = (
    full_mean_tr[:, None].astype(np.float32),
    full_mean_te[:, None].astype(np.float32))
for name, (label, idx, covered) in CUTS.items():
    conditions[f"prefix_mean_{name}"] = (
        Htr[:, idx].mean(1)[:, None].astype(np.float32),
        Hte[:, idx].mean(1)[:, None].astype(np.float32))
    conditions[f"prefix_hist6_{name}"] = (hist6(Htr[:, idx]), hist6(Hte[:, idx]))

results = {}
for cname, feats in conditions.items():
    if feats is None:
        p = predict(Xtr, ytr, Xte)
    else:
        ftr, fte = feats
        p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                    np.concatenate([Xte, fte], axis=1))
    results[cname] = metrics(yte, p)
    print(cname, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in results[cname].items()}, flush=True)

# 土台の再現の確認
base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
print("base gain (should be ~0.1178):", round(base_gain, 4), flush=True)
assert abs(results["static_only"]["auprc"] - 0.3637) < 5e-4, results["static_only"]
assert abs(results["base_hist_D6"]["auprc"] - 0.4815) < 5e-4, results["base_hist_D6"]

recovery = {}
for cname, m in results.items():
    if cname == "static_only":
        continue
    recovery[cname] = float((m["auprc"] - results["static_only"]["auprc"]) / base_gain)

out = {
    "cuts": {k: v for k, v in CUTS.items()},
    "residual_spread_vs_full_mean": residual,
    "references": {
        "real_history_gain_1ns": 0.1178,
        "replica_lower_bound_A": 0.249,
        "generator_mean_error_A": 0.483,
    },
    "metrics_1ns": results,
    "base_gain_1ns": base_gain,
    "recovery_vs_real_history_gain": recovery,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("saved", OUT / "results.json")
