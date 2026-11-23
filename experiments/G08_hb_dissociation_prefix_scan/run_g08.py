"""G08: 水素結合の解離で、切り取りの長さを細かく見る.

G06 では水素結合の解離で、0.1 ns 平均の回復率が 0.532 と 5割の線
ぎりぎりだった。0.25 ns（フレーム [0,1,2]）の切り取りで、平均だけと
6統計量の両方を測る。0.5 ns の 2条件は G06 の値（平均 0.589、
6統計量 0.772）をそのまま使う（再計算しない）。

切り取りの定義（G02/G06 と同じ、結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 0.25 ns -> フレーム 0, 1, 2（3フレーム、実際に覆う長さは 0.2 ns）
- 起点フレーム10（1 ns）は変えない。ラベルは 1 ns先（フレーム20）。

測るもの（1 ns先、水素結合の解離、K46 のキャッシュ）:
- 土台の再現: 静的のみ・全部入りが K46 の記録と小数第4位まで一致すること。
- 切った平均だけ（静的特徴＋切った履歴の平均だけ）の利得の回復率。
  回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得)。
- 参考: 切った6統計量全部の条件、切った平均と 0〜1 ns の平均との
  残りのばらつき（G06 の 0.1 ns 0.297 Å・0.5 ns 0.163 Å と並べる）。

参照: 水素結合の解離の実履歴の利得 +0.1785（K46、1 ns先）。
判定（先に固定）:
- 0.25 ns の 6統計量で回復率 6割以上なら、その切り取りが実用的な選択肢。
- 0.25 ns で 5割未満なら、解離では 0.5 ns まで伸ばすのが要る。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 0  # 1 ns先
BOUND = 3.5
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
CUT = {"0.25ns": [0, 1, 2]}
REF_STATIC_AUPRC = 0.4160
REF_FULL_AUPRC = 0.5945
REF_GAIN = 0.1785


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            parts.append({k: z[k] for k in z.files})
    X = np.concatenate([p["X"] for p in parts])
    H = np.concatenate([p["hist"].T for p in parts])
    Y = np.concatenate([p["y"] for p in parts])
    return X, H, Y


def hist6(series, bound):
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


Xtr, Htr, Ytr = load(TRAIN_P)
Xte, Hte, Yte = load(TEST_P)
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)

full_mean_tr = Htr.mean(1)
full_mean_te = Hte.mean(1)

residual = {}
for name, idx in CUT.items():
    pm_tr = Htr[:, idx].mean(1)
    pm_te = Hte[:, idx].mean(1)
    residual[name] = dict(
        frames=idx,
        train_residual_std=float(np.std(pm_tr - full_mean_tr)),
        test_residual_std=float(np.std(pm_te - full_mean_te)),
        train_residual_mean=float(np.mean(pm_tr - full_mean_tr)),
        test_residual_mean=float(np.mean(pm_te - full_mean_te)),
    )
    print(name, "residual std test", round(residual[name]["test_residual_std"], 4),
          "train", round(residual[name]["train_residual_std"], 4), flush=True)

conditions = {"static_only": None}
conditions["full_hist6"] = (hist6(Htr, BOUND), hist6(Hte, BOUND))
conditions["mean_only_full"] = (
    full_mean_tr[:, None].astype(np.float32),
    full_mean_te[:, None].astype(np.float32))
for name, idx in CUT.items():
    conditions[f"prefix_mean_{name}"] = (
        Htr[:, idx].mean(1)[:, None].astype(np.float32),
        Hte[:, idx].mean(1)[:, None].astype(np.float32))
    conditions[f"prefix_hist6_{name}"] = (
        hist6(Htr[:, idx], BOUND), hist6(Hte[:, idx], BOUND))

mets = {}
for cname, feats in conditions.items():
    if feats is None:
        p = predict(Xtr, ytr, Xte)
    else:
        ftr, fte = feats
        p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                    np.concatenate([Xte, fte], axis=1))
    mets[cname] = metrics(yte, p)
    print(cname, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in mets[cname].items()}, flush=True)

base_gain = mets["full_hist6"]["auprc"] - mets["static_only"]["auprc"]
print("base gain (ref %.4f):" % REF_GAIN, round(base_gain, 4), flush=True)
assert abs(mets["static_only"]["auprc"] - REF_STATIC_AUPRC) < 5e-4, mets["static_only"]
assert abs(mets["full_hist6"]["auprc"] - REF_FULL_AUPRC) < 5e-4, mets["full_hist6"]

recovery = {c: float((m["auprc"] - mets["static_only"]["auprc"]) / base_gain)
            for c, m in mets.items() if c != "static_only"}
print("recovery", {k: round(v, 3) for k, v in recovery.items()}, flush=True)

out = {
    "phenomenon": "hb_dissociation",
    "horizon_ns": 1,
    "cuts": {k: v for k, v in CUT.items()},
    "residual_spread_vs_full_mean": residual,
    "metrics_1ns": mets,
    "base_gain_1ns": float(base_gain),
    "reference_gain_1ns": REF_GAIN,
    "recovery_vs_real_history_gain": recovery,
    "g06_reference": {
        "prefix_mean_0.1ns": 0.532, "prefix_hist6_0.1ns": 0.590,
        "prefix_mean_0.5ns": 0.589, "prefix_hist6_0.5ns": 0.772,
        "residual_std_test_0.1ns": 0.297, "residual_std_test_0.5ns": 0.163,
    },
}
(HERE / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

rows = []
for cname, m in mets.items():
    rows.append(dict(phenomenon="hb_dissociation", condition=cname, horizon_ns=1,
                     recovery=recovery.get(cname, ""), **m))
with open(HERE / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", HERE / "results.json")
