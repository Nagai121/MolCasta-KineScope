"""G149: 接触の形成で、起点のフレームを入れるかどうかで平均だけの
回復率がどう変わるか（資料Dの条件、接触の形成）.

G144 で水素結合の形成では、起点のフレームを入れるかどうかで回復率が
10 ns先で +0.269、50 ns先で -0.158 と大きく変わった。接触の形成でも
同じように分かれるかどうかを見る。

2 つの条件（結果を見る前に固定）:
- mean10: フレーム [0..9] の 10 フレームの平均だけ。実際に覆う長さは
  0.9 ns。起点フレーム 10 は入らない。
- mean11: フレーム [0..10] の 11 フレーム全部の平均だけ。0〜1 ns の
  履歴全部。起点フレーム 10 が入る。

平均だけの条件は、他の履歴統計量を入れずに平均の列だけを足す K21 の形。
回復率 = (その条件の AUPRC - 静的のみ) / (その予測幅での実履歴の利得)。
土台の利得は K13・K16 の記録値（再計算でも確認する）:
  1 ns +0.2271、5 ns +0.2322、10 ns +0.2030、30 ns +0.1518、
  50 ns +0.2034、90 ns +0.1850。

読み方（事前に固定）: 2 つの平均だけの回復率を予測幅ごとに並べ、差が
大きい予測幅を挙げる。差の大きさは回復率の差と AUPRC の差の両方で見る。
G144 との比較（水素結合の形成と接触の形成で、起点のフレームの影響は
同じ形か）も記録する。
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

OUT = Path(__file__).resolve().parent
K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)

# 切り取り: 名前 -> フレームの列（G65 と同じ、照合用）
CUTS = {
    "0.1ns": [0, 1],
    "0.25ns": [0, 1, 2],
    "0.5ns": [0, 1, 2, 3, 4, 5],
    "1.0ns": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}

# K13・K16 の記録値（土台の利得、AUPRC の差）
REF_GAIN = {1: 0.2271, 5: 0.2322, 10: 0.2030, 30: 0.1518, 50: 0.2034, 90: 0.1850}

# K16 の記録値（静的のみ・全部入りの AUPRC、照合用）
K16_REF = {
    1: (0.3398, 0.5669),
    5: (0.2870, 0.5192),
    10: (0.3204, 0.5234),
    30: (0.2868, 0.4386),
    50: (0.2604, 0.4638),
    90: (0.2872, 0.4722),
}

# G65 の記録値（切り取り平均だけの回復率、照合用）
REF_G65 = {
    1: {"0.1ns": 0.432, "0.25ns": 0.512, "0.5ns": 0.773},
    5: {"0.1ns": 0.641, "0.25ns": 0.784, "0.5ns": 0.767},
    10: {"0.1ns": 0.434, "0.25ns": 0.573, "0.5ns": 0.727},
    30: {"0.1ns": 0.578, "0.25ns": 0.839, "0.5ns": 1.008},
    50: {"0.1ns": 0.501, "0.25ns": 0.746, "0.5ns": 0.633},
    90: {"0.1ns": 0.502, "0.25ns": 0.680, "0.5ns": 0.665},
}


def hist6(series, bound=4.5):
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
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K13 / f"{p}_R{r}_contact.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


tr = load(TRAIN_P)
te = load(TEST_P)
Xtr = np.concatenate([p["X"] for p in tr]).astype(np.float64)
Xte = np.concatenate([p["X"] for p in te]).astype(np.float64)
Htr = np.concatenate([p["hist"] for p in tr])
Hte = np.concatenate([p["hist"] for p in te])
Ytr = np.concatenate([p["y"] for p in tr])
Yte = np.concatenate([p["y"] for p in te])
print("contact train n:", len(Ytr), "test n:", len(Yte), flush=True)

# 履歴の最後の値（フレーム 10 の距離）が静的特徴の 0 列目と一致することの確認。
# K13 の extract.py では静的特徴の 0 列目は hist[-1]（フレーム 10 の距離）。
print("max |X[:,0] - hist[:,-1]| train",
      float(np.max(np.abs(Xtr[:, 0] - Htr[:, -1]))), flush=True)
print("max |X[:,0] - hist[:,-1]| test",
      float(np.max(np.abs(Xte[:, 0] - Hte[:, -1]))), flush=True)

full_mean_tr = Htr.mean(1)
full_mean_te = Hte.mean(1)

# (a) 2 つの平均の関係（11フレームの平均は 10フレームの平均とフレーム 10 の距離の加重平均）
mean10_tr = Htr[:, :10].mean(1)
mean10_te = Hte[:, :10].mean(1)
print("mean11 - mean10: test std",
      round(float(np.std(full_mean_te - mean10_te)), 4),
      "mean", round(float(np.mean(full_mean_te - mean10_te)), 4), flush=True)

all_metrics = {}
all_recovery = {}
all_gain = {}
for hi, h in enumerate(HORIZONS):
    ytr, yte = Ytr[:, hi], Yte[:, hi]
    conditions = {}
    conditions["static_only"] = None
    conditions["base_hist_D6"] = (hist6(Htr), hist6(Hte))
    conditions["mean_only_full"] = (
        full_mean_tr[:, None].astype(np.float32),
        full_mean_te[:, None].astype(np.float32))
    for name, idx in CUTS.items():
        conditions[f"prefix_mean_{name}"] = (
            Htr[:, idx].mean(1)[:, None].astype(np.float32),
            Hte[:, idx].mean(1)[:, None].astype(np.float32))

    results = {}
    for cname, feats in conditions.items():
        if feats is None:
            p = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = feats
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                        np.concatenate([Xte, fte], axis=1))
        results[cname] = metrics(yte, p)
    all_metrics[str(h)] = results

    base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
    all_gain[str(h)] = float(base_gain)
    print(f"{h} ns: static {results['static_only']['auprc']:.4f} "
          f"full {results['base_hist_D6']['auprc']:.4f} "
          f"gain {base_gain:.4f} (ref {REF_GAIN[h]})", flush=True)
    recovery = {}
    for cname, m in results.items():
        if cname == "static_only":
            continue
        recovery[cname] = float((m["auprc"] - results["static_only"]["auprc"]) / base_gain)
    all_recovery[str(h)] = recovery
    print(f"  {h} ns mean10 (prefix_mean_1.0ns): auprc "
          f"{results['prefix_mean_1.0ns']['auprc']:.4f} "
          f"recovery {recovery['prefix_mean_1.0ns']:.3f}", flush=True)
    print(f"  {h} ns mean11 (mean_only_full): auprc "
          f"{results['mean_only_full']['auprc']:.4f} "
          f"recovery {recovery['mean_only_full']:.3f}", flush=True)

# 土台の再現の確認（全予測幅で K16 の記録と一致）
for h in HORIZONS:
    ref_s, ref_f = K16_REF[h]
    assert abs(all_metrics[str(h)]["static_only"]["auprc"] - ref_s) < 5e-4, (h, "static")
    assert abs(all_metrics[str(h)]["base_hist_D6"]["auprc"] - ref_f) < 5e-4, (h, "full")
# G65 との照合（0.1・0.25・0.5 ns の切り取り）
for h in HORIZONS:
    for name, ref in REF_G65[h].items():
        got = all_recovery[str(h)][f"prefix_mean_{name}"]
        assert abs(got - ref) < 5e-3, (h, name, got, ref)
# 静的特徴の 0 列目がフレーム 10 の距離であることの確認
assert np.max(np.abs(Xtr[:, 0] - Htr[:, -1])) < 1e-6
assert np.max(np.abs(Xte[:, 0] - Hte[:, -1])) < 1e-6

# 差（11フレームの平均だけ − 10フレームの平均だけ）
diff = {}
for h in HORIZONS:
    m = all_metrics[str(h)]
    r = all_recovery[str(h)]
    diff[str(h)] = dict(
        recovery_diff=float(r["mean_only_full"] - r["prefix_mean_1.0ns"]),
        auprc_diff=float(m["mean_only_full"]["auprc"] - m["prefix_mean_1.0ns"]["auprc"]),
    )
    print(f"{h} ns: recovery diff {diff[str(h)]['recovery_diff']:+.3f} "
          f"auprc diff {diff[str(h)]['auprc_diff']:+.4f}", flush=True)

out = {
    "conditions": {
        "mean10": {"label": "フレーム [0..9] の 10 フレームの平均だけ",
                   "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   "covered_length": "0.9 ns"},
        "mean11": {"label": "フレーム [0..10] の 11 フレーム全部の平均だけ",
                   "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "covered_length": "1.0 ns"},
    },
    "horizons": list(HORIZONS),
    "references": {
        "base_gain_ref": REF_GAIN,
        "base_gain_recomputed": all_gain,
        "k16_static_full_auprc": {str(h): K16_REF[h] for h in HORIZONS},
        "g65_prefix_mean_recovery": REF_G65,
    },
    "metrics": all_metrics,
    "base_gain": all_gain,
    "recovery_vs_real_history_gain": all_recovery,
    "diff_mean11_minus_mean10": diff,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "base_gain", "static_auprc", "full_auprc",
                "mean10_auprc", "mean10_recovery",
                "mean11_auprc", "mean11_recovery",
                "recovery_diff_mean11_minus_mean10",
                "auprc_diff_mean11_minus_mean10"])
    for h in HORIZONS:
        m = all_metrics[str(h)]
        r = all_recovery[str(h)]
        d = diff[str(h)]
        w.writerow([h, round(all_gain[str(h)], 4),
                    round(m["static_only"]["auprc"], 4),
                    round(m["base_hist_D6"]["auprc"], 4),
                    round(m["prefix_mean_1.0ns"]["auprc"], 4),
                    round(r["prefix_mean_1.0ns"], 3),
                    round(m["mean_only_full"]["auprc"], 4),
                    round(r["mean_only_full"], 3),
                    round(d["recovery_diff"], 3),
                    round(d["auprc_diff"], 4)])
print("saved", OUT / "results.json", OUT / "results.csv")
