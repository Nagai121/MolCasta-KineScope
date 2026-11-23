"""G152: 水素結合の解離で、起点のフレームを入れるかどうかの影響を確かめる
（資料Dの条件、水素結合の解離）.

G144 は水素結合の形成で、切り取り 1.0 ns（フレーム [0,1,2,3,4,5,6,7,8,9]、
10フレーム）の平均だけと、0〜1 ns の平均だけ（フレーム [0,1,2,3,4,5,6,7,8,9,10]、
11フレーム全部）とで、平均だけの回復率が大きく違うことを見つけました。
10 ns先では +0.269（0.839 -> 1.108）、50 ns先では -0.158
（1.340 -> 1.182）と、予測幅によって分かれ方が違いました。
ここでは水素結合の解離でも同じように分かれるかどうかを見ます。

2 つの条件（結果を見る前に固定）:
- mean10: フレーム [0..9] の 10 フレームの平均だけ。実際に覆う長さは
  0.9 ns。起点フレーム 10 は入らない（G137 の「1.0 ns の切り取り」と同じ）。
- mean11: フレーム [0..10] の 11 フレーム全部の平均だけ。0〜1 ns の
  履歴全部。起点フレーム 10 が入る（G70・G06 の「0〜1 ns の平均だけ」と同じ）。

平均だけの条件は、他の履歴統計量を入れずに平均の列だけを足す K21 の形。
回復率 = (その条件の AUPRC - 静的のみ) / (その予測幅での実履歴の利得)。
土台の利得は K46 の記録値（再計算でも確認する）:
  1 ns +0.1785、5 ns +0.1757、10 ns +0.1478、30 ns +0.1322、
  50 ns +0.1142、90 ns +0.0679。

読み方（事前に固定）: 2 つの平均だけの回復率を予測幅ごとに並べ、差が
大きい予測幅を挙げる。差の大きさは回復率の差と AUPRC の差の両方で見る。

起点のフレーム（フレーム 10）は予測時点そのものの構造なので、静的な
入力の一部として入れるのは正当です。フレーム 10 の距離は静的特徴の
0 列目にも入っています（K46 のキャッシュでも確認済み）。
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)

# 切り取り: 名前 -> (表示する長さ, フレームの列, 実際に覆う長さ)
CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
    "1.0ns": ("1.0 ns", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "0.9 ns"),
}

# K46 の記録値（土台の利得、AUPRC の差）
REF_GAIN = {1: 0.1785, 5: 0.1757, 10: 0.1478, 30: 0.1322, 50: 0.1142, 90: 0.0679}

# K46 の記録値（静的のみ・全部入りの AUPRC、照合用）
REF_K46 = {1: (0.4160, 0.5945), 5: (0.4346, 0.6103), 10: (0.4496, 0.5974),
           30: (0.4409, 0.5731), 50: (0.4088, 0.5231), 90: (0.5118, 0.5797)}

# G70・G06 の記録値（切り取り平均だけの回復率、照合用）
# 1 ns先の 0.1 ns は G06（0.532）、5〜90 ns先の 0.1 ns は G70、
# 0.25・0.5 ns は G137 の AUPRC から計算した回復率
REF_G70 = {
    1: {"0.1ns": 0.532, "0.25ns": 0.573, "0.5ns": 0.589},
    5: {"0.1ns": 0.668, "0.25ns": 0.729, "0.5ns": 0.921},
    10: {"0.1ns": 0.553, "0.25ns": 0.797, "0.5ns": 0.893},
    30: {"0.1ns": 0.586, "0.25ns": 0.665, "0.5ns": 0.815},
    50: {"0.1ns": 0.563, "0.25ns": 0.556, "0.5ns": 0.647},
    90: {"0.1ns": 0.709, "0.25ns": 1.058, "0.5ns": 1.488},
}

# G137 の記録値（1.0 ns の切り取り＝フレーム 0〜9 の平均だけの回復率、照合用）
REF_G137_10F = {1: 0.935, 5: 0.985, 10: 0.961, 30: 0.902, 50: 0.688, 90: 1.258}


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
print("train", Xtr.shape, "test", Xte.shape, flush=True)

# 履歴の最後の値（フレーム 10 の距離）が静的特徴の 0 列目と一致することの確認。
# K46 のキャッシュでも静的特徴の 0 列目は hist[-1]（フレーム 10 の距離）。
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
    for name, (label, idx, covered) in CUTS.items():
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

# 土台の再現の確認（K46 の記録値）
for h in HORIZONS:
    ref_s, ref_f = REF_K46[h]
    assert abs(all_metrics[str(h)]["static_only"]["auprc"] - ref_s) < 5e-4, (h, "static")
    assert abs(all_metrics[str(h)]["base_hist_D6"]["auprc"] - ref_f) < 5e-4, (h, "full")
# G70・G06・G137 との照合（0.1・0.25・0.5 ns の切り取り）
for h in HORIZONS:
    for name, ref in REF_G70[h].items():
        got = all_recovery[str(h)][f"prefix_mean_{name}"]
        assert abs(got - ref) < 5e-3, (h, name, got, ref)
# G137 との照合（フレーム 0〜9 の平均だけ）
for h in HORIZONS:
    got = all_recovery[str(h)]["prefix_mean_1.0ns"]
    assert abs(got - REF_G137_10F[h]) < 5e-3, (h, got, REF_G137_10F[h])
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
        "g70_prefix_mean_recovery": REF_G70,
        "g137_mean10_recovery": REF_G137_10F,
        "k46_static_full": REF_K46,
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
