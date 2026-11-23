"""G156: 水素結合の解離で、6 個の統計量の切り取りに起点のフレームを入れる
かどうかの影響を確かめる（資料Dの条件、水素結合の解離）.

G155 は水素結合の形成で、切り取り 1.0 ns（フレーム [0,1,2,3,4,5,6,7,8,9]、
10フレーム）の 6 個の統計量と、0〜1 ns の 6 個の統計量（フレーム
[0,1,2,3,4,5,6,7,8,9,10]、11フレーム全部）とで、回復率が大きく違う
ことを見つけました。30 ns先では -0.403（1.403 -> 1.000）、10 ns先では
+0.286（0.714 -> 1.000）と、予測幅によって分かれ方が違いました。
ここでは水素結合の解離でも同じように分かれるかどうかを見ます。

2 つの条件（結果を見る前に固定）:
- hist6_10f: フレーム [0..9] の 10 フレームで計算した 6 個の統計量。
  実際に覆う長さは 0.9 ns です。起点フレーム 10 は入らない
  （G137 の「1.0 ns の切り取り」と同じ）。
- hist6_11f: フレーム [0..10] の 11 フレーム全部で計算した 6 個の統計量。
  0〜1 ns の履歴全部です。起点フレーム 10 が入ります。

6 個の統計量は、最初の値、平均、標準偏差、最小値、最後の値 − 最初の値、
3.5 Å 以下の割合です。解離では K46 の全部入りが「静的＋実履歴6統計量」
なので、フレーム [0..10] の 11 フレーム全部で計算した 6 個の統計量は
全部入りの条件そのもののはずで、回復率は 1.0 になるはずです。実際に
測って確かめます。もし全部入りと一致しなければ、その理由を調べます。

回復率 = (その条件の AUPRC - 静的のみ) / (その予測幅での実履歴の利得)。
土台の利得は K46 の記録値（再計算でも確認する）:
  1 ns +0.1785、5 ns +0.1757、10 ns +0.1478、30 ns +0.1322、
  50 ns +0.1142、90 ns +0.0679。

読み方（事前に固定）: 2 つの 6 個の統計量の回復率を予測幅ごとに並べ、
差が大きい予測幅を挙げる。差の大きさは回復率の差と AUPRC の差の両方で見る。

起点のフレーム（フレーム 10）は予測時点そのものの構造なので、静的な
入力の一部として入れるのは正当です。フレーム 10 の距離は静的特徴の
0 列目にも入っています（K46 のキャッシュでも確認済み）。

注意: K46 のキャッシュの hist は (11, n) の並びで保存されているため、
読み込み時に転置して (n, 11) に直す（G70・G72 と同じ）。
y の列は 0=1 ns、1=5 ns、2=10 ns、3=30 ns、4=50 ns、5=90 ns
（G72 と同じ対応）。
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
# y の列番号と予測幅の対応（G72 と同じ）
Y_COL = {1: 0, 5: 1, 10: 2, 30: 3, 50: 4, 90: 5}

# 切り取り: 名前 -> (表示する長さ, フレームの列, 実際に覆う長さ)
CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
    "1.0ns": ("1.0 ns", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "0.9 ns"),
}

# K46 の記録値（土台の利得、AUPRC の差）
REF_GAIN = {1: 0.1785, 5: 0.1757, 10: 0.1478, 30: 0.1322, 50: 0.1142, 90: 0.0679}

# G70 の記録値（切り取り平均だけの回復率、照合用）
REF_G70 = {
    1: {"0.1ns": 0.532, "0.25ns": 0.573, "0.5ns": 0.589},
    5: {"0.1ns": 0.668, "0.25ns": 0.729, "0.5ns": 0.921},
    10: {"0.1ns": 0.553, "0.25ns": 0.797, "0.5ns": 0.893},
    30: {"0.1ns": 0.586, "0.25ns": 0.665, "0.5ns": 0.815},
    50: {"0.1ns": 0.563, "0.25ns": 0.556, "0.5ns": 0.647},
    90: {"0.1ns": 0.709, "0.25ns": 1.058, "0.5ns": 1.488},
}

# G72 の記録値（切り取りの 6 個の統計量の回復率、照合用。1 ns先は G72 では測っていない）
REF_G72 = {
    5: {"0.1ns": 0.670, "0.25ns": 0.646, "0.5ns": 0.949},
    10: {"0.1ns": 0.673, "0.25ns": 0.993, "0.5ns": 0.979},
    30: {"0.1ns": 0.582, "0.25ns": 0.834, "0.5ns": 0.926},
    50: {"0.1ns": 0.584, "0.25ns": 0.634, "0.5ns": 0.780},
    90: {"0.1ns": 0.183, "0.25ns": 0.899, "0.5ns": 0.916},
}

# K46 の記録値（静的のみ・全部入りの AUPRC、照合用）
REF_K46 = {1: (0.4160, 0.5945), 5: (0.4346, 0.6103), 10: (0.4496, 0.5974),
           30: (0.4409, 0.5731), 50: (0.4088, 0.5231), 90: (0.5118, 0.5797)}

# G137 の記録値（1.0 ns の切り取り＝フレーム 0〜9 の平均だけの回復率、照合用）
REF_G137_10F = {1: 0.935, 5: 0.985, 10: 0.961, 30: 0.902, 50: 0.688, 90: 1.258}

# G152 の記録値（0〜1 ns の平均だけ＝フレーム 0〜10 の11フレーム全部の回復率、照合用）
REF_G152_11F = {1: 0.858, 5: 0.993, 10: 0.973, 30: 0.887, 50: 0.838, 90: 1.304}


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
for h in HORIZONS:
    hi = Y_COL[h]
    ytr, yte = Ytr[:, hi], Yte[:, hi]
    conditions = {}
    conditions["static_only"] = None
    conditions["base_hist_D6"] = (hist6(Htr), hist6(Hte))
    conditions["hist6_10f"] = (hist6(Htr[:, :10]), hist6(Hte[:, :10]))
    conditions["hist6_11f"] = (hist6(Htr), hist6(Hte))
    conditions["mean_only_full"] = (
        full_mean_tr[:, None].astype(np.float32),
        full_mean_te[:, None].astype(np.float32))
    for name, (label, idx, covered) in CUTS.items():
        conditions[f"prefix_mean_{name}"] = (
            Htr[:, idx].mean(1)[:, None].astype(np.float32),
            Hte[:, idx].mean(1)[:, None].astype(np.float32))
        conditions[f"prefix_hist6_{name}"] = (
            hist6(Htr[:, idx]), hist6(Hte[:, idx]))

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
    print(f"  {h} ns hist6_10f: auprc "
          f"{results['hist6_10f']['auprc']:.4f} "
          f"recovery {recovery['hist6_10f']:.3f}", flush=True)
    print(f"  {h} ns hist6_11f: auprc "
          f"{results['hist6_11f']['auprc']:.4f} "
          f"recovery {recovery['hist6_11f']:.3f}", flush=True)

# 土台の再現の確認（K46 の記録値）
for h in HORIZONS:
    ref_s, ref_f = REF_K46[h]
    assert abs(all_metrics[str(h)]["static_only"]["auprc"] - ref_s) < 5e-4, (h, "static")
    assert abs(all_metrics[str(h)]["base_hist_D6"]["auprc"] - ref_f) < 5e-4, (h, "full")
# G70 との照合（0.1・0.25・0.5 ns の切り取りの平均だけ）
for h in HORIZONS:
    for name, ref in REF_G70[h].items():
        got = all_recovery[str(h)][f"prefix_mean_{name}"]
        assert abs(got - ref) < 5e-3, (h, name, got, ref)
# G72 との照合（0.1・0.25・0.5 ns の切り取りの 6 個の統計量）
for h in HORIZONS:
    if h not in REF_G72:
        continue
    for name, ref in REF_G72[h].items():
        got = all_recovery[str(h)][f"prefix_hist6_{name}"]
        assert abs(got - ref) < 5e-3, (h, name, got, ref)
# G137 との照合（フレーム 0〜9 の平均だけ）
for h in HORIZONS:
    got = all_recovery[str(h)]["prefix_mean_1.0ns"]
    assert abs(got - REF_G137_10F[h]) < 5e-3, (h, got, REF_G137_10F[h])
# G152 との照合（フレーム 0〜10 の11フレーム全部の平均だけ）
for h in HORIZONS:
    got = all_recovery[str(h)]["mean_only_full"]
    assert abs(got - REF_G152_11F[h]) < 5e-3, (h, got, REF_G152_11F[h])
# フレーム 0〜10 の 6 個の統計量は全部入りの条件そのものなので一致する
for h in HORIZONS:
    assert abs(all_metrics[str(h)]["hist6_11f"]["auprc"]
               - all_metrics[str(h)]["base_hist_D6"]["auprc"]) < 1e-9, h
    assert abs(all_recovery[str(h)]["hist6_11f"] - 1.0) < 1e-9, h
# 静的特徴の 0 列目がフレーム 10 の距離であることの確認
assert np.max(np.abs(Xtr[:, 0] - Htr[:, -1])) < 1e-6
assert np.max(np.abs(Xte[:, 0] - Hte[:, -1])) < 1e-6

# 差（11フレームの 6 個の統計量 − 10フレームの 6 個の統計量）
diff = {}
for h in HORIZONS:
    m = all_metrics[str(h)]
    r = all_recovery[str(h)]
    diff[str(h)] = dict(
        recovery_diff=float(r["hist6_11f"] - r["hist6_10f"]),
        auprc_diff=float(m["hist6_11f"]["auprc"] - m["hist6_10f"]["auprc"]),
    )
    print(f"{h} ns: recovery diff {diff[str(h)]['recovery_diff']:+.3f} "
          f"auprc diff {diff[str(h)]['auprc_diff']:+.4f}", flush=True)

out = {
    "conditions": {
        "hist6_10f": {"label": "フレーム [0..9] の 10 フレームで計算した 6 個の統計量",
                      "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      "covered_length": "0.9 ns"},
        "hist6_11f": {"label": "フレーム [0..10] の 11 フレーム全部で計算した 6 個の統計量",
                      "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      "covered_length": "1.0 ns"},
    },
    "horizons": list(HORIZONS),
    "references": {
        "base_gain_ref": REF_GAIN,
        "base_gain_recomputed": all_gain,
        "g70_prefix_mean_recovery": REF_G70,
        "g72_prefix_hist6_recovery": REF_G72,
        "g137_mean10_recovery": REF_G137_10F,
        "g152_mean11_recovery": REF_G152_11F,
        "k46_static_full": REF_K46,
    },
    "metrics": all_metrics,
    "base_gain": all_gain,
    "recovery_vs_real_history_gain": all_recovery,
    "diff_hist6_11f_minus_hist6_10f": diff,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "base_gain", "static_auprc", "full_auprc",
                "hist6_10f_auprc", "hist6_10f_recovery",
                "hist6_11f_auprc", "hist6_11f_recovery",
                "recovery_diff_hist6_11f_minus_hist6_10f",
                "auprc_diff_hist6_11f_minus_hist6_10f"])
    for h in HORIZONS:
        m = all_metrics[str(h)]
        r = all_recovery[str(h)]
        d = diff[str(h)]
        w.writerow([h, round(all_gain[str(h)], 4),
                    round(m["static_only"]["auprc"], 4),
                    round(m["base_hist_D6"]["auprc"], 4),
                    round(m["hist6_10f"]["auprc"], 4),
                    round(r["hist6_10f"], 3),
                    round(m["hist6_11f"]["auprc"], 4),
                    round(r["hist6_11f"], 3),
                    round(d["recovery_diff"], 3),
                    round(d["auprc_diff"], 4)])
print("saved", OUT / "results.json", OUT / "results.csv")
