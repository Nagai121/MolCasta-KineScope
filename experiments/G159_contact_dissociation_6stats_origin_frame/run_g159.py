"""G159: 接触の解離で、6 個の統計量の切り取りに起点のフレームを入れる
かどうかの影響を確かめる（資料Dの条件、接触の解離）.

G155 は水素結合の形成で、切り取り 1.0 ns（フレーム [0,1,2,3,4,5,6,7,8,9]、
10フレーム）の 6 個の統計量と、0〜1 ns の 6 個の統計量（フレーム
[0,1,2,3,4,5,6,7,8,9,10]、11フレーム全部）とで、回復率が大きく違うことを
見つけました。30 ns先では -0.403（1.403 -> 1.000）、10 ns先では +0.286
（0.714 -> 1.000）と、予測幅によって分かれ方が違いました。
G158 では塩橋の解離で、5 ns先が -0.317、30 ns先が -0.308、50 ns先が
+0.299 と大きく変わりました。ここでは接触の解離でも同じように
分かれるかどうかを見ます。

2 つの条件（結果を見る前に固定）:
- hist6_10f: フレーム [0..9] の 10 フレームで計算した 6 個の統計量。
  実際に覆う長さは 0.9 ns です。起点フレーム 10 は入らない
  （G131・G134 の「1.0 ns の切り取り」と同じ。ただし G131・G134 は
  生成系列と重なる候補だけの部分集合で測ったもので、ここでは
  全部の候補で測るので、値が違ってもおかしくありません）。
- hist6_11f: フレーム [0..10] の 11 フレーム全部で計算した 6 個の統計量。
  0〜1 ns の履歴全部です。起点フレーム 10 が入ります。
  11 フレーム全部の 6 個の統計量は、全部入りの条件そのものなので、
  回復率は定義により 1.0 です。つまりこの比べ方は、10 フレームの
  6 個の統計量が全部入りの利得をどれだけ取り込めているかを見るのと
  同じことです。

6 個の統計量は、最初の値、平均、標準偏差、最小値、最後の値 − 最初の値、
4.5 Å 以下の割合です（接触の解離のしきい値）。
回復率 = (その条件の AUPRC - 静的のみ) / (その予測幅での実履歴の利得)。
土台の利得は K50 の記録値（再計算でも確認する）:
  1 ns +0.2142、5 ns +0.2506、10 ns +0.2022、30 ns +0.2368、
  50 ns +0.1606、90 ns +0.1864。

読み方（事前に固定）: 2 つの 6 個の統計量の回復率を予測幅ごとに並べ、
差が大きい予測幅を挙げる。差の大きさは回復率の差と AUPRC の差の両方で見る。

起点のフレーム（フレーム 10）は予測時点そのものの構造なので、静的な
入力の一部として入れるのは正当です。フレーム 10 の距離は静的特徴の
0 列目にも入っています（K50 のキャッシュでも確認済み）。
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
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

# K50 の記録値（土台の利得、AUPRC の差）
REF_GAIN = {1: 0.2142, 5: 0.2506, 10: 0.2022, 30: 0.2368, 50: 0.1606, 90: 0.1864}

# K50 の記録値（静的のみ・全部入りの AUPRC、照合用）
REF_K50 = {1: (0.3105, 0.5246), 5: (0.3022, 0.5528), 10: (0.2772, 0.4794),
           30: (0.2756, 0.5124), 50: (0.2927, 0.4533), 90: (0.2886, 0.4750)}

# G71・G06 の記録値（切り取り平均だけの回復率、照合用）
# 1 ns先の 0.1 ns は G06（0.919）、5〜90 ns先は G71
REF_G71 = {
    1: {"0.1ns": 0.919, "0.25ns": 0.906, "0.5ns": 0.945},
    5: {"0.1ns": 0.802, "0.25ns": 0.620, "0.5ns": 0.677},
    10: {"0.1ns": 0.578, "0.25ns": 0.588, "0.5ns": 0.802},
    30: {"0.1ns": 0.796, "0.25ns": 0.795, "0.5ns": 0.827},
    50: {"0.1ns": 0.612, "0.25ns": 0.665, "0.5ns": 0.827},
    90: {"0.1ns": 0.935, "0.25ns": 0.861, "0.5ns": 0.930},
}

# G140 の記録値（1.0 ns の切り取り＝フレーム 0〜9 の平均だけの回復率、照合用）
REF_G140_10F = {1: 0.915, 5: 0.844, 10: 0.951, 30: 0.840, 50: 0.952, 90: 0.917}

# G131・G134 の記録値（1.0 ns の切り取り＝フレーム 0〜9 の 6 個の統計量の回復率、
# 生成系列と重なる候補だけの部分集合での値。全部の候補での値とは違うので照合には使わない）
REF_G131_G134_10F_MATCHED = {1: 1.023, 5: 1.037, 10: 0.875, 30: 1.034, 50: 0.975, 90: 0.920}


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


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
# K50 のキャッシュでも静的特徴の 0 列目は hist[-1]（フレーム 10 の距離）。
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
    conditions["hist6_10f"] = (hist6(Htr[:, :10]), hist6(Hte[:, :10]))
    conditions["hist6_11f"] = (hist6(Htr), hist6(Hte))
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
    print(f"  {h} ns hist6_10f: auprc "
          f"{results['hist6_10f']['auprc']:.4f} "
          f"recovery {recovery['hist6_10f']:.3f}", flush=True)
    print(f"  {h} ns hist6_11f: auprc "
          f"{results['hist6_11f']['auprc']:.4f} "
          f"recovery {recovery['hist6_11f']:.3f}", flush=True)

# 土台の再現の確認（K50 の記録値）
for h in HORIZONS:
    ref_s, ref_f = REF_K50[h]
    assert abs(all_metrics[str(h)]["static_only"]["auprc"] - ref_s) < 5e-4, (h, "static")
    assert abs(all_metrics[str(h)]["base_hist_D6"]["auprc"] - ref_f) < 5e-4, (h, "full")
# G71・G06 との照合（0.1・0.25・0.5 ns の切り取り）
for h in HORIZONS:
    for name, ref in REF_G71[h].items():
        got = all_recovery[str(h)][f"prefix_mean_{name}"]
        assert abs(got - ref) < 5e-3, (h, name, got, ref)
# G140 との照合（フレーム 0〜9 の平均だけ）
for h in HORIZONS:
    got = all_recovery[str(h)]["prefix_mean_1.0ns"]
    assert abs(got - REF_G140_10F[h]) < 5e-3, (h, got, REF_G140_10F[h])
# フレーム 0〜9 の 6 個の統計量は、全部の候補での記録がまだない新しい測定。
# G131・G134 の部分集合での値は参考として出すだけにする
for h in HORIZONS:
    print(f"  {h} ns hist6_10f recovery {all_recovery[str(h)]['hist6_10f']:.3f} "
          f"(G131/G134 matched-subset ref {REF_G131_G134_10F_MATCHED[h]:.3f})", flush=True)
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
        "g71_prefix_mean_recovery": REF_G71,
        "g140_mean10_recovery": REF_G140_10F,
        "g131_g134_hist6_10f_recovery_matched_subset": REF_G131_G134_10F_MATCHED,
        "k50_static_full": REF_K50,
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
