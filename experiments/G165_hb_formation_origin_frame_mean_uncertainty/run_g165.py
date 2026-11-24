"""G165: 水素結合の形成で、起点のフレームを入れるかどうかでの平均だけの回復率の
差に、区間を付ける（資料Dの条件、水素結合の形成、1・5・10・30・50・90 ns先）.

G144 では、フレーム [0..9] の 10 フレームの平均だけ（mean10）とフレーム
[0..10] の 11 フレーム全部の平均だけ（mean11）の回復率の差が、1 ns先
-0.071、5 ns先 +0.071、10 ns先 +0.269、30 ns先 +0.136、50 ns先 -0.158、
90 ns先 -0.054 でした。ここでは 6 予測幅すべてについて、その差の
ブートストラップ 95% 区間を出します（G162・G163・G164 と同じ段取り）。

再標本化の単位はタンパク質。ただし土台は D 条件でテストが 2 タンパク質
（1mix_A・2qt4_A）だけなので、区間は 3 つの値の最小値と最大値の形で
粗くなります。2 タンパク質から復元抽出で 2 タンパク質を選び直すと、
取りうる組み合わせは (1mix, 1mix)、(1mix, 2qt4)、(2qt4, 2qt4) の
3通りだけで、区間はその3値の最小値〜最大値に等しくなります。
「0 を含む／含まない」は実質的に「2タンパク質の差の向きが一致しているか／
食い違っているか」と同じ読みです（G23・G49・G59・G62・G67・G114・G130
と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G73・G49・G59・G62・G67・G23・G147・G148・G162・G163・G164 と同じです。

段取り:
- G144 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
- 回復率の差の定義: (mean11 の回復率) - (mean10 の回復率)。各
  ブートストラップ標本の中でも、静的のみ・全部入り・mean10・mean11 の
  AUPRC を測り直し、利得（全部入り - 静的のみ）と 2 つの回復率をその
  標本の中で計算し直す。つまり回復率の分母の土台の利得は、標本ごとに
  測り直す（G147・G148・G162・G163・G164 と同じ）。土台の利得が小さいので、
  AUPRC の差（mean11 - mean10）の区間も同じ段取りで出す。
- 計算の工夫: 1回の抽出で 2 タンパク質を復元抽出し、その行の集合に
  ついて、6つの予測幅 ×（静的のみ・全部入り・mean10・mean11）の AUPRC
  を測る。AUPRC の計算は sklearn の average_precision_score と同じ
  段取り（しきい値で区切って精度と再現率を取り、再現率の増え分で重み
  付けする）を numpy で書き直したもので、実際の予測確率で sklearn
  の値と 1e-12 まで一致することを確かめてから使う。抽出の乱数 seed は
  G23 と同じ 42 である。

フレームの注記: 「1.0 ns」の切り取りは、呼び名の 1.0 ns に対して
実際は 0.9 ns の範囲である（フレーム [0,1,2,3,4,5,6,7,8,9] は時刻
0〜0.9 ns で、起点のフレーム10 は入らない）。

読み方（先に固定）:
- 回復率の差の区間が 0 を含むかどうかを、予測幅ごとに見てください。
- 10 ns先は差が +0.269、50 ns先は -0.158 と大きく、区間とあわせて記録する。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 3.5
MEAN10_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り（G136・G144 と同じ）
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23・G147・G148・G162 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23・G147・G148・G162 と同じ）

# G144 の記録との照合用の期待値（合算 AUPRC、G144 の README §4 と results.json）
EXPECTED = {
    1: {"static_only": 0.3637, "full_hist": 0.4815, "mean10": 0.4949, "mean11": 0.4865,
        "full_gain": 0.1178, "mean10_recovery": 1.114, "mean11_recovery": 1.042,
        "recovery_diff": -0.071, "auprc_diff": -0.0084},
    5: {"static_only": 0.3493, "full_hist": 0.4648, "mean10": 0.4785, "mean11": 0.4868,
        "full_gain": 0.1155, "mean10_recovery": 1.119, "mean11_recovery": 1.190,
        "recovery_diff": 0.071, "auprc_diff": 0.0082},
    10: {"static_only": 0.3548, "full_hist": 0.4377, "mean10": 0.4243, "mean11": 0.4467,
         "full_gain": 0.0829, "mean10_recovery": 0.839, "mean11_recovery": 1.108,
         "recovery_diff": 0.269, "auprc_diff": 0.0223},
    30: {"static_only": 0.3639, "full_hist": 0.4264, "mean10": 0.4340, "mean11": 0.4426,
         "full_gain": 0.0626, "mean10_recovery": 1.121, "mean11_recovery": 1.258,
         "recovery_diff": 0.136, "auprc_diff": 0.0085},
    50: {"static_only": 0.3342, "full_hist": 0.4081, "mean10": 0.4332, "mean11": 0.4216,
         "full_gain": 0.0739, "mean10_recovery": 1.340, "mean11_recovery": 1.182,
         "recovery_diff": -0.158, "auprc_diff": -0.0117},
    90: {"static_only": 0.3035, "full_hist": 0.4246, "mean10": 0.4035, "mean11": 0.3970,
         "full_gain": 0.1211, "mean10_recovery": 0.826, "mean11_recovery": 0.772,
         "recovery_diff": -0.054, "auprc_diff": -0.0066},
}


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                       max_leaf_nodes=31, min_samples_leaf=15,
                                       l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def auprc(y, p):
    return float(average_precision_score(y, p))


def ap_fast(y, s):
    """sklearn の average_precision_score と同じ段取りの numpy 版。
    予測確率の同じ値（タイブ）をしきい値でひとまとめにして扱う。
    実際の予測確率で sklearn の値と 1e-12 まで一致することを確かめてから使う。"""
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    total_pos = ys.sum()
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(ys)
    change = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[change[1:] - 1, len(ys) - 1]
    tp_t = tp[ends]
    prec = tp_t / (ends + 1)
    rec = tp_t / total_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def load(prots):
    """水素結合の形成は K01 キャッシュ（G144 と同じ）。hist は (11, n) で保存
    されているので転置して (n, 11) にする（G144 の cat と同じ）。"""
    real = []
    proteins = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: (z[k].T if k.startswith("hist") else z[k]) for k in z.files}
            real.append(d)
            proteins.extend([p] * len(z["y"]))
    return real, np.array(proteins)


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
Htr = np.concatenate([d["hist"] for d in train]).astype(np.float64)
Hte = np.concatenate([d["hist"] for d in test]).astype(np.float64)
Ytr = np.concatenate([d["y"] for d in train])
Yte = np.concatenate([d["y"] for d in test])
print("train", Xtr.shape, "test", Xte.shape, "test proteins", len(set(test_protein)),
      flush=True)

# 履歴の最後の値（フレーム 10 の距離）が静的特徴の 0 列目と一致することの確認。
print("max |X[:,0] - hist[:,-1]| train",
      float(np.max(np.abs(Xtr[:, 0] - Htr[:, -1]))), flush=True)
print("max |X[:,0] - hist[:,-1]| test",
      float(np.max(np.abs(Xte[:, 0] - Hte[:, -1]))), flush=True)

mean10_tr = Htr[:, MEAN10_FRAMES].mean(1)
mean10_te = Hte[:, MEAN10_FRAMES].mean(1)
mean11_tr = Htr.mean(1)
mean11_te = Hte.mean(1)
print("mean11 - mean10: test std",
      round(float(np.std(mean11_te - mean10_te)), 4),
      "mean", round(float(np.mean(mean11_te - mean10_te)), 4), flush=True)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

CONDS = ["static_only", "full_hist", "mean10", "mean11"]


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname == "mean10":
        return (np.column_stack([Xtr, mean10_tr[:, None].astype(np.float32)]),
                np.column_stack([Xte, mean10_te[:, None].astype(np.float32)]))
    return (np.column_stack([Xtr, mean11_tr[:, None].astype(np.float32)]),
            np.column_stack([Xte, mean11_te[:, None].astype(np.float32)]))


results = {}
preds = {}  # horizon_ns -> condition -> test 予測確率
for hidx, hns in HORIZONS.items():
    ytr, yte = Ytr[:, hidx], Yte[:, hidx]
    p = {}
    for cname in CONDS:
        xtr, xte = feats_for(cname)
        p[cname] = predict(xtr, ytr, xte)
    preds[hns] = p
    a = {c: auprc(yte, p[c]) for c in CONDS}
    # ap_fast が sklearn と一致することの確認（ブートストラップで使うため）
    for c in CONDS:
        assert abs(ap_fast(yte, p[c]) - a[c]) < 1e-12, (hns, c)
    gain = a["full_hist"] - a["static_only"]
    rec10 = (a["mean10"] - a["static_only"]) / gain if gain else float("nan")
    rec11 = (a["mean11"] - a["static_only"]) / gain if gain else float("nan")
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        if len(idx) == 0:
            continue
        n_pos = int(yte[idx].sum())
        if n_pos == 0:
            per_protein[prot] = {
                "n": int(len(idx)), "positives": 0,
                "auprc": {c: float("nan") for c in CONDS},
                "full_gain": float("nan"),
                "recovery_diff": float("nan"), "auprc_diff": float("nan"),
            }
            continue
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": n_pos,
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "recovery_diff": ((ap["mean11"] - ap["mean10"]) / g
                              if g else float("nan")),
            "auprc_diff": ap["mean11"] - ap["mean10"],
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain,
        "mean10_recovery": rec10, "mean11_recovery": rec11,
        "recovery_diff": rec11 - rec10,
        "auprc_diff": a["mean11"] - a["mean10"],
        "per_protein": per_protein,
    }
    # G144 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist", "mean10", "mean11"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(gain - exp["full_gain"]) < 5e-4, (hns, gain)
    assert abs(rec10 - exp["mean10_recovery"]) < 5e-3, (hns, rec10)
    assert abs(rec11 - exp["mean11_recovery"]) < 5e-3, (hns, rec11)
    assert abs((rec11 - rec10) - exp["recovery_diff"]) < 5e-3, (hns, rec11 - rec10)
    assert abs((a["mean11"] - a["mean10"]) - exp["auprc_diff"]) < 5e-4, (hns, a["mean11"] - a["mean10"])
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"mean10 {a['mean10']:.4f} mean11 {a['mean11']:.4f} gain {gain:.4f} "
          f"recovery_diff {rec11 - rec10:+.3f} (record {exp['recovery_diff']:+.3f}) "
          f"auprc_diff {a['mean11'] - a['mean10']:+.4f}", flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("checkpoint: saved test_predictions.npz", flush=True)

# ---- タンパク質単位のブートストラップ（2 タンパク質からの復元抽出） ----
# 1回の抽出につき行の集合を1回だけ作り、静的のみ・全部入り・mean10・mean11 の
# AUPRC は予測幅ごとに1回だけ測る。回復率の差は、その標本の中で利得と
# 2 つの回復率を計算し直してから取る（G147・G148・G162・G163・G164 と同じ）。
# 2 タンパク質だけなので、取りうる組み合わせは (1mix, 1mix)、(1mix, 2qt4)、
# (2qt4, 2qt4) の 3通りだけで、区間はその3値の最小値〜最大値に等しくなる。
rng = np.random.default_rng(BOOT_SEED)
prot_list = list(TEST_P)
prot_rows = [prot_idx[p] for p in prot_list]
draws = rng.integers(0, len(prot_list), size=(N_BOOT, len(prot_list)))
vals_rec = {hns: np.empty(N_BOOT) for hns in HORIZONS.values()}
vals_auprc = {hns: np.empty(N_BOOT) for hns in HORIZONS.values()}
for b in range(N_BOOT):
    idx = np.concatenate([prot_rows[i] for i in draws[b]])
    for hidx, hns in HORIZONS.items():
        yte = Yte[:, hidx]
        p = preds[hns]
        y = yte[idx]
        ap_s = ap_fast(y, p["static_only"][idx])
        ap_f = ap_fast(y, p["full_hist"][idx])
        ap_10 = ap_fast(y, p["mean10"][idx])
        ap_11 = ap_fast(y, p["mean11"][idx])
        gain = ap_f - ap_s
        vals_rec[hns][b] = (ap_11 - ap_10) / gain if gain else float("nan")
        vals_auprc[hns][b] = ap_11 - ap_10
    if (b + 1) % 2000 == 0:
        print(f"bootstrap {b + 1}/{N_BOOT} draws done", flush=True)
boot_rec = {}
boot_auprc = {}
for hns in HORIZONS.values():
    v = vals_rec[hns]
    lo, hi = np.percentile(v, [2.5, 97.5])
    boot_rec[hns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[hns]["recovery_diff"],
        "ci_lower": float(lo), "ci_upper": float(hi),
        "contains_zero": bool(lo <= 0 <= hi),
        "n_distinct_boot_values": int(len(np.unique(np.round(v, 12)))),
    }
    va = vals_auprc[hns]
    loa, hia = np.percentile(va, [2.5, 97.5])
    boot_auprc[hns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[hns]["auprc_diff"],
        "ci_lower": float(loa), "ci_upper": float(hia),
        "contains_zero": bool(loa <= 0 <= hia),
        "n_distinct_boot_values": int(len(np.unique(np.round(va, 12)))),
    }
    print(f"{hns}ns bootstrap recovery diff: {boot_rec[hns]['point_estimate']:+.3f} "
          f"[{boot_rec[hns]['ci_lower']:+.3f}, {boot_rec[hns]['ci_upper']:+.3f}] "
          f"contains_zero={boot_rec[hns]['contains_zero']} "
          f"n_distinct={boot_rec[hns]['n_distinct_boot_values']}", flush=True)
    print(f"{hns}ns bootstrap (AUPRC diff): {boot_auprc[hns]['point_estimate']:+.4f} "
          f"[{boot_auprc[hns]['ci_lower']:+.4f}, {boot_auprc[hns]['ci_upper']:+.4f}] "
          f"zero={boot_auprc[hns]['contains_zero']} "
          f"n_distinct={boot_auprc[hns]['n_distinct_boot_values']}", flush=True)

# 回復率の差の区間が 0 を含むかどうか（先に固定した読み方）
contains_zero = {f"{h}ns": boot_rec[h]["contains_zero"] for h in HORIZONS.values()}
print("recovery-diff CIs contain zero:", contains_zero, flush=True)

out = {
    "label": "HB formation, origin-frame (mean10 vs mean11) recovery-difference uncertainty across horizons",
    "phenomenon": "HB formation (D conditions, K01 cache)",
    "conditions": {
        "mean10": {"label": "フレーム [0..9] の 10 フレームの平均だけ",
                   "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   "covered_length": "0.9 ns"},
        "mean11": {"label": "フレーム [0..10] の 11 フレーム全部の平均だけ",
                   "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "covered_length": "1.0 ns"},
    },
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (2 test proteins, sampled with "
                  "replacement; only 3 distinct resampling combinations exist, "
                  "so each interval equals the min-max of those 3 values — "
                  "the same coarseness as G23/G49/G114/G130)",
                  "note": "AUPRC inside the bootstrap uses a numpy reimplementation "
                  "of sklearn's average_precision_score (same threshold-grouping "
                  "treatment of ties); it was verified to agree with sklearn to "
                  "1e-12 on all fitted conditions before use. Models were retrained "
                  "per condition and horizon; pooled AUPRCs were verified against "
                  "G144 records to 4 decimal places. Within each bootstrap sample, "
                  "the base gain and the two recoveries are recomputed inside the "
                  "sample before taking their difference (same as G147/G148/G162)."},
    "horizons": {str(h): results[h] for h in HORIZONS.values()},
    "bootstrap_results_recovery_diff": {str(h): boot_rec[h] for h in HORIZONS.values()},
    "bootstrap_results_auprc_diff": {str(h): boot_auprc[h] for h in HORIZONS.values()},
    "recovery_diff_cis_contain_zero": contains_zero,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "test_positives", "auprc_static",
                "auprc_full", "auprc_mean10", "auprc_mean11",
                "full_gain", "mean10_recovery", "mean11_recovery",
                "recovery_diff", "boot_rec_ci_lower", "boot_rec_ci_upper",
                "rec_ci_contains_zero",
                "auprc_diff", "boot_auprc_ci_lower", "boot_auprc_ci_upper",
                "auprc_ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        b = boot_rec[h]
        ba = boot_auprc[h]
        w.writerow([h, r["test_positives"],
                    r["auprc"]["static_only"], r["auprc"]["full_hist"],
                    r["auprc"]["mean10"], r["auprc"]["mean11"],
                    r["full_gain"], r["mean10_recovery"], r["mean11_recovery"],
                    r["recovery_diff"], b["ci_lower"], b["ci_upper"],
                    b["contains_zero"],
                    r["auprc_diff"], ba["ci_lower"], ba["ci_upper"],
                    ba["contains_zero"]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
