"""G170: 水素結合の解離で、起点のフレームを入れるかどうかでの 6 個の統計量の
回復率の差に、区間を付ける（G156 の 6 予測幅すべて）.

G156 では、切り取り 1.0 ns の 6 個の統計量の回復率が、フレーム [0..9]
の 10 フレームで計算したものと、フレーム [0..10] の 11 フレーム全部で
計算したものとで、1 ns先 −0.058、5 ns先 −0.023、10 ns先 +0.012、
30 ns先 −0.048、50 ns先 −0.088、90 ns先 −0.090 と、6 予測幅すべてで
小さく違いました。ここでは 6 予測幅すべてについて、その差の
ブートストラップ 95% 区間を出します（G162・G163・G164 と同じやり方）。

再標本化の単位はタンパク質。ただし土台は D 条件でテストが 2 タンパク質
（1mix_A と 2qt4_A）だけなので、ブートストラップの取りうる組み合わせは
3通りしかありません（1mix_A だけを 2 回、2qt4_A だけを 2 回、1回ずつ）。
区間は 3 つの値の最小値と最大値の形で粗くなります。G49・G59・G62・G67・
G167 と同じ粗さです。その点も README に書きます。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G73・G49・G59・G62・G67・G23・G96・G117・G129・G148・G162・G163・
G164・G167 と同じです。

段取り:
- G156 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
- 回復率の定義: (その条件の AUPRC - 静的のみの AUPRC) /
  (全部入りの AUPRC - 静的のみの AUPRC)。回復率の分母は、その予測幅での
  実履歴の利得。各ブートストラップ標本の中でも、静的のみ・11フレームの
  6 個の統計量・10フレームの 6 個の統計量の AUPRC を測り直し、利得と
  回復率をその標本の中で計算し直す。11 フレーム全部の 6 個の統計量は
  全部入りの条件そのものなので、その回復率は標本の中でも 1.0 です。
  差は 1.0 − 10フレームの 6 個の統計量の回復率と同じことです。
  AUPRC の差（11フレームの 6 個の統計量 − 10フレームの 6 個の統計量）
  の区間も同じ段取りで出す。
- 計算の工夫: 1回の抽出で 2 タンパク質を復元抽出し、その行の集合に
  ついて、6つの予測幅 ×（静的のみ・11フレームの 6 個の統計量・
  10フレームの 6 個の統計量）の AUPRC を測ります。AUPRC の計算は
  sklearn の average_precision_score と同じ段取り（しきい値で区切って
  精度と再現率を取り、再現率の増え分で重み付けする）を numpy で書き直した
  もので、実際の予測確率で sklearn の値と 1e-12 まで一致することを
  確かめてから使います。抽出の乱数 seed は G23 と同じ 42 です。

読み方（先に固定）:
- G156 の 6 予測幅すべてについて、回復率の差の区間が 0 を含むかどうかを見る。
- G156 の差は 6 予測幅すべてで小さいので、区間とあわせて記録する。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

K46_CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 3.5
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り（G143 と同じ）
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）

# G156 の記録との照合用の期待値（合算 AUPRC）
# hist6_11f は 11 フレーム全部の 6 個の統計量で、全部入りの条件そのもの。
# その AUPRC は全部入りの AUPRC と一致し、回復率は 1.0。
EXPECTED = {
    1: {"static_only": 0.4160, "hist6_11f": 0.5945, "hist6_10f": 0.6049,
        "full_gain": 0.1785, "recovery_10f": 1.058, "recovery_11f": 1.000,
        "recovery_diff": -0.058, "auprc_diff": -0.0104},
    5: {"static_only": 0.4346, "hist6_11f": 0.6103, "hist6_10f": 0.6144,
        "full_gain": 0.1757, "recovery_10f": 1.023, "recovery_11f": 1.000,
        "recovery_diff": -0.023, "auprc_diff": -0.0041},
    10: {"static_only": 0.4496, "hist6_11f": 0.5974, "hist6_10f": 0.5956,
         "full_gain": 0.1478, "recovery_10f": 0.988, "recovery_11f": 1.000,
         "recovery_diff": 0.012, "auprc_diff": 0.0018},
    30: {"static_only": 0.4409, "hist6_11f": 0.5731, "hist6_10f": 0.5795,
         "full_gain": 0.1322, "recovery_10f": 1.048, "recovery_11f": 1.000,
         "recovery_diff": -0.048, "auprc_diff": -0.0064},
    50: {"static_only": 0.4088, "hist6_11f": 0.5231, "hist6_10f": 0.5332,
         "full_gain": 0.1142, "recovery_10f": 1.088, "recovery_11f": 1.000,
         "recovery_diff": -0.088, "auprc_diff": -0.0101},
    90: {"static_only": 0.5118, "hist6_11f": 0.5797, "hist6_10f": 0.5858,
         "full_gain": 0.0679, "recovery_10f": 1.090, "recovery_11f": 1.000,
         "recovery_diff": -0.090, "auprc_diff": -0.0061},
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
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K46_CACHE / f"{p}_R{r}.npz")
            parts.append({k: z[k] for k in z.files})
            proteins.append(np.full(len(z["y"]), p))
    return parts, np.concatenate(proteins)


def cat(parts, key):
    """K46 の hist は (11, N) で入っているので転置する (G156 の cat と同じ)。"""
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr = cat(train, "X").astype(np.float64)
Xte = cat(test, "X").astype(np.float64)
Htr = cat(train, "hist").astype(np.float64)
Hte = cat(test, "hist").astype(np.float64)
Ytr = cat(train, "y")
Yte = cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, "test proteins", len(set(test_protein)),
      flush=True)

prefix_hist6_tr = hist6(Htr[:, PREFIX_FRAMES])
prefix_hist6_te = hist6(Hte[:, PREFIX_FRAMES])

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in list(prot_idx.items())[:5]}, "... total",
      sum(len(v) for v in prot_idx.values()), flush=True)

CONDS = ["static_only", "hist6_11f", "hist6_10f"]


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "hist6_11f":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    return (np.column_stack([Xtr, prefix_hist6_tr]),
            np.column_stack([Xte, prefix_hist6_te]))


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
    gain = a["hist6_11f"] - a["static_only"]
    rec10f = (a["hist6_10f"] - a["static_only"]) / gain
    rec11f = (a["hist6_11f"] - a["static_only"]) / gain
    rec_diff = rec11f - rec10f
    auprc_diff = a["hist6_11f"] - a["hist6_10f"]
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        if len(idx) == 0:
            continue
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["hist6_11f"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "recovery_10f": (ap["hist6_10f"] - ap["static_only"]) / g if g else float("nan"),
            "recovery_11f": (ap["hist6_11f"] - ap["static_only"]) / g if g else float("nan"),
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain,
        "recovery_10f": rec10f, "recovery_11f": rec11f,
        "recovery_diff": rec_diff, "auprc_diff": auprc_diff,
        "per_protein": per_protein,
    }
    # G156 の記録との照合
    exp = EXPECTED[hns]
    for c in CONDS:
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(gain - exp["full_gain"]) < 5e-4, (hns, gain)
    assert abs(rec10f - exp["recovery_10f"]) < 5e-3, (hns, rec10f)
    assert abs(rec11f - exp["recovery_11f"]) < 5e-3, (hns, rec11f)
    assert abs(rec_diff - exp["recovery_diff"]) < 5e-3, (hns, rec_diff)
    assert abs(auprc_diff - exp["auprc_diff"]) < 5e-4, (hns, auprc_diff)
    print(f"{hns}ns: static {a['static_only']:.4f} hist6_11f {a['hist6_11f']:.4f} "
          f"hist6_10f {a['hist6_10f']:.4f} gain {gain:.4f} "
          f"rec10f {rec10f:.3f} rec11f {rec11f:.3f} diff {rec_diff:+.3f} "
          f"auprc diff {auprc_diff:+.4f}", flush=True)
    pos_count = sum(1 for prot in per_protein
                    if per_protein[prot]["recovery_10f"] == per_protein[prot]["recovery_10f"]
                    and per_protein[prot]["recovery_10f"] > 0.5)
    print(f"  per-protein recovery_10f > 0.5: {pos_count}/{len(per_protein)}", flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
# 実行が途中で止まっても、学習し直しからやり直さずに済むようにする。
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("checkpoint: saved test_predictions.npz", flush=True)

# ---- タンパク質単位のブートストラップ（30 タンパク質からの復元抽出） ----
# 1回の抽出につき行の集合を1回だけ作り、静的のみ・全部入り・切り取り
# 6統計量だけの AUPRC は予測幅ごとに1回だけ測る。
rng = np.random.default_rng(BOOT_SEED)
prot_list = list(TEST_P)
prot_rows = [prot_idx[p] for p in prot_list]
# 抽出はタンパク質の添字で行い、行はタンパク質ごとにまとめてつなげる
draws = rng.integers(0, len(prot_list), size=(N_BOOT, len(prot_list)))
vals = {hns: np.empty(N_BOOT) for hns in HORIZONS.values()}
vals_auprc = {hns: np.empty(N_BOOT) for hns in HORIZONS.values()}
for b in range(N_BOOT):
    idx = np.concatenate([prot_rows[i] for i in draws[b]])
    for hidx, hns in HORIZONS.items():
        yte = Yte[:, hidx]
        p = preds[hns]
        y = yte[idx]
        ap_s = ap_fast(y, p["static_only"][idx])
        ap_11f = ap_fast(y, p["hist6_11f"][idx])
        ap_10f = ap_fast(y, p["hist6_10f"][idx])
        gain = ap_11f - ap_s
        rec10f = (ap_10f - ap_s) / gain if gain else float("nan")
        rec11f = (ap_11f - ap_s) / gain if gain else float("nan")
        vals[hns][b] = rec11f - rec10f
        vals_auprc[hns][b] = ap_11f - ap_10f
    if (b + 1) % 2000 == 0:
        print(f"bootstrap {b + 1}/{N_BOOT} draws done", flush=True)
boot = {}
boot_auprc = {}
for hns in HORIZONS.values():
    v = vals[hns]
    lo, hi = np.percentile(v, [2.5, 97.5])
    boot[hns] = {
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
    print(f"{hns}ns bootstrap recovery diff: {boot[hns]['point_estimate']:+.3f} "
          f"[{boot[hns]['ci_lower']:+.3f}, {boot[hns]['ci_upper']:+.3f}] "
          f"contains zero={boot[hns]['contains_zero']}", flush=True)
    print(f"{hns}ns bootstrap (AUPRC diff): {boot_auprc[hns]['point_estimate']:+.4f} "
          f"[{boot_auprc[hns]['ci_lower']:+.4f}, {boot_auprc[hns]['ci_upper']:+.4f}] "
          f"zero={boot_auprc[hns]['contains_zero']}", flush=True)

# 回復率の差の区間が 0 を含むかどうか（先に固定した読み方）
contains_zero = {f"{h}ns": boot[h]["contains_zero"] for h in HORIZONS.values()}
print("recovery diff CI contains zero:", contains_zero, flush=True)
contains_zero_auprc = {f"{h}ns": boot_auprc[h]["contains_zero"] for h in HORIZONS.values()}
print("AUPRC diff CI contains zero:", contains_zero_auprc, flush=True)

out = {
    "label": "Hydrogen-bond dissociation, origin-frame 6-statistic recovery difference uncertainty across horizons (G156)",
    "phenomenon": "hydrogen-bond dissociation (D conditions, K46 cache)",
    "prefix_frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (2 test proteins, sampled with "
                  "replacement; only 3 distinct draws exist: both 1mix_A, "
                  "both 2qt4_A, one of each, so the percentile interval is "
                  "the min and max of the 3 values, as in G49/G59/G62/G67)",
                  "note": "AUPRC inside the bootstrap uses a numpy reimplementation "
                  "of sklearn's average_precision_score (same threshold-grouping "
                  "treatment of ties); it was verified to agree with sklearn to "
                  "1e-12 on all fitted conditions before use. Models were retrained "
                  "per condition and horizon; pooled AUPRCs were verified against "
                  "G156 records to 4 decimal places. The 11-frame 6-statistic "
                  "condition is the full-history condition itself, so its recovery "
                  "is 1.0 by definition, in every bootstrap replicate as well; "
                  "the recovery difference is 1.0 minus the 10-frame recovery."},
    "horizons": {str(h): results[h] for h in HORIZONS.values()},
    "bootstrap_results": {str(h): boot[h] for h in HORIZONS.values()},
    "bootstrap_results_auprc": {str(h): boot_auprc[h] for h in HORIZONS.values()},
    "recovery_diff_ci_contains_zero": contains_zero,
    "auprc_diff_ci_contains_zero": contains_zero_auprc,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "test_positives", "auprc_static", "auprc_hist6_11f",
                "auprc_hist6_10f", "full_gain", "recovery_10f", "recovery_11f",
                "recovery_diff", "boot_ci_lower", "boot_ci_upper", "ci_contains_zero",
                "auprc_diff", "boot_auprc_ci_lower",
                "boot_auprc_ci_upper", "auprc_ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        b = boot[h]
        ba = boot_auprc[h]
        w.writerow([h, r["test_positives"],
                    r["auprc"]["static_only"], r["auprc"]["hist6_11f"],
                    r["auprc"]["hist6_10f"], r["full_gain"],
                    r["recovery_10f"], r["recovery_11f"],
                    r["recovery_diff"], b["ci_lower"], b["ci_upper"],
                    b["contains_zero"],
                    r["auprc_diff"], ba["ci_lower"], ba["ci_upper"],
                    ba["contains_zero"]])
print("saved to", OUT)
