"""G172: 接触の解離で、起点のフレームを入れるかどうかでの 6 個の
統計量の回復率の差に、区間を付ける.

G159 では、6 個の統計量の回復率の 11 フレーム（フレーム [0..10]、起点の
フレーム 10 を含む）と 10 フレーム（フレーム [0..9]）の差が、1 ns先
-0.027、5 ns先 -0.029、10 ns先 +0.109、30 ns先 +0.168、50 ns先
+0.104、90 ns先 +0.031 でした。AUPRC の差は、1 ns先 -0.0058、
5 ns先 -0.0073、10 ns先 +0.0220、30 ns先 +0.0399、50 ns先 +0.0167、
90 ns先 +0.0058 でした。6 つの予測幅すべてについて、この差の
ブートストラップ 95% 区間を出します（G162・G163・G164 と同じやり方）。

再標本化の単位はタンパク質。ただし土台は D 条件で、テストのタンパク質は
1mix_A と 2qt4_A の 2つだけなので、タンパク質単位の再標本化で取りうる
組み合わせは {1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通りしか
ありません。そのためブートストラップ分布も3つの値しか取らず、95% 区間は
その3値の最小値〜最大値になります。区間が粗いことは留保として記録する
（G49・G59・G62・G67・G118・G133 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G73・G49・G59・G62・G67・G23 と同じです。

段取り:
- G159 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
- 回復率の定義: (切り取りの 6 個の統計量の AUPRC - 静的のみの AUPRC) /
  (全部入りの AUPRC - 静的のみの AUPRC)。各ブートストラップ標本の
  中でも、静的のみ・全部入り・切り取りの 6 個の統計量（10 フレームと
  11 フレーム）の AUPRC を測り直し、利得と回復率をその標本の中で
  計算し直し、回復率の差（11 フレーム - 10 フレーム）を取る。土台の
  利得が小さいので、AUPRC の差（11 フレームの 6 個の統計量 - 10 フレームの
  6 個の統計量）の区間も同じ段取りで出す。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42。
- AUPRC の計算は sklearn の average_precision_score と同じ段取り
  （しきい値で区切って精度と再現率を取り、再現率の増え分で重み付け
  する）を numpy で書き直したもので、実際の予測確率で sklearn の値と
  1e-12 まで一致することを確かめてから使います。

フレームの注記: 「1.0 ns」の切り取りは、呼び名の 1.0 ns に対して
実際は 0.9 ns の範囲である（フレーム [0,1,2,3,4,5,6,7,8,9] は時刻
0〜0.9 ns で、起点のフレーム10 は入らない）。11 フレームの 6 個の
統計量はフレーム [0..10] で、起点のフレーム 10 を含む。

候補の範囲の注記: G159 は全部の候補で測っており、G131・G134 の
76% の部分集合とは条件が違う。G131・G134 は G18 の方法22の生成系列と
重なる候補だけ（テストの 76%）で測っているが、G159 は重なりの有無を
問わず全部の候補で測っている。

読み方（先に固定）:
- 回復率の差の区間が 0 を含むかどうかを、予測幅ごとに見てください。
  30 ns先は差が +0.168 と大きく、区間とあわせて記録してください。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
PREFIX_FRAMES_10F = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 フレーム（G159 と同じ）
PREFIX_FRAMES_11F = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 11 フレーム、起点のフレーム 10 を含む
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))

# G159 の記録との照合用の期待値（合算 AUPRC、G159 の results.json から）
EXPECTED = {
    1: {"static_only": 0.3105, "full_hist": 0.5246, "prefix_hist6_10f": 0.5304,
        "prefix_hist6_11f": 0.5246, "full_gain": 0.2142,
        "recovery_10f": 1.0269, "recovery_11f": 1.0000,
        "recovery_diff": -0.0269, "auprc_diff": -0.0058},
    5: {"static_only": 0.3022, "full_hist": 0.5528, "prefix_hist6_10f": 0.5602,
        "prefix_hist6_11f": 0.5528, "full_gain": 0.2506,
        "recovery_10f": 1.0293, "recovery_11f": 1.0000,
        "recovery_diff": -0.0293, "auprc_diff": -0.0073},
    10: {"static_only": 0.2772, "full_hist": 0.4794, "prefix_hist6_10f": 0.4575,
         "prefix_hist6_11f": 0.4794, "full_gain": 0.2022,
         "recovery_10f": 0.8914, "recovery_11f": 1.0000,
         "recovery_diff": 0.1086, "auprc_diff": 0.0220},
    30: {"static_only": 0.2756, "full_hist": 0.5124, "prefix_hist6_10f": 0.4725,
         "prefix_hist6_11f": 0.5124, "full_gain": 0.2368,
         "recovery_10f": 0.8317, "recovery_11f": 1.0000,
         "recovery_diff": 0.1683, "auprc_diff": 0.0399},
    50: {"static_only": 0.2927, "full_hist": 0.4533, "prefix_hist6_10f": 0.4366,
         "prefix_hist6_11f": 0.4533, "full_gain": 0.1606,
         "recovery_10f": 0.8963, "recovery_11f": 1.0000,
         "recovery_diff": 0.1037, "auprc_diff": 0.0167},
    90: {"static_only": 0.2886, "full_hist": 0.4750, "prefix_hist6_10f": 0.4692,
         "prefix_hist6_11f": 0.4750, "full_gain": 0.1864,
         "recovery_10f": 0.9687, "recovery_11f": 1.0000,
         "recovery_diff": 0.0313, "auprc_diff": 0.0058},
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
    """実側は k50_cache（G159 と同じ）。hist は (N, 11) で保存されているので転置しない。"""
    real = []
    proteins = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
            d = {k: z[k] for k in z.files}
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

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, "total",
      sum(len(v) for v in prot_idx.values()), flush=True)

CONDS = ["static_only", "full_hist", "prefix_hist6_10f", "prefix_hist6_11f"]


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname == "prefix_hist6_10f":
        return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES_10F])]),
                np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES_10F])]))
    return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES_11F])]),
            np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES_11F])]))


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
    pgain10 = a["prefix_hist6_10f"] - a["static_only"]
    pgain11 = a["prefix_hist6_11f"] - a["static_only"]
    rec10 = pgain10 / gain if gain else float("nan")
    rec11 = pgain11 / gain if gain else float("nan")
    rec_diff = rec11 - rec10
    auprc_diff = a["prefix_hist6_11f"] - a["prefix_hist6_10f"]
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        if len(idx) == 0:
            continue
        n_pos = int(yte[idx].sum())
        if n_pos == 0:
            # そのタンパク質に陽性が1件もない予測幅では、AUPRC は定義されない。
            # sklearn は警告を出して 0.0 を返すが、ここでは nan と記録する。
            per_protein[prot] = {
                "n": int(len(idx)), "positives": 0,
                "auprc": {c: float("nan") for c in CONDS},
                "full_gain": float("nan"),
                "prefix_gain_10f": float("nan"), "prefix_gain_11f": float("nan"),
                "recovery_10f": float("nan"), "recovery_11f": float("nan"),
                "recovery_diff": float("nan"), "auprc_diff": float("nan"),
            }
            continue
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist"] - ap["static_only"]
        r10 = ((ap["prefix_hist6_10f"] - ap["static_only"]) / g
               if g else float("nan"))
        r11 = ((ap["prefix_hist6_11f"] - ap["static_only"]) / g
               if g else float("nan"))
        per_protein[prot] = {
            "n": int(len(idx)), "positives": n_pos,
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "prefix_gain_10f": ap["prefix_hist6_10f"] - ap["static_only"],
            "prefix_gain_11f": ap["prefix_hist6_11f"] - ap["static_only"],
            "recovery_10f": r10, "recovery_11f": r11,
            "recovery_diff": r11 - r10,
            "auprc_diff": ap["prefix_hist6_11f"] - ap["prefix_hist6_10f"],
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain,
        "prefix_gain_10f": pgain10, "prefix_gain_11f": pgain11,
        "recovery_10f": rec10, "recovery_11f": rec11,
        "recovery_diff": rec_diff, "auprc_diff": auprc_diff,
        "per_protein": per_protein,
    }
    # G159 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist", "prefix_hist6_10f", "prefix_hist6_11f"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(gain - exp["full_gain"]) < 5e-4, (hns, gain)
    assert abs(rec10 - exp["recovery_10f"]) < 5e-3, (hns, rec10)
    assert abs(rec11 - exp["recovery_11f"]) < 5e-3, (hns, rec11)
    assert abs(rec_diff - exp["recovery_diff"]) < 5e-3, (hns, rec_diff)
    assert abs(auprc_diff - exp["auprc_diff"]) < 5e-4, (hns, auprc_diff)
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"prefix_hist6_10f {a['prefix_hist6_10f']:.4f} prefix_hist6_11f "
          f"{a['prefix_hist6_11f']:.4f} gain {gain:.4f} "
          f"rec10 {rec10:.3f} rec11 {rec11:.3f} diff {rec_diff:+.3f} "
          f"(record {exp['recovery_diff']:+.3f}) auprc_diff {auprc_diff:+.4f}",
          flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("checkpoint: saved test_predictions.npz", flush=True)


def recovery_diff_for(y, p_s, p_f, p_10f, p_11f):
    ap_s = ap_fast(y, p_s)
    ap_f = ap_fast(y, p_f)
    ap_10 = ap_fast(y, p_10f)
    ap_11 = ap_fast(y, p_11f)
    gain = ap_f - ap_s
    r10 = (ap_10 - ap_s) / gain if gain else float("nan")
    r11 = (ap_11 - ap_s) / gain if gain else float("nan")
    return r11 - r10, ap_11 - ap_10


# ---- タンパク質単位のブートストラップ（G118・G133 と同じ、3通りの組み合わせへの写像） ----
rng = np.random.default_rng(BOOT_SEED)
boot = {}
boot_auprc = {}
for hidx, hns in HORIZONS.items():
    yte = Yte[:, hidx]
    p = preds[hns]
    three_rec = {}
    three_auprc = {}
    for combo in COMBOS:
        idx = np.concatenate([prot_idx[prot] for prot in combo])
        rd, ad = recovery_diff_for(
            yte[idx], p["static_only"][idx], p["full_hist"][idx],
            p["prefix_hist6_10f"][idx], p["prefix_hist6_11f"][idx])
        three_rec["+".join(combo)] = rd
        three_auprc["+".join(combo)] = ad
    combo_idx = rng.integers(0, len(COMBOS), size=N_BOOT)
    vals = np.array([three_rec["+".join(COMBOS[i])] for i in combo_idx])
    vals_a = np.array([three_auprc["+".join(COMBOS[i])] for i in combo_idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    loa, hia = np.percentile(vals_a, [2.5, 97.5])
    boot[hns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[hns]["recovery_diff"],
        "ci_lower": float(lo), "ci_upper": float(hi),
        "contains_zero": bool(lo <= 0 <= hi),
        "three_resample_values": three_rec,
        "n_distinct_boot_values": int(len(np.unique(np.round(vals, 12)))),
    }
    boot_auprc[hns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[hns]["auprc_diff"],
        "ci_lower": float(loa), "ci_upper": float(hia),
        "contains_zero": bool(loa <= 0 <= hia),
        "three_resample_values": three_auprc,
        "n_distinct_boot_values": int(len(np.unique(np.round(vals_a, 12)))),
    }
    print(f"{hns}ns bootstrap recovery diff: {boot[hns]['point_estimate']:+.3f} "
          f"[{boot[hns]['ci_lower']:+.3f}, {boot[hns]['ci_upper']:+.3f}] "
          f"contains_zero={boot[hns]['contains_zero']}", flush=True)
    print(f"{hns}ns bootstrap (AUPRC diff): {boot_auprc[hns]['point_estimate']:+.4f} "
          f"[{boot_auprc[hns]['ci_lower']:+.4f}, {boot_auprc[hns]['ci_upper']:+.4f}] "
          f"zero={boot_auprc[hns]['contains_zero']}", flush=True)

# 回復率の差の区間が 0 を含むかどうか（先に固定した読み方）
contains_zero = {f"{h}ns": boot[h]["contains_zero"] for h in HORIZONS.values()}
print("recovery-diff CIs contain 0:", contains_zero, flush=True)
contains_zero_auprc = {f"{h}ns": boot_auprc[h]["contains_zero"]
                       for h in HORIZONS.values()}
print("AUPRC-diff CIs contain 0:", contains_zero_auprc, flush=True)

out = {
    "label": "Contact dissociation, origin-frame effect on 6-statistic recovery uncertainty across horizons (G159)",
    "phenomenon": "contact dissociation (Doc D, k50_cache)",
    "prefix_frames_10f": PREFIX_FRAMES_10F, "prefix_frames_11f": PREFIX_FRAMES_11F,
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (1mix_A, 2qt4_A; 2 proteins only, "
                  "so only 3 resample combinations exist and the bootstrap "
                  "distribution takes only 3 distinct values; the 95% CI is "
                  "the min-max of the 3 values)",
                  "note": "The 10,000 draws are mapped onto the 3 exact combination "
                  "values; this is mathematically identical to recomputing AUPRC "
                  "on the resampled rows each time, because the difference depends "
                  "only on the chosen combination. AUPRC inside the bootstrap uses "
                  "a numpy reimplementation of sklearn's average_precision_score "
                  "(same threshold-grouping treatment of ties); it was verified "
                  "to agree with sklearn to 1e-12 on all fitted conditions "
                  "before use. Models were retrained per condition and horizon; "
                  "pooled AUPRCs were verified against G159 records to "
                  "4 decimal places."},
    "horizons": {str(h): results[h] for h in HORIZONS.values()},
    "bootstrap_results_recovery_diff": {str(h): boot[h] for h in HORIZONS.values()},
    "bootstrap_results_auprc_diff": {str(h): boot_auprc[h] for h in HORIZONS.values()},
    "recovery_diff_cis_contain_zero": contains_zero,
    "auprc_diff_cis_contain_zero": contains_zero_auprc,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_hist6_10f", "auprc_prefix_hist6_11f",
                "full_gain", "recovery_10f", "recovery_11f",
                "recovery_diff", "auprc_diff",
                "boot_diff_ci_lower", "boot_diff_ci_upper", "diff_ci_contains_zero",
                "boot_auprc_diff_ci_lower", "boot_auprc_diff_ci_upper",
                "auprc_diff_ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        b = boot[h]
        ba = boot_auprc[h]
        w.writerow([h, r["test_positives"],
                    r["auprc"]["static_only"], r["auprc"]["full_hist"],
                    r["auprc"]["prefix_hist6_10f"], r["auprc"]["prefix_hist6_11f"],
                    r["full_gain"], r["recovery_10f"], r["recovery_11f"],
                    r["recovery_diff"], r["auprc_diff"],
                    b["ci_lower"], b["ci_upper"], b["contains_zero"],
                    ba["ci_lower"], ba["ci_upper"], ba["contains_zero"]])

print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
