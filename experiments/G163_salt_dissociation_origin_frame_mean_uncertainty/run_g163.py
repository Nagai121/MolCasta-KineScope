"""G163: 塩橋の解離で、起点のフレームを入れるかどうかでの平均だけの
回復率の差に、区間を付ける.

G151 では、平均だけの回復率の 11 フレーム（フレーム [0..10]、起点の
フレーム 10 を含む）と 10 フレーム（フレーム [0..9]）の差が、1 ns先
-0.130、5 ns先 +0.055、10 ns先 -0.057、30 ns先 +0.141、50 ns先
+0.009、90 ns先 +0.152 でした。AUPRC の差は、1 ns先 -0.0233、
5 ns先 +0.0063、10 ns先 -0.0101、30 ns先 +0.0089、50 ns先 +0.0012、
90 ns先 +0.0088 でした。6 つの予測幅すべてについて、この差の
ブートストラップ 95% 区間を出します（G147・G148 と同じやり方）。

再標本化の単位はタンパク質。テストは 30 タンパク質（K15 の広げた条件の
テスト分け）なので、2 タンパク質の土台（G49・G59・G62・G67）のような
「取りうる組み合わせは3通りしかない」という粗さはありません。
30 タンパク質から復元抽出で 30 タンパク質を選び直す、ふつうの
ブートストラップです。区間は 2 タンパク質の土台よりは細かい、という
留保を記録します（G73・G96 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G73・G49・G59・G62・G67・G23 と同じです。

段取り:
- G151 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
- 回復率の定義: (切り取りの平均だけの AUPRC - 静的のみの AUPRC) /
  (全部入りの AUPRC - 静的のみの AUPRC)。各ブートストラップ標本の
  中でも、静的のみ・全部入り・切り取りの平均だけ（10 フレームと
  11 フレーム）の AUPRC を測り直し、利得と回復率をその標本の中で
  計算し直し、回復率の差（11 フレーム - 10 フレーム）を取る。土台の
  利得が小さいので、AUPRC の差（11 フレームの平均だけ - 10 フレームの
  平均だけ）の区間も同じ段取りで出す。
- 計算の工夫: 1回の抽出で 30 タンパク質を復元抽出し、その行の集合に
  ついて、6つの予測幅 ×（静的のみ・全部入り・切り取りの平均だけ
  10 フレーム・切り取りの平均だけ 11 フレーム）の AUPRC を測ります。
  AUPRC の計算は sklearn の average_precision_score と同じ段取り
  （しきい値で区切って精度と再現率を取り、再現率の増え分で重み付け
  する）を numpy で書き直したもので、実際の予測確率で sklearn の値と
  1e-12 まで一致することを確かめてから使います。抽出の乱数 seed は
  G23 と同じ 42 です。

フレームの注記: 「1.0 ns」の切り取りは、呼び名の 1.0 ns に対して
実際は 0.9 ns の範囲である（フレーム [0,1,2,3,4,5,6,7,8,9] は時刻
0〜0.9 ns で、起点のフレーム10 は入らない）。11 フレームの平均だけは
フレーム [0..10] で、起点のフレーム 10 を含む。

読み方（先に固定）:
- 回復率の差の区間が 0 を含むかどうかを、予測幅ごとに見てください。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

K53 = Path("/home/hatch/workspace/kinescope/work/k53_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
SPLIT = json.loads((K15_DIR / "split.json").read_text())
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
TRAIN_P = ORIG_TRAIN + tuple(SPLIT["train_additions"])
TEST_P = ORIG_TEST + tuple(SPLIT["test_additions"])
BOUND = 4.0
PREFIX_FRAMES_10F = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 フレーム（G142/G151 と同じ）
PREFIX_FRAMES_11F = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 11 フレーム、起点のフレーム 10 を含む
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）

# G151 の記録との照合用の期待値（合算 AUPRC、G151 の results.json から）
EXPECTED = {
    1: {"static_only": 0.3734, "full_hist": 0.5522, "prefix_mean_10f": 0.5614,
        "prefix_mean_11f": 0.5382, "full_gain": 0.1788,
        "recovery_10f": 1.0515, "recovery_11f": 0.9215,
        "recovery_diff": -0.1300, "auprc_diff": -0.0233},
    5: {"static_only": 0.4641, "full_hist": 0.5780, "prefix_mean_10f": 0.5731,
        "prefix_mean_11f": 0.5794, "full_gain": 0.1139,
        "recovery_10f": 0.9572, "recovery_11f": 1.0123,
        "recovery_diff": 0.0551, "auprc_diff": 0.0063},
    10: {"static_only": 0.3572, "full_hist": 0.5353, "prefix_mean_10f": 0.5402,
         "prefix_mean_11f": 0.5302, "full_gain": 0.1781,
         "recovery_10f": 1.0278, "recovery_11f": 0.9712,
         "recovery_diff": -0.0565, "auprc_diff": -0.0101},
    30: {"static_only": 0.4799, "full_hist": 0.5432, "prefix_mean_10f": 0.5584,
         "prefix_mean_11f": 0.5673, "full_gain": 0.0633,
         "recovery_10f": 1.2399, "recovery_11f": 1.3810,
         "recovery_diff": 0.1411, "auprc_diff": 0.0089},
    50: {"static_only": 0.4863, "full_hist": 0.6165, "prefix_mean_10f": 0.5833,
         "prefix_mean_11f": 0.5845, "full_gain": 0.1302,
         "recovery_10f": 0.7450, "recovery_11f": 0.7541,
         "recovery_diff": 0.0091, "auprc_diff": 0.0012},
    90: {"static_only": 0.4824, "full_hist": 0.5405, "prefix_mean_10f": 0.5342,
         "prefix_mean_11f": 0.5430, "full_gain": 0.0582,
         "recovery_10f": 0.8916, "recovery_11f": 1.0432,
         "recovery_diff": 0.1516, "auprc_diff": 0.0088},
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
    """実側は K53 キャッシュ（G151 と同じ）。"""
    real = []
    proteins = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K53 / f"{p}_R{r}_salt_diss.npz")
            real.append({k: z[k] for k in z.files})
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

prefix_mean_10f_tr = Htr[:, PREFIX_FRAMES_10F].mean(1)
prefix_mean_10f_te = Hte[:, PREFIX_FRAMES_10F].mean(1)
prefix_mean_11f_tr = Htr[:, PREFIX_FRAMES_11F].mean(1)
prefix_mean_11f_te = Hte[:, PREFIX_FRAMES_11F].mean(1)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in list(prot_idx.items())[:5]}, "... total",
      sum(len(v) for v in prot_idx.values()), flush=True)

CONDS = ["static_only", "full_hist", "prefix_mean_10f", "prefix_mean_11f"]


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname == "prefix_mean_10f":
        return (np.column_stack([Xtr, prefix_mean_10f_tr[:, None].astype(np.float32)]),
                np.column_stack([Xte, prefix_mean_10f_te[:, None].astype(np.float32)]))
    return (np.column_stack([Xtr, prefix_mean_11f_tr[:, None].astype(np.float32)]),
            np.column_stack([Xte, prefix_mean_11f_te[:, None].astype(np.float32)]))


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
    pgain10 = a["prefix_mean_10f"] - a["static_only"]
    pgain11 = a["prefix_mean_11f"] - a["static_only"]
    rec10 = pgain10 / gain if gain else float("nan")
    rec11 = pgain11 / gain if gain else float("nan")
    rec_diff = rec11 - rec10
    auprc_diff = a["prefix_mean_11f"] - a["prefix_mean_10f"]
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
        r10 = ((ap["prefix_mean_10f"] - ap["static_only"]) / g
               if g else float("nan"))
        r11 = ((ap["prefix_mean_11f"] - ap["static_only"]) / g
               if g else float("nan"))
        per_protein[prot] = {
            "n": int(len(idx)), "positives": n_pos,
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "prefix_gain_10f": ap["prefix_mean_10f"] - ap["static_only"],
            "prefix_gain_11f": ap["prefix_mean_11f"] - ap["static_only"],
            "recovery_10f": r10, "recovery_11f": r11,
            "recovery_diff": r11 - r10,
            "auprc_diff": ap["prefix_mean_11f"] - ap["prefix_mean_10f"],
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain,
        "prefix_gain_10f": pgain10, "prefix_gain_11f": pgain11,
        "recovery_10f": rec10, "recovery_11f": rec11,
        "recovery_diff": rec_diff, "auprc_diff": auprc_diff,
        "per_protein": per_protein,
    }
    # G151 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist", "prefix_mean_10f", "prefix_mean_11f"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(gain - exp["full_gain"]) < 5e-4, (hns, gain)
    assert abs(rec10 - exp["recovery_10f"]) < 5e-3, (hns, rec10)
    assert abs(rec11 - exp["recovery_11f"]) < 5e-3, (hns, rec11)
    assert abs(rec_diff - exp["recovery_diff"]) < 5e-3, (hns, rec_diff)
    assert abs(auprc_diff - exp["auprc_diff"]) < 5e-4, (hns, auprc_diff)
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"prefix_mean_10f {a['prefix_mean_10f']:.4f} prefix_mean_11f "
          f"{a['prefix_mean_11f']:.4f} gain {gain:.4f} "
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

# ---- タンパク質単位のブートストラップ（30 タンパク質からの復元抽出） ----
# 1回の抽出につき行の集合を1回だけ作り、静的のみ・全部入り・切り取りの
# 平均だけ（10 フレーム・11 フレーム）の AUPRC は予測幅ごとに1回だけ測る。
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
        ap_f = ap_fast(y, p["full_hist"][idx])
        ap_m10 = ap_fast(y, p["prefix_mean_10f"][idx])
        ap_m11 = ap_fast(y, p["prefix_mean_11f"][idx])
        gain = ap_f - ap_s
        r10 = (ap_m10 - ap_s) / gain if gain else float("nan")
        r11 = (ap_m11 - ap_s) / gain if gain else float("nan")
        vals[hns][b] = r11 - r10
        vals_auprc[hns][b] = ap_m11 - ap_m10
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
    "label": "Salt-bridge dissociation, origin-frame effect on mean-only recovery uncertainty across horizons (G151)",
    "phenomenon": "salt-bridge dissociation (K15 expanded conditions, K53 cache)",
    "prefix_frames_10f": PREFIX_FRAMES_10F, "prefix_frames_11f": PREFIX_FRAMES_11F,
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (30 test proteins, sampled with "
                  "replacement; unlike the 2-protein bases, the bootstrap "
                  "distribution takes many distinct values)",
                  "note": "AUPRC inside the bootstrap uses a numpy reimplementation "
                  "of sklearn's average_precision_score (same threshold-grouping "
                  "treatment of ties); it was verified to agree with sklearn to "
                  "1e-12 on all fitted conditions before use. Models were retrained "
                  "per condition and horizon; pooled AUPRCs were verified against "
                  "G151 records to 4 decimal places."},
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
                "auprc_full", "auprc_prefix_mean_10f", "auprc_prefix_mean_11f",
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
                    r["auprc"]["prefix_mean_10f"], r["auprc"]["prefix_mean_11f"],
                    r["full_gain"], r["recovery_10f"], r["recovery_11f"],
                    r["recovery_diff"], r["auprc_diff"],
                    b["ci_lower"], b["ci_upper"], b["contains_zero"],
                    ba["ci_lower"], ba["ci_upper"], ba["contains_zero"]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
