"""G174: 塩橋の形成で、起点のフレームを入れるかどうかでの 6 個の
統計量の回復率の差に、区間を付ける.

G160 では、6 個の統計量の回復率の 11 フレーム（フレーム [0..10]、起点の
フレーム 10 を含む）と 10 フレーム（フレーム [0..9]）の差が、1 ns先
+0.027、5 ns先 -0.127、10 ns先 -0.264、30 ns先 +0.202、50 ns先
+0.424、90 ns先 +0.184 でした。AUPRC の差は、1 ns先 +0.0039、
5 ns先 -0.0100、10 ns先 -0.0157、30 ns先 +0.0179、50 ns先 +0.0116、
90 ns先 +0.0111 でした。6 つの予測幅すべてについて、この差の
ブートストラップ 95% 区間を出します（G162・G163・G164 と同じやり方）。

再標本化の単位はタンパク質で、テストは K15 の 30 タンパク質です。
ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G96・G73 と同じです。

段取り:
- G160 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
- 回復率の定義: (切り取りの 6 個の統計量の AUPRC - 静的のみの AUPRC) /
  (全部入りの AUPRC - 静的のみの AUPRC)。各ブートストラップ標本の
  中でも、静的のみ・全部入り・切り取りの 6 個の統計量（10 フレームと
  11 フレーム）の AUPRC を測り直し、利得と回復率をその標本の中で
  計算し直し、回復率の差（11 フレーム - 10 フレーム）を取る。
  土台の利得が小さいので、AUPRC の差（11 フレームの 6 個の統計量 -
  10 フレームの 6 個の統計量）の区間も同じ段取りで出す。
- AUPRC の計算は sklearn の average_precision_score と同じ段取り
  （しきい値で区切って精度と再現率を取り、再現率の増え分で重み付け
  する）を numpy で書き直したもので、実際の予測確率で sklearn の値と
  1e-12 まで一致することを確かめてから使います。タンパク質ごとに
  陽性が 0 件の予測幅がある場合は、そのタンパク質の AUPRC は未定義
  として nan にし、平均には入れない（sklearn の警告つきの 0.0 ではなく）。

フレームの注記: 「1.0 ns」の切り取りは、呼び名の 1.0 ns に対して
実際は 0.9 ns の範囲である（フレーム [0,1,2,3,4,5,6,7,8,9] は時刻
0〜0.9 ns で、起点のフレーム10 は入らない）。11 フレームの 6 個の
統計量はフレーム [0..10] で、起点のフレーム 10 を含む。

読み方（先に固定）:
- 回復率の差の区間が 0 を含むかどうかを、予測幅ごとに見てください。
  50 ns先は土台の利得が +0.0274 と小さく、回復率の差 +0.424 は
  AUPRC の差では +0.0116 と小さい点も、区間とあわせて記録して
  ください。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
SPLIT = json.loads((K15_DIR / "split.json").read_text())
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
TRAIN_P = ORIG_TRAIN + tuple(SPLIT["train_additions"])
TEST_P = ORIG_TEST + tuple(SPLIT["test_additions"])
BOUND = 4.0
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り（G160 と同じ）
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）

# G160 の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    1: {"static_only": 0.2402, "full_hist": 0.3842, "prefix_hist6_10f": 0.3803,
        "prefix_hist6_11f": 0.3842, "full_gain": 0.1440,
        "recovery_10f": 0.9727, "recovery_11f": 1.0000,
        "recovery_diff": 0.0269, "auprc_diff": 0.0039},
    5: {"static_only": 0.2513, "full_hist": 0.3301, "prefix_hist6_10f": 0.3401,
        "prefix_hist6_11f": 0.3301, "full_gain": 0.0788,
        "recovery_10f": 1.1269, "recovery_11f": 1.0000,
        "recovery_diff": -0.1269, "auprc_diff": -0.0100},
    10: {"static_only": 0.2034, "full_hist": 0.2627, "prefix_hist6_10f": 0.2784,
         "prefix_hist6_11f": 0.2627, "full_gain": 0.0594,
         "recovery_10f": 1.2638, "recovery_11f": 1.0000,
         "recovery_diff": -0.2638, "auprc_diff": -0.0157},
    30: {"static_only": 0.1582, "full_hist": 0.2466, "prefix_hist6_10f": 0.2287,
         "prefix_hist6_11f": 0.2466, "full_gain": 0.0884,
         "recovery_10f": 0.7977, "recovery_11f": 1.0000,
         "recovery_diff": 0.2023, "auprc_diff": 0.0179},
    50: {"static_only": 0.2295, "full_hist": 0.2569, "prefix_hist6_10f": 0.2453,
         "prefix_hist6_11f": 0.2569, "full_gain": 0.0274,
         "recovery_10f": 0.5755, "recovery_11f": 1.0000,
         "recovery_diff": 0.4245, "auprc_diff": 0.0116},
    90: {"static_only": 0.2136, "full_hist": 0.2736, "prefix_hist6_10f": 0.2626,
         "prefix_hist6_11f": 0.2736, "full_gain": 0.0601,
         "recovery_10f": 0.8156, "recovery_11f": 1.0000,
         "recovery_diff": 0.1844, "auprc_diff": 0.0111},
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
    """実側は K15 キャッシュ（G160 と同じ）。"""
    real = []
    proteins = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15 / f"{p}_R{r}_salt.npz")
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

# 11 フレームの 6 個の統計量は、全部入りの条件そのもの（hist6(H) と同じ）。
# 10 フレームの 6 個の統計量は、フレーム [0..9] で計算する。

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in list(prot_idx.items())[:5]}, "... total",
      sum(len(v) for v in prot_idx.values()), flush=True)

CONDS = ["static_only", "full_hist", "prefix_hist6_10f", "prefix_hist6_11f"]


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist" or cname == "prefix_hist6_11f":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
            np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])]))


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
    rec = rec11 - rec10
    pgain = pgain11 - pgain10  # AUPRC の差（11 フレーム - 10 フレーム）
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
                "full_gain": float("nan"), "prefix_gain_10f": float("nan"),
                "prefix_gain_11f": float("nan"), "recovery_10f": float("nan"),
                "recovery_11f": float("nan"), "recovery_diff": float("nan"),
                "auprc_diff": float("nan"),
            }
            continue
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist"] - ap["static_only"]
        pg10 = ap["prefix_hist6_10f"] - ap["static_only"]
        pg11 = ap["prefix_hist6_11f"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": n_pos,
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "prefix_gain_10f": pg10,
            "prefix_gain_11f": pg11,
            "recovery_10f": (pg10 / g if g else float("nan")),
            "recovery_11f": (pg11 / g if g else float("nan")),
            "recovery_diff": ((pg11 - pg10) / g if g else float("nan")),
            "auprc_diff": pg11 - pg10,
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain,
        "prefix_gain_10f": pgain10, "prefix_gain_11f": pgain11,
        "recovery_10f": rec10, "recovery_11f": rec11,
        "recovery_diff": rec, "auprc_diff": pgain,
        "per_protein": per_protein,
    }
    # G160 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist", "prefix_hist6_10f", "prefix_hist6_11f"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(gain - exp["full_gain"]) < 5e-4, (hns, gain)
    assert abs(rec10 - exp["recovery_10f"]) < 5e-3, (hns, rec10)
    assert abs(rec11 - exp["recovery_11f"]) < 5e-3, (hns, rec11)
    assert abs(rec - exp["recovery_diff"]) < 5e-3, (hns, rec)
    assert abs(pgain - exp["auprc_diff"]) < 5e-4, (hns, pgain)
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"prefix_hist6_10f {a['prefix_hist6_10f']:.4f} "
          f"prefix_hist6_11f {a['prefix_hist6_11f']:.4f} gain {gain:.4f} "
          f"rec10 {rec10:.3f} rec11 {rec11:.3f} diff {rec:+.3f} "
          f"(record {exp['recovery_diff']:+.3f}) auprc_diff {pgain:+.4f}",
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
# 平均だけの AUPRC は予測幅ごとに1回だけ測る。
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
        ap_10 = ap_fast(y, p["prefix_hist6_10f"][idx])
        ap_11 = ap_fast(y, p["prefix_hist6_11f"][idx])
        gain = ap_f - ap_s
        rec10 = (ap_10 - ap_s) / gain if gain else float("nan")
        rec11 = (ap_11 - ap_s) / gain if gain else float("nan")
        vals[hns][b] = rec11 - rec10
        vals_auprc[hns][b] = ap_11 - ap_10
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
contains_0 = {f"{h}ns": boot[h]["contains_zero"] for h in HORIZONS.values()}
print("recovery-diff CIs contain 0:", contains_0, flush=True)

out = {
    "label": "Salt-bridge formation, origin-frame effect on 6-statistic recovery uncertainty across horizons",
    "phenomenon": "salt-bridge formation (K15 expanded conditions, K15 cache)",
    "prefix_frames": PREFIX_FRAMES,
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (30 test proteins, sampled with "
                  "replacement; unlike the 2-protein bases, the bootstrap "
                  "distribution takes many distinct values)",
                  "note": "AUPRC inside the bootstrap uses a numpy reimplementation "
                  "of sklearn's average_precision_score (same threshold-grouping "
                  "treatment of ties); it was verified to agree with sklearn to "
                  "1e-12 on all fitted conditions before use. Models were retrained "
                  "per condition and horizon; pooled AUPRCs were verified against "
                  "G160 records to 4 decimal places."},
    "horizons": {str(h): results[h] for h in HORIZONS.values()},
    "bootstrap_results_recovery_diff": {str(h): boot[h] for h in HORIZONS.values()},
    "bootstrap_results_auprc_diff": {str(h): boot_auprc[h] for h in HORIZONS.values()},
    "recovery_diff_cis_contain_zero": contains_0,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_hist6_10f", "auprc_prefix_hist6_11f",
                "full_gain", "recovery_10f", "recovery_11f", "recovery_diff",
                "auprc_diff",
                "boot_ci_lower", "boot_ci_upper", "ci_contains_zero",
                "boot_auprc_ci_lower", "boot_auprc_ci_upper",
                "auprc_ci_contains_zero"])
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

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
