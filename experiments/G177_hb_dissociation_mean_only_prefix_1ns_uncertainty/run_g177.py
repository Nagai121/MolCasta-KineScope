"""G177: 水素結合の解離で、切り取り 1.0 ns の平均だけの回復率に、区間を付ける.

G137 では、切り取り 1.0 ns の平均だけの回復率が、1 ns先 0.935、
5 ns先 0.985、10 ns先 0.961、30 ns先 0.902、50 ns先 0.688、90 ns先
1.258 でした。6 つの予測幅すべてについて、回復率のブートストラップ
95% 区間を出します（G165〜G172 と同じやり方）。

再標本化の単位はタンパク質。ただし土台は D 条件で、テストのタンパク質は
1mix_A と 2qt4_A の 2つだけなので、タンパク質単位の再標本化で取りうる
組み合わせは {1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通りしか
ありません。そのためブートストラップ分布も3つの値しか取らず、95% 区間は
その3値の最小値〜最大値になります。1 万回のブートストラップは意味が
ないので、区間は 3 つの値（合算、1mix_A、2qt4_A）の最小値と最大値に
します。区間が粗いことは留保として記録する（G49・G59・G62・G67・
G118・G133 と同じ留保）。

段取り:
- G137 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこから区間を出す。
- 回復率の定義: (切り取りの平均だけの AUPRC - 静的のみの AUPRC) /
  (全部入りの AUPRC - 静的のみの AUPRC)。各組み合わせの中でも、
  静的のみ・全部入り・切り取りの平均だけの AUPRC を測り直し、利得と
  回復率をその組み合わせの中で計算し直す。土台の利得が小さいので、
  AUPRC の差（切り取りの平均だけ - 静的のみ）の区間も同じ段取りで出す。
- AUPRC の計算は sklearn の average_precision_score と同じ段取り
  （しきい値で区切って精度と再現率を取り、再現率の増え分で重み付け
  する）を numpy で書き直したもので、実際の予測確率で sklearn の値と
  1e-12 まで一致することを確かめてから使います。

フレームの注記: 「1.0 ns」の切り取りは、呼び名の 1.0 ns に対して
実際は 0.9 ns の範囲である（フレーム [0,1,2,3,4,5,6,7,8,9] は時刻
0〜0.9 ns で、起点のフレーム10 は入らない）。

読み方（先に固定）:
- 回復率の区間が 0.5 を含むかどうかを、予測幅ごとに見てください。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 3.5
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り（G137 と同じ）
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))

# G137 の記録との照合用の期待値（合算 AUPRC、G137 の results.json から）
EXPECTED = {
    1: {"static_only": 0.4160, "full_hist": 0.5945, "prefix_mean_1.0ns": 0.5829,
        "full_gain": 0.1785, "prefix_gain": 0.1669, "recovery": 0.9349},
    5: {"static_only": 0.4346, "full_hist": 0.6103, "prefix_mean_1.0ns": 0.6076,
        "full_gain": 0.1757, "prefix_gain": 0.1730, "recovery": 0.9846},
    10: {"static_only": 0.4496, "full_hist": 0.5974, "prefix_mean_1.0ns": 0.5916,
         "full_gain": 0.1478, "prefix_gain": 0.1420, "recovery": 0.9609},
    30: {"static_only": 0.4409, "full_hist": 0.5731, "prefix_mean_1.0ns": 0.5601,
         "full_gain": 0.1322, "prefix_gain": 0.1192, "recovery": 0.9015},
    50: {"static_only": 0.4088, "full_hist": 0.5231, "prefix_mean_1.0ns": 0.4874,
         "full_gain": 0.1142, "prefix_gain": 0.0786, "recovery": 0.6880},
    90: {"static_only": 0.5118, "full_hist": 0.5797, "prefix_mean_1.0ns": 0.5972,
         "full_gain": 0.0679, "prefix_gain": 0.0854, "recovery": 1.2581},
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
    """実側は data_cache_k46（G137 と同じ）。hist は (11, N) で保存されているので転置する。"""
    real = []
    proteins = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: (z[k].T if k == "hist" else z[k]) for k in z.files}
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

CONDS = ["static_only", "full_hist", "prefix_mean_1.0ns"]


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    return (np.column_stack([Xtr, Htr[:, PREFIX_FRAMES].mean(1)[:, None].astype(np.float32)]),
            np.column_stack([Xte, Hte[:, PREFIX_FRAMES].mean(1)[:, None].astype(np.float32)]))


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
    pgain = a["prefix_mean_1.0ns"] - a["static_only"]
    rec = pgain / gain if gain else float("nan")
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
                "prefix_gain": float("nan"), "recovery": float("nan"),
            }
            continue
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist"] - ap["static_only"]
        pg = ap["prefix_mean_1.0ns"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": n_pos,
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "prefix_gain": pg,
            "recovery": (pg / g if g else float("nan")),
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain,
        "prefix_gain": pgain, "recovery": rec,
        "per_protein": per_protein,
    }
    # G137 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist", "prefix_mean_1.0ns"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(gain - exp["full_gain"]) < 5e-4, (hns, gain)
    assert abs(pgain - exp["prefix_gain"]) < 5e-4, (hns, pgain)
    assert abs(rec - exp["recovery"]) < 5e-3, (hns, rec)
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"prefix_mean_1.0ns {a['prefix_mean_1.0ns']:.4f} gain {gain:.4f} "
          f"prefix_gain {pgain:+.4f} recovery {rec:.3f} "
          f"(record {exp['recovery']:.3f})",
          flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("checkpoint: saved test_predictions.npz", flush=True)


def recovery_for(y, p_s, p_f, p_m):
    ap_s = ap_fast(y, p_s)
    ap_f = ap_fast(y, p_f)
    ap_m = ap_fast(y, p_m)
    gain = ap_f - ap_s
    return (ap_m - ap_s) / gain if gain else float("nan"), ap_m - ap_s


# ---- タンパク質単位のブートストラップ（G118・G133 と同じ、3通りの組み合わせ） ----
# 2 タンパク質だけなので、1 万回のブートストラップは意味がない。
# 区間は 3 つの値（合算、1mix_A、2qt4_A）の最小値と最大値にする。
boot = {}
boot_auprc = {}
for hidx, hns in HORIZONS.items():
    yte = Yte[:, hidx]
    p = preds[hns]
    three_rec = {}
    three_auprc = {}
    for combo in COMBOS:
        idx = np.concatenate([prot_idx[prot] for prot in combo])
        r, ad = recovery_for(
            yte[idx], p["static_only"][idx], p["full_hist"][idx],
            p["prefix_mean_1.0ns"][idx])
        three_rec["+".join(combo)] = r
        three_auprc["+".join(combo)] = ad
    lo, hi = min(three_rec.values()), max(three_rec.values())
    loa, hia = min(three_auprc.values()), max(three_auprc.values())
    boot[hns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[hns]["recovery"],
        "ci_lower": float(lo), "ci_upper": float(hi),
        "contains_0.5": bool(lo <= 0.5 <= hi),
        "three_resample_values": three_rec,
        "n_distinct_boot_values": int(len(set(round(v, 12) for v in three_rec.values()))),
    }
    boot_auprc[hns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[hns]["prefix_gain"],
        "ci_lower": float(loa), "ci_upper": float(hia),
        "contains_zero": bool(loa <= 0 <= hia),
        "three_resample_values": three_auprc,
        "n_distinct_boot_values": int(len(set(round(v, 12) for v in three_auprc.values()))),
    }
    print(f"{hns}ns bootstrap recovery: {boot[hns]['point_estimate']:+.3f} "
          f"[{boot[hns]['ci_lower']:+.3f}, {boot[hns]['ci_upper']:+.3f}] "
          f"contains_0.5={boot[hns]['contains_0.5']}", flush=True)
    print(f"{hns}ns bootstrap (AUPRC diff): {boot_auprc[hns]['point_estimate']:+.4f} "
          f"[{boot_auprc[hns]['ci_lower']:+.4f}, {boot_auprc[hns]['ci_upper']:+.4f}] "
          f"zero={boot_auprc[hns]['contains_zero']}", flush=True)

# 回復率の区間が 0.5 を含むかどうか（先に固定した読み方）
contains_05 = {f"{h}ns": boot[h]["contains_0.5"] for h in HORIZONS.values()}
print("recovery CIs contain 0.5:", contains_05, flush=True)
contains_zero_auprc = {f"{h}ns": boot_auprc[h]["contains_zero"]
                       for h in HORIZONS.values()}
print("AUPRC-diff CIs contain 0:", contains_zero_auprc, flush=True)

out = {
    "label": "Hydrogen-bond dissociation, 1.0ns prefix mean-only recovery uncertainty across horizons (G137)",
    "phenomenon": "hydrogen-bond dissociation (Doc D, data_cache_k46)",
    "prefix_frames": PREFIX_FRAMES,
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (1mix_A, 2qt4_A; 2 proteins only, "
                  "so only 3 resample combinations exist and the bootstrap "
                  "distribution takes only 3 distinct values; the 95% CI is "
                  "the min-max of the 3 values)",
                  "note": "With 2 test proteins only, 10,000 bootstrap draws are "
                  "meaningless: the 95% CI is the min-max of the 3 exact "
                  "combination values (pooled, 1mix_A-only, 2qt4_A-only). "
                  "AUPRC inside the bootstrap uses a numpy reimplementation "
                  "of sklearn's average_precision_score (same threshold-grouping "
                  "treatment of ties); it was verified to agree with sklearn "
                  "to 1e-12 on all fitted conditions before use. Models were "
                  "retrained per condition and horizon; pooled AUPRCs were "
                  "verified against G137 records to 4 decimal places."},
    "horizons": {str(h): results[h] for h in HORIZONS.values()},
    "bootstrap_results_recovery": {str(h): boot[h] for h in HORIZONS.values()},
    "bootstrap_results_auprc_diff": {str(h): boot_auprc[h] for h in HORIZONS.values()},
    "recovery_cis_contain_0.5": contains_05,
    "auprc_diff_cis_contain_zero": contains_zero_auprc,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_mean_1.0ns",
                "full_gain", "prefix_gain", "recovery",
                "boot_ci_lower", "boot_ci_upper", "ci_contains_0.5",
                "boot_auprc_diff_ci_lower", "boot_auprc_diff_ci_upper",
                "auprc_diff_ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        b = boot[h]
        ba = boot_auprc[h]
        w.writerow([h, r["test_positives"],
                    r["auprc"]["static_only"], r["auprc"]["full_hist"],
                    r["auprc"]["prefix_mean_1.0ns"], r["full_gain"],
                    r["prefix_gain"], r["recovery"],
                    b["ci_lower"], b["ci_upper"], b["contains_0.5"],
                    ba["ci_lower"], ba["ci_upper"], ba["contains_zero"]])

print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
