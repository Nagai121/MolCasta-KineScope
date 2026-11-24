"""G178: 接触の形成で、切り取り 1.0 ns の平均だけの回復率に区間を付ける
（資料Dの条件、接触の形成、1・5・10・30・50・90 ns先）.

G138 では、切り取り 1.0 ns（フレーム [0..9] の 10 フレーム）の平均だけの
回復率が、1 ns先 0.915、5 ns先 0.784、10 ns先 0.641、30 ns先 1.079、
50 ns先 0.748、90 ns先 0.738 でした。ここでは 6 予測幅すべてについて、
その回復率の区間を出します。

区間の付け方（G176 と同じ）:
- 区間は 3 つの値（合算、1mix_A、2qt4_A）の最小値と最大値です。
- テストが 2 タンパク質だけなので、取りうる組み合わせは 3通りだけで、
  1 万回のブートストラップは意味がありません。G165〜G172 の
  ブートストラップ区間も、実質的には同じ 3 値の最小値〜最大値に
  等しくなっています（G23・G49・G114・G130 と同じ留保）。
- 「区間が 0.5 を含む／含まない」は実質的に「合算と 2 タンパク質の
  回復率が 0.5 の同じ側にあるか／またぐか」と同じ読みです。
- 回復率の定義: (切り取りの平均だけの AUPRC - 静的のみの AUPRC) /
  (全部入りの AUPRC - 静的のみの AUPRC)。タンパク質ごとの回復率も、
  そのタンパク質の中で利得（全部入り - 静的のみ）を測り直してから
  計算します（G147・G148・G162・G165 と同じ）。
- 土台の利得が小さい予測幅（30 ns先 +0.1518）があるので、AUPRC の差
  （切り取りの平均だけ - 静的のみ）の区間も同じ段取りで出します。

段取り:
- G138 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存します。
- G149 で起点のフレームを入れるかどうかで回復率が 1 ns先 −0.206、
  5 ns先 −0.091、10 ns先 +0.005、30 ns先 −0.088、50 ns先 −0.093、
  90 ns先 −0.092 と、10 ns先を除く 5 予測幅で下がったため、区間と
  あわせて記録する。

フレームの注記: 「1.0 ns」の切り取りは、呼び名の 1.0 ns に対して
実際は 0.9 ns の範囲である（フレーム [0,1,2,3,4,5,6,7,8,9] は時刻
0〜0.9 ns で、起点のフレーム10 は入らない）。0.25 ns の切り取りが
0.2 ns なのと同じ数え方である（G138 の README にも同じ注記がある）。

読み方（先に固定）:
- 各予測幅で、区間が 0.5 を含むかどうかを見てください。
- AUPRC の差の区間が 0 を含むかどうかも、予測幅ごとに見てください。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り（G138 と同じ）
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns

# G138 の記録との照合用の期待値（合算 AUPRC、G138 の results.json の全精度値）
EXPECTED = {
    1: {"static_only": 0.33977026863870574, "full_hist": 0.5668799691676633,
        "prefix_mean_1.0ns": 0.5475669980864415,
        "full_gain": 0.22710970052895751, "recovery": 0.9149619279306862},
    5: {"static_only": 0.2869855867812501, "full_hist": 0.5191989452346799,
        "prefix_mean_1.0ns": 0.46907135385636795,
        "full_gain": 0.23221335845342983, "recovery": 0.7841313190930616},
    10: {"static_only": 0.320384938464356, "full_hist": 0.5233798114377418,
         "prefix_mean_1.0ns": 0.45056934427409895,
         "full_gain": 0.2029948729733858, "recovery": 0.6413186890035943},
    30: {"static_only": 0.28681357061495694, "full_hist": 0.4386046069734208,
         "prefix_mean_1.0ns": 0.4505265968272691,
         "full_gain": 0.15179103635846386, "recovery": 1.0785421204035643},
    50: {"static_only": 0.26038453431471853, "full_hist": 0.46379209531119275,
         "prefix_mean_1.0ns": 0.412544662471832,
         "full_gain": 0.20340756099647422, "recovery": 0.7480554184500102},
    90: {"static_only": 0.2872126453371935, "full_hist": 0.47216777524969433,
         "prefix_mean_1.0ns": 0.423699232818619,
         "full_gain": 0.18495512991250085, "recovery": 0.7379443195006},
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


def load(prots):
    """接触の形成は K13 キャッシュ（G138 と同じ）。hist は (n, 11) で
    保存されているので転置しない。ファイル名は {p}_R{r}_contact.npz。"""
    real = []
    proteins = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact.npz")
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

prefix_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_te = Hte[:, PREFIX_FRAMES].mean(1)
print("prefix mean (frames 0..9) test: mean",
      round(float(prefix_te.mean()), 4), flush=True)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

CONDS = ["static_only", "full_hist", "prefix_mean_1.0ns"]


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    return (np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32)]),
            np.column_stack([Xte, prefix_te[:, None].astype(np.float32)]))


def interval_of_3(pooled, per_protein):
    """3 つの値（合算、1mix_A、2qt4_A）の最小値と最大値を区間とする。
    G165〜G172 のブートストラップ区間も、2 タンパク質だけの土台では
    実質的にこの 3 値の最小値〜最大値に等しい。"""
    vals = [pooled] + [per_protein[p] for p in TEST_P]
    return float(min(vals)), float(max(vals))


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
    gain = a["full_hist"] - a["static_only"]
    rec = (a["prefix_mean_1.0ns"] - a["static_only"]) / gain if gain else float("nan")
    adiff = a["prefix_mean_1.0ns"] - a["static_only"]
    per_protein = {}
    rec_by_prot = {}
    adiff_by_prot = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        n_pos = int(yte[idx].sum())
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist"] - ap["static_only"]
        r = (ap["prefix_mean_1.0ns"] - ap["static_only"]) / g if g else float("nan")
        d = ap["prefix_mean_1.0ns"] - ap["static_only"]
        rec_by_prot[prot] = r
        adiff_by_prot[prot] = d
        per_protein[prot] = {
            "n": int(len(idx)), "positives": n_pos,
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "recovery": r,
            "auprc_diff": d,
        }
    rec_lo, rec_hi = interval_of_3(rec, rec_by_prot)
    ad_lo, ad_hi = interval_of_3(adiff, adiff_by_prot)
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain,
        "prefix_mean_1.0ns_recovery": rec,
        "auprc_diff": adiff,
        "per_protein": per_protein,
        "recovery_interval": {
            "method": "min-max of 3 values (pooled, 1mix_A, 2qt4_A); "
                      "no 10,000-draw bootstrap (meaningless with 2 test proteins)",
            "n_values": 3,
            "values": {"pooled": rec,
                       "1mix_A": rec_by_prot["1mix_A"],
                       "2qt4_A": rec_by_prot["2qt4_A"]},
            "ci_lower": rec_lo, "ci_upper": rec_hi,
            "contains_0.5": bool(rec_lo <= 0.5 <= rec_hi),
        },
        "auprc_diff_interval": {
            "method": "min-max of 3 values (pooled, 1mix_A, 2qt4_A)",
            "n_values": 3,
            "values": {"pooled": adiff,
                       "1mix_A": adiff_by_prot["1mix_A"],
                       "2qt4_A": adiff_by_prot["2qt4_A"]},
            "ci_lower": ad_lo, "ci_upper": ad_hi,
            "contains_zero": bool(ad_lo <= 0 <= ad_hi),
        },
    }
    # G138 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist", "prefix_mean_1.0ns"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(gain - exp["full_gain"]) < 5e-4, (hns, gain)
    assert abs(rec - exp["recovery"]) < 5e-3, (hns, rec)
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"prefix1.0 {a['prefix_mean_1.0ns']:.4f} gain {gain:.4f} "
          f"recovery {rec:.3f} (record {exp['recovery']:.3f}) "
          f"CI [{rec_lo:.3f}, {rec_hi:.3f}] contains_0.5="
          f"{results[hns]['recovery_interval']['contains_0.5']} "
          f"auprc_diff {adiff:+.4f} CI [{ad_lo:+.4f}, {ad_hi:+.4f}] "
          f"contains_zero={results[hns]['auprc_diff_interval']['contains_zero']}",
          flush=True)

contains_05 = {f"{h}ns": results[h]["recovery_interval"]["contains_0.5"]
               for h in HORIZONS.values()}
contains_zero = {f"{h}ns": results[h]["auprc_diff_interval"]["contains_zero"]
                 for h in HORIZONS.values()}
print("recovery CIs contain 0.5:", contains_05, flush=True)
print("auprc-diff CIs contain zero:", contains_zero, flush=True)

out = {
    "label": "Contact formation, 1.0ns-prefix mean-only recovery uncertainty across horizons",
    "phenomenon": "Contact formation (D conditions, K13 cache)",
    "conditions": {
        "prefix_mean_1.0ns": {"label": "フレーム [0..9] の 10 フレームの平均だけ",
                              "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                              "covered_length": "0.9 ns"},
    },
    "interval": {"method": "min-max of 3 values (pooled, 1mix_A, 2qt4_A)",
                 "note": "テストが 2 タンパク質だけなので、取りうる組み合わせは "
                         "3通りだけで、1 万回のブートストラップは意味がない。G165〜G172 "
                         "のブートストラップ区間も、2 タンパク質の土台では実質的に "
                         "同じ 3 値の最小値〜最大値に等しい。区間が 0.5 を含むかどうかは、"
                         "合算と 2 タンパク質の回復率が 0.5 の同じ側にあるか／またぐかと "
                         "同じ読みになる（G23・G49・G114・G130 と同じ留保）。タンパク質 "
                         "ごとの回復率は、そのタンパク質の中で利得を測り直してから計算した "
                         "（G147・G148・G162・G165 と同じ）。",
                 "n_values": 3},
    "horizons": {str(h): results[h] for h in HORIZONS.values()},
    "recovery_cis_contain_0.5": contains_05,
    "auprc_diff_cis_contain_zero": contains_zero,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_mean_1.0ns",
                "full_gain", "prefix_mean_1.0ns_recovery",
                "recovery_ci_lower", "recovery_ci_upper",
                "recovery_ci_contains_0.5",
                "auprc_diff", "auprc_diff_ci_lower", "auprc_diff_ci_upper",
                "auprc_diff_ci_contains_zero",
                "recovery_1mix_A", "recovery_2qt4_A"])
    for h in HORIZONS.values():
        r = results[h]
        ri = r["recovery_interval"]
        ai = r["auprc_diff_interval"]
        w.writerow([h, r["test_positives"],
                    r["auprc"]["static_only"], r["auprc"]["full_hist"],
                    r["auprc"]["prefix_mean_1.0ns"],
                    r["full_gain"], r["prefix_mean_1.0ns_recovery"],
                    ri["ci_lower"], ri["ci_upper"], ri["contains_0.5"],
                    r["auprc_diff"], ai["ci_lower"], ai["ci_upper"],
                    ai["contains_zero"],
                    r["per_protein"]["1mix_A"]["recovery"],
                    r["per_protein"]["2qt4_A"]["recovery"]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
