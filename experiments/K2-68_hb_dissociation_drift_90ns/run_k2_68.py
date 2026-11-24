"""K2-68: K2-15 の問い（漂いが足しになるか）を水素結合の解離の 90 ns先で測る.

K2-15 では、水素結合の解離の 1 ns先で、履歴の漂い（前半の平均 −
後半の平均）を全部の平均の上に足すと足しになった（(c) 0.5923 は (b)
0.5691 を +0.0232 上回る）。30 ns先（K2-25）でも足しになった
（(c) 0.5736 は (b) 0.5582 を +0.0154 上回る）。ここでは同じ問いを
90 ns先で測る。長い予測幅では平均の担い手が変わるため（K46 の 90 ns先の
全部の平均だけの回復率は 1.304 で、1 ns先の 0.858・30 ns先の 0.887 とは
違う）、漂いの役割も変わるかもしれない。なお、水素結合の形成の 30 ns先では
漂いは足しにならなかった（K2-16、(c) − (b) = −0.0014）。

条件は資料Dの条件（6 タンパク質×3 レプリカ、起点 1 ns、履歴 0〜1 ns、
HistGBM を条件ごとに学習し直し、学習 4／テスト 1mix_A＋2qt4_A）で、
水素結合の解離、90 ns先。

入力は、静的な特徴量に、つぎの形で比べる:
- (a) 漂いだけ: 静的特徴＋漂い
- (b) 全部の平均だけ: 静的特徴＋全部（フレーム [0..10]）の距離の平均
- (c) 全部の平均＋漂い: 静的特徴＋全部の平均＋漂い

漂いの定義（K2-06 と同じ）:
- 漂い = 前半の平均 − 後半の平均。前半はフレーム [0..4]（0〜0.4 ns）
  の距離の平均、後半はフレーム [6..10]（0.6〜1.0 ns）の距離の平均。
- 符号: 漂いが正なら「前半のほうが距離が大きい」（履歴のなかで距離が
  縮む方向に動いた）、負なら「後半のほうが距離が大きい」（距離が開く
  方向に動いた）という意味。
- フレーム 5（0.5 ns）は前半にも後半にも入らない（K2-01 と同じ分け方）。

土台の再現: 静的のみと全部の 6 個の統計量の AUPRC が、K46 / G70 の
90 ns先の記録（静的のみ 0.5118、全部 0.5797、利得 +0.0679、テスト
2,080件・陽性 413件）と一致することを確かめる。全部の平均は K46 の
記録（0.6003）と照らす。

読み方（K2-06 と同じく先に固定）:
- (c) 全部の平均＋漂いが (b) 全部の平均だけを AUPRC で +0.01 以上
  上回れば「漂いは足しになる」、未満なら「足しにならない」。
- (a) 漂いだけの利得も記録し、全部入りの利得に対する割合もあわせて
  記録する。
- K2-51 の 90 ns先の「前半と後半の両方」（0.6044、割合 1.364）と (c) の
  結果の関係も記録する。(c) は (b) の上に漂いを足した形で、K2-51 の
  「前半と後半の両方」は前半と後半の平均を 2 つ入れた形。2 つの条件の
  AUPRC の差も記録する。

割合の分母は全部入りの利得（静的＋履歴 6 統計量の利得）を主とし、
あわせて全部の平均の利得に対する割合も記録する（分母の確認のため）。

実行はローカルの仮想マシンで行う（6 条件の学習で数十秒。K2 系列では
重い計算は Colab に逃がす決め事があるが、この規模は重い計算には
あたらないと判断した）。sklearn の学習はスレッド数を 1 に絞る。
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
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 3.5
HIDX = 5  # 90 ns先（予測幅は 1/5/10/30/50/90 ns で 0〜5）
FRAMES_FIRST = [0, 1, 2, 3, 4]       # 前半（0〜0.4 ns）
FRAMES_SECOND = [6, 7, 8, 9, 10]     # 後半（0.6〜1.0 ns）
FRAMES_FULL = list(range(11))        # 全部（0〜1.0 ns）
LINE = 0.01  # 先に固定した線（AUPRC の差）

# K46 / G70 の記録との照合用の期待値（90 ns先、合算）
EXPECTED = {"static_only": 0.5118, "full_hist6": 0.5797,
            "mean_full": 0.6003, "test_n": 2080, "test_positives": 413}


def hist6(series, bound=BOUND):
    """series は (N, F) の距離系列。6 個の統計量。"""
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


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            # hist は (11, N) で保存されているので転置する
            d["hist"] = d["hist"].T
            d["protein"] = np.full(len(d["y"]), p)
            parts.append(d)
    return parts


train = load(TRAIN_P)
test = load(TEST_P)
Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
Htr = np.concatenate([d["hist"] for d in train]).astype(np.float64)
Hte = np.concatenate([d["hist"] for d in test]).astype(np.float64)
Ytr = np.concatenate([d["y"] for d in train])
Yte = np.concatenate([d["y"] for d in test])
Pte = np.concatenate([d["protein"] for d in test])
print("train", Xtr.shape, "test", Xte.shape, flush=True)


def mean_of(H, fr):
    return H[:, fr].mean(1, keepdims=True)


def drift_of(H):
    """漂い = 前半の平均 − 後半の平均。(N, 1)。"""
    return mean_of(H, FRAMES_FIRST) - mean_of(H, FRAMES_SECOND)


CONDS = ["static_only", "full_hist6", "drift_only", "mean_full",
         "mean_full_plus_drift", "mean_both_halves",
         "mean_first_half", "mean_second_half"]  # 最後の2つは前半・後半の照合用（K2-51 の 90 ns先の記録：前半 0.6012、後半 0.5835）


def feats_for(cname):
    if cname == "static_only":
        return Xtr, Xte
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr[:, FRAMES_FULL])]),
                np.column_stack([Xte, hist6(Hte[:, FRAMES_FULL])]))
    if cname == "drift_only":
        return (np.column_stack([Xtr, drift_of(Htr)]),
                np.column_stack([Xte, drift_of(Hte)]))
    if cname == "mean_full":
        return (np.column_stack([Xtr, mean_of(Htr, FRAMES_FULL)]),
                np.column_stack([Xte, mean_of(Hte, FRAMES_FULL)]))
    if cname == "mean_full_plus_drift":
        return (np.column_stack([Xtr, mean_of(Htr, FRAMES_FULL), drift_of(Htr)]),
                np.column_stack([Xte, mean_of(Hte, FRAMES_FULL), drift_of(Hte)]))
    if cname == "mean_both_halves":
        return (np.column_stack([Xtr, mean_of(Htr, FRAMES_FIRST),
                                 mean_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, mean_of(Hte, FRAMES_FIRST),
                                 mean_of(Hte, FRAMES_SECOND)]))
    if cname == "mean_first_half":
        return (np.column_stack([Xtr, mean_of(Htr, FRAMES_FIRST)]),
                np.column_stack([Xte, mean_of(Hte, FRAMES_FIRST)]))
    if cname == "mean_second_half":
        return (np.column_stack([Xtr, mean_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, mean_of(Hte, FRAMES_SECOND)]))
    raise ValueError(cname)


ytr, yte = Ytr[:, HIDX], Yte[:, HIDX]
res = {}
per_protein = {}
for cname in CONDS:
    xtr, xte = feats_for(cname)
    p = predict(xtr, ytr, xte)
    res[cname] = metrics(yte, p)
    per_protein[cname] = {}
    for prot in TEST_P:
        m = Pte == prot
        per_protein[cname][prot] = metrics(yte[m], p[m])
    print(cname, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in res[cname].items()}, flush=True)

# K46 / G70 の記録との照合
assert res["static_only"]["n"] == EXPECTED["test_n"]
assert res["static_only"]["positives"] == EXPECTED["test_positives"]
for c in ("static_only", "full_hist6", "mean_full"):
    assert abs(res[c]["auprc"] - EXPECTED[c]) < 5e-4, (c, res[c]["auprc"])
print("base reproduction vs K46/G70: OK", flush=True)

gain_full = res["full_hist6"]["auprc"] - res["static_only"]["auprc"]
gain_mean_full = res["mean_full"]["auprc"] - res["static_only"]["auprc"]
gains = {c: res[c]["auprc"] - res["static_only"]["auprc"] for c in CONDS}
ratios = {c: gains[c] / gain_full if gain_full else float("nan") for c in CONDS}
ratios_vs_mean_full = {c: gains[c] / gain_mean_full for c in CONDS}

# 先に固定した読み方：(c) が (b) を AUPRC で +0.01 以上上回れば足しになる
diff_c_b = res["mean_full_plus_drift"]["auprc"] - res["mean_full"]["auprc"]
reading = "漂いは足しになる" if diff_c_b >= LINE else "漂いは足しにならない"
diff_c_both = (res["mean_full_plus_drift"]["auprc"]
               - res["mean_both_halves"]["auprc"])
print(f"gain_full {gain_full:+.4f} gain_drift {gains['drift_only']:+.4f} "
      f"(ratio {ratios['drift_only']:.3f}) gain_mean_full {gain_mean_full:+.4f} "
      f"(ratio {ratios['mean_full']:.3f}) gain_c {gains['mean_full_plus_drift']:+.4f} "
      f"(ratio {ratios['mean_full_plus_drift']:.3f}) "
      f"diff (c)-(b) {diff_c_b:+.4f} reading: {reading} "
      f"diff (c)-(both halves) {diff_c_both:+.4f}", flush=True)

out = {
    "label": "HB dissociation, drift (first-half mean minus second-half mean) on top of the full mean (K2-15 question on HB dissociation at 90 ns)",
    "phenomenon": "HB dissociation (Doc D, data_cache_k46)",
    "horizon_ns": 90,
    "frames": {"first_half": FRAMES_FIRST, "second_half": FRAMES_SECOND,
               "full": FRAMES_FULL,
               "note": "frame 5 belongs to neither half"},
    "drift_definition": "first-half mean minus second-half mean; positive = distance shrank over the history",
    "bound": BOUND,
    "line_auprc_diff": LINE,
    "metrics": res,
    "per_protein_metrics": per_protein,
    "gains": gains,
    "gain_ratios_vs_full_hist6": ratios,
    "gain_ratios_vs_mean_full": ratios_vs_mean_full,
    "diff_c_minus_b_auprc": float(diff_c_b),
    "diff_c_minus_both_halves_auprc": float(diff_c_both),
    "reading_rule": ("(c) mean_full_plus_drift beats (b) mean_full by AUPRC >= +0.01 "
                     "-> drift adds; else it does not"),
    "reading": reading,
    "checks_vs_K46_G70": {
        "static_only_auprc_expected": EXPECTED["static_only"],
        "full_hist6_auprc_expected": EXPECTED["full_hist6"],
        "mean_full_auprc_expected": EXPECTED["mean_full"]},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auc", "auprc", "brier",
                "logloss", "gain_vs_static", "gain_ratio_vs_full_hist6",
                "gain_ratio_vs_mean_full"])
    for c in CONDS:
        m = res[c]
        w.writerow([c, m["n"], m["positives"], m["auc"], m["auprc"],
                    m["brier"], m["logloss"], gains[c], ratios[c],
                    ratios_vs_mean_full[c]])
print("saved", OUT / "results.json", OUT / "results.csv", flush=True)
