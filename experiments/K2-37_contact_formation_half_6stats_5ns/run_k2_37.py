"""K2-37: 接触の形成で、履歴の前半と後半のどちらに指導力があるか（5 ns先）.

K2-04 の問い（接触の形成の前半と後半、6 個の統計量）を、
5 ns先で測る。1 ns先（K2-04）では後半 0.980・前半 0.792、
30 ns先（K2-19）では前半 1.204・後半 1.026、90 ns先（K2-28）では
前半 0.826・後半 0.649 だった。5 ns先で前半と後半の形がどうなるかを見る。

条件は資料Dの条件（6 タンパク質×3 レプリカ、起点 1 ns、履歴 0〜1 ns、
HistGBM を条件ごとに学習し直し、学習 4／テスト 1mix_A＋2qt4_A）で、
接触の形成、5 ns先、6 個の統計量。

入力は、静的な特徴量に、つぎのどれか 1 つを足した形で比べる:
- 前半（フレーム [0..4]）の 6 個の統計量
- 後半（フレーム [6..10]）の 6 個の統計量
- 前半と後半の両方（6 個の統計量を 2 つ並べたもの）
- 全部（フレーム [0..10]）の 6 個の統計量
6 個の統計量は、与えられたフレームの範囲について
[最初, 平均, 標準偏差, 最小, 最後−最初, 4.5 Å 以下の割合] を計算する
（資料Dの D6 と同じ定義を、その範囲に当てたもの）。前半と後半の
あいだのフレーム 5 は、どちらの半分にも入れない。

土台の再現: 静的のみと全部入り（6 個の統計量）の AUPRC が、接触の形成
の 5 ns先の記録（静的のみ 0.2870、全部 0.5192、テスト 7,384件・
陽性 214件）と一致することを確かめる。

読み方（K2-01 と同じく先に固定）。「全部入り」は全部の 6 個の統計量の
利得で、割合はその利得で割る。
- 後半の利得が全部入りの 8 割以上で、前半の利得が全部入りの 5 割以下
  なら「今の状態が効いている」。
- 前半も後半も全部入りの 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する。
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

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
HIDX = 1  # 5 ns先
FRAMES_FIRST = [0, 1, 2, 3, 4]       # 前半（0〜0.4 ns）
FRAMES_SECOND = [6, 7, 8, 9, 10]     # 後半（0.6〜1.0 ns）
FRAMES_FULL = list(range(11))        # 全部（0〜1.0 ns）

# 接触の形成の 5 ns先の記録との照合用の期待値（合算）
EXPECTED = {"static_only": 0.2870, "full_hist6": 0.5192,
            "test_n": 7384, "test_positives": 214}


def hist6(series, bound=BOUND):
    """series は (N, F) の距離系列。与えられたフレーム範囲の 6 個の統計量。"""
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
            z = np.load(CACHE / f"{p}_R{r}_contact.npz")
            d = {k: z[k] for k in z.files}
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

CONDS = ["static_only", "hist6_first_half", "hist6_second_half",
         "hist6_both_halves", "full_hist6"]


def feats_for(cname):
    if cname == "static_only":
        return Xtr, Xte
    if cname == "hist6_first_half":
        return (np.column_stack([Xtr, hist6(Htr[:, FRAMES_FIRST])]),
                np.column_stack([Xte, hist6(Hte[:, FRAMES_FIRST])]))
    if cname == "hist6_second_half":
        return (np.column_stack([Xtr, hist6(Htr[:, FRAMES_SECOND])]),
                np.column_stack([Xte, hist6(Hte[:, FRAMES_SECOND])]))
    if cname == "hist6_both_halves":
        return (np.column_stack([Xtr, hist6(Htr[:, FRAMES_FIRST]),
                                 hist6(Htr[:, FRAMES_SECOND])]),
                np.column_stack([Xte, hist6(Hte[:, FRAMES_FIRST]),
                                 hist6(Hte[:, FRAMES_SECOND])]))
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr[:, FRAMES_FULL])]),
                np.column_stack([Xte, hist6(Hte[:, FRAMES_FULL])]))
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

# 接触の形成の 5 ns先の記録との照合
assert res["static_only"]["n"] == EXPECTED["test_n"]
assert res["static_only"]["positives"] == EXPECTED["test_positives"]
for c in ("static_only", "full_hist6"):
    assert abs(res[c]["auprc"] - EXPECTED[c]) < 5e-4, (c, res[c]["auprc"])
print("base reproduction vs K27 (5 ns): OK", flush=True)

gain_full = res["full_hist6"]["auprc"] - res["static_only"]["auprc"]
gains = {c: res[c]["auprc"] - res["static_only"]["auprc"] for c in CONDS}
ratios = {c: gains[c] / gain_full if gain_full else float("nan") for c in CONDS}
ratio_first = ratios["hist6_first_half"]
ratio_second = ratios["hist6_second_half"]

# 先に固定した読み方（分母は全部の 6 個の統計量の利得、K2-01 と同じ）
if ratio_second >= 0.8 and ratio_first <= 0.5:
    reading = "今の状態が効いている"
elif ratio_first >= 0.8 and ratio_second >= 0.8:
    reading = "続いている状態が効いている"
else:
    reading = "どちらでもない（数値をそのまま記録）"
print(f"gain_full {gain_full:+.4f} gain_first {gains['hist6_first_half']:+.4f} "
      f"(ratio {ratio_first:.3f}) gain_second {gains['hist6_second_half']:+.4f} "
      f"(ratio {ratio_second:.3f}) gain_both {gains['hist6_both_halves']:+.4f} "
      f"(ratio {ratios['hist6_both_halves']:.3f}) reading: {reading}", flush=True)

out = {
    "label": "Contact formation, first-half vs second-half history statistics at 5 ns (K2-04 question)",
    "phenomenon": "contact formation (Doc D, k13_cache)",
    "horizon_ns": 5,
    "frames": {"first_half": FRAMES_FIRST, "second_half": FRAMES_SECOND,
               "full": FRAMES_FULL,
               "note": "frame 5 belongs to neither half"},
    "bound": BOUND,
    "metrics": res,
    "per_protein_metrics": per_protein,
    "gains": gains,
    "gain_ratios_vs_full_hist6": ratios,
    "reading_rule": ("ratios are vs the full hist6 gain (K2-01 convention); "
                     "second >= 0.8 and first <= 0.5 -> current state; "
                     "both >= 0.8 -> persistent state; else record numbers"),
    "reading": reading,
    "checks_vs_K27_5ns": {
        "static_only_auprc_expected": EXPECTED["static_only"],
        "full_hist6_auprc_expected": EXPECTED["full_hist6"]},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auc", "auprc", "brier",
                "logloss", "gain_vs_static", "gain_ratio_vs_full_hist6"])
    for c in CONDS:
        m = res[c]
        w.writerow([c, m["n"], m["positives"], m["auc"], m["auprc"],
                    m["brier"], m["logloss"], gains[c], ratios[c]])
print("saved", OUT / "results.json", OUT / "results.csv", flush=True)
