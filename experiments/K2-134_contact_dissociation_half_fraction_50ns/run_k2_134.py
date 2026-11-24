"""K2-134: 接触の解離で、履歴の前半と後半のどちらに指導力があるか（割合だけ）.

K2-14 の問い（割合の前半と後半）を、接触の解離の 50 ns先で測る。
割合は、ペアの最小重原子間距離が接触の閾値（4.5 Å）以下の時間の
割合で、6 個の統計量のうちの「4.5 Å 以下の割合」と同じ定義である。
接触の解離の割合の前半と後半は、1・5・10・30・90 ns先（K2-129・K2-132・
K2-133・K2-131・K2-130）で測り終えた。50 ns先ではどうなるかを見る。
平均だけの前半と後半は 1 ns先で測り終えた（K2-12：前半 0.865、
後半 0.818 で、読み方では「続いている状態が効いている」になった）。

条件は資料Dの条件（6 タンパク質×3 レプリカ、起点 1 ns、履歴 0〜1 ns、
HistGBM を条件ごとに学習し直し、学習 4／テスト 1mix_A＋2qt4_A）で、
接触の解離、50 ns先、割合だけ。

入力は、静的な特徴量に、つぎのどれか 1 つを足した形で比べる:
- 前半（フレーム [0..4]）の割合
- 後半（フレーム [6..10]）の割合
- 前半と後半の両方（2 つの割合を並べたもの）
- 全部（フレーム [0..10]）の割合
前半と後半のあいだのフレーム 5 は、どちらの半分にも入れない。

土台の再現: 静的のみと全部入り（6 個の統計量）の AUPRC が、K50 の
記録（静的のみ 0.2927、全部 0.4533、テスト 2,915件・陽性 267件）と
一致することを確かめる。K50 には割合だけの記録がないので、全部の
割合は照合ではなく記録として扱う。タンパク質ごとの記録も K2-12 と
同じく入れる。

読み方（先に固定）:
- 後半の利得が全部入りの 8 割以上で、前半の利得が全部入りの 5 割以下
  なら「今の状態が効いている」。
- 前半も後半も全部入りの 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する。
- 分母を全部の割合の利得にしても判定が変わる場合は、両方を記録する。
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

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
HIDX = 4  # 50 ns先
FRAMES_FIRST = [0, 1, 2, 3, 4]       # 前半（0〜0.4 ns）
FRAMES_SECOND = [6, 7, 8, 9, 10]     # 後半（0.6〜1.0 ns）
FRAMES_FULL = list(range(11))        # 全部（0〜1.0 ns）

# K50 の記録との照合用の期待値（50 ns先、合算）
EXPECTED = {"static_only": 0.2927, "full_hist6": 0.4533,
            "test_n": 2915, "test_positives": 267}


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
            z = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
            d = {k: z[k] for k in z.files}
            # hist は (N, 11) で保存されているので転置しない
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


def frac_of(H, fr):
    return (H[:, fr] <= BOUND).mean(1, keepdims=True)

CONDS = ["static_only", "frac_first_half", "frac_second_half",
         "frac_both_halves", "frac_full", "full_hist6"]


def feats_for(cname):
    if cname == "static_only":
        return Xtr, Xte
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr[:, FRAMES_FULL])]),
                np.column_stack([Xte, hist6(Hte[:, FRAMES_FULL])]))
    if cname == "frac_first_half":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FIRST)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FIRST)]))
    if cname == "frac_second_half":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_SECOND)]))
    if cname == "frac_both_halves":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FIRST),
                                 frac_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FIRST),
                                 frac_of(Hte, FRAMES_SECOND)]))
    if cname == "frac_full":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FULL)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FULL)]))
    raise ValueError(cname)


ytr, yte = Ytr[:, HIDX], Yte[:, HIDX]
res = {}
per_protein = {}
preds = {}
for cname in CONDS:
    xtr, xte = feats_for(cname)
    p = predict(xtr, ytr, xte)
    preds[cname] = p
    res[cname] = metrics(yte, p)
    per_protein[cname] = {}
    for prot in TEST_P:
        m = Pte == prot
        per_protein[cname][prot] = metrics(yte[m], p[m])
    print(cname, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in res[cname].items()}, flush=True)

# K50 の記録との照合
assert res["static_only"]["n"] == EXPECTED["test_n"]
assert res["static_only"]["positives"] == EXPECTED["test_positives"]
for c in ("static_only", "full_hist6"):
    assert abs(res[c]["auprc"] - EXPECTED[c]) < 5e-4, (c, res[c]["auprc"])
print("base reproduction vs K50: OK", flush=True)

gain_full = res["full_hist6"]["auprc"] - res["static_only"]["auprc"]
gain_frac_full = res["frac_full"]["auprc"] - res["static_only"]["auprc"]
gains = {c: res[c]["auprc"] - res["static_only"]["auprc"] for c in CONDS}
ratios = {c: gains[c] / gain_full if gain_full else float("nan") for c in CONDS}
ratios_vs_frac_full = {c: gains[c] / gain_frac_full if gain_frac_full
                       else float("nan") for c in CONDS}
ratio_first = ratios["frac_first_half"]
ratio_second = ratios["frac_second_half"]


def reading_for(r_first, r_second):
    if r_second >= 0.8 and r_first <= 0.5:
        return "今の状態が効いている"
    if r_first >= 0.8 and r_second >= 0.8:
        return "続いている状態が効いている"
    return "どちらでもない（数値をそのまま記録）"


reading = reading_for(ratio_first, ratio_second)
reading_vs_frac = reading_for(ratios_vs_frac_full["frac_first_half"],
                              ratios_vs_frac_full["frac_second_half"])
print(f"gain_full {gain_full:+.4f} gain_first {gains['frac_first_half']:+.4f} "
      f"(ratio {ratio_first:.3f}) gain_second {gains['frac_second_half']:+.4f} "
      f"(ratio {ratio_second:.3f}) gain_both {gains['frac_both_halves']:+.4f} "
      f"(ratio {ratios['frac_both_halves']:.3f}) "
      f"gain_frac_full {gain_frac_full:+.4f} "
      f"(ratio {ratios['frac_full']:.3f}) reading: {reading}", flush=True)
print(f"reading with denominator = full-fraction gain: {reading_vs_frac}",
      flush=True)

# タンパク質ごとの形の数え上げ（K2-07 と同じ基準。ただしテストは 2タンパク質だけ）
per_protein_readings = {}
for prot in TEST_P:
    g_full = (per_protein["full_hist6"][prot]["auprc"]
              - per_protein["static_only"][prot]["auprc"])
    if g_full <= 0.01:
        per_protein_readings[prot] = "利得が小さく判定から外したもの"
        continue
    r1 = ((per_protein["frac_first_half"][prot]["auprc"]
           - per_protein["static_only"][prot]["auprc"]) / g_full)
    r2 = ((per_protein["frac_second_half"][prot]["auprc"]
           - per_protein["static_only"][prot]["auprc"]) / g_full)
    per_protein_readings[prot] = reading_for(r1, r2)
counts = {"今の状態": 0, "続いている状態": 0, "どちらでもない": 0,
          "利得が小さく判定から外したもの": 0}
for v in per_protein_readings.values():
    key = ("今の状態" if v == "今の状態が効いている"
           else "続いている状態" if v == "続いている状態が効いている"
           else "どちらでもない" if v.startswith("どちらでもない")
           else "利得が小さく判定から外したもの")
    counts[key] += 1
print("per-protein readings:", per_protein_readings, "counts:", counts,
      flush=True)

out = {
    "label": "Contact dissociation, first-half vs second-half history fraction (K2-14 question on contact dissociation, fraction only)",
    "phenomenon": "contact dissociation (Doc D, k50_cache)",
    "horizon_ns": 50,
    "frames": {"first_half": FRAMES_FIRST, "second_half": FRAMES_SECOND,
               "full": FRAMES_FULL,
               "note": "frame 5 belongs to neither half"},
    "bound": BOUND,
    "metrics": res,
    "per_protein_metrics": per_protein,
    "per_protein_readings": per_protein_readings,
    "per_protein_reading_counts": counts,
    "gains": gains,
    "gain_ratios_vs_full_hist6": ratios,
    "gain_ratios_vs_frac_full": ratios_vs_frac_full,
    "reading_rule": ("ratios are vs the full hist6 gain (K2-01 convention); "
                     "second >= 0.8 and first <= 0.5 -> current state; "
                     "both >= 0.8 -> persistent state; else record numbers"),
    "reading": reading,
    "reading_vs_frac_full_gain": reading_vs_frac,
    "checks_vs_K50": {"static_only_auprc_expected": EXPECTED["static_only"],
                      "full_hist6_auprc_expected": EXPECTED["full_hist6"]},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auc", "auprc", "brier",
                "logloss", "gain_vs_static", "gain_ratio_vs_full_hist6",
                "gain_ratio_vs_frac_full"])
    for c in CONDS:
        m = res[c]
        w.writerow([c, m["n"], m["positives"], m["auc"], m["auprc"],
                    m["brier"], m["logloss"], gains[c], ratios[c],
                    ratios_vs_frac_full[c]])
np.savez_compressed(OUT / "test_predictions.npz", y_test=yte,
                    protein_test=Pte, **{f"pred_{c}": preds[c] for c in CONDS})
print("saved", OUT / "results.json", OUT / "results.csv",
      OUT / "test_predictions.npz", flush=True)
