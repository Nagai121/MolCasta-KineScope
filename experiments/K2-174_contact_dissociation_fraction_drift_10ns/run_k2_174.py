"""K2-174: 接触の解離で、割合の「漂い」が足しになるか（10 ns先、割合だけ）.

K2-15 の問い（漂い）を割合で測る。漂いは、前半の割合から後半の割合を
引いたもの（前半の割合 − 後半の割合）で、割合はペアの最小重原子間距離
が接触の閾値（4.5 Å）以下の時間の割合である。

条件は資料Dの条件（6 タンパク質×3 レプリカ、起点 1 ns、履歴 0〜1 ns、
HistGBM を条件ごとに学習し直し、学習 4／テスト 1mix_A＋2qt4_A）で、
接触の解離、10 ns先、割合だけ。

入力は、静的な特徴量に、つぎのどれか 1 つを足した形で比べる:
- 前半（フレーム [0..4]）の割合
- 後半（フレーム [6..10]）の割合
- 前半と後半の両方（2 つの割合を並べたもの）
- 全部（フレーム [0..10]）の割合
- 漂い（前半の割合 − 後半の割合）だけ
- 全部の割合＋漂い
前半と後半のあいだのフレーム 5 は、どちらの半分にも入れない。

土台の再現: 静的のみと全部入り（6 個の統計量）の AUPRC が、接触の解離
の 10 ns先の記録（K50、静的のみ 0.2772、全部 0.4794、テスト 2,915件・
陽性 228件）と一致することを確かめる。全部の平均だけは K50 の
mean_only（10 ns先、0.4630）とも照らす。前半・後半・両方・全部の割合
だけは K2-133 の記録（10 ns先、前半 0.4039・後半 0.4276・両方 0.4302・
全部 0.4429）とも照らす。

読み方（K2-15 と同じく先に固定）:
- 全部の割合＋漂いが全部の割合だけを AUPRC で +0.01 以上上回れば
  「漂いが足しになる」、そうでなければ「足しにならない」。
- 漂いだけの利得も記録する。
タンパク質ごとの (c)−(b) の差も記録する。
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
HIDX = 2  # 10 ns先
FRAMES_FIRST = [0, 1, 2, 3, 4]       # 前半（0〜0.4 ns）
FRAMES_SECOND = [6, 7, 8, 9, 10]     # 後半（0.6〜1.0 ns）
FRAMES_FULL = list(range(11))        # 全部（0〜1.0 ns）

# 接触の解離の 10 ns先の記録との照合用の期待値（合算）
EXPECTED = {"static_only": 0.2772, "full_hist6": 0.4794,
            "mean_full": 0.4630,  # K50 の mean_only（10 ns先）
            "first_half_frac": 0.4039, "second_half_frac": 0.4276,
            "both_halves_frac": 0.4302, "full_frac": 0.4429,  # K2-133 の記録（10 ns先）
            "test_n": 2915, "test_positives": 228}


def hist6(series, bound=BOUND):
    """series は (N, F) の距離系列。与えられたフレーム範囲の 6 個の統計量。"""
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def frac_of(series, frames, bound=BOUND):
    """series は (N, F) の距離系列。指定フレームの境界以下割合を返す (N, 1)。"""
    return (series[:, frames] <= bound).mean(axis=1, keepdims=True).astype(np.float32)


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

CONDS = ["static_only", "full_hist6", "mean_full", "first_half_frac",
         "second_half_frac", "both_halves_frac", "full_frac",
         "drift_only", "full_frac_plus_drift"]


def feats_for(cname):
    if cname == "static_only":
        return Xtr, Xte
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr[:, FRAMES_FULL])]),
                np.column_stack([Xte, hist6(Hte[:, FRAMES_FULL])]))
    if cname == "mean_full":
        return (np.column_stack([Xtr, Htr[:, FRAMES_FULL].mean(1, keepdims=True)]),
                np.column_stack([Xte, Hte[:, FRAMES_FULL].mean(1, keepdims=True)]))
    if cname == "first_half_frac":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FIRST)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FIRST)]))
    if cname == "second_half_frac":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_SECOND)]))
    if cname == "both_halves_frac":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FIRST),
                                 frac_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FIRST),
                                 frac_of(Hte, FRAMES_SECOND)]))
    if cname == "full_frac":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FULL)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FULL)]))
    if cname == "drift_only":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FIRST)
                                 - frac_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FIRST)
                                 - frac_of(Hte, FRAMES_SECOND)]))
    if cname == "full_frac_plus_drift":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FULL),
                                 frac_of(Htr, FRAMES_FIRST)
                                 - frac_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FULL),
                                 frac_of(Hte, FRAMES_FIRST)
                                 - frac_of(Hte, FRAMES_SECOND)]))
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

# 接触の解離の 10 ns先の記録との照合
assert res["static_only"]["n"] == EXPECTED["test_n"]
assert res["static_only"]["positives"] == EXPECTED["test_positives"]
for c in ("static_only", "full_hist6"):
    assert abs(res[c]["auprc"] - EXPECTED[c]) < 5e-4, (c, res[c]["auprc"])
assert abs(res["full_frac"]["auprc"] - EXPECTED["full_frac"]) < 5e-4, res["full_frac"]
print("base reproduction vs K50 (10 ns): OK", flush=True)
assert abs(res["mean_full"]["auprc"] - EXPECTED["mean_full"]) < 5e-4, res["mean_full"]
print("full mean vs K50 mean_only (10 ns): OK", flush=True)
for c in ("first_half_frac", "second_half_frac", "both_halves_frac", "full_frac"):
    assert abs(res[c]["auprc"] - EXPECTED[c]) < 5e-4, (c, res[c]["auprc"])
print("frac halves vs K2-133 (10 ns): OK", flush=True)

gain_full = res["full_hist6"]["auprc"] - res["static_only"]["auprc"]
gain_frac_full = res["full_frac"]["auprc"] - res["static_only"]["auprc"]
gains = {c: res[c]["auprc"] - res["static_only"]["auprc"] for c in CONDS}
ratios = {c: gains[c] / gain_full if gain_full else float("nan") for c in CONDS}
ratios_vs_frac = {c: gains[c] / gain_frac_full if gain_frac_full else float("nan")
                  for c in CONDS}


# ---- 漂いの読み（K2-15 と同じく先に固定） ----
a_frac = res["full_frac"]["auprc"]
a_frac_drift = res["full_frac_plus_drift"]["auprc"]
diff_c_minus_b = a_frac_drift - a_frac
reading = "漂いは足しになる" if diff_c_minus_b >= 0.01 else "漂いは足しにならない"
print(f"gain_full {gain_full:+.4f} gain_first {gains['first_half_frac']:+.4f} "
      f"(ratio {ratios['first_half_frac']:.3f}) "
      f"gain_second {gains['second_half_frac']:+.4f} "
      f"(ratio {ratios['second_half_frac']:.3f}) "
      f"gain_both {gains['both_halves_frac']:+.4f} "
      f"(ratio {ratios['both_halves_frac']:.3f}) "
      f"gain_full_frac {gain_frac_full:+.4f}", flush=True)
print("drift reading: (c) full_frac+drift", round(a_frac_drift, 4),
      "- (b) full_frac", round(a_frac, 4), "= diff", round(diff_c_minus_b, 4),
      "->", reading, flush=True)
print("drift only gain:", round(gains["drift_only"], 4), flush=True)

# タンパク質ごとの (c)−(b) の差
per_protein_diff = {}
for prot in TEST_P:
    a_b = per_protein["full_frac"][prot]["auprc"]
    a_c = per_protein["full_frac_plus_drift"][prot]["auprc"]
    per_protein_diff[prot] = {"auprc_full_frac": a_b,
                              "auprc_full_frac_plus_drift": a_c,
                              "diff_c_minus_b": a_c - a_b}
print("per-protein diff (c)-(b):", {p: round(v["diff_c_minus_b"], 4)
                                    for p, v in per_protein_diff.items()}, flush=True)

out = {
    "label": "Contact dissociation 10ns: does the fraction drift add to the full fraction?",
    "phenomenon": "contact dissociation (Doc D, k50_cache)",
    "experiment": "K2-174",
    "horizon_ns": 10,
    "bound_A": BOUND,
    "fraction": "fraction of frames with min heavy-atom distance <= 4.5 A",
    "frames": {"first_half": FRAMES_FIRST, "second_half": FRAMES_SECOND,
               "full": FRAMES_FULL,
               "note": "frame 5 belongs to neither half"},
    "bound": BOUND,
    "metrics": res,
    "per_protein_metrics": per_protein,
    "gains": gains,
    "gain_ratios_vs_full_hist6": ratios,
    "gain_ratios_vs_full_frac": ratios_vs_frac,
    "full_frac_gain": float(gain_frac_full),
    "drift_definition": "drift = first-half fraction [0..4] minus second-half fraction "
                        "[6..10], fraction = frames with min heavy-atom distance <= 4.5 A",
    "reading_rule_fixed_in_advance": {
        "rule": "(c) full_frac+drift minus (b) full_frac >= +0.01 AUPRC "
                "-> 'drift adds'; otherwise 'drift does not add' (K2-15's rule).",
        "auprc_full_frac": a_frac,
        "auprc_full_frac_plus_drift": a_frac_drift,
        "diff_c_minus_b_auprc": diff_c_minus_b,
        "reading": reading,
        "per_protein_diff_c_minus_b": per_protein_diff,
    },
    "checks_vs_K50_10ns": {
        "static_only_auprc_expected": EXPECTED["static_only"],
        "full_hist6_auprc_expected": EXPECTED["full_hist6"],
        "full_frac_auprc_expected": EXPECTED["full_frac"]},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auc", "auprc", "brier",
                "logloss", "gain_vs_static", "gain_ratio_vs_full_hist6",
                "gain_ratio_vs_full_frac"])
    for c in CONDS:
        m = res[c]
        w.writerow([c, m["n"], m["positives"], m["auc"], m["auprc"],
                    m["brier"], m["logloss"], gains[c], ratios[c],
                    ratios_vs_frac[c]])
print("saved", OUT / "results.json", OUT / "results.csv", flush=True)
# 予測確率は float32 に落として圧縮して保存する（ディスク割り当ての都合で
# 大きな npz が書けないことがあるため、保存に失敗しても results.json は残る）
try:
    np.savez_compressed(OUT / "test_predictions.npz",
                        y_test=yte,
                        **{f"pred_{c}": preds[c].astype(np.float32) for c in CONDS})
    print("saved", OUT / "test_predictions.npz", flush=True)
except OSError as e:
    print(f"WARNING: test_predictions.npz not saved ({e}); "
          "predictions are regenerable by re-running this script", flush=True)
