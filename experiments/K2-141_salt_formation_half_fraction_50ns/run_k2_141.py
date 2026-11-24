"""K2-141: 塩橋の形成で、履歴の前半と後半のどちらに指導力があるか（50 ns先、割合だけ）.

K2-14 の問い（割合の前半と後半）を、塩橋の形成の 50 ns先で測る。
割合は、ペアの荷電重原子間の最小距離が塩橋の閾値（4.0 Å）以下の時間
の割合で、6 個の統計量のうちの「4.0 Å 以下だった割合」と同じ定義で
ある。塩橋の形成の割合の前半と後半は 1・30・90 ns先（K2-135・K2-137・
K2-136）で測り終えた。50 ns先ではどうなるかを見る。

条件は K15 の条件（58 タンパク質で学習、30 タンパク質でテスト、起点
1 ns、履歴 0〜1 ns、HistGBM を条件ごとに学習し直し）で、塩橋の形成、
50 ns先、割合だけ。

入力は、静的な特徴量に、つぎのどれか 1 つを足した形で比べる:
- 前半（フレーム [0..4]）の割合
- 後半（フレーム [6..10]）の割合
- 前半と後半の両方（2 つの割合を並べたもの）
- 全部（フレーム [0..10]）の割合
前半と後半のあいだのフレーム 5 は、どちらの半分にも入れない。

土台の再現: 静的のみと全部入り（6 個の統計量）の AUPRC が、塩橋の形成
の 50 ns先の記録（K15、静的のみ 0.2295、全部 0.2569、テスト 1,170件・
陽性 137件）と一致することを確かめる。50 ns先では全部の平均だけの
記録がないため、平均の照合はない。

読み方（K2-01 と同じく先に固定）。割合の分母は全部入りの利得を主とし、
あわせて全部の割合の利得に対する割合も記録する。分母を全部の割合の
利得にしても判定が変わる場合は、両方を記録する。
- 後半の利得が全部入りの 8 割以上で、前半の利得が全部入りの 5 割以下
  なら「今の状態が効いている」。
- 前半も後半も全部入りの 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する。
タンパク質ごとの形の数え上げ（「今の状態」・「続いている状態」・
どちらでもない・利得が小さく判定から外したもの）も K2-07 と同じく
記録する。
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

CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
BOUND = 4.0
HIDX = 4  # 50 ns先
FRAMES_FIRST = [0, 1, 2, 3, 4]       # 前半（0〜0.4 ns）
FRAMES_SECOND = [6, 7, 8, 9, 10]     # 後半（0.6〜1.0 ns）
FRAMES_FULL = list(range(11))        # 全部（0〜1.0 ns）

# K15 の記録との照合用の期待値（塩橋の形成、50 ns先、合算）
EXPECTED = {"static_only": 0.2295, "full_hist6": 0.2569,
            "test_n": 1170, "test_positives": 137}


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
    spans = []
    for p in prots:
        n = 0
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
            parts.append({k: z[k] for k in z.files})
            n += len(z["X"])
        spans.append((p, n))
    return parts, spans


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


split = json.loads((K15_DIR / "split.json").read_text())
train_p = ORIG_TRAIN + tuple(split["train_additions"])
test_p = ORIG_TEST + tuple(split["test_additions"])
print("train proteins", len(train_p), "test proteins", len(test_p), flush=True)

train, _ = load(train_p)
test, test_spans = load(test_p)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
Pte = np.concatenate([np.full(n, p) for p, n in test_spans])
print("train", Xtr.shape, "test", Xte.shape, flush=True)

CONDS = ["static_only", "full_hist6", "first_half_frac",
         "second_half_frac", "both_halves_frac", "full_frac"]


def feats_for(cname):
    if cname == "static_only":
        return Xtr, Xte
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr[:, FRAMES_FULL])]),
                np.column_stack([Xte, hist6(Hte[:, FRAMES_FULL])]))
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
    for prot in test_p:
        m = Pte == prot
        per_protein[cname][prot] = metrics(yte[m], p[m])
    print(cname, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in res[cname].items()}, flush=True)

# 塩橋の形成の 50 ns先の記録との照合
assert res["static_only"]["n"] == EXPECTED["test_n"]
assert res["static_only"]["positives"] == EXPECTED["test_positives"]
for c in ("static_only", "full_hist6"):
    assert abs(res[c]["auprc"] - EXPECTED[c]) < 5e-4, (c, res[c]["auprc"])
print("base reproduction vs K15 (50 ns): OK", flush=True)

gain_full = res["full_hist6"]["auprc"] - res["static_only"]["auprc"]
gain_frac_full = res["full_frac"]["auprc"] - res["static_only"]["auprc"]
gains = {c: res[c]["auprc"] - res["static_only"]["auprc"] for c in CONDS}
ratios = {c: gains[c] / gain_full if gain_full else float("nan") for c in CONDS}
ratios_vs_frac = {c: gains[c] / gain_frac_full if gain_frac_full else float("nan")
                  for c in CONDS}


def reading(first, second):
    if second >= 0.8 and first <= 0.5:
        return "今の状態が効いている"
    if first >= 0.8 and second >= 0.8:
        return "続いている状態が効いている"
    return "どちらでもない（数値をそのまま記録）"


judgment = {
    "vs_full_hist6": reading(ratios["first_half_frac"], ratios["second_half_frac"]),
    "vs_full_frac": reading(ratios_vs_frac["first_half_frac"],
                           ratios_vs_frac["second_half_frac"]),
}
print(f"gain_full {gain_full:+.4f} gain_first {gains['first_half_frac']:+.4f} "
      f"(ratio {ratios['first_half_frac']:.3f}) "
      f"gain_second {gains['second_half_frac']:+.4f} "
      f"(ratio {ratios['second_half_frac']:.3f}) "
      f"gain_both {gains['both_halves_frac']:+.4f} "
      f"(ratio {ratios['both_halves_frac']:.3f}) "
      f"gain_full_frac {gain_frac_full:+.4f} judgment: {judgment}", flush=True)

# タンパク質ごとの形の数え上げ（K2-07 と同じ。分母はそのタンパク質の全部入りの利得）
shapes = {"now_state": 0, "continuing": 0, "other": 0, "excluded_gain_le_0.01": 0}
prot_frac = {}
for prot in test_p:
    gf = (per_protein["full_hist6"][prot]["auprc"]
          - per_protein["static_only"][prot]["auprc"])
    if gf <= 0.01:
        shapes["excluded_gain_le_0.01"] += 1
        continue
    f = (per_protein["first_half_frac"][prot]["auprc"]
         - per_protein["static_only"][prot]["auprc"]) / gf
    s = (per_protein["second_half_frac"][prot]["auprc"]
         - per_protein["static_only"][prot]["auprc"]) / gf
    prot_frac[prot] = {"first": float(f), "second": float(s)}
    if s >= 0.8 and f <= 0.5:
        shapes["now_state"] += 1
    elif f >= 0.8 and s >= 0.8:
        shapes["continuing"] += 1
    else:
        shapes["other"] += 1
print("per-protein shapes", shapes, flush=True)

out = {
    "label": "Salt-bridge formation, first-half vs second-half history fraction at 50 ns (K2-14 question at 50 ns)",
    "phenomenon": "salt-bridge formation (K15 conditions, k15_cache)",
    "horizon_ns": 50,
    "bound_A": BOUND,
    "fraction": "fraction of frames with min charged heavy-atom distance <= 4.0 A",
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
    "reading_rule": ("ratios are vs the full hist6 gain (K2-01 convention); "
                     "second >= 0.8 and first <= 0.5 -> current state; "
                     "both >= 0.8 -> persistent state; else record numbers"),
    "judgment": judgment,
    "per_protein_shapes": shapes,
    "per_protein_fractions": prot_frac,
    "checks_vs_K15_50ns": {
        "static_only_auprc_expected": EXPECTED["static_only"],
        "full_hist6_auprc_expected": EXPECTED["full_hist6"]},
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
