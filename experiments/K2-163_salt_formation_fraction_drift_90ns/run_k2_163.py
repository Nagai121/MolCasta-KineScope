"""K2-163: 90 ns先で「割合」の漂いを測る
（K15 の条件、塩橋の形成、90 ns先）.

K2-15 の問い（漂い）では、割合の漂いは水素結合の形成の 1・30・90 ns先、
水素結合の解離・接触の形成・接触の解離の 1・30 ns先、塩橋の形成の
1・30 ns先、塩橋の解離の 1 ns先で測り終えている。ここでは同じ問いを
塩橋の形成の 90 ns先で測る。

条件（K15 の条件、K2-136 と同じ土台）:
- 学習は K15 の 58 タンパク質、テストは K15 の 30 タンパク質
  （split.json の orig_train 4 + train_additions 54、
  orig_test 2 + test_additions 28）。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K15 のキャッシュ（~/workspace/kinescope/work/k15_cache/、
  候補ごとの荷電重原子間の最小距離の系列）を再利用する。履歴は
  (N, 11) で保存されているため、読み込み時に転置しない（K2-135 と同じ）。
  特徴量は K2-10・K2-135 と同じく、保存された dtype（float32）のまま
  sklearn に渡す。
- 予測幅は 90 ns先（horizon インデックス 5、HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）。

割合と漂いの定義:
- 割合は、ペアの荷電重原子間の最小距離が塩橋の閾値（4.0 Å）以下の
  時間の割合である。割合は、境界以下だったフレームの数 ÷ その区間の
  フレーム数（前半・後半は 5、全部は 11）。
- 漂いは、前半の割合から後半の割合を引いたもの
  （前半の割合 − 後半の割合）である。正の漂いは、履歴の前半のほうが
  後半より 4.0 Å 以下の時間が長かった（ペアが前半で近かった）ことを
  表す。平均の漂い（K2-15）と同じ向きの定義である。

比べる条件（静的特徴5個に、つぎのどれかを足す）:
- static_only:         静的特徴だけ（土台の照合用）
- base_hist_D6:        静的＋履歴 6 統計量（全部入り、土台の照合用）
- full_frac:           静的＋全部（フレーム [0..10]）の 4.0 Å以下割合（b）
- full_frac_plus_drift: 静的＋全部の割合＋漂い（c）

土台の再現の期待値（K15・K2-136 の 90 ns先の記録）:
- 静的のみ AUPRC 0.2136、全部入り 0.2736、利得 +0.0601、
  全部の割合だけ 0.2688、テスト 1,170件・陽性 149件。

読み方（結果を見る前に固定、K2-15 と同じ）:
- (c) 全部の割合＋漂いが (b) 全部の割合だけを AUPRC で +0.01 以上
  上回れば「漂いが足しになる」、そうでなければ「足しにならない」。
  タンパク質ごとの (c)−(b) の差も記録する（30 タンパク質）。

タンパク質ごとの注意:
- 陽性が 0 件のタンパク質では AUPRC 自体が定義できない（sklearn は
  警告とともに 0.0 を返すが、それは値ではない）。K2-57 の教訓どおり
  auprc_or_nan で守る。90 ns先の塩橋の形成では、陽性が 0 件の
  テストタンパク質が 5 つある（K2-136 の記録どおり）。
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import csv
import json
import math

import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
HORIZON_IDX = 5  # 90 ns先（HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）
BOUND = 4.0  # 割合の境界（塩橋の閾値。6 個の統計量の「4.0 Å 以下だった割合」と同じ）

FIRST_FRAMES = list(range(0, 5))    # フレーム [0..4]、0〜0.4 ns
SECOND_FRAMES = list(range(6, 11))  # フレーム [6..10]、0.6〜1.0 ns
FULL_FRAMES = list(range(0, 11))    # フレーム [0..10]、0〜1.0 ns


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
            d = {k: z[k] for k in z.files}
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    # hist は (N, 11) で保存されているので転置しない（K2-135 と同じ）
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
    return np.stack(
        [series[:, 0], series.mean(1), series.std(1), series.min(1),
         series[:, -1] - series[:, 0], (series <= bound).mean(1)],
        axis=1).astype(np.float32)


def frac_cols(series, frames, bound=BOUND):
    return (series[:, frames] <= bound).mean(1)[:, None].astype(np.float32)


def drift_of(series, bound=BOUND):
    """前半の割合 − 後半の割合。"""
    f1 = (series[:, FIRST_FRAMES] <= bound).mean(1)
    f2 = (series[:, SECOND_FRAMES] <= bound).mean(1)
    return (f1 - f2)[:, None].astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def auprc(y, p):
    return float(average_precision_score(y, p))


def auprc_or_nan(y, p):
    """陽性が 0 件か全陽性のときは AUPRC は定義できないので nan を返す。
    sklearn は警告とともに 0.0 を返すが、それは値ではない（K2-57 の教訓）。"""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, p))


split = json.loads((K15_DIR / "split.json").read_text())
TRAIN_P = tuple(split["orig_train"]) + tuple(split["train_additions"])
TEST_P = tuple(split["orig_test"]) + tuple(split["test_additions"])
print("train proteins", len(TRAIN_P), "test proteins", len(TEST_P), flush=True)
train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
# K2-10・K2-135 と同じく、特徴量は保存された dtype（float32）のまま sklearn に渡す
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 90ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 2557 and Xte.shape[0] == 1170, (Xtr.shape, Xte.shape)
# 起点フレームの距離が履歴の最終フレームと一致すること（K2-135 と同じ確認）
assert np.max(np.abs(Xtr[:, 0] - Htr[:, -1])) < 1e-6
assert np.max(np.abs(Xte[:, 0] - Hte[:, -1])) < 1e-6

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

# 全部の割合が 6 統計量の 6 列目（4.5 Å 以下割合）と一致することの確認
h6_tr = hist6(Htr)
h6_te = hist6(Hte)
full_frac_tr = frac_cols(Htr, FULL_FRAMES)
full_frac_te = frac_cols(Hte, FULL_FRAMES)
print("full frac vs hist6 col 5: train max diff",
      float(np.abs(full_frac_tr[:, 0] - h6_tr[:, 5]).max()),
      "test", float(np.abs(full_frac_te[:, 0] - h6_te[:, 5]).max()), flush=True)
assert float(np.abs(full_frac_tr[:, 0] - h6_tr[:, 5]).max()) < 1e-6
assert float(np.abs(full_frac_te[:, 0] - h6_te[:, 5]).max()) < 1e-6

drift_tr = drift_of(Htr)
drift_te = drift_of(Hte)
print("drift (test): mean", round(float(drift_te.mean()), 4),
      "std", round(float(drift_te.std()), 4),
      "min", round(float(drift_te.min()), 3),
      "max", round(float(drift_te.max()), 3), flush=True)

# ---- 条件ごとの学習と予測 ----
preds = {}
preds["static_only"] = predict(Xtr, ytr, Xte)
preds["base_hist_D6"] = predict(np.column_stack([Xtr, h6_tr]), ytr,
                                np.column_stack([Xte, h6_te]))
preds["full_frac"] = predict(np.column_stack([Xtr, full_frac_tr]), ytr,
                             np.column_stack([Xte, full_frac_te]))
preds["full_frac_plus_drift"] = predict(
    np.column_stack([Xtr, full_frac_tr, drift_tr]), ytr,
    np.column_stack([Xte, full_frac_te, drift_te]))

results = {}
for name, p in preds.items():
    results[name] = metrics(yte, p)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in results[name].items()}, flush=True)

# ---- 土台の再現の確認（K15・K2-136 の 90 ns先の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
a_fullfrac = results["full_frac"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.0601):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.2136) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.2736) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.0601) < 5e-4, full_gain
assert abs(a_fullfrac - 0.2688) < 5e-4, results["full_frac"]
assert results["static_only"]["positives"] == 149, results["static_only"]
print("base reproduction OK (static 0.2136 / full6 0.2736 / gain +0.0601 / full frac 0.2688)",
      flush=True)

# ---- 読み方（結果を見る前に固定、K2-15 と同じ） ----
diff_c_minus_b = results["full_frac_plus_drift"]["auprc"] - results["full_frac"]["auprc"]
print("(c) - (b) =", round(diff_c_minus_b, 4), flush=True)
if diff_c_minus_b >= 0.01:
    verdict = "漂いが足しになる"
else:
    verdict = "漂いが足しにならない"
print("verdict:", verdict, flush=True)

# ---- タンパク質ごとの値 ----
per_protein = {}
for name in preds:
    row = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        row[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc": auprc_or_nan(yte[idx], preds[name][idx]),
        }
    per_protein[name] = row
per_protein_diff = {}
for prot in TEST_P:
    a_b = per_protein["full_frac"][prot]["auprc"]
    a_c = per_protein["full_frac_plus_drift"][prot]["auprc"]
    per_protein_diff[prot] = {
        "n": per_protein["full_frac"][prot]["n"],
        "positives": per_protein["full_frac"][prot]["positives"],
        "auprc_full_frac": a_b,
        "auprc_full_frac_plus_drift": a_c,
        "diff_c_minus_b": (a_c - a_b) if not (math.isnan(a_b) or math.isnan(a_c)) else float("nan"),
    }
    print(prot, "diff (c)-(b):", round(per_protein_diff[prot]["diff_c_minus_b"], 4)
          if not math.isnan(per_protein_diff[prot]["diff_c_minus_b"]) else "nan", flush=True)

# ---- 結果の保存 ----
out = {
    "label": "Salt formation 90ns: drift of the fraction (first-half fraction minus second-half fraction) on top of the full fraction",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-163",
    "horizon_ns": 90,
    "bound_A": BOUND,
    "conditions_definition": {
        "base": "K15 conditions: train 58 proteins (orig_train 4 + train_additions 54), test 30 proteins (orig_test 2 + test_additions 28) pooled, origin frame 10 (1 ns), history frames 0-10 (100 ps spacing), HistGBM (120 iters, lr 0.075, 31 leaves, min_samples_leaf 15, L2 0.02, seed 42) refit per condition, 5 static features. Salt formation, k15_cache (features passed in stored float32 dtype, as in K2-10/K2-135).",
        "fraction": "fraction of frames with pair min charged heavy-atom distance <= 4.0 A (salt-bridge threshold; same definition as the 'fraction <= 4.0 A' of the 6 statistics) within the given frame range, divided by the number of frames in that range (first/second half: 5, full: 11)",
        "drift": "first-half fraction minus second-half fraction; positive = the pair spent more time within 4.0 A in the first half than in the second half",
        "first_half": {"frames": FIRST_FRAMES, "span_ns": "0-0.4 ns"},
        "second_half": {"frames": SECOND_FRAMES, "span_ns": "0.6-1.0 ns"},
        "full": {"frames": FULL_FRAMES, "span_ns": "0-1.0 ns"},
        "note": "Frame 5 (0.5 ns) belongs to neither half by design.",
    },
    "base_reproduction": {
        "expected_from_K15_K2_136_90ns": {"static_only_auprc": 0.2136, "base_hist_D6_auprc": 0.2736,
                              "full6_gain": 0.0601, "full_frac_auprc": 0.2688,
                              "test_n": 1170, "test_positives": 149},
        "observed": {"static_only_auprc": a_static, "base_hist_D6_auprc": a_full6,
                     "full6_gain": full_gain, "full_frac_auprc": a_fullfrac},
        "match": "all within 5e-4",
        "n_train": int(Xtr.shape[0]),
    },
    "metrics_90ns": results,
    "gains": {name: {"gain": results[name]["auprc"] - a_static,
                     "ratio_vs_full6_gain": (results[name]["auprc"] - a_static) / full_gain}
              for name in results},
    "drift": {
        "diff_c_minus_b_auprc": diff_c_minus_b,
        "reading_rule_fixed_in_advance": "(c) full_frac_plus_drift beats (b) full_frac by AUPRC >= +0.01 -> drift adds; else it does not",
        "verdict": verdict,
        "test_drift_stats": {"mean": float(drift_te.mean()), "std": float(drift_te.std()),
                             "min": float(drift_te.min()), "max": float(drift_te.max())},
    },
    "per_protein": per_protein,
    "per_protein_diff_c_minus_b": per_protein_diff,
    "reading_rule_fixed_in_advance": {
        "rule": "(c) full_frac_plus_drift beats (b) full_frac by AUPRC >= +0.01 -> drift adds; else it does not",
        "diff_c_minus_b_auprc": diff_c_minus_b,
        "verdict": verdict,
    },
}
(OUT / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auprc", "auc", "brier", "logloss"])
    for name in preds:
        m = results[name]
        w.writerow([name, m["n"], m["positives"], f'{m["auprc"]:.6f}',
                    f'{m["auc"]:.6f}', f'{m["brier"]:.6f}', f'{m["logloss"]:.6f}'])
    w.writerow([])
    w.writerow(["protein", "n", "positives", "auprc_full_frac",
                "auprc_full_frac_plus_drift", "diff_c_minus_b"])
    for prot in TEST_P:
        d = per_protein_diff[prot]
        fmt = lambda v: "" if math.isnan(v) else f"{v:.6f}"
        w.writerow([prot, d["n"], d["positives"], fmt(d["auprc_full_frac"]),
                    fmt(d["auprc_full_frac_plus_drift"]), fmt(d["diff_c_minus_b"])])

try:
    np.savez_compressed(OUT / "test_predictions.npz",
                        y_30ns=yte.astype(np.uint8),
                        **{f"pred_{k}": v.astype(np.float32) for k, v in preds.items()})
except OSError as e:
    print("WARNING: could not save test_predictions.npz:", e, flush=True)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz", flush=True)
