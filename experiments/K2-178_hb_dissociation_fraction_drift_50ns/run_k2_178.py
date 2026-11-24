"""K2-178: 50 ns先で「割合」の漂いを測る
（資料D の条件、水素結合の解離、50 ns先）.

K2-15 の問い（漂い）では、割合の漂いは水素結合の形成の 1・30・90 ns先、
水素結合の解離の 1・90 ns先、接触の形成の 1・10・30 ns先、接触の解離の
1・5・30 ns先、塩橋の形成の 1・30・90 ns先、塩橋の解離の 1 ns先で測り
終えている。ここでは同じ問いを水素結合の解離の 50 ns先で測る。

条件（資料D の条件、K2-110 と同じ土台）:
- 学習は 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テストは 1mix_A・2qt4_A
  の合算（K46 のキャッシュの分割と同じ）。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K46 のキャッシュ（~/workspace/kinescope/work/data_cache_k46/、
  候補ごとのドナー・アクセプター距離の系列）を再利用する。履歴は
  (11, N) で保存されているため、読み込み時に転置する（K2-149 と同じ）。
  特徴量は K2-149 と同じく、float32 にそろえて sklearn に渡す。
- 予測幅は 50 ns先（horizon インデックス 4、HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）。

割合と漂いの定義:
- 割合は、ペアの距離が水素結合の閾値（3.5 Å）以下の時間の割合である。
  割合は、境界以下だったフレームの数 ÷ その区間のフレーム数
  （前半・後半は 5、全部は 11）。
- 漂いは、前半の割合から後半の割合を引いたもの
  （前半の割合 − 後半の割合）である。正の漂いは、履歴の前半のほうが
  後半より 3.5 Å 以下の時間が長かった（ペアが前半で近かった）ことを
  表す。平均の漂い（K2-15）と同じ向きの定義である。

比べる条件（静的特徴7個に、つぎのどれかを足す）:
- static_only:         静的特徴だけ（土台の照合用）
- base_hist_D6:        静的＋履歴 6 統計量（全部入り、土台の照合用）
- first_half_frac:     静的＋前半（フレーム [0..4]）の 3.5 Å以下割合
- second_half_frac:    静的＋後半（フレーム [6..10]）の 3.5 Å以下割合
- both_halves_frac:    静的＋前半の割合＋後半の割合
- full_frac:           静的＋全部（フレーム [0..10]）の 3.5 Å以下割合（b）
- drift_only:          静的＋漂いだけ
- full_frac_plus_drift: 静的＋全部の割合＋漂い（c）

土台の再現の期待値（K46・K2-110 の 50 ns先の記録）:
- 静的のみ AUPRC 0.4088、全部入り 0.5231、利得 +0.1142、
  前半の割合だけ 0.4929、後半の割合だけ 0.4423、
  全部の割合だけ 0.4827、テスト 2,080件・陽性 362件。

読み方（結果を見る前に固定、K2-15 と同じ）:
- (c) 全部の割合＋漂いが (b) 全部の割合だけを AUPRC で +0.01 以上
  上回れば「漂いが足しになる」、そうでなければ「足しにならない」。
  タンパク質ごとの (c)−(b) の差も記録する（2 タンパク質）。

タンパク質ごとの注意:
- 陽性が 0 件のタンパク質では AUPRC 自体が定義できない（sklearn は
  警告とともに 0.0 を返すが、それは値ではない）。K2-57 の教訓どおり
  auprc_or_nan で守り、results.json では空欄（null）で記録する。
  50 ns先の水素結合の解離では、テストの 2 タンパク質とも陽性がある
  （1mix_A 232 件、2qt4_A 130 件）ので、値はどちらも決まる。
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 4  # 50 ns先（HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）
BOUND = 3.5  # 割合の境界（形成と同じ）

FIRST_FRAMES = list(range(0, 5))    # フレーム [0..4]、0〜0.4 ns
SECOND_FRAMES = list(range(6, 11))  # フレーム [6..10]、0.6〜1.0 ns
FULL_FRAMES = list(range(0, 11))    # フレーム [0..10]、0〜1.0 ns


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)


def hist6(series, bound=3.5):
    return np.stack(
        [series[:, 0], series.mean(1), series.std(1), series.min(1),
         series[:, -1] - series[:, 0], (series <= bound).mean(1)],
        axis=1).astype(np.float32)


def frac_cols(series, frames, bound=BOUND):
    return (series[:, frames] <= bound).mean(1)[:, None].astype(np.float32)


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
    # 陽性が 0 件（または全件陽性）のタンパク質では AUPRC は定義できない
    # （sklearn は警告とともに 0.0 を返すが、それは値ではない）。
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, p))


def clean_nan(o):
    # results.json に NaN の文字を残さない（K2-163 と同じ扱い）。
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_nan(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 50ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 4051 and Xte.shape[0] == 2080, (Xtr.shape, Xte.shape)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

# ---- 条件ごとの特徴量 ----
f6tr, f6te = hist6(Htr), hist6(Hte)
FEATURES = {
    "static_only": None,
    "base_hist_D6": (f6tr, f6te),
    "first_half_frac": (frac_cols(Htr, FIRST_FRAMES), frac_cols(Hte, FIRST_FRAMES)),
    "second_half_frac": (frac_cols(Htr, SECOND_FRAMES), frac_cols(Hte, SECOND_FRAMES)),
    "both_halves_frac": (np.concatenate([frac_cols(Htr, FIRST_FRAMES), frac_cols(Htr, SECOND_FRAMES)], axis=1),
                         np.concatenate([frac_cols(Hte, FIRST_FRAMES), frac_cols(Hte, SECOND_FRAMES)], axis=1)),
    "full_frac": (frac_cols(Htr, FULL_FRAMES), frac_cols(Hte, FULL_FRAMES)),
    "drift_only": (frac_cols(Htr, FIRST_FRAMES) - frac_cols(Htr, SECOND_FRAMES),
                   frac_cols(Hte, FIRST_FRAMES) - frac_cols(Hte, SECOND_FRAMES)),
    "full_frac_plus_drift": (
        np.concatenate([frac_cols(Htr, FULL_FRAMES),
                        frac_cols(Htr, FIRST_FRAMES) - frac_cols(Htr, SECOND_FRAMES)], axis=1),
        np.concatenate([frac_cols(Hte, FULL_FRAMES),
                        frac_cols(Hte, FIRST_FRAMES) - frac_cols(Hte, SECOND_FRAMES)], axis=1)),
}

preds, results = {}, {}
for name, feats in FEATURES.items():
    if feats is None:
        preds[name] = predict(Xtr, ytr, Xte)
    else:
        ftr, fte = feats
        preds[name] = predict(
            np.concatenate([Xtr, ftr], axis=1), ytr,
            np.concatenate([Xte, fte], axis=1))
    results[name] = metrics(yte, preds[name])
    print(name, {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in results[name].items()}, flush=True)

drift_te = (frac_cols(Hte, FIRST_FRAMES) - frac_cols(Hte, SECOND_FRAMES)).ravel()
print("drift (test): mean", round(float(drift_te.mean()), 4), "std",
      round(float(drift_te.std()), 4), "min", round(float(drift_te.min()), 4),
      "max", round(float(drift_te.max()), 4), flush=True)

# ---- 土台の再現の確認（K46・K2-110 の 50 ns先の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.1142):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.4088) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.5231) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.1142) < 5e-4, full_gain
# K2-110 の記録との照合（50 ns先、割合だけ）
assert abs(results["first_half_frac"]["auprc"] - 0.4929) < 5e-4, results["first_half_frac"]
assert abs(results["second_half_frac"]["auprc"] - 0.4423) < 5e-4, results["second_half_frac"]
a_fullfrac = results["full_frac"]["auprc"]
assert abs(a_fullfrac - 0.4827) < 5e-4, results["full_frac"]
assert results["static_only"]["positives"] == 362, results["static_only"]
print("base reproduction OK (static 0.4088 / full6 0.5231 / gain +0.1142 /",
      "first 0.4929 / second 0.4423 / full frac", round(a_fullfrac, 4), ")", flush=True)

# ---- 利得と割合 ----
fullfrac_gain = a_fullfrac - a_static
gains = {}
for name in FEATURES:
    g = results[name]["auprc"] - a_static
    gains[name] = {
        "gain": float(g),
        "ratio_vs_full6_gain": float(g / full_gain),
        "ratio_vs_fullfrac_gain": float(g / fullfrac_gain),
    }
    print(name, "gain", round(g, 4),
          "ratio vs full6", round(g / full_gain, 4),
          "ratio vs full-frac", round(g / fullfrac_gain, 4), flush=True)

# ---- 漂いの読み（K2-15 と同じく先に固定） ----
a_frac = results["full_frac"]["auprc"]
a_frac_drift = results["full_frac_plus_drift"]["auprc"]
diff_c_minus_b = a_frac_drift - a_frac
reading = "漂いは足しになる" if diff_c_minus_b >= 0.01 else "漂いは足しにならない"
print("drift reading: (c) full_frac+drift", round(a_frac_drift, 4),
      "- (b) full_frac", round(a_frac, 4), "= diff", round(diff_c_minus_b, 4),
      "->", reading, flush=True)
print("drift only gain:", round(results["drift_only"]["auprc"] - a_static, 4), flush=True)

# ---- タンパク質ごとの値 ----
per_protein = {}
for name in FEATURES:
    row = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        a_s = auprc_or_nan(yte[idx], preds["static_only"][idx])
        a_f = auprc_or_nan(yte[idx], preds["base_hist_D6"][idx])
        a_c = auprc_or_nan(yte[idx], preds[name][idx])
        g_full = a_f - a_s
        row[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc_static": a_s, "auprc_full6": a_f, "auprc_condition": a_c,
            "full6_gain": g_full,
            "gain": a_c - a_s,
            "ratio_vs_full6_gain": ((a_c - a_s) / g_full if g_full and a_c == a_c
                                    and a_s == a_s else float("nan")),
        }
    per_protein[name] = row

per_protein_diff = {}
for prot in TEST_P:
    idx = prot_idx[prot]
    a_b = auprc_or_nan(yte[idx], preds["full_frac"][idx])
    a_c = auprc_or_nan(yte[idx], preds["full_frac_plus_drift"][idx])
    per_protein_diff[prot] = {"auprc_full_frac": a_b, "auprc_full_frac_plus_drift": a_c,
                              "diff_c_minus_b": a_c - a_b}
print("per-protein diff (c)-(b):", {p: round(v["diff_c_minus_b"], 4)
                                    for p, v in per_protein_diff.items()}, flush=True)

out = {
    "label": "HB dissociation 50ns: does the fraction drift add to the full fraction?",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-178",
    "horizon_ns": 50,
    "bound_A": BOUND,
    "conditions_definition": {
        "base": "Material D conditions (K46 dissociation cache): train 1bgf_A/1ifg_A/1j8r_A/2b0a_A, "
                "test 1mix_A+2qt4_A pooled, origin frame 10 (1 ns), history frames 0-10 "
                "(100 ps spacing), HistGBM (120 iters, lr 0.075, 31 leaves, "
                "min_samples_leaf 15, L2 0.02, seed 42) refit per condition, "
                "7 static features.",
        "fraction": "fraction of frames with pair distance <= 3.5 A "
                    "(boundary from K71; same as formation) within the given frame range, "
                    "divided by the number of frames in that range "
                    "(first/second half: 5, full: 11)",
        "first_half": {"frames": FIRST_FRAMES, "span_ns": "0.0-0.4",
                       "feature": "fraction of frames [0..4] with distance <= 3.5 A"},
        "second_half": {"frames": SECOND_FRAMES, "span_ns": "0.6-1.0",
                        "feature": "fraction of frames [6..10] with distance <= 3.5 A"},
        "note": "Frame 5 (0.5 ns) belongs to neither half by design.",
        "both_halves_frac": "static + first-half fraction + second-half fraction "
                            "(2 features; K2-110 measured it at 50 ns: 0.4867)",
        "full_frac": "static + fraction over frames [0..10] (K2-110 measured it at 50 ns: 0.4827)",
        "drift": "first-half fraction minus second-half fraction "
                 "(fraction drift; K2-15's question in fraction form)",
        "drift_only": "static + drift only",
        "full_frac_plus_drift": "static + full fraction + drift",
        "base_hist_D6": "static + 6 statistics of the full 11-frame history (all-in reference)",
    },
    "base_reproduction": {
        "expected_from_K46_K2_110_50ns": {"static_only_auprc": 0.4088, "base_hist_D6_auprc": 0.5231,
                                    "full6_gain": 0.1142,
                                    "first_half_frac_auprc": 0.4929,
                                    "second_half_frac_auprc": 0.4423,
                                    "full_frac_auprc": 0.4827,
                                    "test_n": 2080, "test_positives": 362},
        "observed": {"static_only_auprc": a_static, "base_hist_D6_auprc": a_full6,
                     "full6_gain": full_gain,
                     "first_half_frac_auprc": results["first_half_frac"]["auprc"],
                     "second_half_frac_auprc": results["second_half_frac"]["auprc"],
                     "full_frac_auprc": a_fullfrac},
        "match": "all within 5e-4 (identical to 4 decimal places)",
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "n_test_positives": int(yte.sum()),
    },
    "metrics_50ns": results,
    "gains": gains,
    "drift_definition": "drift = first-half fraction [0..4] minus second-half fraction "
                        "[6..10], fraction = frames with pair distance <= 3.5 A",
    "reading_rule_fixed_in_advance": {
        "rule": "(c) full_frac+drift minus (b) full_frac >= +0.01 AUPRC "
                "-> 'drift adds'; otherwise 'drift does not add' (K2-15's rule).",
        "auprc_full_frac": a_frac,
        "auprc_full_frac_plus_drift": a_frac_drift,
        "diff_c_minus_b_auprc": diff_c_minus_b,
        "reading": reading,
        "per_protein_diff_c_minus_b": per_protein_diff,
    },
    "per_protein": per_protein,
}
(OUT / "results.json").write_text(json.dumps(clean_nan(out), indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auprc", "auc", "brier", "logloss",
                "gain_vs_static", "ratio_vs_full6_gain", "ratio_vs_fullfrac_gain"])
    for name in FEATURES:
        m, g = results[name], gains[name]
        w.writerow([name, m["n"], m["positives"], m["auprc"], m["auc"], m["brier"],
                    m["logloss"], g["gain"], g["ratio_vs_full6_gain"],
                    g["ratio_vs_fullfrac_gain"]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    y_50ns=yte,
    **{f"p_{name}": preds[name] for name in FEATURES},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
