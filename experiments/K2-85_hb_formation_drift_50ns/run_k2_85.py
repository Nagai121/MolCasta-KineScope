"""K2-85: 履歴の「動き」自体が指導力を担うか（資料Dの条件、水素結合の形成、50 ns先）.

K2 系列（指導力の正体は何かを探る系列）。K2-01 で、前半と後半の両方を
入れると割合 1.074 で全部の平均（1.042）も全部入りも上回った。
前半と後半を分けて入れたときの上積みは、前半と後半の平均の差
（漂い）の情報が入ったからかもしれない。ここでは漂いを 1 つの
特徴量として切り出し、全部の平均の上に足しになるかを測る。
水素結合の形成では、1 ns先（K2-06、(c) − (b) = −0.0031）と 30 ns先
（K2-16、(c) − (b) = −0.0014）のどちらでも漂いは足しにならなかった。
ここでは同じ問いを 50 ns先で測る。

条件（資料Dの条件、K2-01 と同じ土台）:
- 学習 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K01 のキャッシュ（~/workspace/kinescope/work/data_cache/、
  変換済み実軌道18本から作った候補ごとの距離系列）を再利用する。
  距離はフレームごとの箱を軸ごとに扱って計算済み。

漂いの定義（結果を見る前に固定）:
- 漂い = 前半の平均 − 後半の平均。
  前半はフレーム [0..4]（0〜0.4 ns）の距離の平均、
  後半はフレーム [6..10]（0.6〜1.0 ns）の距離の平均。
  フレーム 5（0.5 ns）は前半にも後半にも入らない（K2-01 と同じ設計）。
- 漂いが正なら「前半のほうが距離が大きい」（履歴のなかで距離が縮む
  方向に動いた）、負なら「後半のほうが距離が大きい」（距離が開く
  方向に動いた）という意味。

比べる条件（静的特徴7個に、つぎのどれかを足す）:
- static_only:         静的特徴だけ（土台の照合用）
- base_hist_D6:        静的＋履歴 6 統計量（全部入り、土台の照合用）
- drift_only:          静的＋漂いだけ (a)
- full_mean:           静的＋全部（フレーム [0..10]）の距離の平均だけ (b)
- full_mean_plus_drift: 静的＋全部の平均＋漂い (c)
- both_halves_mean:    静的＋前半の平均＋後半の平均の両方（K2-45 の
                       50 ns先の記録との照合用。K2-45 では前半の平均だけ
                       AUPRC 0.4133。両方入れた形は K2-45 にはないため
                       照合は前半の平均だけの値で行う）

利得と割合:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。
- 割合は、全部入りの利得（base_hist_D6 の利得、K2-45 の記録で
  +0.0739）に対する割合を主とし、あわせて全部の平均（full_mean）
  の利得に対する割合も記録する。

読み方（結果を見る前に固定）:
- (c) が (b) を AUPRC で +0.01 以上上回れば「漂いは足しになる」、
  未満なら「足しにならない」。
- (a) の利得（漂いだけの利得）も記録し、全部入りの利得に対する
  割合もあわせて記録する。
- K2-01 の「前半と後半の両方」（割合 1.074）と (c) の結果の関係も
  記録する。(c) は (b) の上に漂いを足した形で、K2-01 の「前半と
  後半の両方」は前半と後半の平均を 2 つ入れた形。2 つの条件の
  AUPRC の差も記録する。

土台の再現の期待値（G74 / G136 / K2-45 の記録、50 ns先）:
- 静的のみ AUPRC 0.3342、全部入り 0.4081、利得 +0.0739。
- 全部の平均は AUPRC 0.4216、利得 +0.0874。
- 前半の平均だけは AUPRC 0.4133、利得 +0.0791。
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

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 4  # 50 ns先

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


def mean_cols(series, frames):
    return series[:, frames].mean(1)[:, None].astype(np.float32)


def drift_cols(series):
    """漂い = 前半の平均 − 後半の平均。フレーム 5 はどちらにも入らない。"""
    first = series[:, FIRST_FRAMES].mean(1)
    second = series[:, SECOND_FRAMES].mean(1)
    return (first - second)[:, None].astype(np.float32)


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


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 50ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 9610 and Xte.shape[0] == 4972, (Xtr.shape, Xte.shape)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

# ---- 条件ごとの特徴量 ----
f6tr, f6te = hist6(Htr), hist6(Hte)
fm_tr, fm_te = mean_cols(Htr, FULL_FRAMES), mean_cols(Hte, FULL_FRAMES)
dr_tr, dr_te = drift_cols(Htr), drift_cols(Hte)
fh_tr, fh_te = mean_cols(Htr, FIRST_FRAMES), mean_cols(Hte, FIRST_FRAMES)
sh_tr, sh_te = mean_cols(Htr, SECOND_FRAMES), mean_cols(Hte, SECOND_FRAMES)
FEATURES = {
    "static_only": None,
    "base_hist_D6": (f6tr, f6te),
    "drift_only": (dr_tr, dr_te),
    "full_mean": (fm_tr, fm_te),
    "full_mean_plus_drift": (
        np.concatenate([fm_tr, dr_tr], axis=1),
        np.concatenate([fm_te, dr_te], axis=1)),
    "both_halves_mean": (
        np.concatenate([fh_tr, sh_tr], axis=1),
        np.concatenate([fh_te, sh_te], axis=1)),
    "front_mean": (fh_tr, fh_te),
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

# ---- 土台の再現の確認（K2-45 の 50 ns先の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.0739):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.3342) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.4081) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.0739) < 5e-4, full_gain
# 全部の平均は K2-45 の 50 ns先の記録と同じ条件
a_fullmean = results["full_mean"]["auprc"]
assert abs(a_fullmean - 0.4216) < 5e-4, results["full_mean"]
# 前半の平均だけは K2-45 の 50 ns先の記録と同じ条件
a_front = results["front_mean"]["auprc"]
assert abs(a_front - 0.4133) < 5e-4, results["front_mean"]
a_both = results["both_halves_mean"]["auprc"]
print("base reproduction OK (static 0.3342 / full6 0.4081 / gain +0.0739 / "
      "full mean 0.4216 / front-half mean 0.4133)", flush=True)

# ---- 利得と割合 ----
fullmean_gain = a_fullmean - a_static
gains = {}
for name in FEATURES:
    g = results[name]["auprc"] - a_static
    gains[name] = {
        "gain": float(g),
        "ratio_vs_full6_gain": float(g / full_gain),
        "ratio_vs_fullmean_gain": float(g / fullmean_gain),
    }
    print(name, "gain", round(g, 4),
          "ratio vs full6", round(g / full_gain, 4),
          "ratio vs full-mean", round(g / fullmean_gain, 4), flush=True)

# ---- 読み方（結果を見る前に固定） ----
diff_c_b = results["full_mean_plus_drift"]["auprc"] - a_fullmean
if diff_c_b >= 0.01:
    verdict = "漂いは足しになる"
else:
    verdict = "漂いは足しにならない"
print("AUPRC (c) - (b):", round(diff_c_b, 4), "verdict:", verdict, flush=True)
diff_c_both = results["full_mean_plus_drift"]["auprc"] - a_both
print("AUPRC (c) - both_halves:", round(diff_c_both, 4), flush=True)

# ---- タンパク質ごとの値 ----
per_protein = {}
for name in FEATURES:
    row = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        a_s = auprc(yte[idx], preds["static_only"][idx])
        a_f = auprc(yte[idx], preds["base_hist_D6"][idx])
        a_c = auprc(yte[idx], preds[name][idx])
        g_full = a_f - a_s
        row[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc_static": a_s, "auprc_full6": a_f, "auprc_condition": a_c,
            "full6_gain": g_full,
            "gain": a_c - a_s,
            "ratio_vs_full6_gain": (a_c - a_s) / g_full if g_full else float("nan"),
        }
    per_protein[name] = row

out = {
    "label": "HB formation 50ns: does the drift (first-half mean minus second-half "
             "mean) add on top of the full mean?",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-85",
    "horizon_ns": 50,
    "conditions_definition": {
        "base": "Material D conditions: train 1bgf_A/1ifg_A/1j8r_A/2b0a_A, "
                "test 1mix_A+2qt4_A pooled, origin frame 10 (1 ns), history frames 0-10 "
                "(100 ps spacing), HistGBM (120 iters, lr 0.075, 31 leaves, "
                "min_samples_leaf 15, L2 0.02, seed 42) refit per condition, "
                "7 static features.",
        "drift": {"definition": "first-half mean minus second-half mean "
                                "(mean distance over frames [0..4] minus mean over "
                                "frames [6..10])",
                  "sign": "positive drift = distance larger in the first half "
                          "(pair moving closer during the history); negative drift = "
                          "distance larger in the second half (pair moving apart)",
                  "first_half": {"frames": FIRST_FRAMES, "span_ns": "0.0-0.4"},
                  "second_half": {"frames": SECOND_FRAMES, "span_ns": "0.6-1.0"},
                  "note": "Frame 5 (0.5 ns) belongs to neither half by design "
                          "(same as K2-01)."},
        "drift_only": "static + drift only (a)",
        "full_mean": "static + mean over frames [0..10] only (b)",
        "full_mean_plus_drift": "static + full mean + drift (c)",
        "both_halves_mean": "static + first-half mean + second-half mean "
                            "(same form as K2-01/K2-06 at 1 ns; no K2-45 reference "
                            "at 50 ns, recorded for comparison with (c))",
        "front_mean": "static + first-half mean only "
                      "(cross-check vs K2-45 at 50 ns, AUPRC 0.4133, ratio 1.070)",
        "base_hist_D6": "static + 6 statistics of the full 11-frame history "
                        "(all-in reference)",
    },
    "base_reproduction": {
        "expected_from_K2-45_50ns": {"static_only_auprc": 0.3342,
                                     "base_hist_D6_auprc": 0.4081,
                                     "full6_gain": 0.0739,
                                     "full_mean_auprc": 0.4216,
                                     "front_half_mean_auprc": 0.4133},
        "observed": {"static_only_auprc": a_static,
                     "base_hist_D6_auprc": a_full6,
                     "full6_gain": full_gain,
                     "full_mean_auprc": a_fullmean,
                     "front_half_mean_auprc": a_front,
                     "both_halves_auprc": a_both},
        "match": "all within 5e-4 (identical to 4 decimal places)",
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "n_test_positives": int(yte.sum()),
    },
    "metrics_50ns": results,
    "gains": gains,
    "reading_rule_fixed_in_advance": {
        "rule": "(c) full_mean_plus_drift beats (b) full_mean by AUPRC >= +0.01 "
                "-> 'drift adds on top of the full mean'; otherwise 'drift does not add'.",
        "auprc_c_minus_b": float(diff_c_b),
        "auprc_c_minus_both_halves": float(diff_c_both),
        "verdict": verdict,
    },
    "per_protein": per_protein,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auprc", "auc", "brier", "logloss",
                "gain_vs_static", "ratio_vs_full6_gain", "ratio_vs_fullmean_gain"])
    for name in FEATURES:
        m, g = results[name], gains[name]
        w.writerow([name, m["n"], m["positives"], m["auprc"], m["auc"], m["brier"],
                    m["logloss"], g["gain"], g["ratio_vs_full6_gain"],
                    g["ratio_vs_fullmean_gain"]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    y_50ns=yte,
    **{f"p_{name}": preds[name] for name in FEATURES},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
