"""K2-167: K2-15 の問い（漂い）を割合で測る
（資料Dの条件、接触の形成、5 ns先）.

K2-15 では、水素結合の解離の 1 ns先で、履歴の漂い（前半の平均 −
後半の平均）を全部の平均の上に足すと足しになった（(c) 0.5923 は (b)
0.5691 を +0.0232 上回る）。接触の形成では、平均の漂いは 1 ns先
（K2-34、+0.0367）、5 ns先（K2-95、+0.0282）、10 ns先（K2-90、
+0.0114）、30 ns先（K2-55、+0.0116）、50 ns先（K2-75、+0.0331）
では足しになったが、90 ns先（K2-71、+0.0087）では足しにならなかった。
ここでは同じ問いを割合の形で 5 ns先で測る。割合の漂いは 1 ns先
（K2-150、−0.0004）と 30 ns先（K2-154、+0.0073）と 90 ns先
（K2-161、−0.0013）では足しにならなかった。

割り当て文の食い違い（元の記録で確かめて直した）:
割り当て文は「接触の形成の 5 ns先の平均の漂いは K2-75 で測り終えて
おり、(c) − (b) = +0.0331」としていたが、K2-75 は 50 ns先の実験で
ある（experiments/K2-75_contact_formation_drift_50ns/results.json
の horizon_ns は 50、(c) − (b) = +0.0331）。5 ns先の平均の漂いは
K2-95 で、(c) − (b) = +0.0282 である（experiments/
K2-95_contact_formation_drift_5ns/results.json）。このスクリプトと
README では、正しい K2-95 の値と比較する。

漂いの定義（割合の形）:
- 割合 = ペアの最小重原子間距離が接触の閾値（4.5 Å）以下の時間の
  割合。6 個の統計量のうちの「4.5 Å 以下だった割合」と同じ定義。
- 漂い = 前半の割合 − 後半の割合。前半はフレーム [0..4]（0〜0.4 ns）
  の 4.5 Å以下割合、後半はフレーム [6..10]（0.6〜1.0 ns）の
  4.5 Å以下割合。フレーム 5（0.5 ns）は前半にも後半にも入らない
  （K2-01 と同じ）。
- 漂いの値の範囲は −1 から +1 である。漂いが正なら、前半のほうが
  接触の閾値以下だった時間の割合が多い（履歴のなかで距離が縮む
  方向に動いた）。

条件（資料Dの条件、K2-01・K2-37・K27 と同じ土台）:
- 学習 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K13 のキャッシュ（~/workspace/kinescope/work/k13_cache/、
  接触の形成の候補ごとの距離系列）を再利用する。距離はフレームごと
  の箱を軸ごとに扱って計算済み。
- 予測幅は 5 ns先（horizon インデックス 1、HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）。

比べる条件（静的特徴に、つぎのどれかを足す）:
- static_only:       静的特徴だけ（土台の照合用）
- base_hist_D6:      静的＋履歴 6 統計量（全部入り、土台の照合用）
- drift_only:        静的＋漂いだけ（前半の割合 − 後半の割合）
- frac_full:         静的＋全部（フレーム [0..10]）の 4.5 Å以下割合
- frac_full_plus_drift: 静的＋全部の割合＋漂い
- first_half_frac / second_half_frac / both_halves_frac: 照合用（K2-126）

割合の定義: 境界（4.5 Å、K27 と同じ）以下だったフレームの数 ÷
その区間のフレーム数（前半・後半は 5、全部は 11）。

利得と割合:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。
- 割合は、全部入りの利得（base_hist_D6 の利得、K2-126 の 5 ns先の
  記録で +0.2322）に対する割合を主とし、あわせて全部の割合
  （frac_full）の利得に対する割合も記録する（読み方の分母の確認のため）。

読み方（結果を見る前に固定、K2-15 と同じ）:
- (c) 全部の割合＋漂いが (b) 全部の割合だけを AUPRC で +0.01 以上
  上回れば「漂いは足しになる」、未満なら「足しにならない」。
- (a) 漂いだけの利得も記録し、全部入りの利得に対する割合もあわせて
  記録する。
- タンパク質ごとの値も記録する。テストは 2 タンパク質（1mix_A・2qt4_A）
  のため、タンパク質ごとの値は粗い。

土台の再現の期待値（K2-126 の 5 ns先の記録）:
- 静的のみ AUPRC 0.2870、全部入り 0.5192、利得 +0.2322、テスト 7,384件・陽性 214件。
- 全部の割合だけは AUPRC 0.5008（K2-126、K27 の only_frac45 と同じ）。
- 前半の割合だけ 0.5089、後半の割合だけ 0.4514、前半と後半の両方 0.4945（K2-126）。
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

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 1  # 5 ns先（HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）
BOUND = 4.5  # 割合の境界（K27 と同じ）

FIRST_FRAMES = list(range(0, 5))    # フレーム [0..4]、0〜0.4 ns
SECOND_FRAMES = list(range(6, 11))  # フレーム [6..10]、0.6〜1.0 ns
FULL_FRAMES = list(range(0, 11))    # フレーム [0..10]、0〜1.0 ns


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact.npz")
            d = {k: z[k] for k in z.files}
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
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


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 5ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 15926 and Xte.shape[0] == 7384, (Xtr.shape, Xte.shape)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

# ---- 条件ごとの特徴量 ----
f6tr, f6te = hist6(Htr), hist6(Hte)
dr_tr = (frac_cols(Htr, FIRST_FRAMES) - frac_cols(Htr, SECOND_FRAMES)).astype(np.float32)
dr_te = (frac_cols(Hte, FIRST_FRAMES) - frac_cols(Hte, SECOND_FRAMES)).astype(np.float32)
ff_tr, ff_te = frac_cols(Htr, FULL_FRAMES), frac_cols(Hte, FULL_FRAMES)
FEATURES = {
    "static_only": None,
    "base_hist_D6": (f6tr, f6te),
    "drift_only": (dr_tr, dr_te),
    "frac_full": (ff_tr, ff_te),
    "frac_full_plus_drift": (
        np.concatenate([ff_tr, dr_tr], axis=1),
        np.concatenate([ff_te, dr_te], axis=1)),
    "first_half_frac": (frac_cols(Htr, FIRST_FRAMES), frac_cols(Hte, FIRST_FRAMES)),
    "second_half_frac": (frac_cols(Htr, SECOND_FRAMES), frac_cols(Hte, SECOND_FRAMES)),
    "both_halves_frac": (np.column_stack([frac_cols(Htr, FIRST_FRAMES),
                                          frac_cols(Htr, SECOND_FRAMES)]),
                         np.column_stack([frac_cols(Hte, FIRST_FRAMES),
                                          frac_cols(Hte, SECOND_FRAMES)])),
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

# ---- 土台の再現の確認（K2-126 の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.2322):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.2870) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.5192) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.2322) < 5e-4, full_gain
assert results["static_only"]["positives"] == 214, results["static_only"]
# 全部の割合だけの 5 ns先の値は K2-126 の記録と照らす
a_fullfrac = results["frac_full"]["auprc"]
assert abs(a_fullfrac - 0.5008) < 5e-4, results["frac_full"]
assert abs(results["first_half_frac"]["auprc"] - 0.5089) < 5e-4, results["first_half_frac"]
assert abs(results["second_half_frac"]["auprc"] - 0.4514) < 5e-4, results["second_half_frac"]
assert abs(results["both_halves_frac"]["auprc"] - 0.4945) < 5e-4, results["both_halves_frac"]
print("base reproduction OK (static 0.2870 / full6 0.5192 / gain +0.2322 / full frac 0.5008)",
      flush=True)

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

drift_adds = results["frac_full_plus_drift"]["auprc"] - results["frac_full"]["auprc"]
verdict = "漂いは足しになる" if drift_adds >= 0.01 else "漂いは足しにならない"
diff_c_both = (results["frac_full_plus_drift"]["auprc"]
               - results["both_halves_frac"]["auprc"])
print("drift adds on top of full fraction (AUPRC diff):", round(drift_adds, 4),
      "verdict:", verdict,
      "diff (c)-(both halves):", round(diff_c_both, 4), flush=True)

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

# 漂いの足しのタンパク質ごとの差（frac_full_plus_drift − frac_full の AUPRC）
drift_diff_per_protein = {}
for prot in TEST_P:
    idx = prot_idx[prot]
    drift_diff_per_protein[prot] = float(
        auprc(yte[idx], preds["frac_full_plus_drift"][idx])
        - auprc(yte[idx], preds["frac_full"][idx]))

out = {
    "label": "Contact formation 5ns: does history fraction drift (first-half fraction minus second-half fraction) add on top of the full fraction?",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-167",
    "horizon_ns": 5,
    "bound_A": BOUND,
    "conditions_definition": {
        "base": "Material D conditions: train 1bgf_A/1ifg_A/1j8r_A/2b0a_A, "
                "test 1mix_A+2qt4_A pooled, origin frame 10 (1 ns), history frames 0-10 "
                "(100 ps spacing), HistGBM (120 iters, lr 0.075, 31 leaves, "
                "min_samples_leaf 15, L2 0.02, seed 42) refit per condition, "
                "contact formation (k13_cache).",
        "fraction": "fraction of frames with pair min heavy-atom distance <= 4.5 A "
                    "(boundary from K27) within the given frame range, "
                    "divided by the number of frames in that range "
                    "(first/second half: 5, full: 11)",
        "first_half": {"frames": FIRST_FRAMES, "span_ns": "0.0-0.4",
                       "feature": "fraction of frames [0..4] with min distance <= 4.5 A"},
        "second_half": {"frames": SECOND_FRAMES, "span_ns": "0.6-1.0",
                        "feature": "fraction of frames [6..10] with min distance <= 4.5 A"},
        "note": "Frame 5 (0.5 ns) belongs to neither half by design.",
        "drift_only": "static + drift only (fraction over frames [0..4] minus fraction over "
                      "frames [6..10])",
        "frac_full": "static + fraction over frames [0..10] (K2-126 reference)",
        "frac_full_plus_drift": "static + full fraction + drift",
        "both_halves_frac": "static + first-half fraction and second-half fraction (2 features)",
        "base_hist_D6": "static + 6 statistics of the full 11-frame history (all-in reference)",
    },
    "base_reproduction": {
        "expected_from_K2-126_5ns": {"static_only_auprc": 0.2870, "base_hist_D6_auprc": 0.5192,
                              "full6_gain": 0.2322, "full_frac_auprc": 0.5008,
                              "first_half_frac_auprc": 0.5089,
                              "second_half_frac_auprc": 0.4514,
                              "both_halves_frac_auprc": 0.4945,
                              "test_n": 7384, "test_positives": 214},
        "note": "Base reproduction is checked against K2-126 (5ns), which matches the K27 record to 4 decimal places.",
        "observed": {"static_only_auprc": a_static, "base_hist_D6_auprc": a_full6,
                     "full6_gain": full_gain, "full_frac_auprc": a_fullfrac},
        "match": "all within 5e-4 (identical to 4 decimal places)",
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "n_test_positives": int(yte.sum()),
    },
    "metrics_5ns": results,
    "gains": gains,
    "drift_adds": float(drift_adds),
    "diff_c_minus_both_halves_auprc": float(diff_c_both),
    "reading_rule_fixed_in_advance": {
        "rule": "(c) frac_full_plus_drift beats (b) frac_full by AUPRC >= +0.01 "
                "-> 'drift adds'; otherwise 'drift does not add'.",
        "drift_adds_auprc": float(drift_adds),
        "verdict": verdict,
    },
    "per_protein": per_protein,
    "drift_diff_per_protein": drift_diff_per_protein,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

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
    y_5ns=yte,
    **{f"p_{name}": preds[name] for name in FEATURES},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
