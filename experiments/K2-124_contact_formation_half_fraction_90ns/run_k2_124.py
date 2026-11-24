"""K2-124: 90 ns先で「割合」の前半と後半を測る
（資料Dの条件、接触の形成、90 ns先）.

K2-14 の問い（割合の前半と後半）では、接触の形成の割合の前半・後半は
まだどの予測幅でも測っていない。6 個の統計量の形では、90 ns先
（K2-28）は前半 0.826・後半 0.649 と、どちらにも当てはまらなかった。
ここでは同じ問いを割合だけを使って 90 ns先で測る。

条件（資料Dの条件、K2-01・K2-05 と同じ土台）:
- 学習 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K13 のキャッシュ（~/workspace/kinescope/work/k13_cache/、
  候補ごとの最小重原子間距離の系列）を再利用する。履歴は (N, 11) で
  保存されているため、読み込み時に転置しない（K2-102・K2-104 と同じ）。
- 予測幅は 90 ns先（horizon インデックス 5、HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）。

比べる条件（静的特徴5個に、つぎのどれかを足す）:
- static_only:       静的特徴だけ（土台の照合用）
- base_hist_D6:      静的＋履歴 6 統計量（全部入り、土台の照合用）
- first_half_frac:   静的＋前半（フレーム [0..4]、0〜0.4 ns）の 4.5 Å以下割合
- second_half_frac:  静的＋後半（フレーム [6..10]、0.6〜1.0 ns）の 4.5 Å以下割合
- both_halves_frac:  静的＋前半の割合と後半の割合の 2 つ
- full_frac:         静的＋全部（フレーム [0..10]）の 4.5 Å以下割合

割合の定義: 境界（4.5 Å、接触の閾値。6 個の統計量のうちの
「4.5 Å 以下の割合」と同じ定義）以下だったフレームの数 ÷
その区間のフレーム数（前半・後半は 5、全部は 11）。
フレーム 5（0.5 ns）は前半にも後半にも入らない設計（K2-01・K2-05 と同じ）。

利得と割合:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。
- 割合は、全部入りの利得（base_hist_D6 の利得、K13 の 90 ns先の記録で
  +0.1850）に対する割合を主とし、あわせて全部の割合（full_frac）の
  利得に対する割合も記録する（読み方の分母の確認のため）。
  土台の再現は K13・K27（90 ns先：静的 0.2872 / 全部入り 0.4722 / 利得
  +0.1850、テスト 7,384件・陽性 257件）と照らす。全部の割合だけは
  K27 の記録（only_frac45 0.4198）とも照らす。

読み方（結果を見る前に固定、K2-01 と同じ）:
- 後半の割合（対 全部入りの利得）が 8 割以上で、前半が 5 割以下なら
  「今の状態が効いている」。
- 前半も後半も 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する（無理に結論を付けない）。

土台の再現の期待値（K13・K27 の 90 ns先の記録）:
- 静的のみ AUPRC 0.2872、全部入り 0.4722、利得 +0.1850、テスト 7,384件・陽性 257件。
- 全部の割合だけは AUPRC 0.4198（K27 only_frac45）。
"""
import csv
import json
import math
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
HORIZON_IDX = 5  # 90 ns先（HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）
BOUND = 4.5  # 割合の境界（接触の閾値。6 個の統計量の「4.5 Å 以下の割合」と同じ）

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
    # hist は (N, 11) で保存されているので転置しない（K2-102・K2-104 と同じ）
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
Xtr, Xte = cat(train, "X").astype(np.float64), cat(test, "X").astype(np.float64)
Htr, Hte = cat(train, "hist").astype(np.float64), cat(test, "hist").astype(np.float64)
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 90ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 15926 and Xte.shape[0] == 7384, (Xtr.shape, Xte.shape)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

# ---- 条件ごとの特徴量 ----
f6tr, f6te = hist6(Htr), hist6(Hte)
FEATURES = {
    "static_only": None,
    "base_hist_D6": (f6tr, f6te),
    "first_half_frac": (frac_cols(Htr, FIRST_FRAMES), frac_cols(Hte, FIRST_FRAMES)),
    "second_half_frac": (frac_cols(Htr, SECOND_FRAMES), frac_cols(Hte, SECOND_FRAMES)),
    "both_halves_frac": (np.column_stack([frac_cols(Htr, FIRST_FRAMES),
                                          frac_cols(Htr, SECOND_FRAMES)]),
                         np.column_stack([frac_cols(Hte, FIRST_FRAMES),
                                          frac_cols(Hte, SECOND_FRAMES)])),
    "full_frac": (frac_cols(Htr, FULL_FRAMES), frac_cols(Hte, FULL_FRAMES)),
}

# 前提の確認：全部の割合が 6 統計量の 6 列目（4.5 Å 以下割合）と一致すること
d_tr = float(np.abs(frac_cols(Htr, FULL_FRAMES)[:, 0] - hist6(Htr)[:, 5]).max())
d_te = float(np.abs(frac_cols(Hte, FULL_FRAMES)[:, 0] - hist6(Hte)[:, 5]).max())
print(f"full frac vs hist6 col 5: train max diff {d_tr:.2e}, test {d_te:.2e}", flush=True)
assert d_tr < 1e-4 and d_te < 1e-4

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

# ---- 土台の再現の確認（K13・K27 の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.1850):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.2872) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.4722) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.1850) < 5e-4, full_gain
assert results["static_only"]["positives"] == 257, results["static_only"]
a_fullfrac = results["full_frac"]["auprc"]
assert abs(a_fullfrac - 0.4198) < 5e-4, results["full_frac"]
print("base reproduction OK (static 0.2872 / full6 0.4722 / gain +0.1850 / "
      "full-frac 0.4198)", flush=True)

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

r_first = gains["first_half_frac"]["ratio_vs_full6_gain"]
r_second = gains["second_half_frac"]["ratio_vs_full6_gain"]
if r_second >= 0.8 and r_first <= 0.5:
    verdict = "今の状態が効いている"
elif r_first >= 0.8 and r_second >= 0.8:
    verdict = "続いている状態が効いている"
else:
    verdict = "どちらでもない（数値をそのまま記録）"
print("verdict:", verdict, flush=True)

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

# ---- タンパク質ごとの形の数え上げ（K2-07 と同じ） ----
shapes = {"now_state": 0, "continuing": 0, "other": 0, "excluded_gain_le_0.01": 0}
for prot in TEST_P:
    row_f = per_protein["first_half_frac"][prot]
    row_s = per_protein["second_half_frac"][prot]
    g_full = row_f["full6_gain"]
    if abs(g_full) <= 0.01:
        shapes["excluded_gain_le_0.01"] += 1
        continue
    f1 = row_f["ratio_vs_full6_gain"]
    f2 = row_s["ratio_vs_full6_gain"]
    if f2 >= 0.8 and f1 <= 0.5:
        shapes["now_state"] += 1
    elif f1 >= 0.8 and f2 >= 0.8:
        shapes["continuing"] += 1
    else:
        shapes["other"] += 1
print("per-protein shapes:", shapes, flush=True)

out = {
    "label": "Contact formation 90ns: guidance of first-half fraction vs second-half fraction of the history",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-124",
    "horizon_ns": 90,
    "bound_A": BOUND,
    "conditions_definition": {
        "base": "Material D conditions: train 1bgf_A/1ifg_A/1j8r_A/2b0a_A, "
                "test 1mix_A+2qt4_A pooled, origin frame 10 (1 ns), history frames 0-10 "
                "(100 ps spacing), HistGBM (120 iters, lr 0.075, 31 leaves, "
                "min_samples_leaf 15, L2 0.02, seed 42) refit per condition, "
                "5 static features. Contact formation, k13_cache.",
        "fraction": "fraction of frames with pair min heavy-atom distance <= 4.5 A "
                    "(contact threshold; same definition as the '4.5 A below' "
                    "statistic in the 6 statistics) within the given frame range, "
                    "divided by the number of frames in that range "
                    "(first/second half: 5, full: 11)",
        "first_half": {"frames": FIRST_FRAMES, "span_ns": "0.0-0.4",
                       "feature": "fraction of frames [0..4] with distance <= 4.5 A"},
        "second_half": {"frames": SECOND_FRAMES, "span_ns": "0.6-1.0",
                        "feature": "fraction of frames [6..10] with distance <= 4.5 A"},
        "note": "Frame 5 (0.5 ns) belongs to neither half by design.",
        "both_halves_frac": "static + first-half fraction and second-half fraction (2 features)",
        "full_frac": "static + fraction over frames [0..10] (K27 only_frac45 0.4198)",
        "base_hist_D6": "static + 6 statistics of the full 11-frame history (all-in reference)",
    },
    "base_reproduction": {
        "expected_from_K13_K27_90ns": {"static_only_auprc": 0.2872, "base_hist_D6_auprc": 0.4722,
                              "full6_gain": 0.1850, "full_frac_auprc": 0.4198,
                              "test_n": 7384, "test_positives": 257},
        "note": "Base reproduction is checked against K13/K27 (90ns); full-fraction is checked against K27 only_frac45.",
        "observed": {"static_only_auprc": a_static, "base_hist_D6_auprc": a_full6,
                     "full6_gain": full_gain, "full_frac_auprc": a_fullfrac},
        "match": "all within 5e-4 (identical to 4 decimal places)",
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "n_test_positives": int(yte.sum()),
    },
    "metrics_90ns": results,
    "gains": gains,
    "reading_rule_fixed_in_advance": {
        "rule": "second-half ratio (vs full6 gain) >= 0.8 and first-half ratio <= 0.5 "
                "-> 'current state carries it'; both ratios >= 0.8 -> 'persistent state "
                "carries it'; otherwise record the numbers as they are.",
        "first_half_ratio_vs_full6": r_first,
        "second_half_ratio_vs_full6": r_second,
        "verdict": verdict,
    },
    "per_protein": per_protein,
    "per_protein_shapes": shapes,
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
    y_90ns=yte,
    **{f"p_{name}": preds[name] for name in FEATURES},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
