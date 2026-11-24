"""K2-144: 10 ns先で「割合」の前半と後半を測る
（K15 の条件、塩橋の解離、10 ns先）.

K2-14 の問い（割合の前半と後半）では、塩橋の解離の割合の前半と後半
は 1 ns先（K2-138、前半 0.716・後半 0.818）と 5 ns先（K2-142、
前半 0.817・後半 0.642）と 90 ns先（K2-143、前半 0.557・
後半 0.966）で測り終えた。ここでは同じ問いを割合だけの形で
10 ns先で測る。

条件（K15 の条件）:
- 学習 58 タンパク質、テスト 30 タンパク質（K15 の split.json に従う）。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
 - データは K53 のキャッシュ（~/workspace/kinescope/work/k53_cache/、
  塩橋の解離の候補ごとの荷電重原子間の最小距離の系列）を再利用する。
  距離はフレームごとの箱を軸ごとに扱って計算済み。履歴は (n, 11) で
  保存されているため、転置はしない（K46 のキャッシュとは違う）。
  学習・テストのタンパク質の分け方は K15 の split.json（学習 58・
  テスト 30 タンパク質）に従う。
- 予測幅は 10 ns先（horizon インデックス 2、HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）。

比べる条件（静的特徴5個に、つぎのどれかを足す）:
- static_only:       静的特徴だけ（土台の照合用）
- base_hist_D6:      静的＋履歴 6 統計量（全部入り、土台の照合用）
 - first_half_frac:   静的＋前半（フレーム [0..4]、0〜0.4 ns）の 4.0 Å以下割合
- second_half_frac:  静的＋後半（フレーム [6..10]、0.6〜1.0 ns）の 4.0 Å以下割合
- both_halves_frac:  静的＋前半の割合と後半の割合の 2 つ
- full_frac:         静的＋全部（フレーム [0..10]）の 4.0 Å以下割合

割合の定義: 境界（4.0 Å、6 個の統計量のうちの「4.0 Å 以下だった割合」と
同じ定義）以下だったフレームの数 ÷ その区間のフレーム数（前半・後半
は 5、全部は 11）。ペアの距離は最小重原子間距離の系列である。
フレーム 5（0.5 ns）は前半にも後半にも入らない設計（K2-01・K2-05 と同じ）。

利得と割合:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。
 - 割合は、全部入りの利得（base_hist_D6 の利得、塩橋の解離の 10 ns先
  の記録で +0.1781）に対する割合を主とし、あわせて全部の割合
  （full_frac）の利得に対する割合も記録する（読み方の分母の確認のため）。
  土台の再現は塩橋の解離の 10 ns先の記録（静的 0.3572 / 全部入り
  0.5353 / 利得 +0.1781、テスト 360件・陽性 131件）と照らす。

読み方（結果を見る前に固定、K2-01 と同じ）:
- 後半の割合（対 全部入りの利得）が 8 割以上で、前半が 5 割以下なら
  「今の状態が効いている」。
- 前半も後半も 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する（無理に結論を付けない）。

土台の再現の期待値（塩橋の解離の 10 ns先の記録）:
- 静的のみ AUPRC 0.3572、全部入り 0.5353、利得 +0.1781、テスト 360件・陽性 131件。
- 全部の割合だけの 10 ns先の値は既存の記録にないため、照合はしない。
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

CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
OUT = Path(__file__).resolve().parent
split = json.loads((K15_DIR / "split.json").read_text())
TRAIN_P = ORIG_TRAIN + tuple(split["train_additions"])
TEST_P = ORIG_TEST + tuple(split["test_additions"])
HORIZON_IDX = 2  # 10 ns先（HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）
BOUND = 4.0  # 割合の境界（6 個の統計量の「4.0 Å 以下だった割合」と同じ）

FIRST_FRAMES = list(range(0, 5))    # フレーム [0..4]、0〜0.4 ns
SECOND_FRAMES = list(range(6, 11))  # フレーム [6..10]、0.6〜1.0 ns
FULL_FRAMES = list(range(0, 11))    # フレーム [0..10]、0〜1.0 ns


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt_diss.npz")
            d = {k: z[k] for k in z.files}
            d["protein"] = np.full(len(d["y"]), p)
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    # K53 のキャッシュの履歴は (n, 11) で保存されているため、転置はしない
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
print("train", Xtr.shape, "test", Xte.shape, "test pos 10ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 735 and Xte.shape[0] == 360, (Xtr.shape, Xte.shape)

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

# ---- 土台の再現の確認（塩橋の解離の 10 ns先の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.1781):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.3572) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.5353) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.1781) < 5e-4, full_gain
assert results["static_only"]["positives"] == 131, results["static_only"]
# 全部の割合だけの 10 ns先の値は既存の記録にないため、照合はしない
a_fullfrac = results["full_frac"]["auprc"]
print("base reproduction OK (static 0.3572 / full6 0.5353 / gain +0.1781)",
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
shapes = {"now_state": 0, "continuing": 0, "other": 0,
          "excluded_gain_le_0.01": 0}
for prot in TEST_P:
    gf = per_protein["base_hist_D6"][prot]["full6_gain"]
    if gf <= 0.01:
        shapes["excluded_gain_le_0.01"] += 1
        continue
    rf = per_protein["first_half_frac"][prot]["ratio_vs_full6_gain"]
    rs = per_protein["second_half_frac"][prot]["ratio_vs_full6_gain"]
    if rs >= 0.8 and rf <= 0.5:
        shapes["now_state"] += 1
    elif rf >= 0.8 and rs >= 0.8:
        shapes["continuing"] += 1
    else:
        shapes["other"] += 1
print("per-protein shapes:", shapes, flush=True)

out = {
    "label": "Salt dissociation 10ns: guidance of first-half fraction vs second-half fraction of the history",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-144",
    "horizon_ns": 10,
    "bound_A": BOUND,
    "conditions_definition": {
        "base": "K15 conditions: train 58 proteins, test 30 proteins (split.json), "
                "origin frame 10 (1 ns), history frames 0-10 "
                "(100 ps spacing), HistGBM (120 iters, lr 0.075, 31 leaves, "
                "min_samples_leaf 15, L2 0.02, seed 42) refit per condition, "
                "5 static features. Salt dissociation, K53 cache.",
        "fraction": "fraction of frames with pair min charged heavy-atom distance <= 4.0 A "
                    "(same definition as the 'fraction <= 4.0 A' of the 6 statistics) "
                    "within the given frame range, "
                    "divided by the number of frames in that range "
                    "(first/second half: 5, full: 11)",
        "first_half": {"frames": FIRST_FRAMES, "span_ns": "0.0-0.4",
                       "feature": "fraction of frames [0..4] with min distance <= 4.0 A"},
        "second_half": {"frames": SECOND_FRAMES, "span_ns": "0.6-1.0",
                        "feature": "fraction of frames [6..10] with min distance <= 4.0 A"},
        "note": "Frame 5 (0.5 ns) belongs to neither half by design.",
        "both_halves_frac": "static + first-half fraction and second-half fraction (2 features)",
        "full_frac": "static + fraction over frames [0..10] (no prior 10ns salt-dissociation value)",
        "base_hist_D6": "static + 6 statistics of the full 11-frame history (all-in reference)",
    },
    "base_reproduction": {
        "expected_from_K53_10ns": {"static_only_auprc": 0.3572, "base_hist_D6_auprc": 0.5353,
                              "full6_gain": 0.1781, "test_n": 360, "test_positives": 131},
        "note": "Base reproduction is checked against the salt-dissociation 10ns record. There is no prior full-fraction 10ns salt-dissociation value.",
        "observed": {"static_only_auprc": a_static, "base_hist_D6_auprc": a_full6,
                     "full6_gain": full_gain, "full_frac_auprc": a_fullfrac},
        "match": "all within 5e-4 (identical to 4 decimal places)",
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "n_test_positives": int(yte.sum()),
    },
    "metrics_10ns": results,
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
    y_10ns=yte,
    **{f"p_{name}": preds[name] for name in FEATURES},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
