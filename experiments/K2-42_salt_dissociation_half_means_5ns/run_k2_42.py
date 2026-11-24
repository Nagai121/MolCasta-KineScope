"""K2-42: 履歴の前半の平均と後半の平均で、指導力はどう違うか
（K15 の広げた条件、塩橋の解離、5 ns先）.

K2 系列（指導力の正体は何かを探る系列）の最初の実験。K 系列が
「指導力はどこにあるか」の切り分けだったのに対して、K2 は
「短い切り取りの平均が何の影なのか」を問う。L01 で 0.6 ns の段が
見えたことと、切り取りを長くすると回復率が下がる条件があったこと
（G 系列）を受けて、履歴の中の「いつの平均」が指導力を担っているかを
前半と後半に分けて測る。

条件（K15 の広げた条件、K53 と同じ土台）:
- 学習 58 タンパク質（元の 4 ＋追加 54、735件）、テスト 30 タンパク質
  （元の 2 ＋追加 28、360件）。タンパク質の分け方は K15 の split.json
  の凍結した集合。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K53 のキャッシュ（~/workspace/kinescope/work/k53_cache/、
  変換済み実軌道 88 タンパク質×3 レプリカから作った解離の候補ごとの
  距離系列）を再利用する。距離は荷電重原子ペアの最小値で、境界は
  4.0 Å。候補は起点で塩橋ができているペア。

比べる条件（静的特徴5個に、つぎのどれかを足す）:
- static_only:      静的特徴だけ（土台の照合用）
- base_hist_D6:     静的＋履歴 6 統計量（全部入り、土台の照合用）
- first_half_mean:  静的＋前半（フレーム [0..4]、0〜0.4 ns）の距離の平均
- second_half_mean: 静的＋後半（フレーム [6..10]、0.6〜1.0 ns）の距離の平均
- both_halves_mean: 静的＋前半の平均と後半の平均の両方
- full_mean:        静的＋全部（フレーム [0..10]）の距離の平均

フレーム 5（0.5 ns）は前半にも後半にも入らない設計。

利得と割合:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。AUPRC の差もあわせて
  記録する。
- 割合は、全部入りの利得（base_hist_D6 の利得、K53 の記録で +0.1139）に
  対する割合を主とし、あわせて全部の平均（full_mean）の利得に対する
  割合も記録する（読み方の分母の確認のため）。
- 塩橋の解離は土台の利得が小さく（5 ns先で +0.1139）、テストも
  360件と小さいため、回復率の読みには留保がつく。タンパク質ごとの
  候補数は平均で約 12 件と少なく、タンパク質ごとの AUPRC はぶれやすい。
  陽性が 0 件のタンパク質では AUPRC が決まらないため、割合は記録しない。

読み方（結果を見る前に固定）:
- 後半の割合（対 全部入りの利得）が 8 割以上で、前半が 5 割以下なら
  「今の状態が効いている」。
- 前半も後半も 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する（無理に結論を付けない）。
- 「前半と後半の両方」の利得も、全部入りの利得と比べて記録する。

土台の再現の期待値（K53 の記録、5 ns先）:
- 静的のみ AUPRC 0.4641、全部入り 0.5780、利得 +0.1139。
- 全部の平均は K53 の mean_only と同じ条件で、AUPRC 0.5794 のはず。
"""
import csv
import json
import math
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
OUT = Path(__file__).resolve().parent
SPLIT = json.loads(Path(
    "/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json"
).read_text())
TRAIN_P = list(SPLIT["orig_train"]) + list(SPLIT["train_additions"])
TEST_P = list(SPLIT["orig_test"]) + list(SPLIT["test_additions"])
HORIZON_IDX = 1  # 5 ns先
BOUND = 4.0

FIRST_FRAMES = list(range(0, 5))    # フレーム [0..4]、0〜0.4 ns
SECOND_FRAMES = list(range(6, 11))  # フレーム [6..10]、0.6〜1.0 ns
FULL_FRAMES = list(range(0, 11))    # フレーム [0..10]、0〜1.0 ns


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt_diss.npz")
            if len(z["X"]) == 0:
                continue
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


def mean_cols(series, frames):
    return series[:, frames].mean(1)[:, None].astype(np.float32)


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


def auprc_or_nan(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, p))


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 5ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 735 and Xte.shape[0] == 360, (Xtr.shape, Xte.shape)
assert len(TRAIN_P) == 58 and len(TEST_P) == 30, (len(TRAIN_P), len(TEST_P))

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print("test proteins:", len(prot_idx),
      "candidates per protein: min", min(len(v) for v in prot_idx.values()),
      "max", max(len(v) for v in prot_idx.values()), flush=True)

# ---- 条件ごとの特徴量 ----
f6tr, f6te = hist6(Htr), hist6(Hte)
FEATURES = {
    "static_only": None,
    "base_hist_D6": (f6tr, f6te),
    "first_half_mean": (mean_cols(Htr, FIRST_FRAMES), mean_cols(Hte, FIRST_FRAMES)),
    "second_half_mean": (mean_cols(Htr, SECOND_FRAMES), mean_cols(Hte, SECOND_FRAMES)),
    "both_halves_mean": (
        np.concatenate([mean_cols(Htr, FIRST_FRAMES), mean_cols(Htr, SECOND_FRAMES)], axis=1),
        np.concatenate([mean_cols(Hte, FIRST_FRAMES), mean_cols(Hte, SECOND_FRAMES)], axis=1)),
    "full_mean": (mean_cols(Htr, FULL_FRAMES), mean_cols(Hte, FULL_FRAMES)),
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

# ---- 土台の再現の確認（K53 の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.1139):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.4641) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.5780) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.1139) < 5e-4, full_gain
# 全部の平均は K53 の mean_only と同じ条件
a_fullmean = results["full_mean"]["auprc"]
assert abs(a_fullmean - 0.5794) < 5e-4, results["full_mean"]
print("base reproduction OK (static 0.4641 / full6 0.5780 / gain +0.1139 / full mean 0.5794)",
      flush=True)

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

r_first = gains["first_half_mean"]["ratio_vs_full6_gain"]
r_second = gains["second_half_mean"]["ratio_vs_full6_gain"]
if r_second >= 0.8 and r_first <= 0.5:
    verdict = "今の状態が効いている"
elif r_first >= 0.8 and r_second >= 0.8:
    verdict = "続いている状態が効いている"
else:
    verdict = "どちらでもない（数値をそのまま記録）"
print("verdict:", verdict, flush=True)

# ---- タンパク質ごとの値（陽性が 0 件のタンパク質では割合は記録しない） ----
per_protein = {}
for name in FEATURES:
    row = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        a_s = auprc_or_nan(yte[idx], preds["static_only"][idx])
        a_f = auprc_or_nan(yte[idx], preds["base_hist_D6"][idx])
        a_c = auprc_or_nan(yte[idx], preds[name][idx])
        if math.isnan(a_s) or math.isnan(a_f) or math.isnan(a_c):
            g_full = float("nan")
            ratio = float("nan")
            gain = float("nan")
        else:
            g_full = a_f - a_s
            gain = a_c - a_s
            ratio = gain / g_full if g_full else float("nan")
        row[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc_static": a_s, "auprc_full6": a_f, "auprc_condition": a_c,
            "full6_gain": g_full,
            "gain": gain,
            "ratio_vs_full6_gain": ratio,
        }
    per_protein[name] = row

out = {
    "label": "Salt dissociation 5ns: guidance of first-half mean vs second-half mean of the history",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-42",
    "horizon_ns": 5,
    "conditions_definition": {
        "base": "K15 expanded conditions: train 58 proteins (orig 4 + 54 additions), "
                "test 30 proteins (orig 2 + 28 additions), frozen split from K15 split.json, "
                "origin frame 10 (1 ns), history frames 0-10 (100 ps spacing), "
                "HistGBM (120 iters, lr 0.075, 31 leaves, min_samples_leaf 15, L2 0.02, "
                "seed 42) refit per condition, distance = minimum over charged "
                "heavy-atom pairs, bound 4.0 A, candidates = pairs salt-bridged at "
                "the origin frame.",
        "static_only": "static features only (5 columns)",
        "base_hist_D6": "static + 6 history statistics (first, mean, std, min, "
                        "last-first, fraction <= bound) over frames 0-10",
        "first_half_mean": "static + mean over frames [0..4]",
        "second_half_mean": "static + mean over frames [6..10]",
        "both_halves_mean": "static + first-half mean + second-half mean",
        "full_mean": "static + mean over frames [0..10]",
    },
    "base_reproduction": {
        "static_only_auprc": a_static,
        "base_hist_D6_auprc": a_full6,
        "full6_gain": full_gain,
        "full_mean_auprc": a_fullmean,
        "reference": "K53 saltbridge_dissociation_carrier results.json (5 ns: static 0.4641, "
                     "full hist6 0.5780, gain +0.1139, mean_only 0.5794)",
    },
    "caution": "Base gain is small (+0.1139) and the test set is small (360 candidates, "
               "139 positives), so ratios against the full gain are noisy. Per-protein "
               "candidate counts average ~12, so per-protein AUPRCs are noisy; proteins "
               "with 0 positives have no defined AUPRC and their ratios are not recorded.",
    "verdict": verdict,
    "results": results,
    "gains": gains,
    "per_protein": per_protein,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

np.savez(OUT / "test_predictions.npz",
         **{f"pred_{k}": v for k, v in preds.items()},
         y_test=yte, test_protein=test_protein)


def fmt(v):
    return "" if (isinstance(v, float) and math.isnan(v)) else round(v, 4)


with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auprc", "auc", "brier", "logloss",
                "gain", "ratio_vs_full6_gain", "ratio_vs_fullmean_gain"])
    for name in FEATURES:
        r = results[name]
        g = gains[name]
        w.writerow([name, r["n"], r["positives"], round(r["auprc"], 4),
                    round(r["auc"], 4), round(r["brier"], 4), round(r["logloss"], 4),
                    round(g["gain"], 4), round(g["ratio_vs_full6_gain"], 4),
                    round(g["ratio_vs_fullmean_gain"], 4)])
    w.writerow([])
    w.writerow(["condition", "protein", "n", "positives", "auprc_static",
                "auprc_full6", "auprc_condition", "full6_gain", "gain",
                "ratio_vs_full6_gain"])
    for name in FEATURES:
        for prot in TEST_P:
            d = per_protein[name][prot]
            w.writerow([name, prot, d["n"], d["positives"],
                        fmt(d["auprc_static"]), fmt(d["auprc_full6"]),
                        fmt(d["auprc_condition"]), fmt(d["full6_gain"]),
                        fmt(d["gain"]), fmt(d["ratio_vs_full6_gain"])])
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
