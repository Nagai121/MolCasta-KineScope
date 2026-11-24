"""K2-01: 履歴の前半の平均と後半の平均で、指導力はどう違うか
（資料Dの条件、水素結合の形成、1 ns先）.

K2 系列（指導力の正体は何かを探る系列）の最初の実験。K 系列が
「指導力はどこにあるか」の切り分けだったのに対して、K2 は
「短い切り取りの平均が何の影なのか」を問う。L01 で 0.6 ns の段が
見えたことと、切り取りを長くすると回復率が下がる条件があったこと
（G 系列）を受けて、履歴の中の「いつの平均」が指導力を担っているかを
前半と後半に分けて測る。

条件（資料Dの条件、L01・K01 と同じ土台）:
- 学習 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K01 のキャッシュ（~/workspace/kinescope/work/data_cache/、
  変換済み実軌道18本から作った候補ごとの距離系列）を再利用する。
  距離はフレームごとの箱を軸ごとに扱って計算済み。

比べる条件（静的特徴7個に、つぎのどれかを足す）:
- static_only:      静的特徴だけ（土台の照合用）
- base_hist_D6:     静的＋履歴 6 統計量（全部入り、土台の照合用）
- first_half_mean:  静的＋前半（フレーム [0..4]、0〜0.4 ns）の距離の平均
- second_half_mean: 静的＋後半（フレーム [6..10]、0.6〜1.0 ns）の距離の平均
- both_halves_mean: 静的＋前半の平均と後半の平均の両方
- full_mean:        静的＋全部（フレーム [0..10]）の距離の平均

フレーム 5（0.5 ns）は前半にも後半にも入らない設計。

利得と割合:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。
- 割合は、全部入りの利得（base_hist_D6 の利得、L01 の記録で +0.1178）に
  対する割合を主とし、あわせて全部の平均（full_mean）の利得に対する
  割合も記録する（読み方の分母の確認のため）。

読み方（結果を見る前に固定）:
- 後半の割合（対 全部入りの利得）が 8 割以上で、前半が 5 割以下なら
  「今の状態が効いている」。
- 前半も後半も 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する（無理に結論を付けない）。
- 「前半と後半の両方」の利得も、全部入りの利得と比べて記録する。

土台の再現の期待値（L01 の記録、1 ns先）:
- 静的のみ AUPRC 0.3637、全部入り 0.4815、利得 +0.1178。
- 全部の平均は L01 の覆う長さ 1.0 ns と同じ条件で、AUPRC 0.4865、
  回復率（対 全部入りの利得）1.042 のはず。
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 0  # 1 ns先

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
print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 9610 and Xte.shape[0] == 4972, (Xtr.shape, Xte.shape)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

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

# ---- 土台の再現の確認（L01 の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.1178):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.3637) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.4815) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.1178) < 5e-4, full_gain
# 全部の平均は L01 の覆う長さ 1.0 ns と同じ条件
a_fullmean = results["full_mean"]["auprc"]
assert abs(a_fullmean - 0.4865) < 5e-4, results["full_mean"]
print("base reproduction OK (static 0.3637 / full6 0.4815 / gain +0.1178 / full mean 0.4865)",
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
    "label": "HB formation 1ns: guidance of first-half mean vs second-half mean of the history",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-01",
    "horizon_ns": 1,
    "conditions_definition": {
        "base": "Material D conditions: train 1bgf_A/1ifg_A/1j8r_A/2b0a_A, "
                "test 1mix_A+2qt4_A pooled, origin frame 10 (1 ns), history frames 0-10 "
                "(100 ps spacing), HistGBM (120 iters, lr 0.075, 31 leaves, "
                "min_samples_leaf 15, L2 0.02, seed 42) refit per condition, "
                "7 static features.",
        "first_half": {"frames": FIRST_FRAMES, "span_ns": "0.0-0.4",
                       "feature": "mean of the pair distance over frames [0..4]"},
        "second_half": {"frames": SECOND_FRAMES, "span_ns": "0.6-1.0",
                        "feature": "mean of the pair distance over frames [6..10]"},
        "note": "Frame 5 (0.5 ns) belongs to neither half by design.",
        "both_halves": "static + first-half mean + second-half mean (2 features)",
        "full_mean": "static + mean over frames [0..10] (same as L01 covered length 1.0 ns)",
        "base_hist_D6": "static + 6 statistics of the full 11-frame history (all-in reference)",
    },
    "base_reproduction": {
        "expected_from_L01": {"static_only_auprc": 0.3637, "base_hist_D6_auprc": 0.4815,
                              "full6_gain": 0.1178, "full_mean_auprc": 0.4865},
        "observed": {"static_only_auprc": a_static, "base_hist_D6_auprc": a_full6,
                     "full6_gain": full_gain, "full_mean_auprc": a_fullmean},
        "match": "all within 5e-4 (identical to 4 decimal places)",
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "n_test_positives": int(yte.sum()),
    },
    "metrics_1ns": results,
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
    y_1ns=yte,
    **{f"p_{name}": preds[name] for name in FEATURES},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
