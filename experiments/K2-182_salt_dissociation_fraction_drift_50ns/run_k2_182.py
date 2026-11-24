"""K2-182: 履歴の漂いを割合で測る（前半の割合 − 後半の割合）が、全部の
割合の上に足しになるか（K15 の広げた条件、塩橋の解離、50 ns先）.

K2 系列（指導力の正体は何かを探る系列）の漂いの問い。割合の漂いの
地図は 50 ns先で 6 通りのうち 5 通りが埋まっており、残るのは塩橋の
解離だけである（K2-177 の記録）。塩橋の解離の 50 ns先では、平均の漂いは足しに
ならなかった（K2-98 の地図の表、+0.0032）。ここでは、同じ問いを割合の漂いの形
で 50 ns先で測る。

条件（K15 の広げた条件、K2-13 と同じ土台）:
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
- drift_only:       静的＋漂いだけ（前半 [0..4] の割合 − 後半 [6..10] の割合）
- full_frac:        静的＋全部（フレーム [0..10]）の 4.0 Å以下割合
- full_frac_drift:  静的＋全部の割合＋漂い

割合の定義: 境界（4.0 Å、6 個の統計量のうちの「4.0 Å 以下だった割合」
と同じ定義）以下だったフレームの数 ÷ その区間のフレーム数（前半・
後半は 5、全部は 11）。漂いは前半の割合から後半の割合を引いたもの
（前半の割合 − 後半の割合）である。
フレーム 5（0.5 ns）は前半にも後半にも入らない設計。

利得と読み方（結果を見る前に固定）:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。AUPRC の差もあわせて
  記録する。
- (c) 全部の割合＋漂いが (b) 全部の割合だけを AUPRC で +0.01 以上
  上回れば「漂いは足しになる」、未満なら「足しにならない」と記録する。
- (a) 漂いだけの利得も、全部入りの利得に対する割合とあわせて記録する。
- 塩橋の解離は土台の利得が小さく（50 ns先で +0.1302）、テストも
  360件・陽性 166件と小さいため、AUPRC の差の読みには留保がつく。

土台の再現の期待値（K53・K2-146 の記録、50 ns先）:
- 静的のみ AUPRC 0.4863、全部入り 0.6165、利得 +0.1302。
- 全部の割合は K2-146 と同じ条件で、AUPRC 0.5869（利得 +0.1006）の
  はず。全部の平均だけ 0.5845、前半と後半の両方 0.5731 とも照らす。
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
HORIZON_IDX = 4  # 50 ns先（y の列は [1, 5, 10, 30, 50, 90] ns の順）
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
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, p))


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 50ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 735 and Xte.shape[0] == 360, (Xtr.shape, Xte.shape)
assert len(TRAIN_P) == 58 and len(TEST_P) == 30, (len(TRAIN_P), len(TEST_P))

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print("test proteins:", len(prot_idx),
      "candidates per protein: min", min(len(v) for v in prot_idx.values()),
      "max", max(len(v) for v in prot_idx.values()), flush=True)

# ---- 条件ごとの特徴量 ----
f6tr, f6te = hist6(Htr), hist6(Hte)
ff_tr, ff_te = frac_cols(Htr, FULL_FRAMES), frac_cols(Hte, FULL_FRAMES)
dr_tr = (frac_cols(Htr, FIRST_FRAMES) - frac_cols(Htr, SECOND_FRAMES)).astype(np.float32)
dr_te = (frac_cols(Hte, FIRST_FRAMES) - frac_cols(Hte, SECOND_FRAMES)).astype(np.float32)
fh_tr = frac_cols(Htr, FIRST_FRAMES)
fh_te = frac_cols(Hte, FIRST_FRAMES)
sh_tr = frac_cols(Htr, SECOND_FRAMES)
sh_te = frac_cols(Hte, SECOND_FRAMES)
mean_tr = Htr.mean(1)[:, None].astype(np.float32)
mean_te = Hte.mean(1)[:, None].astype(np.float32)
FEATURES = {
    "static_only": None,
    "base_hist_D6": (f6tr, f6te),
    "drift_only": (dr_tr, dr_te),
    "full_frac": (ff_tr, ff_te),
    "full_frac_drift": (
        np.concatenate([ff_tr, dr_tr], axis=1),
        np.concatenate([ff_te, dr_te], axis=1)),
    # 照合用（K2-146 の記録と同じ条件）
    "mean_full": (mean_tr, mean_te),
    "both_halves_frac": (
        np.concatenate([fh_tr, sh_tr], axis=1),
        np.concatenate([fh_te, sh_te], axis=1)),
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

# ---- 土台の再現の確認（K15 の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.1788):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.4863) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.6165) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.1302) < 5e-4, full_gain
assert results["static_only"]["positives"] == 166, results["static_only"]
# 全部の割合は K2-138 と同じ条件（K15 の results.json には full_frac の記録がない）
a_fullfrac = results["full_frac"]["auprc"]
assert abs(a_fullfrac - 0.5869) < 5e-4, results["full_frac"]
assert abs(results["mean_full"]["auprc"] - 0.5845) < 5e-4, results["mean_full"]
assert abs(results["both_halves_frac"]["auprc"] - 0.5731) < 5e-4, results["both_halves_frac"]
print("base reproduction OK (static 0.4863 / full6 0.6165 / gain +0.1302 / full frac 0.5869 / mean 0.5845 / both halves 0.5731)",
      flush=True)

# ---- 利得と漂いの足し ----
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

drift_adds = results["full_frac_drift"]["auprc"] - a_fullfrac
print("drift adds on top of full fraction (AUPRC diff):", round(drift_adds, 4), flush=True)
verdict = "漂いは足しになる" if drift_adds >= 0.01 else "漂いは足しにならない"
print("verdict:", verdict, flush=True)

# ---- タンパク質ごとの値（漂いの足しの差も） ----
# 陽性が 0 件または全件のタンパク質では AUPRC が決まらないため、割合は
# 記録しない（sklearn は警告を出して 0.0 を返すが、それは値ではない）。
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
            gain = float("nan")
            ratio = float("nan")
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
# 漂いの足しのタンパク質ごとの差（full_frac_drift − full_frac の AUPRC）
drift_diff_per_protein = {}
for prot in TEST_P:
    idx = prot_idx[prot]
    a_b = auprc_or_nan(yte[idx], preds["full_frac"][idx])
    a_c = auprc_or_nan(yte[idx], preds["full_frac_drift"][idx])
    drift_diff_per_protein[prot] = (
        float("nan") if (math.isnan(a_b) or math.isnan(a_c)) else float(a_c - a_b))

out = {
    "label": "Salt dissociation 50ns: does fraction drift (first-half fraction minus second-half fraction) add on top of the full fraction?",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-182",
    "horizon_ns": 50,
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
        "drift_only": "static + drift only (fraction over frames [0..4] minus fraction "
                      "over frames [6..10], bound 4.0 A)",
        "full_frac": "static + fraction <= 4.0 A over frames [0..10]",
        "full_frac_drift": "static + full fraction + drift",
    },
    "base_reproduction": {
        "static_only_auprc": a_static,
        "base_hist_D6_auprc": a_full6,
        "full6_gain": full_gain,
        "full_frac_auprc": a_fullfrac,
        "reference": "K53 saltbridge_dissociation_carrier results.json dissociation/ (50 ns: static 0.4863, "
                     "full hist6 0.6165, gain +0.1302); full fraction 0.5869, mean 0.5845, both halves 0.5731 "
                     "from K2-146 (same conditions, 50 ns)",
    },
    "caution": "Base gain is small (+0.1302) and the test set is small (360 candidates, "
               "166 positives), so AUPRC differences near the +0.01 line are noisy. "
               "Per-protein candidate counts average ~12, so per-protein AUPRCs are noisy.",
    "drift_adds": float(drift_adds),
    "verdict": verdict,
    "results": results,
    "gains": gains,
    "per_protein": per_protein,
    "drift_diff_per_protein": drift_diff_per_protein,
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
                "gain", "ratio_vs_full6_gain", "ratio_vs_fullfrac_gain"])
    for name in FEATURES:
        r = results[name]
        g = gains[name]
        w.writerow([name, r["n"], r["positives"], round(r["auprc"], 4),
                    round(r["auc"], 4), round(r["brier"], 4), round(r["logloss"], 4),
                    round(g["gain"], 4), round(g["ratio_vs_full6_gain"], 4),
                    round(g["ratio_vs_fullfrac_gain"], 4)])
    w.writerow([])
    w.writerow(["drift_adds_auprc", round(drift_adds, 4), "verdict", verdict])
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
    w.writerow([])
    w.writerow(["protein", "drift_diff_auprc"])
    for prot in TEST_P:
        w.writerow([prot, fmt(drift_diff_per_protein[prot])])
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
