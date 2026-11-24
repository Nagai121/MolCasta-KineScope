"""K2-175: 履歴の漂い（前半の割合 − 後半の割合）が、全部の割合の上に
足しになるか（K15 の条件、塩橋の形成、10 ns先）.

K2 系列（指導力の正体は何かを探る系列）の漂いの問い。K2-15 では、
水素結合の解離の 1 ns先で、漂い（前半の平均 − 後半の平均）は足しに
なった（全部の平均＋漂い 0.5923 は全部の平均だけ 0.5691 を +0.0232
上回る）。塩橋の形成の 30 ns先（K2-57）では、平均の漂いは足しに
ならなかった（−0.0205）。ここでは、同じ問いを割合の形で 10 ns先で
測る。塩橋の形成の 10 ns先では、割合の漂いはまだ測っていない。K2-150 では、接触の形成
の 1 ns先で、割合の漂いは足しにならなかった（−0.0004）。K2-155 では、
塩橋の形成の 30 ns先で、割合の漂いは足しにならなかった（−0.0013）。
K2-160 では、水素結合の解離の 90 ns先で、割合の漂いは足しにならなかっ
た（+0.0081、線のすぐ下）。K2-165 では、水素結合の形成の 5 ns先で、
割合の漂いは足しにならなかった（−0.0090）。K2-170 では、塩橋の解離の
5 ns先で、割合の漂いは足しにならなかった（−0.0133）。

漂いの定義（割合の形）:
- 割合 = ペアの荷電重原子間の最小距離が塩橋の閾値（4.0 Å）以下の
  時間の割合。6 個の統計量のうちの「4.0 Å 以下だった割合」と同じ定義。
- 漂い = 前半の割合 − 後半の割合。前半はフレーム [0..4]
  （0〜0.4 ns）、後半はフレーム [6..10]（0.6〜1.0 ns）。
  フレーム 5（0.5 ns）は前半にも後半にも入らない（K2-01 と同じ）。

条件（K15 の条件、K2-137 と同じ土台）:
- 学習 58 タンパク質（元の 4 ＋追加 54、2,557件）、テスト 30 タンパク質
  （元の 2 ＋追加 28、1,170件）。タンパク質の分け方は K15 の split.json
  の凍結した集合。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K15 のキャッシュ（~/workspace/kinescope/work/k15_cache/、
  変換済み実軌道 88 タンパク質×3 レプリカから作った形成の候補ごとの
  距離系列）を再利用する。距離は荷電重原子ペアの最小値で、境界は
  4.0 Å。候補は起点で塩橋ができていないペア。

比べる条件（静的特徴5個に、つぎのどれかを足す）:
- static_only:      静的特徴だけ（土台の照合用）
- base_hist_D6:     静的＋履歴 6 統計量（全部入り、土台の照合用）
- drift_only:       静的＋漂いだけ（前半 [0..4] の割合 − 後半 [6..10] の割合）
- frac_full:        静的＋全部（フレーム [0..10]）の 4.0 Å以下割合
- frac_full_plus_drift: 静的＋全部の割合＋漂い
 - frac_first_half / frac_second_half / frac_both_halves: 照合用（K2-140）

フレーム 5（0.5 ns）は前半にも後半にも入らない設計。

利得と読み方（結果を見る前に固定）:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。AUPRC の差もあわせて
  記録する。
- (c) 全部の割合＋漂いが (b) 全部の割合だけを AUPRC で +0.01 以上
  上回れば「漂いは足しになる」、未満なら「足しにならない」と記録する。
- (a) 漂いだけの利得も、全部入りの利得に対する割合とあわせて記録する。
- 塩橋の形成は土台の利得が小さく（10 ns先で +0.0594）、テストも
  1,170件と小さいため、AUPRC の差の読みには留保がつく。

土台の再現の期待値（K15 / K2-140 の記録、10 ns先）:
- 静的のみ AUPRC 0.2034、全部入り 0.2627、利得 +0.0594。
- 全部の割合は K2-140 と同じ条件で、AUPRC 0.2326（利得 +0.0292）の
  はず。K15 の results.json には frac_only の記録がないため、K2-140 の
  記録と照らす。
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

CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
OUT = Path(__file__).resolve().parent
SPLIT = json.loads(Path(
    "/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json"
).read_text())
TRAIN_P = list(SPLIT["orig_train"]) + list(SPLIT["train_additions"])
TEST_P = list(SPLIT["orig_test"]) + list(SPLIT["test_additions"])
HORIZON_IDX = 2  # 10 ns先（y の列は [1, 5, 10, 30, 50, 90] ns の順）
BOUND = 4.0

FIRST_FRAMES = list(range(0, 5))    # フレーム [0..4]、0〜0.4 ns
SECOND_FRAMES = list(range(6, 11))  # フレーム [6..10]、0.6〜1.0 ns
FULL_FRAMES = list(range(0, 11))    # フレーム [0..10]、0〜1.0 ns


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
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
print("train", Xtr.shape, "test", Xte.shape, "test pos 10ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 2557 and Xte.shape[0] == 1170, (Xtr.shape, Xte.shape)
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
f1_tr, f1_te = frac_cols(Htr, FIRST_FRAMES), frac_cols(Hte, FIRST_FRAMES)
f2_tr, f2_te = frac_cols(Htr, SECOND_FRAMES), frac_cols(Hte, SECOND_FRAMES)
FEATURES = {
    "static_only": None,
    "base_hist_D6": (f6tr, f6te),
    "drift_only": (dr_tr, dr_te),
    "frac_full": (ff_tr, ff_te),
    "frac_full_plus_drift": (
        np.concatenate([ff_tr, dr_tr], axis=1),
        np.concatenate([ff_te, dr_te], axis=1)),
    "frac_first_half": (f1_tr, f1_te),
    "frac_second_half": (f2_tr, f2_te),
    "frac_both_halves": (
        np.concatenate([f1_tr, f2_tr], axis=1),
        np.concatenate([f1_te, f2_te], axis=1)),
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
print("base gain (should be ~0.0594):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.2034) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.2627) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.0594) < 5e-4, full_gain
assert results["static_only"]["positives"] == 131, results["static_only"]
# 全部の割合は K2-140 と同じ条件（K15 の results.json には frac_only の記録がない）
a_fullfrac = results["frac_full"]["auprc"]
assert abs(a_fullfrac - 0.2326) < 5e-4, results["frac_full"]
print("base reproduction OK (static 0.2034 / full6 0.2627 / gain +0.0594 / full frac 0.2326)",
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

drift_adds = results["frac_full_plus_drift"]["auprc"] - a_fullfrac
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
# 漂いの足しのタンパク質ごとの差（frac_full_plus_drift − frac_full の AUPRC）
drift_diff_per_protein = {}
for prot in TEST_P:
    idx = prot_idx[prot]
    a_b = auprc_or_nan(yte[idx], preds["frac_full"][idx])
    a_c = auprc_or_nan(yte[idx], preds["frac_full_plus_drift"][idx])
    drift_diff_per_protein[prot] = (
        float("nan") if (math.isnan(a_b) or math.isnan(a_c)) else float(a_c - a_b))

out = {
    "label": "Salt formation 10ns: does history fraction drift (first-half fraction minus second-half fraction) add on top of the full fraction?",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-175",
    "horizon_ns": 10,
    "conditions_definition": {
        "base": "K15 expanded conditions: train 58 proteins (orig 4 + 54 additions), "
                "test 30 proteins (orig 2 + 28 additions), frozen split from K15 split.json, "
                "origin frame 10 (1 ns), history frames 0-10 (100 ps spacing), "
                "HistGBM (120 iters, lr 0.075, 31 leaves, min_samples_leaf 15, L2 0.02, "
                "seed 42) refit per condition, distance = minimum over charged "
                "heavy-atom pairs, bound 4.0 A, candidates = pairs NOT salt-bridged at "
                "the origin frame.",
        "static_only": "static features only (5 columns)",
        "base_hist_D6": "static + 6 history statistics (first, mean, std, min, "
                        "last-first, fraction <= bound) over frames 0-10",
        "drift_only": "static + drift only (fraction over frames [0..4] minus fraction over "
                      "frames [6..10])",
        "frac_full": "static + fraction over frames [0..10] (fraction of frames with "
                    "min distance <= 4.0 A)",
        "frac_full_plus_drift": "static + full fraction + drift",
        "frac_first_half": "static + fraction over frames [0..4] (K2-140 reference)",
        "frac_second_half": "static + fraction over frames [6..10] (K2-140 reference)",
        "frac_both_halves": "static + first-half fraction and second-half fraction (K2-140 reference)",
    },
    "base_reproduction": {
        "static_only_auprc": a_static,
        "base_hist_D6_auprc": a_full6,
        "full6_gain": full_gain,
        "full_frac_auprc": a_fullfrac,
        "reference": "K15 saltbridge_expansion results.json full/ (10 ns: static 0.2034, "
                     "full hist6 0.2627, gain +0.0594); full fraction 0.2326 from K2-140 "
                     "(same conditions, 10 ns)",
    },
    "caution": "Base gain is small (+0.0594) and the test set is small (1,170 candidates, "
               "131 positives), so AUPRC differences near the +0.01 line are noisy. "
               "Per-protein candidate counts average ~39, so per-protein AUPRCs are noisy.",
    "drift_adds": float(drift_adds),
    "verdict": verdict,
    "results": results,
    "gains": gains,
    "per_protein": per_protein,
    "drift_diff_per_protein": drift_diff_per_protein,
}
def clean_nan(x):
    """AUPRC が定義できないタンパク質の nan を None に直す（K2-135・K2-140 と
    同じ扱い）。json.dumps は nan を NaN と書いてしまい、正しい JSON になら
    ないため、書き出す前に直す。"""
    if isinstance(x, dict):
        return {k: clean_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_nan(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


(OUT / "results.json").write_text(json.dumps(clean_nan(out), indent=2, ensure_ascii=False))

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
