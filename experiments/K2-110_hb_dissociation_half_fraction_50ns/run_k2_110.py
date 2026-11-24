"""K2-110: 50 ns先で担い手になる「割合」の前半と後半を測る
（資料Dの条件、水素結合の解離、50 ns先）.

形成では割合の前半と後半を測り終えた（K2-14、90 ns先は前半 0.816・
後半 0.676）。解離の割合の前半と後半は 1 ns先（K2-99）、5 ns先
（K2-107）、90 ns先（K2-100、前半 1.307・後半 0.909）で測り終えた。
ここでは解離の 50 ns先で割合の形の前半と後半を測る。解離の土台は
K46 の条件で、候補は起点で 3.5 Å 以下のペア（解離の候補）、ラベルは
予測幅の先で 3.5 Å を超えたかどうか。割合の境界は形成と同じ 3.5 Å。

条件（資料Dの条件、K46 と同じ土台）:
- 学習 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K46 のキャッシュ（~/workspace/kinescope/work/data_cache_k46/、
  解離の候補ごとの距離系列）を再利用する。
  距離はフレームごとの箱を軸ごとに扱って計算済み。
- 予測幅は 50 ns先（horizon インデックス 4、HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）。

比べる条件（静的特徴7個に、つぎのどれかを足す）:
- static_only:       静的特徴だけ（土台の照合用）
- base_hist_D6:      静的＋履歴 6 統計量（全部入り、土台の照合用）
- first_half_frac:   静的＋前半（フレーム [0..4]、0〜0.4 ns）の 3.5 Å以下割合
- second_half_frac:  静的＋後半（フレーム [6..10]、0.6〜1.0 ns）の 3.5 Å以下割合
- both_halves_frac:  静的＋前半の割合＋後半の割合の両方
- full_frac:         静的＋全部（フレーム [0..10]）の 3.5 Å以下割合

割合の定義: 境界（3.5 Å、形成と同じ）以下だったフレームの数 ÷
その区間のフレーム数（前半・後半は 5、全部は 11）。
フレーム 5（0.5 ns）は前半にも後半にも入らない設計（K2-01・K2-05 と同じ）。

利得と割合:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。
- 割合は、全部入りの利得（base_hist_D6 の利得、K46 の記録で +0.1142）に
  対する割合を主とし、あわせて全部の割合（full_frac）の利得に対する
  割合も記録する（読み方の分母の確認のため）。分母を全部の割合の利得に
  しても判定が変わる場合は、両方を記録する。

読み方（結果を見る前に固定、K2-01 と同じ）:
- 後半の割合（対 全部入りの利得）が 8 割以上で、前半が 5 割以下なら
  「今の状態が効いている」。
- 前半も後半も 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する（無理に結論を付けない）。

土台の再現の期待値（K46 の記録、50 ns先）:
- 静的のみ AUPRC 0.4088、全部入り 0.5231、利得 +0.1142。
- 全部の割合だけは解離の 50 ns先ではまだ測った記録がないため、照合はしない。
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

# ---- 土台の再現の確認（K46 の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.1142):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.4088) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.5231) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.1142) < 5e-4, full_gain
# 全部の割合だけは解離の 50 ns先ではまだ測った記録がないため、照合はしない
a_fullfrac = results["full_frac"]["auprc"]
print("base reproduction OK (static 0.4088 / full6 0.5231 / gain +0.1142 / full frac", round(a_fullfrac, 4), ")",
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
r_first_ff = gains["first_half_frac"]["ratio_vs_fullfrac_gain"]
r_second_ff = gains["second_half_frac"]["ratio_vs_fullfrac_gain"]


def _judge(rf, rs):
    if rs >= 0.8 and rf <= 0.5:
        return "今の状態が効いている"
    if rf >= 0.8 and rs >= 0.8:
        return "続いている状態が効いている"
    return "どちらでもない（数値をそのまま記録）"


verdict = _judge(r_first, r_second)
verdict_vs_fullfrac = _judge(r_first_ff, r_second_ff)
print("verdict:", verdict, "| verdict vs full-frac gain:", verdict_vs_fullfrac, flush=True)

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
    "label": "HB dissociation 50ns: guidance of first-half fraction vs second-half fraction of the history",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-110",
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
                            "(2 features; no prior record at 50 ns for dissociation)",
        "full_frac": "static + fraction over frames [0..10] (no prior record at 50 ns "
                     "for dissociation)",
        "base_hist_D6": "static + 6 statistics of the full 11-frame history (all-in reference)",
    },
    "base_reproduction": {
        "expected_from_K46": {"static_only_auprc": 0.4088, "base_hist_D6_auprc": 0.5231,
                              "full6_gain": 0.1142, "full_frac_auprc": None},
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
                "carries it'; otherwise record the numbers as they are. "
                "The same rule is also applied with the full-frac gain as denominator.",
        "first_half_ratio_vs_full6": r_first,
        "second_half_ratio_vs_full6": r_second,
        "verdict": verdict,
        "first_half_ratio_vs_fullfrac": r_first_ff,
        "second_half_ratio_vs_fullfrac": r_second_ff,
        "verdict_vs_fullfrac": verdict_vs_fullfrac,
    },
    "per_protein": per_protein,
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
    y_50ns=yte,
    **{f"p_{name}": preds[name] for name in FEATURES},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
