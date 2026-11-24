"""K2-147: 1 ns先で「割合」の漂いを測る
（資料Dの条件、水素結合の形成、1 ns先）.

K2-15 の問い（漂い）を割合の形で測る。平均の漂いの地図は 1 ns先で
6 通りすべて埋まっているが、割合の漂いはまだどの予測幅でも測って
いない。K2-06 では平均の漂いを測った（(a) 漂いだけの利得 +0.0272、
割合 0.231、(c) 全部の平均＋漂いが (b) 全部の平均だけを −0.0031 下回り
「漂いは足しにならない」）。ここでは同じ問いを割合の形で測る。

条件（資料Dの条件、K2-01・K2-06・K2-61・K71 と同じ土台）:
- 学習 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算。
- 起点フレーム10（1 ns）、履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）を条件ごとに学習し直す。
- データは K01 のキャッシュ（~/workspace/kinescope/work/data_cache/、
  変換済み実軌道18本から作った候補ごとの距離系列）を再利用する。
  距離はフレームごとの箱を軸ごとに扱って計算済み。
- 予測幅は 1 ns先（horizon インデックス 0、HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）。

比べる条件（静的特徴7個に、つぎのどれかを足す）:
- static_only:         静的特徴だけ（土台の照合用）
- base_hist_D6:        静的＋履歴 6 統計量（全部入り、土台の照合用）
- (a) drift_frac_only: 静的＋漂い（前半の割合 − 後半の割合）だけ
- (b) full_frac:       静的＋全部（フレーム [0..10]）の 3.5 Å以下割合
- (c) full_frac_drift: 静的＋全部の割合＋漂い
- both_halves_frac:    静的＋前半の割合と後半の割合の 2 つ（K2-61 との比較用）

割合の定義: 境界（3.5 Å、K71 と同じ）以下だったフレームの数 ÷
その区間のフレーム数（前半・後半は 5、全部は 11）。
フレーム 5（0.5 ns）は前半にも後半にも入らない設計（K2-01・K2-06 と同じ）。

漂いの定義（結果を見る前に固定）:
- 漂い = 前半の割合 − 後半の割合。
- 符号の意味は平均の漂い（K2-06：前半の平均 − 後半の平均）とは逆である
  ことに注意する。平均の漂いが正なら「前半のほうが距離が大きい」
  （履歴のなかで距離が縮む方向に動いた）だが、割合の漂いが正なら
  「前半のほうが 3.5 Å 以下だった割合が大きい」（履歴のなかで距離が
  開く方向に動いた）である。割合は距離が小さいほど大きくなるため、
  符号の向きが平均の漂いと逆になる。

利得と割合:
- 利得 = その条件の AUPRC − 静的のみの AUPRC。
- 割合は、全部入りの利得（base_hist_D6 の利得、K2-61 の 1 ns先の記録で
  +0.1178）に対する割合を主とし、あわせて全部の割合（full_frac）の
  利得に対する割合も記録する（読み方の分母の確認のため）。
  土台の再現は K2-61 の 1 ns先の記録（静的 0.3637 / 全部入り 0.4815 /
  利得 +0.1178、全部の割合だけ 0.4829）と照らす。K71 の 1 ns先の記録
  （experiments/K71_90ns_carrier/results.json の conditions の
  static_only / full_hist6 / only_frac_le3.5 の "1"）とも同じ値である。
  割り当て文の初版に書かれていた「静的のみ 0.3398、全部入り 0.5669、
  全部の割合 0.5578」は接触の形成の 1 ns先の値（K2-123 の記録）で、
  水素結合の形成ではない。統括の訂正で K2-61 の値に直された。

読み方（結果を見る前に固定、K2-15 と同じ）:
- (c) 全部の割合＋漂いが (b) 全部の割合だけを AUPRC で +0.01 以上
  上回れば「漂いは足しになる」、未満なら「足しにならない」。
- (a) 漂いだけの利得と割合も記録する。

土台の再現の期待値（K2-61 の 1 ns先の記録、K71 とも一致）:
- 静的のみ AUPRC 0.3637、全部入り 0.4815、利得 +0.1178、テスト 4,972件・陽性 290件。
- 全部の割合だけは AUPRC 0.4829（K2-61 の記録）。
- 前半と後半の両方（割合の形）は AUPRC 0.4864（K2-61 の記録）。
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
HORIZON_IDX = 0  # 1 ns先（HORIZONS_ALL = (1, 5, 10, 30, 50, 90)）
HORIZON_NS = 1
BOUND = 3.5  # 割合の境界（K71 と同じ）

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
print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 9610 and Xte.shape[0] == 4972, (Xtr.shape, Xte.shape)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

# ---- 条件ごとの特徴量 ----
f6tr, f6te = hist6(Htr), hist6(Hte)
ff_tr = frac_cols(Htr, FIRST_FRAMES)
ff_te = frac_cols(Hte, FIRST_FRAMES)
fs_tr = frac_cols(Htr, SECOND_FRAMES)
fs_te = frac_cols(Hte, SECOND_FRAMES)
drift_tr = (ff_tr - fs_tr).astype(np.float32)
drift_te = (ff_te - fs_te).astype(np.float32)
full_tr = frac_cols(Htr, FULL_FRAMES)
full_te = frac_cols(Hte, FULL_FRAMES)

FEATURES = {
    "static_only": None,
    "base_hist_D6": (f6tr, f6te),
    "drift_frac_only": (drift_tr, drift_te),
    "full_frac": (full_tr, full_te),
    "full_frac_drift": (np.column_stack([full_tr, drift_tr]),
                        np.column_stack([full_te, drift_te])),
    "both_halves_frac": (np.column_stack([ff_tr, fs_tr]),
                         np.column_stack([ff_te, fs_te])),
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

# ---- 土台の再現の確認（K2-61 の記録と小数第4位まで） ----
a_static = results["static_only"]["auprc"]
a_full6 = results["base_hist_D6"]["auprc"]
full_gain = a_full6 - a_static
print("base gain (should be ~0.1178):", round(full_gain, 4), flush=True)
assert abs(a_static - 0.3637) < 5e-4, results["static_only"]
assert abs(a_full6 - 0.4815) < 5e-4, results["base_hist_D6"]
assert abs(full_gain - 0.1178) < 5e-4, full_gain
a_fullfrac = results["full_frac"]["auprc"]
assert abs(a_fullfrac - 0.4829) < 5e-4, results["full_frac"]
a_both = results["both_halves_frac"]["auprc"]
assert abs(a_both - 0.4864) < 5e-4, results["both_halves_frac"]
print("base reproduction OK (static 0.3637 / full6 0.4815 / gain +0.1178 / "
      "full_frac 0.4829 / both_halves_frac 0.4864)", flush=True)

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

c_minus_b = results["full_frac_drift"]["auprc"] - results["full_frac"]["auprc"]
verdict = ("漂いは足しになる" if c_minus_b >= 0.01 else "漂いは足しにならない")
print("(c) - (b) =", round(c_minus_b, 4), "verdict:", verdict, flush=True)

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

# タンパク質ごとの (c) - (b)
per_protein_diff = {}
for prot in TEST_P:
    idx = prot_idx[prot]
    per_protein_diff[prot] = {
        "auprc_full_frac": auprc(yte[idx], preds["full_frac"][idx]),
        "auprc_full_frac_drift": auprc(yte[idx], preds["full_frac_drift"][idx]),
        "diff_c_minus_b": float(auprc(yte[idx], preds["full_frac_drift"][idx])
                              - auprc(yte[idx], preds["full_frac"][idx])),
        "n": int(len(idx)), "positives": int(yte[idx].sum()),
    }

out = {
    "label": "HB formation 1ns: does fraction drift add on top of full fraction?",
    "series": "K2 (what the guidance is, not where it is)",
    "experiment": "K2-147",
    "horizon_ns": HORIZON_NS,
    "bound_A": BOUND,
    "conditions_definition": {
        "base": "Material D conditions: train 1bgf_A/1ifg_A/1j8r_A/2b0a_A, "
                "test 1mix_A+2qt4_A pooled, origin frame 10 (1 ns), history frames 0-10 "
                "(100 ps spacing), HistGBM (120 iters, lr 0.075, 31 leaves, "
                "min_samples_leaf 15, L2 0.02, seed 42) refit per condition, "
                "7 static features.",
        "fraction": "fraction of frames with pair distance <= 3.5 A "
                    "(boundary from K71) within the given frame range, "
                    "divided by the number of frames in that range "
                    "(first/second half: 5, full: 11)",
        "drift": "first-half fraction minus second-half fraction. "
                 "Note the sign is opposite to the mean drift (K2-06): a positive "
                 "fraction drift means the pair was within 3.5 A more often in the "
                 "first half, i.e. the distance tended to open up; a positive mean "
                 "drift means the first-half mean distance was larger, i.e. the "
                 "distance tended to close.",
        "first_half": {"frames": FIRST_FRAMES, "span_ns": "0.0-0.4",
                       "feature": "fraction of frames [0..4] with distance <= 3.5 A"},
        "second_half": {"frames": SECOND_FRAMES, "span_ns": "0.6-1.0",
                        "feature": "fraction of frames [6..10] with distance <= 3.5 A"},
        "note": "Frame 5 (0.5 ns) belongs to neither half by design.",
        "drift_frac_only": "static + fraction drift (1 feature), condition (a)",
        "full_frac": "static + fraction over frames [0..10], condition (b)",
        "full_frac_drift": "static + full fraction and fraction drift (2 features), "
                           "condition (c)",
        "both_halves_frac": "static + first-half fraction and second-half fraction "
                            "(2 features, comparison with K2-61)",
        "base_hist_D6": "static + 6 statistics of the full 11-frame history (all-in reference)",
    },
    "base_reproduction": {
        "expected_from_K2_61_1ns": {"static_only_auprc": 0.3637, "base_hist_D6_auprc": 0.4815,
                              "full6_gain": 0.1178, "full_frac_auprc": 0.4829,
                              "both_halves_frac_auprc": 0.4864,
                              "test_n": 4972, "test_positives": 290},
        "note": "K2-61 (experiments/K2-61_hb_formation_half_fraction_1ns/results.json) "
                "agrees with K71's 1ns record (experiments/K71_90ns_carrier/results.json, "
                "conditions static_only / full_hist6 / only_frac_le3.5, key '1'). The first "
                "version of the assignment text gave static 0.3398 / full 0.5669 / full "
                "fraction 0.5578, which are the contact-formation 1ns values (K2-123); "
                "the coordinator corrected them to the K2-61 values above.",
        "observed": {"static_only_auprc": a_static, "base_hist_D6_auprc": a_full6,
                     "full6_gain": full_gain, "full_frac_auprc": a_fullfrac,
                     "both_halves_frac_auprc": a_both},
        "match": "all within 5e-4 (identical to 4 decimal places)",
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "n_test_positives": int(yte.sum()),
    },
    "metrics_1ns": results,
    "gains": gains,
    "reading_rule_fixed_in_advance": {
        "rule": "(c) full fraction + drift exceeds (b) full fraction alone by AUPRC "
                "+0.01 or more -> 'drift adds'; otherwise 'drift does not add'.",
        "c_minus_b": float(c_minus_b),
        "verdict": verdict,
    },
    "per_protein": per_protein,
    "per_protein_c_minus_b": per_protein_diff,
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
    y_1ns=yte,
    **{f"p_{name}": preds[name] for name in FEATURES},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
