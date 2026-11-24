"""K2-171: K2-15 の問い（漂い）を割合で測る.

K2-15 では、水素結合の解離で、漂い（前半の平均 − 後半の平均）は
足しになった（(c) 0.5923 は (b) 0.5691 を +0.0232 上回る）。
水素結合の形成では、平均の漂いは 1 ns先（K2-06、−0.0031）と
30 ns先（K2-16、−0.0014）で足しにならなかった。ここでは同じ問いを
割合の漂いの形で 10 ns先で測る。割合の漂いはまだどの予測幅でも
測っていない。

漂いの定義（割合の形）:
- 割合 = ペアの距離が水素結合の閾値（3.5 Å）以下の時間の割合。
  6 個の統計量のうちの「3.5 Å 以下だった割合」と同じ定義。
- 漂い = 前半の割合 − 後半の割合。前半はフレーム [0..4]
  （0〜0.4 ns）、後半はフレーム [6..10]（0.6〜1.0 ns）。
  フレーム 5（0.5 ns）は前半にも後半にも入らない（K2-01 と同じ）。

条件は資料Dの条件（6 タンパク質×3 レプリカ、起点 1 ns、履歴 0〜1 ns、
HistGBM を条件ごとに学習し直し、学習 4／テスト 1mix_A＋2qt4_A）で、
水素結合の形成、10 ns先、割合だけ。静的な特徴量は 7 個である。
水素結合の形成の系列は、ドナー・アクセプター重原子ペアの距離の
11 フレーム（100 ps 間隔）です。入力は資料Dの共有キャッシュ
（~/workspace/kinescope/work/data_cache/）を再利用する。
このキャッシュの履歴は (11, N) で保存されているため、転置して使う
（K13・K15・K53 のキャッシュとは違う）。

土台の再現: 静的のみと全部の 6 個の統計量の AUPRC が、水素結合の
形成の 10 ns先の記録（K2-62、静的のみ 0.3548、全部 0.4377、テスト
4,972件・陽性 325件）と一致することを確かめる。全部の割合は
K2-62 の記録（0.4553）と照らす。

読み方（K2-15 と同じく先に固定）:
- (c) 全部の割合＋漂いが (b) 全部の割合だけを AUPRC で +0.01 以上
  上回れば「漂いは足しになる」、未満なら「足しにならない」。
- (a) 漂いだけの利得も記録し、全部入りの利得に対する割合もあわせて
  記録する。
- K2-62 の「前半と後半の両方」（割合の形での利得 1.100）と (c) の
  結果の関係も記録する。(c) は (b) の上に漂いを足した形で、K2-62
  の「前半と後半の両方」は前半と後半の割合を 2 つ入れた形。
  2 つの条件の AUPRC の差も記録する。
- 割合の分母は全部入りの利得（静的＋履歴 6 統計量の利得）を主とし、
  あわせて全部の割合の利得に対する割合も記録する（分母の確認のため）。
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
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 3.5
HIDX = 2  # 10 ns先（y の列は [1, 5, 10, 30, 50, 90] ns の順）
FRAMES_FIRST = [0, 1, 2, 3, 4]       # 前半（0〜0.4 ns）
FRAMES_SECOND = [6, 7, 8, 9, 10]     # 後半（0.6〜1.0 ns）
FRAMES_FULL = list(range(11))        # 全部（0〜1.0 ns）

# K2-62 の記録との照合用の期待値（10 ns先、合算）
EXPECTED = {"static_only": 0.3548, "full_hist6": 0.4377,
            "frac_full": 0.4553, "test_n": 4972, "test_positives": 325}


def hist6(series, bound=BOUND):
    """series は (N, F) の距離系列。与えられたフレーム範囲の 6 個の統計量。"""
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                       max_leaf_nodes=31, min_samples_leaf=15,
                                       l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            # 資料Dの共有キャッシュの hist は (11, N) で保存されている
            # ため転置する
            parts.append({k: (z[k].T if k.startswith("hist") else z[k])
                          for k in z.files})
    return parts


train = load(TRAIN_P)
test = load(TEST_P)
Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
Htr = np.concatenate([d["hist"] for d in train]).astype(np.float64)
Hte = np.concatenate([d["hist"] for d in test]).astype(np.float64)
Ytr = np.concatenate([d["y"] for d in train])
Yte = np.concatenate([d["y"] for d in test])
print("train", Xtr.shape, "test", Xte.shape, flush=True)


def frac_of(S, fr, bound=BOUND):
    return (S[:, fr] <= bound).mean(axis=1, keepdims=True)


# 転置の向きの確認: hist は (11, N) で保存されているため、
# 転置し忘れると系列の形が (N, 11) にならない。X と同じ行数で
# 11 列であることを確かめる（転置の向きを間違えるとここで止まる）。


def drift_of(S):
    return frac_of(S, FRAMES_FIRST) - frac_of(S, FRAMES_SECOND)


CONDS = ["static_only", "full_hist6", "drift_only", "frac_full",
         "frac_full_plus_drift", "frac_both_halves",
         "frac_first_half", "frac_second_half"]  # 最後の2つは前半・後半の照合用


def feats_for(cname):
    if cname == "static_only":
        return Xtr, Xte
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname == "drift_only":
        return (np.column_stack([Xtr, drift_of(Htr)]),
                np.column_stack([Xte, drift_of(Hte)]))
    if cname == "frac_full":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FULL)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FULL)]))
    if cname == "frac_full_plus_drift":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FULL), drift_of(Htr)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FULL), drift_of(Hte)]))
    if cname == "frac_both_halves":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FIRST),
                                 frac_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FIRST),
                                 frac_of(Hte, FRAMES_SECOND)]))
    if cname == "frac_first_half":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_FIRST)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_FIRST)]))
    if cname == "frac_second_half":
        return (np.column_stack([Xtr, frac_of(Htr, FRAMES_SECOND)]),
                np.column_stack([Xte, frac_of(Hte, FRAMES_SECOND)]))
    raise ValueError(cname)


ytr, yte = Ytr[:, HIDX], Yte[:, HIDX]
res, per_protein, preds = {}, {}, {}
for cname in CONDS:
    xtr, xte = feats_for(cname)
    p = predict(xtr, ytr, xte)
    res[cname] = metrics(yte, p)
    preds[cname] = p
    # タンパク質ごとの指標（テストの 2 タンパク質）。
    # テストの並びは 1mix_A の 3 レプリカ、つぎに 2qt4_A の 3 レプリカ
    per_protein[cname] = {}
    counts = [sum(len(d["y"]) for d in test[i * 3:(i + 1) * 3]) for i in range(2)]
    start = 0
    for prot, n_prot in zip(TEST_P, counts):
        per_protein[cname][prot] = metrics(yte[start:start + n_prot],
                                           p[start:start + n_prot])
        start += n_prot
    print(cname, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in res[cname].items()}, flush=True)

assert Htr.shape == (Xtr.shape[0], 11), Htr.shape
assert Hte.shape == (Xte.shape[0], 11), Hte.shape
print("history shape verified (series is (N, 11) after transpose)", flush=True)

# K2-62 の記録との照合
assert res["static_only"]["n"] == EXPECTED["test_n"]
assert res["static_only"]["positives"] == EXPECTED["test_positives"]
for c in ("static_only", "full_hist6", "frac_full"):
    assert abs(res[c]["auprc"] - EXPECTED[c]) < 5e-4, (c, res[c]["auprc"])
print("base reproduction vs K2-62: OK", flush=True)

gain_full = res["full_hist6"]["auprc"] - res["static_only"]["auprc"]
gain_frac_full = res["frac_full"]["auprc"] - res["static_only"]["auprc"]
gains = {c: res[c]["auprc"] - res["static_only"]["auprc"] for c in CONDS}
ratios_full = {c: gains[c] / gain_full for c in CONDS}
ratios_frac = {c: gains[c] / gain_frac_full for c in CONDS}
diff_c_b = res["frac_full_plus_drift"]["auprc"] - res["frac_full"]["auprc"]
diff_c_both = (res["frac_full_plus_drift"]["auprc"]
               - res["frac_both_halves"]["auprc"])
reading = "漂いは足しになる" if diff_c_b >= 0.01 else "漂いは足しにならない"
print(f"gain_full {gain_full:+.4f} gain_drift {gains['drift_only']:+.4f} "
      f"(ratio {ratios_full['drift_only']:.3f}) "
      f"gain_frac_full {gain_frac_full:+.4f} (ratio {ratios_full['frac_full']:.3f}) "
      f"gain_c {gains['frac_full_plus_drift']:+.4f} "
      f"(ratio {ratios_full['frac_full_plus_drift']:.3f}) "
      f"diff (c)-(b) {diff_c_b:+.4f} reading: {reading} "
      f"diff (c)-(both halves) {diff_c_both:+.4f}", flush=True)

out = {
    "label": "HB formation 10ns, fraction drift (first-half fraction minus second-half fraction) on top of the full fraction (K2-15 question, fraction form)",
    "phenomenon": "hb formation (Doc D, data_cache)",
    "horizon_ns": 10,
    "frames": {"first_half": FRAMES_FIRST, "second_half": FRAMES_SECOND,
               "full": FRAMES_FULL,
               "note": "frame 5 belongs to neither half"},
    "bound": BOUND,
    "fraction_definition": "fraction = share of frames with pair distance <= 3.5 A (same definition as the 'fraction <= 3.5 A' of the 6 statistics) within the given frame range",
    "drift_definition": "drift = first-half fraction minus second-half fraction of the min-distance series",
    "metrics": res,
    "per_protein_metrics": per_protein,
    "gains": gains,
    "gain_ratios_vs_full_hist6": ratios_full,
    "gain_ratios_vs_frac_full": ratios_frac,
    "diff_c_minus_b_auprc": float(diff_c_b),
    "diff_c_minus_both_halves_auprc": float(diff_c_both),
    "reading_rule": ("(c) minus (b) in AUPRC >= +0.01 -> drift adds; "
                     "otherwise it does not (K2-06 convention)"),
    "reading": reading,
    "checks_vs_K2_62": {
        "static_only_auprc_expected": EXPECTED["static_only"],
        "full_hist6_auprc_expected": EXPECTED["full_hist6"],
        "frac_full_auprc_expected": EXPECTED["frac_full"]},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n", "positives", "auc", "auprc", "brier",
                "logloss", "gain", "ratio_vs_full_hist6_gain",
                "ratio_vs_frac_full_gain"])
    for c in CONDS:
        m = res[c]
        w.writerow([c, m["n"], m["positives"], f"{m['auc']:.4f}",
                    f"{m['auprc']:.4f}", f"{m['brier']:.4f}",
                    f"{m['logloss']:.4f}", f"{gains[c]:+.4f}",
                    f"{ratios_full[c]:.3f}", f"{ratios_frac[c]:.3f}"])
counts = [sum(len(d["y"]) for d in test[i * 3:(i + 1) * 3]) for i in range(2)]
test_protein = np.concatenate([np.repeat(p, n) for p, n in zip(TEST_P, counts)])
np.savez(OUT / "test_predictions.npz",
         **{f"pred_{k}": v for k, v in preds.items()},
         y_test=yte, test_protein=test_protein)
print("saved", OUT / "results.json", OUT / "results.csv",
      OUT / "test_predictions.npz", flush=True)
