"""K2-92: K2-06 の問い（漂いが足しになるか）を塩橋の解離の 10 ns先で測る
（K15 の広げた条件、塩橋の解離、10 ns先）.

K2-06 では水素結合の形成で、履歴の漂い（前半の平均 − 後半の平均）を
全部の平均の上に足しても足しになりませんでした（(c) 0.4834 は (b)
0.4865 を −0.0031 下回る）。K2-15 では水素結合の解離で足しになりました
（+0.0232）。K2-30 では塩橋の形成で足しになりませんでした（−0.0178）。
K2-34 では接触の形成で足しになりました（+0.0367）。ここでは同じ問いを
塩橋の解離の 10 ns先で測る。1 ns先（K2-39）では足しになりませんでした
（(c) 0.5213 は (b) 0.5382 を −0.0169 下回る）し、30 ns先（K2-58）
でも足しになりませんでした（(c) 0.5383 は (b) 0.5673 を −0.0290
下回る）。

条件（結果を見る前に固定）:
- static_only: 静的な特徴量だけ
- base_hist_D6: 静的な特徴量＋履歴全部の 6 個の統計量（土台の検証と、
  割合の分母の利得に使う。境界は 4.0 A）
- drift_only (a): 静的な特徴量＋漂いだけ
- mean_full (b): 静的な特徴量＋フレーム [0..10] の平均だけ
- mean_full_plus_drift (c): 静的な特徴量＋全部の平均＋漂い
- mean_first_half: 静的な特徴量＋フレーム [0..4] の平均だけ（K2-50 との照合用）
- mean_second_half: 静的な特徴量＋フレーム [6..10] の平均だけ（K2-50 との照合用）
- mean_both: 静的な特徴量＋前半の平均と後半の平均の 2 つ（K2-50 との照合用）

漂い = 前半の平均 − 後半の平均。前半はフレーム [0..4]（0〜0.4 ns）、
後半はフレーム [6..10]（0.6〜1.0 ns）。フレーム 5 は前半にも後半にも
入らない（K2-01・K2-06 と同じ分け方）。

読み方（事前に固定、K2-06 と同じ）: 利得は各条件の AUPRC −
静的のみの AUPRC。割合の分母は全部入りの利得（静的＋履歴 6 統計量の
利得）を主とし、あわせて全部の平均の利得に対する割合も記録する。
(c) 全部の平均＋漂いが (b) 全部の平均だけを AUPRC で +0.01 以上
上回れば「漂いは足しになる」、未満なら「足しにならない」。
(a) 漂いだけの利得も記録する。AUPRC の差も記録する。

注意: 塩橋の解離は土台の利得が小さい（10 ns先で +0.1781）ため、
割合の読みには留保がつく。AUPRC の差とあわせて読む。
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

K53 = Path("/home/hatch/workspace/kinescope/work/k53_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
BOUND = 4.0
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
HORIZON = 10
HI = 2

FIRST = [0, 1, 2, 3, 4]
SECOND = [6, 7, 8, 9, 10]
FULL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# K53 の記録値（土台の AUPRC と利得、照合用。README と results.json で確認）
REF_BASE = (0.3572, 0.5353)
REF_GAIN = 0.1781
# K2-50 の記録値（照合用。README と results.json で確認）
REF_K2_50 = {
    "mean_first_half": 0.5214,
    "mean_second_half": 0.4564,
    "mean_both": 0.5299,
    "mean_full": 0.5302,
}
# テストの陽性数（K53 の記録、照合用）
REF_TEST_POS = 131


def hist6(series, bound=BOUND):
    return np.stack(
        [series[:, 0], series.mean(1), series.std(1), series.min(1),
         series[:, -1] - series[:, 0], (series <= bound).mean(1)],
        axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def load(prots):
    parts = []
    spans = []
    for p in prots:
        start = sum(s for _, s in spans)
        n = 0
        for r in (1, 2, 3):
            z = np.load(K53 / f"{p}_R{r}_salt_diss.npz")
            parts.append({k: z[k] for k in z.files})
            n += len(z["X"])
        spans.append((p, n))
        assert sum(s for _, s in spans) == start + n
    return parts, spans


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


split = json.loads((K15_DIR / "split.json").read_text())
train_p = ORIG_TRAIN + tuple(split["train_additions"])
test_p = ORIG_TEST + tuple(split["test_additions"])
print("train proteins", len(train_p), "test proteins", len(test_p), flush=True)

train, _ = load(train_p)
test, test_spans = load(test_p)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, flush=True)
assert np.max(np.abs(Xtr[:, 0] - Htr[:, -1])) < 1e-6
assert np.max(np.abs(Xte[:, 0] - Hte[:, -1])) < 1e-6

span_ranges = {}
pos = 0
for name, n in test_spans:
    span_ranges[name] = (pos, pos + n)
    pos += n
assert pos == len(Xte)

ytr, yte = Ytr[:, HI], Yte[:, HI]

drift_tr = (Htr[:, FIRST].mean(1) - Htr[:, SECOND].mean(1))[:, None].astype(np.float32)
drift_te = (Hte[:, FIRST].mean(1) - Hte[:, SECOND].mean(1))[:, None].astype(np.float32)
mean_full_tr = Htr[:, FULL].mean(1)[:, None].astype(np.float32)
mean_full_te = Hte[:, FULL].mean(1)[:, None].astype(np.float32)
mean_first_tr = Htr[:, FIRST].mean(1)[:, None].astype(np.float32)
mean_first_te = Hte[:, FIRST].mean(1)[:, None].astype(np.float32)
mean_second_tr = Htr[:, SECOND].mean(1)[:, None].astype(np.float32)
mean_second_te = Hte[:, SECOND].mean(1)[:, None].astype(np.float32)
mean_both_tr = np.stack([Htr[:, FIRST].mean(1), Htr[:, SECOND].mean(1)], axis=1).astype(np.float32)
mean_both_te = np.stack([Hte[:, FIRST].mean(1), Hte[:, SECOND].mean(1)], axis=1).astype(np.float32)

feats_tr = {
    "base_hist_D6": hist6(Htr),
    "drift_only": drift_tr,
    "mean_full": mean_full_tr,
    "mean_full_plus_drift": np.concatenate([mean_full_tr, drift_tr], axis=1),
    "mean_first_half": mean_first_tr,
    "mean_second_half": mean_second_tr,
    "mean_both": mean_both_tr,
}
feats_te = {
    "base_hist_D6": hist6(Hte),
    "drift_only": drift_te,
    "mean_full": mean_full_te,
    "mean_full_plus_drift": np.concatenate([mean_full_te, drift_te], axis=1),
    "mean_first_half": mean_first_te,
    "mean_second_half": mean_second_te,
    "mean_both": mean_both_te,
}

CONDITIONS = ["static_only", "base_hist_D6", "drift_only", "mean_full",
              "mean_full_plus_drift", "mean_first_half", "mean_second_half",
              "mean_both"]

results = {}
preds = {}
for cname in CONDITIONS:
    if cname == "static_only":
        p = predict(Xtr, ytr, Xte)
    else:
        p = predict(np.concatenate([Xtr, feats_tr[cname]], axis=1), ytr,
                    np.concatenate([Xte, feats_te[cname]], axis=1))
    results[cname] = metrics(yte, p)
    preds[cname] = p.astype(np.float64)

base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
print(f"10 ns: static {results['static_only']['auprc']:.4f} "
      f"full {results['base_hist_D6']['auprc']:.4f} "
      f"gain {base_gain:.4f} (ref {REF_GAIN})", flush=True)

gains = {c: results[c]["auprc"] - results["static_only"]["auprc"] for c in CONDITIONS[1:]}
recovery = {c: float(gains[c] / base_gain) for c in CONDITIONS[1:]}
gfull = gains["mean_full"]
ratio = {c: float(gains[c] / gfull) for c in CONDITIONS[1:]}
for c in CONDITIONS[1:]:
    print(f"  {c}: auprc {results[c]['auprc']:.4f} gain {gains[c]:+.4f} "
          f"recovery {recovery[c]:.3f} ratio_to_full {ratio[c]:.3f}", flush=True)

diff_c_b = results["mean_full_plus_drift"]["auprc"] - results["mean_full"]["auprc"]
verdict = "漂いは足しになる" if diff_c_b >= 0.01 else "足しにならない"
print(f"  (c) - (b) auprc diff: {diff_c_b:+.4f} -> {verdict}", flush=True)
print(f"  (c) - mean_both auprc diff: "
      f"{results['mean_full_plus_drift']['auprc'] - results['mean_both']['auprc']:+.4f}", flush=True)

# タンパク質ごと（30 タンパク質）
pp = {}
for pname, (a, b) in span_ranges.items():
    entry = {}
    for cname in CONDITIONS:
        entry[cname] = metrics(yte[a:b], preds[cname][a:b])
    g_pp = {c: entry[c]["auprc"] - entry["static_only"]["auprc"] for c in CONDITIONS[1:]}
    base_gain_pp = entry["base_hist_D6"]["auprc"] - entry["static_only"]["auprc"]
    entry["_gain"] = {c: float(g_pp[c]) for c in CONDITIONS[1:]}
    entry["_recovery"] = {c: float(g_pp[c] / base_gain_pp) if base_gain_pp != 0 else float("nan")
                          for c in CONDITIONS[1:]}
    entry["_ratio_to_full"] = {c: float(g_pp[c] / g_pp["mean_full"]) if g_pp["mean_full"] != 0 else float("nan")
                               for c in CONDITIONS[1:]}
    entry["_diff_c_minus_b"] = float(entry["mean_full_plus_drift"]["auprc"] - entry["mean_full"]["auprc"])
    pp[pname] = entry
n_pos_diff = sum(1 for e in pp.values() if e["_diff_c_minus_b"] >= 0.01)
n_neg_diff = sum(1 for e in pp.values() if e["_diff_c_minus_b"] < 0)
diffs = sorted(e["_diff_c_minus_b"] for e in pp.values())
med = float(np.median(diffs))
print(f"per-protein: (c)-(b) >= +0.01 in {n_pos_diff}/30, "
      f"(c)-(b) < 0 in {n_neg_diff}/30, median {med:+.4f}, "
      f"range [{diffs[0]:+.4f}, {diffs[-1]:+.4f}]", flush=True)

# 土台の再現の確認
assert abs(results["static_only"]["auprc"] - REF_BASE[0]) < 5e-4
assert abs(results["base_hist_D6"]["auprc"] - REF_BASE[1]) < 5e-4
for cname, ref in REF_K2_50.items():
    assert abs(results[cname]["auprc"] - ref) < 5e-4, (cname, results[cname]["auprc"], ref)
assert results["static_only"]["positives"] == REF_TEST_POS, \
    (results["static_only"]["positives"], REF_TEST_POS)
print("all base checks passed", flush=True)

out = {
    "conditions": {
        "drift_only": {"label": "漂いだけ（前半の平均 − 後半の平均）", "frames_first": FIRST, "frames_second": SECOND},
        "mean_full": {"frames": FULL, "label": "フレーム [0..10] の平均だけ（全部）"},
        "mean_full_plus_drift": {"label": "全部の平均＋漂い"},
        "mean_first_half": {"frames": FIRST, "label": "フレーム [0..4] の平均だけ（前半）"},
        "mean_second_half": {"frames": SECOND, "label": "フレーム [6..10] の平均だけ（後半）"},
        "mean_both": {"frames": [FIRST, SECOND], "label": "前半の平均と後半の平均の 2 つ"},
    },
    "horizon_ns": HORIZON,
    "references": {"base_auprc_ref": REF_BASE, "base_gain_ref": REF_GAIN,
                   "k2_50_auprc_ref": REF_K2_50, "test_positives_ref": REF_TEST_POS},
    "metrics": results,
    "per_protein": pp,
    "base_gain": float(base_gain),
    "gain": {c: float(gains[c]) for c in CONDITIONS[1:]},
    "recovery_vs_real_history_gain": recovery,
    "gain_ratio_to_full_mean": ratio,
    "auprc_diff_c_minus_b": float(diff_c_b),
    "auprc_diff_c_minus_both": float(results["mean_full_plus_drift"]["auprc"] - results["mean_both"]["auprc"]),
    "per_protein_diff_c_minus_b_median": med,
    "per_protein_diff_c_minus_b_range": [float(diffs[0]), float(diffs[-1])],
    "per_protein_diff_c_minus_b_ge_001": int(n_pos_diff),
    "per_protein_diff_c_minus_b_lt_0": int(n_neg_diff),
    "verdict": verdict,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

np.savez_compressed(
    OUT / "test_predictions.npz",
    **{f"{c}": preds[c] for c in CONDITIONS},
    y=yte)

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "auprc", "auc", "brier", "logloss", "gain_vs_static",
                "recovery_vs_full_history_gain", "ratio_to_full_mean_gain"])
    for c in CONDITIONS:
        m = results[c]
        w.writerow([c, round(m["auprc"], 4), round(m["auc"], 4), round(m["brier"], 4),
                    round(m["logloss"], 4),
                    "" if c == "static_only" else round(gains[c], 4),
                    "" if c == "static_only" else round(recovery[c], 3),
                    "" if c == "static_only" else round(ratio[c], 3)])
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
