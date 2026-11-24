"""K2-29: K2-10 の問い（塩橋の形成の前半と後半、平均だけ）を長い
予測幅（90 ns先）で測る（K15 の広げた条件、塩橋の形成、90 ns先、平均だけ）.

K2-10 は塩橋の形成（K15 の広げた条件、1 ns先）で、割合（全部入り
の利得に対して）は前半 0.545、後半 0.600 で、先に固定した読み方の
どちらにも当てはまらなかった。K2-20（30 ns先）では前半 1.115・
後半 0.290 と、長い予測幅では前半が大きくなる形になった。ここでは
同じ形を 90 ns先で測る。K2-05（水素結合の形成、90 ns先は前半
0.903・後半 0.750）では 30 ns先ほど前半が大きくない。

条件（結果を見る前に固定）:
- static_only: 静的な特徴量だけ
- base_hist_D6: 静的な特徴量＋履歴全部の 6 個の統計量（土台の検証と、
  回復率の分母の利得に使う。境界は 4.0 A）
- mean_first_half: 静的な特徴量＋フレーム [0..4] の平均だけ
- mean_second_half: 静的な特徴量＋フレーム [6..10] の平均だけ
- mean_both: 静的な特徴量＋前半の平均と後半の平均の 2 つ
- mean_full: 静的な特徴量＋フレーム [0..10] の平均だけ
- prefix_mean_0.1ns: 静的な特徴量＋フレーム [0, 1] の平均だけ
  （G60 の記録との照合用。読み方の対象ではない）

フレーム 5 は前半にも後半にも入らない（K2-01 と同じ分け方）。

読み方（事前に固定、K2-01 と同じ）: 利得は各条件の AUPRC −
静的のみの AUPRC。割合の分母は全部入りの利得（静的＋履歴 6 統計量の
利得）を主とし、あわせて全部の平均の利得に対する割合も記録する。
後半の割合が 8 割以上で、前半の割合が 5 割以下なら「今の状態が
効いている」。前半も後半も 8 割以上なら「続いている状態が効いている」。
どちらでもなければ、数値をそのまま記録する。AUPRC の差も記録する。

注意: 塩橋の形成は土台の利得が小さい（90 ns先で +0.0601）ため、
回復率の読みには留保がつく。AUPRC の差とあわせて読む。タンパク質
ごとでは利得が 0 のタンパク質があり、回復率が決まらないものは
判定の数え上げから外し、別に数える（K2-20 の記録で利得 0 の
タンパク質の扱いが不明瞭だった反省から）。
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
BOUND = 4.0
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
HORIZON = 90
HI = 5

HALVES = {
    "mean_first_half": ([0, 1, 2, 3, 4], 1),
    "mean_second_half": ([6, 7, 8, 9, 10], 1),
    "mean_full": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1),
    "prefix_mean_0.1ns": ([0, 1], 1),
}

# K15 の記録値（土台の AUPRC と利得、照合用）
REF_BASE = (0.2136, 0.2736)
REF_GAIN = 0.0601
# G60 の記録値（0.1 ns の切り取り平均だけの回復率、90 ns先、照合用）
REF_G60_PREFIX_01 = 0.654
# テストの陽性数（G19 の記録、照合用）
REF_TEST_POS = 149


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


def verdict(ratio_first, ratio_second):
    if ratio_second >= 0.8 and ratio_first <= 0.5:
        return "今の状態が効いている"
    if ratio_first >= 0.8 and ratio_second >= 0.8:
        return "続いている状態が効いている"
    return "どちらでもない"


def load(prots):
    parts = []
    spans = []
    for p in prots:
        start = sum(s for _, s in spans)
        n = 0
        for r in (1, 2, 3):
            z = np.load(K15 / f"{p}_R{r}_salt.npz")
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

feats_tr, feats_te = {}, {}
feats_tr["base_hist_D6"], feats_te["base_hist_D6"] = hist6(Htr), hist6(Hte)
for cname, (idx, _) in HALVES.items():
    feats_tr[cname] = Htr[:, idx].mean(1)[:, None].astype(np.float32)
    feats_te[cname] = Hte[:, idx].mean(1)[:, None].astype(np.float32)
feats_tr["mean_both"] = np.stack(
    [Htr[:, HALVES["mean_first_half"][0]].mean(1),
     Htr[:, HALVES["mean_second_half"][0]].mean(1)], axis=1).astype(np.float32)
feats_te["mean_both"] = np.stack(
    [Hte[:, HALVES["mean_first_half"][0]].mean(1),
     Hte[:, HALVES["mean_second_half"][0]].mean(1)], axis=1).astype(np.float32)

CONDITIONS = ["static_only", "base_hist_D6", "mean_first_half",
              "mean_second_half", "mean_both", "mean_full", "prefix_mean_0.1ns"]

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
print(f"1 ns: static {results['static_only']['auprc']:.4f} "
      f"full {results['base_hist_D6']['auprc']:.4f} "
      f"gain {base_gain:.4f} (ref {REF_GAIN})", flush=True)

gains = {c: results[c]["auprc"] - results["static_only"]["auprc"] for c in CONDITIONS[1:]}
recovery = {c: float(gains[c] / base_gain) for c in CONDITIONS[1:]}
gfull = gains["mean_full"]
ratio = {c: float(gains[c] / gfull) for c in CONDITIONS[1:]}
v = verdict(recovery["mean_first_half"], recovery["mean_second_half"])
v_alt = verdict(ratio["mean_first_half"], ratio["mean_second_half"])
for c in CONDITIONS[1:]:
    print(f"  {c}: auprc {results[c]['auprc']:.4f} gain {gains[c]:+.4f} "
          f"recovery {recovery[c]:.3f} ratio_to_full {ratio[c]:.3f}", flush=True)
print(f"  verdict (vs full history gain): {v}; (vs full mean gain): {v_alt}", flush=True)
print(f"  auprc diff (second - first): "
      f"{results['mean_second_half']['auprc'] - results['mean_first_half']['auprc']:+.4f}", flush=True)
print(f"  auprc diff (both - full): "
      f"{results['mean_both']['auprc'] - results['mean_full']['auprc']:+.4f}", flush=True)

# タンパク質ごと（30 タンパク質）
pp = {}
verdict_counts = {"今の状態が効いている": 0, "続いている状態が効いている": 0,
                  "どちらでもない": 0, "判定なし（利得 0）": 0}
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
    if base_gain_pp == 0:
        entry["_verdict"] = "判定なし（利得 0）"
    else:
        entry["_verdict"] = verdict(entry["_recovery"]["mean_first_half"],
                                    entry["_recovery"]["mean_second_half"])
    verdict_counts[entry["_verdict"]] += 1
    pp[pname] = entry
print("per-protein verdict counts (vs full history gain):", verdict_counts, flush=True)
n_first_ge_08 = sum(1 for e in pp.values() if e["_recovery"]["mean_first_half"] >= 0.8)
n_second_ge_08 = sum(1 for e in pp.values() if e["_recovery"]["mean_second_half"] >= 0.8)
n_first_le_05 = sum(1 for e in pp.values() if e["_recovery"]["mean_first_half"] <= 0.5)
print(f"per-protein: first recovery >= 0.8 in {n_first_ge_08}/30, "
      f"second >= 0.8 in {n_second_ge_08}/30, first <= 0.5 in {n_first_le_05}/30", flush=True)

# 土台の再現の確認
assert abs(results["static_only"]["auprc"] - REF_BASE[0]) < 5e-4
assert abs(results["base_hist_D6"]["auprc"] - REF_BASE[1]) < 5e-4
assert abs(recovery["prefix_mean_0.1ns"] - REF_G60_PREFIX_01) < 5e-3, \
    (recovery["prefix_mean_0.1ns"], REF_G60_PREFIX_01)
assert results["static_only"]["positives"] == REF_TEST_POS, \
    (results["static_only"]["positives"], REF_TEST_POS)
print("all base checks passed", flush=True)

out = {
    "conditions": {
        "mean_first_half": {"frames": HALVES["mean_first_half"][0], "label": "フレーム [0..4] の平均だけ（前半）"},
        "mean_second_half": {"frames": HALVES["mean_second_half"][0], "label": "フレーム [6..10] の平均だけ（後半）"},
        "mean_both": {"frames": [HALVES["mean_first_half"][0], HALVES["mean_second_half"][0]],
                      "label": "前半の平均と後半の平均の 2 つ"},
        "mean_full": {"frames": HALVES["mean_full"][0], "label": "フレーム [0..10] の平均だけ（全部）"},
    },
    "horizon_ns": HORIZON,
    "references": {"base_auprc_ref": REF_BASE, "base_gain_ref": REF_GAIN,
                   "g60_prefix_0.1ns_recovery": REF_G60_PREFIX_01,
                   "test_positives_ref": REF_TEST_POS},
    "metrics": results,
    "per_protein": pp,
    "per_protein_verdict_counts": verdict_counts,
    "base_gain": float(base_gain),
    "gain": {c: float(gains[c]) for c in CONDITIONS[1:]},
    "recovery_vs_real_history_gain": recovery,
    "gain_ratio_to_full_mean": ratio,
    "auprc_diff_second_minus_first": float(results["mean_second_half"]["auprc"] - results["mean_first_half"]["auprc"]),
    "auprc_diff_both_minus_full": float(results["mean_both"]["auprc"] - results["mean_full"]["auprc"]),
    "verdict": {"vs_full_history_gain": v, "vs_full_mean_gain": v_alt},
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
