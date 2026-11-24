"""K2-139: K2-14 の問い（割合の前半と後半）を塩橋の形成の 5 ns先で
測る（K15 の条件、塩橋の形成、5 ns先、割合だけ）.

塩橋の形成の割合の前半と後半は、1 ns先（K2-135：前半 0.650・後半
0.443）、30 ns先（K2-137：前半 0.597・後半 0.312）、90 ns先
（K2-136：前半 0.418・後半 0.444）で測り終えている。ここでは 5 ns先
を測る。平均の前半と後半は 5 ns先ではまだ測っていない。

条件（結果を見る前に固定）:
- static_only: 静的な特徴量だけ
- base_hist_D6: 静的な特徴量＋履歴全部の 6 個の統計量（土台の検証と、
  回復率の分母の利得に使う。境界は 4.0 A）
- frac_first_half: 静的な特徴量＋フレーム [0..4] の割合だけ
- frac_second_half: 静的な特徴量＋フレーム [6..10] の割合だけ
- frac_both: 静的な特徴量＋前半の割合と後半の割合の 2 つ
- frac_full: 静的な特徴量＋フレーム [0..10] の割合だけ

割合は、ペアの荷電重原子間の最小距離が塩橋の閾値（4.0 Å）以下の
時間の割合で、6 個の統計量のうちの「4.0 Å 以下だった割合」と同じ
定義である。フレーム 5 は前半にも後半にも入らない（K2-01 と同じ
分け方）。

読み方（事前に固定、K2-01 と同じ）: 利得は各条件の AUPRC −
静的のみの AUPRC。割合の分母は全部入りの利得（静的＋履歴 6 統計量の
利得）を主とし、あわせて全部の割合の利得に対する割合も記録する。
後半の割合が 8 割以上で、前半の割合が 5 割以下なら「今の状態が
効いている」。前半も後半も 8 割以上なら「続いている状態が効いている」。
どちらでもなければ、数値をそのまま記録する。AUPRC の差も記録する。
分母を全部の割合の利得にしても判定が変わる場合は、両方を記録する。

タンパク質ごとの形の数え上げ（「今の状態」・「続いている状態」・
どちらでもない・利得が小さく判定から外したもの）も K2-07 と同じく
記録する。全部入りの利得が 0.01 を超えるタンパク質だけを分類し、
除いた数を記録する。テストのタンパク質のうち陽性が 0 件のものは
AUPRC が決まらないため、別に数えて記録する（K2-135 と同じ扱い）。

注意: 塩橋の形成は土台の利得が小さい（5 ns先で +0.0788）ため、
回復率の読みには留保がつく。AUPRC の差とあわせて読む。
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

K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
BOUND = 4.0
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
HORIZON = 5
HI = 1

HALVES = {
    "frac_first_half": ([0, 1, 2, 3, 4], 1),
    "frac_second_half": ([6, 7, 8, 9, 10], 1),
    "frac_full": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1),
}

# K15 の記録値（土台の AUPRC と利得、照合用。5 ns先）
REF_BASE = (0.2513, 0.3301)
REF_GAIN = 0.0788
# テストの陽性数（K15 の記録、照合用）
REF_TEST_POS = 129


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

def frac_of(H, idx, bound=BOUND):
    """H は (N, 11) の距離系列。指定フレームの境界以下割合を返す (N, 1)。"""
    return (H[:, idx] <= bound).mean(1)[:, None].astype(np.float32)


feats_tr, feats_te = {}, {}
feats_tr["base_hist_D6"], feats_te["base_hist_D6"] = hist6(Htr), hist6(Hte)
for cname, (idx, _) in HALVES.items():
    feats_tr[cname] = frac_of(Htr, idx)
    feats_te[cname] = frac_of(Hte, idx)
feats_tr["frac_both"] = np.stack(
    [frac_of(Htr, HALVES["frac_first_half"][0])[:, 0],
     frac_of(Htr, HALVES["frac_second_half"][0])[:, 0]], axis=1).astype(np.float32)
feats_te["frac_both"] = np.stack(
    [frac_of(Hte, HALVES["frac_first_half"][0])[:, 0],
     frac_of(Hte, HALVES["frac_second_half"][0])[:, 0]], axis=1).astype(np.float32)

CONDITIONS = ["static_only", "base_hist_D6", "frac_first_half",
              "frac_second_half", "frac_both", "frac_full"]

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
print(f"5 ns: static {results['static_only']['auprc']:.4f} "
      f"full {results['base_hist_D6']['auprc']:.4f} "
      f"gain {base_gain:.4f} (ref {REF_GAIN})", flush=True)

gains = {c: results[c]["auprc"] - results["static_only"]["auprc"] for c in CONDITIONS[1:]}
recovery = {c: float(gains[c] / base_gain) for c in CONDITIONS[1:]}
gfull = gains["frac_full"]
ratio = {c: float(gains[c] / gfull) for c in CONDITIONS[1:]}
v = verdict(recovery["frac_first_half"], recovery["frac_second_half"])
v_alt = verdict(ratio["frac_first_half"], ratio["frac_second_half"])
for c in CONDITIONS[1:]:
    print(f"  {c}: auprc {results[c]['auprc']:.4f} gain {gains[c]:+.4f} "
          f"recovery {recovery[c]:.3f} ratio_to_full {ratio[c]:.3f}", flush=True)
print(f"  verdict (vs full history gain): {v}; (vs full frac gain): {v_alt}", flush=True)
print(f"  auprc diff (second - first): "
      f"{results['frac_second_half']['auprc'] - results['frac_first_half']['auprc']:+.4f}", flush=True)
print(f"  auprc diff (both - full): "
      f"{results['frac_both']['auprc'] - results['frac_full']['auprc']:+.4f}", flush=True)

def metrics_pp(y, p):
    """タンパク質ごとの指標。陽性が 0 件（または全件陽性）のタンパク質は
    AUPRC・AUC が決まらないため nan にする（K2-135 と同じ扱い）。"""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    out = dict(n=int(len(y)), positives=int(y.sum()),
               brier=float(brier_score_loss(y, p)),
               logloss=float(log_loss(y, p, labels=[0, 1])))
    if np.unique(y).size < 2:
        out["auprc"] = float("nan")
        out["auc"] = float("nan")
    else:
        out["auprc"] = float(average_precision_score(y, p))
        out["auc"] = float(roc_auc_score(y, p))
    return out


# タンパク質ごと（30 タンパク質）
pp = {}
undefined_auprc = {"zero_positives": [], "all_positive": []}
verdict_counts = {"今の状態が効いている": 0, "続いている状態が効いている": 0, "どちらでもない": 0}
for pname, (a, b) in span_ranges.items():
    entry = {}
    if np.unique(yte[a:b]).size < 2:
        if int(yte[a:b].sum()) == 0:
            undefined_auprc["zero_positives"].append(pname)
        else:
            undefined_auprc["all_positive"].append(pname)
    for cname in CONDITIONS:
        entry[cname] = metrics_pp(yte[a:b], preds[cname][a:b])
    g_pp = {c: entry[c]["auprc"] - entry["static_only"]["auprc"] for c in CONDITIONS[1:]}
    base_gain_pp = entry["base_hist_D6"]["auprc"] - entry["static_only"]["auprc"]
    entry["_gain"] = {c: float(g_pp[c]) for c in CONDITIONS[1:]}
    entry["_recovery"] = {c: float(g_pp[c] / base_gain_pp) if base_gain_pp != 0 else float("nan")
                          for c in CONDITIONS[1:]}
    entry["_ratio_to_full"] = {c: float(g_pp[c] / g_pp["frac_full"]) if g_pp["frac_full"] != 0 else float("nan")
                               for c in CONDITIONS[1:]}
    entry["_verdict"] = verdict(entry["_recovery"]["frac_first_half"],
                                entry["_recovery"]["frac_second_half"])
    verdict_counts[entry["_verdict"]] += 1
    pp[pname] = entry
print("per-protein verdict counts (vs full history gain):", verdict_counts, flush=True)
print("per-protein undefined AUPRC:", undefined_auprc, flush=True)
n_first_ge_08 = sum(1 for e in pp.values() if e["_recovery"]["frac_first_half"] >= 0.8)
n_second_ge_08 = sum(1 for e in pp.values() if e["_recovery"]["frac_second_half"] >= 0.8)
n_first_le_05 = sum(1 for e in pp.values() if e["_recovery"]["frac_first_half"] <= 0.5)

# タンパク質ごとの形の数え上げ（K2-07 と同じ。分母はそのタンパク質の全部入りの利得）
shapes = {"now_state": 0, "continuing": 0, "other": 0, "excluded_gain_le_0.01": 0,
          "undefined_auprc": 0}
prot_frac = {}
for pname, e in pp.items():
    if np.isnan(e["static_only"]["auprc"]):
        shapes["undefined_auprc"] += 1
        continue
    gf = e["_gain"]["base_hist_D6"]
    if gf <= 0.01:
        shapes["excluded_gain_le_0.01"] += 1
        continue
    f = e["_recovery"]["frac_first_half"]
    s2 = e["_recovery"]["frac_second_half"]
    prot_frac[pname] = {"first": float(f), "second": float(s2)}
    if s2 >= 0.8 and f <= 0.5:
        shapes["now_state"] += 1
    elif f >= 0.8 and s2 >= 0.8:
        shapes["continuing"] += 1
    else:
        shapes["other"] += 1
print("per-protein shapes (K2-07 rule):", shapes, flush=True)
print(f"per-protein: first recovery >= 0.8 in {n_first_ge_08}/30, "
      f"second >= 0.8 in {n_second_ge_08}/30, first <= 0.5 in {n_first_le_05}/30", flush=True)

# 土台の再現の確認
assert abs(results["static_only"]["auprc"] - REF_BASE[0]) < 5e-4
assert abs(results["base_hist_D6"]["auprc"] - REF_BASE[1]) < 5e-4
assert results["static_only"]["positives"] == REF_TEST_POS, \
    (results["static_only"]["positives"], REF_TEST_POS)
print("all base checks passed", flush=True)

out = {
    "conditions": {
        "frac_first_half": {"frames": HALVES["frac_first_half"][0], "label": "フレーム [0..4] の割合だけ（前半）"},
        "frac_second_half": {"frames": HALVES["frac_second_half"][0], "label": "フレーム [6..10] の割合だけ（後半）"},
        "frac_both": {"frames": [HALVES["frac_first_half"][0], HALVES["frac_second_half"][0]],
                      "label": "前半の割合と後半の割合の 2 つ"},
        "frac_full": {"frames": HALVES["frac_full"][0], "label": "フレーム [0..10] の割合だけ（全部）"},
    },
    "horizon_ns": HORIZON,
    "bound_A": BOUND,
    "fraction": "fraction of frames with pair distance <= 4.0 A",
    "references": {"base_auprc_ref": REF_BASE, "base_gain_ref": REF_GAIN,
                   "test_positives_ref": REF_TEST_POS},
    "metrics": results,
    "per_protein": pp,
    "per_protein_verdict_counts": verdict_counts,
    "per_protein_shapes": shapes,
    "undefined_auprc": undefined_auprc,
    "per_protein_fractions": prot_frac,
    "base_gain": float(base_gain),
    "gain": {c: float(gains[c]) for c in CONDITIONS[1:]},
    "recovery_vs_real_history_gain": recovery,
    "gain_ratio_to_full_frac": ratio,
    "auprc_diff_second_minus_first": float(results["frac_second_half"]["auprc"] - results["frac_first_half"]["auprc"]),
    "auprc_diff_both_minus_full": float(results["frac_both"]["auprc"] - results["frac_full"]["auprc"]),
    "verdict": {"vs_full_history_gain": v, "vs_full_frac_gain": v_alt},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

np.savez_compressed(
    OUT / "test_predictions.npz",
    **{f"{c}": preds[c] for c in CONDITIONS},
    y=yte)

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "auprc", "auc", "brier", "logloss", "gain_vs_static",
                "recovery_vs_full_history_gain", "ratio_to_full_frac_gain"])
    for c in CONDITIONS:
        m = results[c]
        w.writerow([c, round(m["auprc"], 4), round(m["auc"], 4), round(m["brier"], 4),
                    round(m["logloss"], 4),
                    "" if c == "static_only" else round(gains[c], 4),
                    "" if c == "static_only" else round(recovery[c], 3),
                    "" if c == "static_only" else round(ratio[c], 3)])
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
