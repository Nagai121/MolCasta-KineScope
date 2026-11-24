"""K2-05: K2-01 の形（履歴の前半と後半で指導力がどう違うか）は、
長い予測幅（30・90 ns先）でも同じか（資料Dの条件、水素結合の形成）.

K2-01 は 1 ns先で、履歴の前半（フレーム [0..4]）の平均・後半
（フレーム [6..10]）の平均・全部（フレーム [0..10]）の平均を、
静的な特徴量に 1 つずつ足して比べた。ここでは同じ形を 30 ns先と
90 ns先で測る。

条件（結果を見る前に固定）:
- static_only: 静的な特徴量だけ
- base_hist_D6: 静的な特徴量＋履歴全部の 6 個の統計量（土台の検証と、
  回復率の分母の利得に使う）
- mean_first_half: 静的な特徴量＋フレーム [0..4] の平均だけ
- mean_second_half: 静的な特徴量＋フレーム [6..10] の平均だけ
- mean_full: 静的な特徴量＋フレーム [0..10] の平均だけ

フレーム 5 は前半にも後半にも入らない（K2-01 と同じ分け方）。

読み方（事前に固定）: 利得は各条件の AUPRC − 静的のみの AUPRC。
割合の分母は K2-01 と同じく、全部入りの利得（静的＋履歴 6 統計量の
利得）を主とし、あわせて全部の平均の利得に対する割合も記録する。
後半の割合が 8 割以上で、前半の割合が 5 割以下なら
「今の状態が効いている」。前半も後半も 8 割以上なら「続いている状態が
効いている」。どちらでもなければ、数値をそのまま記録する。

注意: 90 ns先では、資料Dの条件で担い手は割合（境界 3.5 Å より近い
時間の割合）になることが K71 で分かっている。平均だけで測る今回の
形では、その分は拾えない。
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
HORIZONS_ALL = (1, 5, 10, 30, 50, 90)
HORIZONS = (30, 90)

HALVES = {
    "mean_first_half": [0, 1, 2, 3, 4],
    "mean_second_half": [6, 7, 8, 9, 10],
    "mean_full": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# K01・資料Dの記録値（土台の AUPRC と利得、照合用）
REF_BASE = {30: (0.3639, 0.4264), 90: (0.3035, 0.4246)}
REF_GAIN = {30: 0.0626, 90: 0.1211}
# G74 の記録値（フレーム [0..10] の平均だけの回復率、照合用）
REF_G74_11F = {30: 1.258, 90: 0.772}


def load(prots):
    parts = []
    spans = []
    for p in prots:
        start = sum(s for _, s in spans)
        n = 0
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            parts.append({k: z[k] for k in z.files})
            n += len(z["X"])
        spans.append((p, n))
        assert sum(s for _, s in spans) == start + n
    return parts, spans


def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)


def hist6(series, bound=3.5):
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
    p = np.clip(p, 1e-8, 1 - 1e-8)
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


train, _ = load(TRAIN_P)
test, test_spans = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, "test spans", test_spans, flush=True)
assert np.max(np.abs(Xtr[:, 0] - Htr[:, -1])) < 1e-6
assert np.max(np.abs(Xte[:, 0] - Hte[:, -1])) < 1e-6

# テストのタンパク質ごとの範囲
span_ranges = {}
pos = 0
for name, n in test_spans:
    span_ranges[name] = (pos, pos + n)
    pos += n
assert pos == len(Xte)

feats_tr, feats_te = {}, {}
feats_tr["base_hist_D6"], feats_te["base_hist_D6"] = hist6(Htr), hist6(Hte)
for cname, idx in HALVES.items():
    feats_tr[cname] = Htr[:, idx].mean(1)[:, None].astype(np.float32)
    feats_te[cname] = Hte[:, idx].mean(1)[:, None].astype(np.float32)

all_metrics = {}
all_pp = {}
all_gain = {}
all_recovery = {}
all_ratio = {}
all_verdict = {}
preds_out = {}
for h in HORIZONS:
    hi = HORIZONS_ALL.index(h)
    ytr, yte = Ytr[:, hi], Yte[:, hi]
    conditions = {"static_only": None, "base_hist_D6": (feats_tr["base_hist_D6"], feats_te["base_hist_D6"])}
    for cname in HALVES:
        conditions[cname] = (feats_tr[cname], feats_te[cname])

    results = {}
    preds = {}
    for cname, feats in conditions.items():
        if feats is None:
            p = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = feats
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                        np.concatenate([Xte, fte], axis=1))
        results[cname] = metrics(yte, p)
        preds[cname] = p.astype(np.float64)
    preds_out[str(h)] = preds
    all_metrics[str(h)] = results

    base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
    all_gain[str(h)] = float(base_gain)
    print(f"{h} ns: static {results['static_only']['auprc']:.4f} "
          f"full {results['base_hist_D6']['auprc']:.4f} "
          f"gain {base_gain:.4f} (ref {REF_GAIN[h]})", flush=True)

    gains = {c: results[c]["auprc"] - results["static_only"]["auprc"] for c in HALVES}
    recovery = {c: float(gains[c] / base_gain) for c in HALVES}
    all_recovery[str(h)] = recovery
    gfull = gains["mean_full"]
    ratio = {c: float(gains[c] / gfull) for c in HALVES}
    all_ratio[str(h)] = ratio
    # 主な読み方は全部入りの利得に対する割合（= 回復率）で見る（K2-01 と同じ）
    v = verdict(recovery["mean_first_half"], recovery["mean_second_half"])
    v_alt = verdict(ratio["mean_first_half"], ratio["mean_second_half"])
    all_verdict[str(h)] = {"vs_full_history_gain": v, "vs_full_mean_gain": v_alt}
    for c in HALVES:
        print(f"  {h} ns {c}: auprc {results[c]['auprc']:.4f} gain {gains[c]:+.4f} "
              f"recovery {recovery[c]:.3f} ratio_to_full {ratio[c]:.3f}", flush=True)
    print(f"  {h} ns verdict (vs full history gain): {v}; (vs full mean gain): {v_alt}", flush=True)

    # タンパク質ごと
    pp = {}
    for pname, (a, b) in span_ranges.items():
        entry = {}
        for cname in conditions:
            entry[cname] = metrics(yte[a:b], preds[cname][a:b])
        g_pp = {c: entry[c]["auprc"] - entry["static_only"]["auprc"] for c in HALVES}
        base_gain_pp = entry["base_hist_D6"]["auprc"] - entry["static_only"]["auprc"]
        entry["_gain"] = {c: float(g_pp[c]) for c in HALVES}
        entry["_recovery"] = {c: float(g_pp[c] / base_gain_pp) for c in HALVES}
        entry["_ratio_to_full"] = {c: float(g_pp[c] / g_pp["mean_full"]) for c in HALVES}
        entry["_verdict"] = verdict(entry["_recovery"]["mean_first_half"],
                                    entry["_recovery"]["mean_second_half"])
        entry["_verdict_vs_full_mean"] = verdict(entry["_ratio_to_full"]["mean_first_half"],
                                                 entry["_ratio_to_full"]["mean_second_half"])
        pp[pname] = entry
        print(f"  {h} ns {pname}: first recovery {entry['_recovery']['mean_first_half']:.3f} "
              f"second recovery {entry['_recovery']['mean_second_half']:.3f} "
              f"verdict {entry['_verdict']} / vs full mean {entry['_verdict_vs_full_mean']}", flush=True)
    all_pp[str(h)] = pp

# 土台の再現の確認
for h in HORIZONS:
    s_ref, f_ref = REF_BASE[h]
    assert abs(all_metrics[str(h)]["static_only"]["auprc"] - s_ref) < 5e-4, h
    assert abs(all_metrics[str(h)]["base_hist_D6"]["auprc"] - f_ref) < 5e-4, h
    got = all_recovery[str(h)]["mean_full"]
    assert abs(got - REF_G74_11F[h]) < 5e-3, (h, got, REF_G74_11F[h])
print("all base checks passed", flush=True)

out = {
    "conditions": {
        "mean_first_half": {"frames": HALVES["mean_first_half"], "label": "フレーム [0..4] の平均だけ（前半）"},
        "mean_second_half": {"frames": HALVES["mean_second_half"], "label": "フレーム [6..10] の平均だけ（後半）"},
        "mean_full": {"frames": HALVES["mean_full"], "label": "フレーム [0..10] の平均だけ（全部）"},
    },
    "horizons": list(HORIZONS),
    "references": {"base_auprc_ref": {str(h): REF_BASE[h] for h in HORIZONS},
                   "base_gain_ref": {str(h): REF_GAIN[h] for h in HORIZONS},
                   "g74_mean11_recovery": {str(h): REF_G74_11F[h] for h in HORIZONS}},
    "metrics": all_metrics,
    "per_protein": all_pp,
    "base_gain": all_gain,
    "recovery_vs_real_history_gain": all_recovery,
    "gain_ratio_to_full_mean": all_ratio,
    "verdict": all_verdict,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

np.savez_compressed(
    OUT / "test_predictions.npz",
    **{f"h{h}_{c}": preds_out[str(h)][c] for h in HORIZONS for c in preds_out[str(h)]},
    **{f"y_{h}": Yte[:, HORIZONS_ALL.index(h)] for h in HORIZONS})

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "base_gain", "static_auprc", "full_hist6_auprc",
                "first_half_auprc", "first_half_gain", "first_half_recovery", "first_half_ratio_to_full",
                "second_half_auprc", "second_half_gain", "second_half_recovery", "second_half_ratio_to_full",
                "full_mean_auprc", "full_mean_gain", "full_mean_recovery", "verdict"])
    for h in HORIZONS:
        m = all_metrics[str(h)]
        r = all_recovery[str(h)]
        rt = all_ratio[str(h)]
        st = m["static_only"]["auprc"]
        w.writerow([h, round(all_gain[str(h)], 4),
                    round(st, 4), round(m["base_hist_D6"]["auprc"], 4),
                    round(m["mean_first_half"]["auprc"], 4), round(m["mean_first_half"]["auprc"] - st, 4),
                    round(r["mean_first_half"], 3), round(rt["mean_first_half"], 3),
                    round(m["mean_second_half"]["auprc"], 4), round(m["mean_second_half"]["auprc"] - st, 4),
                    round(r["mean_second_half"], 3), round(rt["mean_second_half"], 3),
                    round(m["mean_full"]["auprc"], 4), round(m["mean_full"]["auprc"] - st, 4),
                    round(r["mean_full"], 3), all_verdict[str(h)]["vs_full_history_gain"]])
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
