"""G74: 水素結合の形成で、真ん中の予測幅でも短い切り取りで指導力が戻るか
（資料Dの条件、水素結合の形成）.

G02 は 1 ns先だけ、G03 は 90 ns先だけだったので、真ん中の予測幅
（5・10・30・50 ns）での形が空白だった。ここを埋める。

切り取りの定義（G02/G03 と同じ、結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 0.1 ns -> フレーム 0, 1（2フレーム、覆う長さ 0.1 ns）
- 0.25 ns -> フレーム 0, 1, 2（3フレーム、覆う長さは 0.2 ns）
- 0.5 ns -> フレーム 0〜5（6フレーム、覆う長さ 0.5 ns）
- 起点フレーム10（1 ns）は変えない。

条件は「静的特徴＋切った平均だけ」（K21 の「平均だけ」と同じ形）。
回復率 = (その条件の AUPRC - 静的のみ) / (その予測幅での実履歴の利得)。
土台の利得は K01・資料Dの記録値（再計算でも確認する）:
  1 ns +0.1178、5 ns +0.1155、10 ns +0.0829、30 ns +0.0626、
  50 ns +0.0739、90 ns +0.1211。

読み方（事前に固定）: 各予測幅で、回復率 0.5 を超える最小の切り取りを出す。
3つの切り取りのどれでも 0.5 を超えなければ「0.5 ns まででも届かない」。
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)

# 切り取り: 名前 -> (表示する長さ, フレームの列, 実際に覆う長さ)
CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
}

# K01・資料Dの記録値（土台の利得、AUPRC の差）
REF_GAIN = {1: 0.1178, 5: 0.1155, 10: 0.0829, 30: 0.0626, 50: 0.0739, 90: 0.1211}


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


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


train = load(TRAIN_P)
test = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, flush=True)

full_mean_tr = Htr.mean(1)
full_mean_te = Hte.mean(1)

# (a) 残りのばらつき（G02 と同じ量）
residual = {}
for name, (label, idx, covered) in CUTS.items():
    pm_tr = Htr[:, idx].mean(1)
    pm_te = Hte[:, idx].mean(1)
    residual[name] = dict(
        cut_label=label, frames=idx, covered_length=covered,
        train_residual_std=float(np.std(pm_tr - full_mean_tr)),
        test_residual_std=float(np.std(pm_te - full_mean_te)),
        train_residual_mean=float(np.mean(pm_tr - full_mean_tr)),
        test_residual_mean=float(np.mean(pm_te - full_mean_te)),
    )
    print(name, "residual std test",
          round(residual[name]["test_residual_std"], 4), flush=True)

# (b) 各予測幅での回復率
all_metrics = {}
all_recovery = {}
all_gain = {}
for hi, h in enumerate(HORIZONS):
    ytr, yte = Ytr[:, hi], Yte[:, hi]
    conditions = {}
    conditions["static_only"] = None
    conditions["base_hist_D6"] = (hist6(Htr), hist6(Hte))
    conditions["mean_only_full"] = (
        full_mean_tr[:, None].astype(np.float32),
        full_mean_te[:, None].astype(np.float32))
    for name, (label, idx, covered) in CUTS.items():
        conditions[f"prefix_mean_{name}"] = (
            Htr[:, idx].mean(1)[:, None].astype(np.float32),
            Hte[:, idx].mean(1)[:, None].astype(np.float32))

    results = {}
    for cname, feats in conditions.items():
        if feats is None:
            p = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = feats
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                        np.concatenate([Xte, fte], axis=1))
        results[cname] = metrics(yte, p)
    all_metrics[str(h)] = results

    base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
    all_gain[str(h)] = float(base_gain)
    print(f"{h} ns: static {results['static_only']['auprc']:.4f} "
          f"full {results['base_hist_D6']['auprc']:.4f} "
          f"gain {base_gain:.4f} (ref {REF_GAIN[h]})", flush=True)
    recovery = {}
    for cname, m in results.items():
        if cname == "static_only":
            continue
        recovery[cname] = float((m["auprc"] - results["static_only"]["auprc"]) / base_gain)
    all_recovery[str(h)] = recovery
    for cname in ("prefix_mean_0.1ns", "prefix_mean_0.25ns", "prefix_mean_0.5ns"):
        print(f"  {h} ns {cname}: auprc {results[cname]['auprc']:.4f} "
              f"recovery {recovery[cname]:.3f}", flush=True)

# 土台の再現の確認（1 ns先と 90 ns先）
assert abs(all_metrics["1"]["static_only"]["auprc"] - 0.3637) < 5e-4
assert abs(all_metrics["1"]["base_hist_D6"]["auprc"] - 0.4815) < 5e-4
assert abs(all_metrics["90"]["static_only"]["auprc"] - 0.3035) < 5e-4
assert abs(all_metrics["90"]["base_hist_D6"]["auprc"] - 0.4246) < 5e-4
# G02・G03 との照合
assert abs(all_recovery["1"]["prefix_mean_0.1ns"] - 0.570) < 5e-3, all_recovery["1"]
assert abs(all_recovery["1"]["prefix_mean_0.25ns"] - 0.647) < 5e-3, all_recovery["1"]
assert abs(all_recovery["1"]["prefix_mean_0.5ns"] - 0.770) < 5e-3, all_recovery["1"]
assert abs(all_recovery["90"]["prefix_mean_0.1ns"] - 0.590) < 5e-3, all_recovery["90"]

# 各予測幅で 0.5 を超える最小の切り取り
min_cut = {}
for h in HORIZONS:
    rec = all_recovery[str(h)]
    found = None
    for name in ("0.1ns", "0.25ns", "0.5ns"):
        if rec[f"prefix_mean_{name}"] > 0.5:
            found = name
            break
    min_cut[str(h)] = found
    print(f"{h} ns: min cut over 0.5 = {found}", flush=True)

out = {
    "cuts": {k: v for k, v in CUTS.items()},
    "horizons": list(HORIZONS),
    "residual_spread_vs_full_mean": residual,
    "references": {
        "base_gain_ref": REF_GAIN,
        "base_gain_recomputed": all_gain,
        "g02_1ns_prefix_mean": {"0.1ns": 0.570, "0.25ns": 0.647, "0.5ns": 0.770},
        "g03_90ns_prefix_mean_0.1ns": 0.590,
    },
    "metrics": all_metrics,
    "base_gain": all_gain,
    "recovery_vs_real_history_gain": all_recovery,
    "min_cut_over_0.5": min_cut,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

# results.csv: 予測幅 × 切り取りの回復率の表
import csv
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "base_gain", "static_auprc", "full_auprc",
                "prefix_mean_0.1ns_auprc", "prefix_mean_0.1ns_recovery",
                "prefix_mean_0.25ns_auprc", "prefix_mean_0.25ns_recovery",
                "prefix_mean_0.5ns_auprc", "prefix_mean_0.5ns_recovery",
                "mean_only_full_auprc", "mean_only_full_recovery",
                "min_cut_over_0.5"])
    for h in HORIZONS:
        m = all_metrics[str(h)]
        r = all_recovery[str(h)]
        w.writerow([h, round(all_gain[str(h)], 4),
                    round(m["static_only"]["auprc"], 4),
                    round(m["base_hist_D6"]["auprc"], 4),
                    round(m["prefix_mean_0.1ns"]["auprc"], 4), round(r["prefix_mean_0.1ns"], 3),
                    round(m["prefix_mean_0.25ns"]["auprc"], 4), round(r["prefix_mean_0.25ns"], 3),
                    round(m["prefix_mean_0.5ns"]["auprc"], 4), round(r["prefix_mean_0.5ns"], 3),
                    round(m["mean_only_full"]["auprc"], 4), round(r["mean_only_full"], 3),
                    min_cut[str(h)]])
print("saved", OUT / "results.json", OUT / "results.csv")
