"""L2-24 h5: 保存済み OOF 予測から回復率のタンパク質ごとのブートストラップ信頼区間。
sklearn の average_precision_score は遅いので、numpy による同じ計算
(塩橋のブートストラップで sklearn と 1e-16 まで一致を確認済みの ap_fast)
で 10,000 回回す。"""
import numpy as np
from pathlib import Path

D = Path("/home/hatch/workspace/kinescope/repo/experiments/L2-24_hb_dissociation_mean_only_e01_public_base_adjacent_window")
z = np.load(D / "oof_predictions_5ns.npz")
y = z["label"].astype(np.float64)
prots = z["proteins"]
ps = z["static_only"].astype(np.float64)
pm = z["window_mean_1.0ns"].astype(np.float64)
pf = z["full_history"].astype(np.float64)


def ap_fast(y, s):
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec[ys == 1]) / ys.sum())


def rec_fast(idx):
    a_s = ap_fast(y[idx], ps[idx])
    a_m = ap_fast(y[idx], pm[idx])
    a_f = ap_fast(y[idx], pf[idx])
    return (a_m - a_s) / (a_f - a_s)


print("point estimate (numpy AP): %.6f" % rec_fast(np.arange(len(y))), flush=True)
uniq = np.unique(prots)
idx_by_prot = {p: np.where(prots == p)[0] for p in uniq}
rng = np.random.default_rng(42)
recs = np.empty(10000)
for b in range(10000):
    chosen = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([idx_by_prot[p] for p in chosen])
    recs[b] = rec_fast(idx)
    if (b + 1) % 2000 == 0:
        print(f"... {b + 1} draws", flush=True)
lo, hi = np.percentile(recs, [2.5, 97.5])
print("bootstrap 95%% CI for recovery (protein-level): [%.4f, %.4f]" % (lo, hi), flush=True)
print("E01 recovery 0.9473 inside CI: %s" % (lo <= 0.947291 <= hi), flush=True)
print("fraction of draws >= E01: %.4f" % float(np.mean(recs >= 0.947291)), flush=True)
