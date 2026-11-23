"""G21: 強い土台で、混合は足しになるか (K06(b) の条件、水素結合の形成、1 ns先)。

G11 (弱い土台、資料D) の強い土台版。0.1 ns の切り取り平均 (G05 で
0.545/0.553) に、第22案の生成平均 (K06(b) の生成6特徴量のうち平均の列)
を足した混合で回復率を測る。比較のため、切り取り平均だけ (G05 の再現)
と生成平均だけの条件も同じ土台で測る。

系列は G05 で作り直した11点系列を使う。生成平均は前方系列 (起点＋
+0.1/0.3/0.5/1.0 ns の5点) の平均で、切り取り平均は後方履歴の平均。
平均は向きに依らない量 (G11 と同じ留保)。

土台 (K06(b)/G05 と同じ): 凍結 T1I2006 + 補正モデル HistGB
(160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、種 20261121)。
較正8タンパク質で学習、未見100タンパク質×3レプリカ (163,074件) で評価。
回復率 = (条件の AUPRC - 静的のみ) / (実履歴の利得)、全体とタンパク質
ごとの平均の両方。

読み方 (先に固定): 混合の回復率が切り取り平均だけより +0.05 以上
上がれば、混合は強い土台でも足しになる。+0.05 未満なら、混合の
効き目は弱い土台 (資料D) に限られる。
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/g05_series")
OUT = Path("/home/hatch/workspace/g21")
SEED = 20261121
H = 0  # 1 ns

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]


def load(proteins):
    static, hist, label, series, gen6 = [], [], [], [], []
    for p in proteins:
        z = np.load(K06B / f"{p}.npz")
        s = np.load(SERIES / f"{p}.npz")
        for r in (1, 2, 3):
            static.append(z[f"r{r}_static"])
            hist.append(z[f"r{r}_history"])
            label.append(z[f"r{r}_label"])
            series.append(s[f"r{r}_series"])
            gen6.append(z[f"r{r}_gen6"])
    return dict(static=np.concatenate(static), hist=np.concatenate(hist),
                label=np.concatenate(label),
                series=np.concatenate(series, axis=1),
                gen6=np.concatenate(gen6))


def learner(X, y):
    return HistGradientBoostingClassifier(
        max_iter=160, learning_rate=.065, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=.15,
        random_state=SEED).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


cal, test = load(CAL), load(TEST)
y_cal, y_test = cal["label"][:, H], test["label"][:, H]
print("cal", len(y_cal), "test", len(y_test), "test pos", int(y_test.sum()),
      flush=True)

prefix_cal = cal["series"][[0, 1]].mean(axis=0)
prefix_test = test["series"][[0, 1]].mean(axis=0)
full_mean_cal = cal["series"].mean(axis=0)
full_mean_test = test["series"].mean(axis=0)
gen_mean_cal = cal["gen6"][:, 1].astype(np.float64)
gen_mean_test = test["gen6"][:, 1].astype(np.float64)

# sanity: generated mean vs real full mean (K43: corr 0.786 on test)
corr_gm = float(np.corrcoef(gen_mean_test, full_mean_test)[0, 1])
print(f"corr(gen_mean, real full mean) test = {corr_gm:.4f} (K43: 0.786)",
      flush=True)

relations = dict(
    corr_prefix_gen_test=float(np.corrcoef(prefix_test, gen_mean_test)[0, 1]),
    gen_minus_prefix_test_mean=float((gen_mean_test - prefix_test).mean()),
    gen_minus_prefix_test_std=float((gen_mean_test - prefix_test).std()),
    corr_gen_fullmean_test=corr_gm,
    corr_prefix_fullmean_test=float(
        np.corrcoef(prefix_test, full_mean_test)[0, 1]),
)

spans, cur = [], 0
for p in TEST:
    z = np.load(K06B / f"{p}.npz")
    n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
    spans.append((p, cur, cur + n))
    cur += n


def col(a):
    return a[:, None].astype(np.float32)


conditions = {
    "static_only": (cal["static"], test["static"]),
    "full_history": (np.column_stack([cal["static"], cal["hist"]]),
                     np.column_stack([test["static"], test["hist"]])),
    "mean_only_full": (np.column_stack([cal["static"], col(full_mean_cal)]),
                       np.column_stack([test["static"], col(full_mean_test)])),
    "prefix_mean_0.1ns": (np.column_stack([cal["static"], col(prefix_cal)]),
                          np.column_stack([test["static"], col(prefix_test)])),
    "gen_mean_only": (np.column_stack([cal["static"], col(gen_mean_cal)]),
                      np.column_stack([test["static"], col(gen_mean_test)])),
    "mixed": (np.column_stack([cal["static"], col(prefix_cal),
                               col(gen_mean_cal)]),
              np.column_stack([test["static"], col(prefix_test),
                               col(gen_mean_test)])),
}

results = {}
for name, (xtr, xte) in conditions.items():
    pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
    m = metrics(y_test, pred)
    macro = [average_precision_score(y_test[s:e], pred[s:e])
             for _, s, e in spans if 0 < y_test[s:e].sum() < e - s]
    m["macro_auprc"] = float(np.mean(macro))
    results[name] = m
    print(f"{name}: pooled {m['auprc']:.4f} macro {m['macro_auprc']:.4f}",
          flush=True)

# base reproduction checks (k06b_results.json / G05, 1 ns)
assert abs(results["static_only"]["auprc"] - 0.4591) < 5e-4
assert abs(results["full_history"]["auprc"] - 0.5293) < 5e-4
assert abs(results["static_only"]["macro_auprc"] - 0.4539) < 5e-4
assert abs(results["full_history"]["macro_auprc"] - 0.5251) < 5e-4
assert abs(results["prefix_mean_0.1ns"]["auprc"] - 0.4974) < 5e-4, \
    results["prefix_mean_0.1ns"]
print("base + G05 prefix reproduction OK", flush=True)

gain_p = results["full_history"]["auprc"] - results["static_only"]["auprc"]
gain_m = (results["full_history"]["macro_auprc"]
          - results["static_only"]["macro_auprc"])
recovery = {}
for name, m in results.items():
    if name == "static_only":
        continue
    recovery[name] = dict(
        pooled=float((m["auprc"] - results["static_only"]["auprc"]) / gain_p),
        macro=float((m["macro_auprc"] - results["static_only"]["macro_auprc"])
                    / gain_m))
    print(f"{name}: recovery pooled {recovery[name]['pooled']:.3f} "
          f"macro {recovery[name]['macro']:.3f}", flush=True)

delta = dict(
    pooled=float(recovery["mixed"]["pooled"]
                 - recovery["prefix_mean_0.1ns"]["pooled"]),
    macro=float(recovery["mixed"]["macro"]
                - recovery["prefix_mean_0.1ns"]["macro"]))
print(f"mixed - prefix: pooled {delta['pooled']:+.3f} macro {delta['macro']:+.3f}",
      flush=True)

out = dict(metrics_1ns=results, recovery=recovery, mixed_minus_prefix=delta,
           relations=relations,
           base_gain_1ns=dict(pooled=float(gain_p), macro=float(gain_m)))
(OUT / "g21_results.json").write_text(json.dumps(out, indent=2))
print("DONE")
