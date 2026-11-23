"""G33: strong base - does mixing help with the LONGER 0.5 ns prefix?
(K06(b) conditions, hydrogen-bond formation, 1 ns)

G21 found on the strong base that mixing the 0.1 ns prefix mean with
the generated mean HURT recovery (-0.083 pooled / -0.062 macro),
opposite to the weak base (G11: +0.186). But the prefix there was
only 0.1 ns. With the 0.5 ns prefix, whose mean alone recovers
0.695/0.686 (G05), the mixture may come back. This experiment asks
the same question with the 0.5 ns prefix.

Series: the 11-point series G05 re-extracted. Generated mean: the
mean column of K06(b)'s saved generated 6 features (r{r}_gen6[:, 1]),
the mean of proposal 22's 5-point forward series (origin +0.1/0.3/
0.5/1.0 ns). The prefix mean is the backward history's first part.
The mean does not depend on direction (same caveat as G11/G21).

Base (K06(b)/G05): frozen T1I2006 + correction HistGB (160 iters,
lr .065, 15 leaves, min_samples_leaf 20, L2 .15, seed 20261121).
Train on the 8 calibration proteins, evaluate on the 100 unseen
proteins x 3 replicas (163,074 candidates, 11,712 positives at 1 ns).
Recovery = (condition AUPRC - static-only) / (real-history gain),
pooled and per-protein macro. Full gain pooled +0.0702, macro +0.0712.

Pre-fixed reading: if the mixed recovery rises >= +0.05 above the
prefix-mean-only recovery, mixing helps on the strong base with a
longer prefix. If below +0.05, mixing does not help on the strong
base even with a longer prefix.
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
OUT = Path(__file__).resolve().parent
SEED = 20261121
H = 0  # 1 ns
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5]  # 0.5 ns の切り取り（G05 と同じ）

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

prefix_cal = cal["series"][PREFIX_FRAMES].mean(axis=0)
prefix_test = test["series"][PREFIX_FRAMES].mean(axis=0)
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
    "prefix_mean_0.5ns": (np.column_stack([cal["static"], col(prefix_cal)]),
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
assert abs(results["prefix_mean_0.5ns"]["auprc"] - 0.5079) < 5e-4, \
    results["prefix_mean_0.5ns"]
assert abs(results["prefix_mean_0.5ns"]["macro_auprc"] - 0.5027) < 5e-4, \
    results["prefix_mean_0.5ns"]
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
                 - recovery["prefix_mean_0.5ns"]["pooled"]),
    macro=float(recovery["mixed"]["macro"]
                - recovery["prefix_mean_0.5ns"]["macro"]))
print(f"mixed - prefix: pooled {delta['pooled']:+.3f} macro {delta['macro']:+.3f}",
      flush=True)

out = dict(metrics_1ns=results, recovery=recovery, mixed_minus_prefix=delta,
           relations=relations,
           base_gain_1ns=dict(pooled=float(gain_p), macro=float(gain_m)))
(OUT / "results.json").write_text(json.dumps(out, indent=2))
print("saved to", OUT)
