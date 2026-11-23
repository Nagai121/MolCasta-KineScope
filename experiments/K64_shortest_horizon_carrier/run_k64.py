"""K64: shortest horizon (0.1 ns) carrier check, Doc D base.

Same base as K01/K27/K31: ATLAS 6 proteins x 3 replicas, train
1bgf_A/1ifg_A/1j8r_A/2b0a_A (9,610 candidates), test 1mix_A + 2qt4_A
pooled (4,972). Origin frame 10 (1 ns), history 0..1 ns (11 frames),
candidates 3.5 < d <= 6.0 A at the origin (K01's extraction; the
reproduced Doc D candidate rule as implemented in extract.py).
Static features 7. History statistics are the 6 of the real 11-frame
series, computed exactly as in K27/K31's hist6():
[first, mean, std, min, last-first, frac <= 3.5 A].
Distances use the per-frame box, axis-wise, as in K01.

Horizons:
  0.1 ns  label = distance at frame 11 <= 3.5 A (extract_k64.py;
          the frame right after the origin, so the history's last
          frame and the label frame are adjacent).
  1 ns    label = y[:, 0] from the K01 cache (reference, must
          reproduce the base: static 0.3637, full 0.4815 AUPRC).

Conditions (features = static + the listed statistics):
  static_only  no history
  full         all 6 statistics
  only_mean    [mean] only (same split as K21's "mean only", but on
               the Doc D base, as in K27/K31)
Learner retrained per condition and horizon, identical to K01/K27:
HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
min_samples_leaf 15, L2 0.02, seed 42). Metrics: AUC, AUPRC, Brier,
Log Loss on the pooled test set. Recovery = (only_mean - static) /
(full - static) in AUPRC.

Results are checkpointed per condition in results_k64_partial.json.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

os.environ.setdefault("OMP_NUM_THREADS", "2")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
Y01 = Path("/home/hatch/workspace/kinescope/work/k64_cache")
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_k64_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 3.5
CONDS = {"static_only": None, "full": [0, 1, 2, 3, 4, 5], "only_mean": [1]}
# horizon key -> (label column source, label index in that source)
HORIZONS = {"0.1": ("y01", None), "1": ("y", 0)}


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            z01 = np.load(Y01 / f"{p}_R{r}_y01.npz")
            # K01 cache stores hist as (11 frames, n candidates)
            parts.append({"X": z["X"], "hist": z["hist"].T, "y": z["y"],
                          "y01": z01["y01"]})
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def main():
    train, test = load(TRAIN_P), load(TEST_P)
    Xtr, Xte = cat(train, "X"), cat(test, "X")
    Ftr, Fte = hist6(cat(train, "hist")), hist6(cat(test, "hist"))
    labels = {}
    for hk, (src, idx) in HORIZONS.items():
        if idx is None:
            labels[hk] = (cat(train, src), cat(test, src))
        else:
            labels[hk] = (cat(train, src)[:, idx], cat(test, src)[:, idx])
    print("train", Xtr.shape, "test", Xte.shape,
          {hk: (int(v[0].sum()), int(v[1].sum())) for hk, v in labels.items()}, flush=True)

    results = {}
    if PARTIAL.exists():
        results = json.loads(PARTIAL.read_text())
        print("resumed:", sorted(results), flush=True)

    for name, cols in CONDS.items():
        if name in results and all(hk in results[name] for hk in HORIZONS):
            print(name, "already done, skipped", flush=True)
            continue
        results.setdefault(name, {})
        for hk in HORIZONS:
            if hk in results[name]:
                continue
            ytr, yte = labels[hk]
            if cols is None:
                p = predict(Xtr, ytr, Xte)
            else:
                p = predict(np.concatenate([Xtr, Ftr[:, cols]], axis=1), ytr,
                            np.concatenate([Xte, Fte[:, cols]], axis=1))
            results[name][hk] = metrics(yte, p)
        print(name, "AUPRC", {hk: round(results[name][hk]["auprc"], 4) for hk in HORIZONS}, flush=True)
        PARTIAL.write_text(json.dumps(results, indent=2))

    gain = {}
    recovery = {}
    for hk in HORIZONS:
        s = results["static_only"][hk]["auprc"]
        f = results["full"][hk]["auprc"]
        gain[hk] = float(f - s)
        recovery[hk] = float((results["only_mean"][hk]["auprc"] - s) / (f - s)) if f != s else float("nan")
    results["_gain"] = gain
    results["_recovery_mean_only"] = recovery
    # base check: 1 ns must reproduce the Doc D base values
    k01 = json.loads((OUT.parent / "K01_time_degradation" / "results.json").read_text())
    results["_checks_vs_K01_1ns"] = {
        "static_max_abs_diff": abs(results["static_only"]["1"]["auprc"] - k01["static_only"]["1"]["auprc"]),
        "full_max_abs_diff": abs(results["full"]["1"]["auprc"] - k01["base_hist_D6"]["1"]["auprc"]),
    }
    print("gain:", {k: round(v, 4) for k, v in gain.items()},
          "recovery:", {k: round(v, 4) for k, v in recovery.items()},
          "checks vs K01:", results["_checks_vs_K01_1ns"], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    rows = []
    for name, by_h in results.items():
        if name.startswith("_"):
            continue
        for hk, m in by_h.items():
            rows.append(dict(condition=name, horizon_ns=hk, **m))
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
