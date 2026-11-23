"""K27: contact formation, single-statistic conditions (Doc D base).

K21 (hydrogen bonds, strong base) found that the mean alone restores
0.87-1.08 of the full history gain. K16 (contact) found that removing
any single statistic barely reduces the gain. K27 measures the
"add one" side for contact, with the same split as K21, on K16's base.

Base (identical to K13/K16): ATLAS 6 proteins x 3 replicas, train
1bgf_A/1ifg_A/1j8r_A/2b0a_A (15,926 candidates), test 1mix_A + 2qt4_A
pooled (7,384). Contact = minimum heavy-atom distance between residue
pairs; candidates are > 4.5 A at the origin (frame 10, 1 ns); the label
is <= 4.5 A at the horizon. Static features 5. History statistics are
the 6 of the real 11-frame series (0..1 ns), computed exactly as in
K16's hist6(): [first, mean, std, min, last-first, frac <= 4.5 A].
Distances come from K13's extracted cache
(~/workspace/kinescope/work/k13_cache/), where the periodic boundary
was applied per axis with each frame's box.

Conditions (features = static + the listed statistics):
  only_mean    [mean]
  only_min     [min]
  only_frac45  [frac <= 4.5 A]
  shape_only   [std, last-first]  (no position statistic)
  full         all 6 statistics = K16's base_hist condition
  static_only  no history (reference)
Learner is retrained per condition and horizon, identical to K16:
HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
min_samples_leaf 15, L2 0.02, seed 42). Metrics: AUC, AUPRC, Brier,
Log Loss on the pooled test set. Recovery = (condition - static) /
(full - static) in AUPRC.

Results are checkpointed per condition in results_k27_partial.json.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_k27_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.5
# statistic indices in the 6-vector [first, mean, std, min, last-first, frac]
CONDS = {"static_only": None, "full": [0, 1, 2, 3, 4, 5],
         "only_mean": [1], "only_min": [3], "only_frac45": [5],
         "shape_only": [2, 4]}


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact.npz")
            parts.append({k: z[k] for k in z.files})
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
    Ytr, Yte = cat(train, "y"), cat(test, "y")
    Ftr, Fte = hist6(cat(train, "hist")), hist6(cat(test, "hist"))
    print("train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)

    results = {}
    if PARTIAL.exists():
        results = json.loads(PARTIAL.read_text())
        print("resumed:", sorted(results), flush=True)

    for name, cols in CONDS.items():
        if name in results:
            print(name, "already done, skipped", flush=True)
            continue
        results[name] = {}
        for hi, h in enumerate(HORIZONS):
            if cols is None:
                p = predict(Xtr, Ytr[:, hi], Xte)
            else:
                p = predict(np.concatenate([Xtr, Ftr[:, cols]], axis=1), Ytr[:, hi],
                            np.concatenate([Xte, Fte[:, cols]], axis=1))
            results[name][str(h)] = metrics(Yte[:, hi], p)
        print(name, "AUPRC", {h: round(results[name][str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
        PARTIAL.write_text(json.dumps(results, indent=2))

    recovery = {}
    for h in HORIZONS:
        s = results["static_only"][str(h)]["auprc"]
        f = results["full"][str(h)]["auprc"]
        recovery[str(h)] = {name: float((results[name][str(h)]["auprc"] - s) / (f - s))
                            for name in CONDS if name not in ("static_only", "full")}
    results["_recovery"] = recovery
    k16 = json.loads((OUT.parent / "K16_contact_channel" / "results.json").read_text())
    results["_checks_vs_K16"] = {
        "static_max_abs_diff": max(abs(results["static_only"][str(h)]["auprc"] - k16["static_only"][str(h)]["auprc"]) for h in HORIZONS),
        "full_max_abs_diff": max(abs(results["full"][str(h)]["auprc"] - k16["base_hist"][str(h)]["auprc"]) for h in HORIZONS),
    }
    print("checks vs K16:", results["_checks_vs_K16"], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    rows = []
    for name, by_h in results.items():
        if name.startswith("_"):
            continue
        for h, m in by_h.items():
            rows.append(dict(condition=name, horizon_ns=h, **m))
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
