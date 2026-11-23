"""K50: contact DISSOCIATION -- does real history guide, and is the
carrier the mean? (Doc D conditions)

Mirror of K46 (hydrogen-bond dissociation) for contact formation's
counterpart. Dissociation definition, fixed before seeing results
(K46's mirror-image logic):
  candidates: same residue-pair rule as K13 contact formation
    (C-alpha distance <= 12.0 A, sequence separation >= 2, i < j);
    at frame 10 (1 ns) the minimum heavy-atom distance is <= 4.5 A
    (contact formed at the origin). Formation candidates (> 4.5 A)
    and dissociation candidates do not overlap; together they are all
    residue pairs in the rule (verified in extract_k50.py).
  label: minimum distance > 4.5 A at the horizon frame (1 = dissociated).
Static features: same 5 as formation (origin min distance, CA distance,
log1p(sep), n_heavy_i, n_heavy_j). History: 11-frame minimum-distance
series (0..1 ns); 6 statistics [first, mean, std, min, last-first,
frac <= 4.5 A], same definition as K13/K16. Distances use the
per-frame box, axis-wise (extraction already applied per-axis PBC).

Conditions (K21/K27/K46 split, mean-only only):
  static_only : static 5 features
  full        : static + all 6 history statistics
  only_mean   : static + the mean statistic only
Model: HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
min_samples_leaf 15, L2 0.02, seed 42), retrained per condition and
horizon. Train = 1bgf_A, 1ifg_A, 1j8r_A, 2b0a_A; test = 1mix_A + 2qt4_A
pooled. Horizons 1/5/10/30/50/90 ns. Metrics: AUC, AUPRC, Brier,
Log Loss. Recovery = (mean-only gain) / (full gain) in AUPRC.

Control: contact FORMATION recomputed with the same code from the K13
cache; it must match K16's recorded values (static 0.3398 -> full
0.5669 at 1 ns).
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")
HERE = Path(__file__).resolve().parent
PARTIAL = HERE / "results_k50_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.5


def load(cache, prots, suffix):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(cache / f"{p}_R{r}_{suffix}.npz")
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


def run_split(cache, suffix, label):
    train, test = load(cache, TRAIN_P, suffix), load(cache, TEST_P, suffix)
    Xtr, Xte = cat(train, "X"), cat(test, "X")
    Htr, Hte = cat(train, "hist"), cat(test, "hist")
    Ytr, Yte = cat(train, "y"), cat(test, "y")
    print(label, "train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)
    Ftr, Fte = hist6(Htr), hist6(Hte)
    res = {}
    for name, cols in (("static_only", None), ("full", [0, 1, 2, 3, 4, 5]), ("only_mean", [1])):
        res[name] = {}
        for hi, h in enumerate(HORIZONS):
            if cols is None:
                p = predict(Xtr, Ytr[:, hi], Xte)
            else:
                p = predict(np.concatenate([Xtr, Ftr[:, cols]], axis=1), Ytr[:, hi],
                            np.concatenate([Xte, Fte[:, cols]], axis=1))
            res[name][str(h)] = metrics(Yte[:, hi], p)
        print(label, name, "AUPRC", {h: round(res[name][str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
    res["_split"] = dict(n_train=int(len(Xtr)), n_test=int(len(Xte)),
                         test_positives=[int(v) for v in Yte.sum(0)],
                         train_positives=[int(v) for v in Ytr.sum(0)])
    return res


def main():
    results = {}
    if PARTIAL.exists():
        results = json.loads(PARTIAL.read_text())
        print("resumed:", sorted(results), flush=True)
    if "dissociation" not in results:
        results["dissociation"] = run_split(CACHE, "contact_diss", "dissociation")
        PARTIAL.write_text(json.dumps(results, indent=2))
    if "formation_control" not in results:
        results["formation_control"] = run_split(K13, "contact", "formation_control")
        PARTIAL.write_text(json.dumps(results, indent=2))

    recovery = {}
    for h in HORIZONS:
        s = results["dissociation"]["static_only"][str(h)]["auprc"]
        f = results["dissociation"]["full"][str(h)]["auprc"]
        m = results["dissociation"]["only_mean"][str(h)]["auprc"]
        recovery[str(h)] = float((m - s) / (f - s)) if abs(f - s) > 1e-9 else None
    results["_recovery_mean_only"] = recovery
    k16 = json.loads((HERE.parent / "K16_contact_channel" / "results.json").read_text())
    results["_checks_vs_K16_formation"] = {
        "static_max_abs_diff": max(
            abs(results["formation_control"]["static_only"][str(h)]["auprc"] - k16["static_only"][str(h)]["auprc"])
            for h in HORIZONS),
        "full_max_abs_diff": max(
            abs(results["formation_control"]["full"][str(h)]["auprc"] - k16["base_hist"][str(h)]["auprc"])
            for h in HORIZONS),
    }
    print("recovery (mean only):", {h: round(v, 3) for h, v in recovery.items()}, flush=True)
    print("checks vs K16 (formation):", results["_checks_vs_K16_formation"], flush=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    rows = []
    for split, res in results.items():
        if split.startswith("_"):
            continue
        for name, by_h in res.items():
            if name.startswith("_"):
                continue
            for h, m in by_h.items():
                rows.append(dict(split=split, condition=name, horizon_ns=h, **m))
    with open(HERE / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved to", HERE)


if __name__ == "__main__":
    main()
