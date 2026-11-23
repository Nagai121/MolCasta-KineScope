"""K53: salt-bridge DISSOCIATION on the K15 expanded set - does real history
guide, and is the carrier the mean?

Conditions (K15 split, salt-bridge dissociation; candidates: origin min
distance <= 4.0 A, label: min distance > 4.0 A at horizon):
  static_only : 5 static features
  full_hist6  : static + 6 history statistics (first, mean, std, min,
                last-first, fraction <= 4.0 A) - same definition as formation
  mean_only   : static + mean statistic only (same split as K21/K27/K31/K46)
Also reproduces the FORMATION base (K15 cache) as a control: static and full
hist6 must match the recorded K15 values (1 ns AUPRC 0.2402 / 0.3842) to
4 decimals. Model: HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
min_samples_leaf 15, L2 0.02, seed 42), retrained per condition and horizon.
Results are checkpointed per horizon into results_k53_partial.json.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

DISS = Path("/home/hatch/workspace/kinescope/work/k53_cache")
FORM = Path("/home/hatch/workspace/kinescope/work/k15_cache")
HERE = Path(__file__).resolve().parent
_SPLIT = json.loads((HERE.parent / "K15_saltbridge_expansion" / "split.json").read_text())
TRAIN_P = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"])
TEST_P = tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.0


def load(cache, prots, suffix):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(cache / f"{p}_R{r}_{suffix}.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts]) if parts else np.empty((0,))


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                       max_leaf_nodes=31, min_samples_leaf=15,
                                       l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def run_one(tag, cache, suffix, partial_name):
    train = load(cache, TRAIN_P, suffix); test = load(cache, TEST_P, suffix)
    Xtr = cat(train, "X"); Xte = cat(test, "X")
    Htr = cat(train, "hist"); Hte = cat(test, "hist")
    Ytr = cat(train, "y"); Yte = cat(test, "y")
    print(tag, "train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)
    print(tag, "origin dist range train", float(Htr[:, -1].min()), float(Htr[:, -1].max()),
          "test", float(Hte[:, -1].min()), float(Hte[:, -1].max()), flush=True)
    # distribution of the <= 4.0 A fraction statistic (mostly saturated for dissociation)
    frac = (Hte <= BOUND).mean(1)
    print(tag, "test frac<=4A mean", float(frac.mean()), "median", float(np.median(frac)),
          "all-11 frac", float((frac == 1.0).mean()), flush=True)
    Ftr = hist6(Htr); Fte = hist6(Hte)
    conds = {
        "static_only": (None, None),
        "full_hist6": (Ftr, Fte),
        "mean_only": (Ftr[:, [1]], Fte[:, [1]]),
    }
    out = {}
    for name, (ftr, fte) in conds.items():
        out[name] = {}
        for hi, h in enumerate(HORIZONS):
            ytr = Ytr[:, hi]; yte = Yte[:, hi]
            if ftr is None:
                p = predict(Xtr, ytr, Xte)
            else:
                p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                            np.concatenate([Xte, fte], axis=1))
            out[name][str(h)] = metrics(yte, p)
            print(tag, name, h, {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in out[name][str(h)].items()}, flush=True)
            (HERE / partial_name).write_text(json.dumps({tag: out}, indent=2))
    return out


results = {}
results["dissociation"] = run_one("dissociation", DISS, "salt_diss", "results_k53_partial.json")
results["formation_control"] = run_one("formation_control", FORM, "salt", "results_k53_partial.json")

rec = {}
for h in HORIZONS:
    s = results["dissociation"]["static_only"][str(h)]["auprc"]
    f = results["dissociation"]["full_hist6"][str(h)]["auprc"]
    m = results["dissociation"]["mean_only"][str(h)]["auprc"]
    rec[str(h)] = dict(gain=float(f - s),
                       mean_only_recovery=float((m - s) / (f - s)) if f != s else None)
results["mean_only_recovery"] = rec
(HERE / "results.json").write_text(json.dumps(results, indent=2))
print("recovery", {h: round(v["mean_only_recovery"], 3) for h, v in rec.items()})

# formation control check against K15 recorded values (1 ns AUPRC)
fc = results["formation_control"]
print("formation control 1ns static", round(fc["static_only"]["1"]["auprc"], 4),
      "full", round(fc["full_hist6"]["1"]["auprc"], 4), "(recorded 0.2402 / 0.3842)")

rows = []
import csv
for tag, res in results.items():
    if tag == "mean_only_recovery":
        continue
    for name, by_h in res.items():
        for h, m in by_h.items():
            rows.append(dict(split=tag, condition=name, horizon_ns=h, **m))
with open(HERE / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved", HERE / "results.json")
