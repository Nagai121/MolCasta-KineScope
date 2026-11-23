"""K46: hydrogen-bond DISSOCIATION - does real history guide, and is the carrier the mean?

Conditions (data D, HB dissociation; candidates: origin distance <= 3.5 A,
label: distance > 3.5 A at horizon):
  static_only : 7 static features
  full_hist6  : static + 6 history statistics (first, mean, std, min,
                last-first, fraction <= 3.5 A) - same definition as formation (K01)
  mean_only   : static + mean statistic only (same split as K21/K27)
Also reproduces the FORMATION base (K01 cache) as a control: static and full
hist6 must match the recorded values (1 ns AUPRC 0.3637 / 0.4815) to 4 decimals.
Model: HistGradientBoosting (120 iters, lr 0.075, 31 leaves, min_samples_leaf 15,
L2 0.02, seed 42), retrained per condition and horizon. Results are checkpointed
per horizon into results_k46_partial.json.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

DISS = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
FORM = Path("/home/hatch/workspace/kinescope/work/data_cache")
HERE = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)

def load(cache, prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(cache / f"{p}_R{r}.npz")
            parts.append({k: z[k] for k in z.files})
    return parts

def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)

def hist6(series, bound=3.5):
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

def run_one(tag, cache):
    train = load(cache, TRAIN_P); test = load(cache, TEST_P)
    Xtr = cat(train, "X"); Xte = cat(test, "X")
    Htr = cat(train, "hist"); Hte = cat(test, "hist")
    Ytr = cat(train, "y"); Yte = cat(test, "y")
    print(tag, "train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)
    # sanity: origin distance (last history frame = frame 10) within the candidate range
    print(tag, "origin dist range train", float(Htr[:, -1].min()), float(Htr[:, -1].max()),
          "test", float(Hte[:, -1].min()), float(Hte[:, -1].max()), flush=True)
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
            (HERE / "results_k46_partial.json").write_text(
                json.dumps({"dissociation": out}, indent=2))
    return out

results = {}
results["dissociation"] = run_one("dissociation", DISS)
results["formation_control"] = run_one("formation_control", FORM)

# recovery of the mean-only condition relative to full history gain
rec = {}
for h in HORIZONS:
    s = results["dissociation"]["static_only"][str(h)]["auprc"]
    f = results["dissociation"]["full_hist6"][str(h)]["auprc"]
    m = results["dissociation"]["mean_only"][str(h)]["auprc"]
    rec[str(h)] = dict(gain=float(f - s), mean_only_recovery=float((m - s) / (f - s)) if f != s else None)
results["mean_only_recovery"] = rec
(HERE / "results.json").write_text(json.dumps(results, indent=2))
print("recovery", {h: round(v["mean_only_recovery"], 3) for h, v in rec.items()})
print("saved", HERE / "results.json")
