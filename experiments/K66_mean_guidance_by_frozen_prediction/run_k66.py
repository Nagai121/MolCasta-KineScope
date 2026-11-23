"""K66: mean-only guidance split by frozen-body prediction level.

Strong base (K06(b)): frozen T1I2006 + correction model (HistGB, 160 iters,
lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained
on 8 calibration proteins, evaluated on 100 unseen proteins x 3 replicas
(163,074 candidates). Reuses K06(b)'s saved features
(~/workspace/k06b/run/k06b_out/). No trajectory re-acquisition, no
checkpoint loading.

Split (fixed before seeing results): candidates are split by the frozen
body's prediction probability (r*_frozen, column 0 of the static features).
Threshold = median of the frozen probability over the FULL evaluation set
(163,074 candidates). The SAME threshold is applied to the calibration set
(so "high" means the same probability cutoff on both sides). Groups:
  high: frozen >= median
  low:  frozen < median
For each group the correction model is retrained on the calibration subset
of that group only, and evaluated on the test subset of that group only
(same as K55's band split).

Conditions (per group, retrained per horizon):
  static:   static 5 features (frozen prob, origin distance, 3 base descriptors)
  history:  static + real history 10 features (K06(b) real-history condition)
  mean_only: static + history mean column only (history[:,1]),
             i.e. K21's condition (a), NOT a K12-style swap: the other
             history statistics are omitted entirely, not replaced.
             Difference from K12: K12 replaced one column of the generated
             6 statistics with the real value; here the mean column alone
             is added to the static features, exactly as in K21.

Metrics: micro AUC/AUPRC/Brier/LogLoss and per-protein macro AUPRC
(K06(b)/K12 definition). Recovery = (mean_only - static) / (history - static).
"""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HORIZONS = (1, 5, 10, 30, 50, 90)
SEED = 20261121
OUTD = Path("/home/hatch/workspace/k06b/run/k06b_out")
HERE = Path(__file__).resolve().parent
PARTIAL = HERE / "results_k66_partial.json"
orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]
MEAN_COL = 1  # history[:,1] = mean of the real 11-frame series


def load(proteins):
    rows = []
    for p in proteins:
        with np.load(OUTD / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({k: z[f"r{r}_{k}"] for k in
                             ("static", "history", "frozen", "label")})
    return rows


def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


CONDS = {
    "static": lambda r: r["static"],
    "history": lambda r: np.column_stack([r["static"], r["history"]]),
    "mean_only": lambda r: np.column_stack([r["static"], r["history"][:, MEAN_COL:MEAN_COL + 1]]),
}


def macro_auprc(y, p, spans):
    vals = [average_precision_score(y[s:e], p[s:e])
            for _, s, e in spans if 0 < y[s:e].sum() < e - s]
    return float(np.mean(vals)), int(len(vals))


def eval_group(cal_rows, test_rows, spans, group):
    """Evaluate the three conditions for one group; returns dict keyed by horizon."""
    out = {}
    y_cal = np.concatenate([r["label"] for r in cal_rows])
    y_test = np.concatenate([r["label"] for r in test_rows])
    frozen_test = np.concatenate([r["frozen"] for r in test_rows])
    for j, h in enumerate(HORIZONS):
        row = {}
        row["n_cal"] = int(len(y_cal))
        row["n_test"] = int(len(y_test))
        row["pos_cal"] = int(y_cal[:, j].sum())
        row["pos_test"] = int(y_test[:, j].sum())
        row["frozen"] = metrics(y_test[:, j], frozen_test)
        fm, fn = macro_auprc(y_test[:, j], frozen_test, spans)
        row["frozen"]["macro_auprc"] = fm
        row["frozen"]["macro_n_proteins"] = fn
        for name, fn_ in CONDS.items():
            Xtr = np.concatenate([fn_(r) for r in cal_rows])
            Xte = np.concatenate([fn_(r) for r in test_rows])
            pred = learner(Xtr, y_cal[:, j]).predict_proba(Xte)[:, 1]
            m = metrics(y_test[:, j], pred)
            mm, mn = macro_auprc(y_test[:, j], pred, spans)
            m["macro_auprc"] = mm
            m["macro_n_proteins"] = mn
            row[name] = m
            print(group, h, name, {k: round(v, 4) for k, v in m.items()}, flush=True)
        out[str(h)] = row
    return out


def main():
    cal, test = load(CAL), load(TEST)
    frozen_test = np.concatenate([r["frozen"] for r in test])
    frozen_cal = np.concatenate([r["frozen"] for r in cal])
    median = float(np.median(frozen_test))
    print("median frozen prob (test set):", median, flush=True)
    print("cal frozen median:", float(np.median(frozen_cal)),
          "cal frac >= median:", float((frozen_cal >= median).mean()), flush=True)
    print("test frac >= median:", float((frozen_test >= median).mean()), flush=True)

    spans, cur = [], 0
    for p in TEST:
        with np.load(OUTD / f"{p}.npz") as z:
            n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n))
        cur += n

    if PARTIAL.is_file():
        results = json.loads(PARTIAL.read_text())
    else:
        results = {"median_frozen": median, "n_cal": int(len(frozen_cal)),
                   "n_test": int(len(frozen_test)), "groups": {}}
    # control: un-split reproduction (static/history/mean_only on full sets)
    if "control" not in results:
        results["control"] = eval_group(cal, test, spans, "control")
        PARTIAL.write_text(json.dumps(results, indent=1))
    for group in ("high", "low"):
        if group in results["groups"]:
            continue
        # per-replica masks, applied per row:
        cal_g = []
        for r in cal:
            m = r["frozen"] >= median if group == "high" else r["frozen"] < median
            cal_g.append({k: v[m] for k, v in r.items()})
        test_g = []
        for r in test:
            m = r["frozen"] >= median if group == "high" else r["frozen"] < median
            test_g.append({k: v[m] for k, v in r.items()})
        # spans for the subgroup (per protein, in test order)
        spans_g, cur = [], 0
        for p, r in zip(TEST, test_g):
            n = len(r["label"])
            spans_g.append((p, cur, cur + n))
            cur += n
        results["groups"][group] = eval_group(cal_g, test_g, spans_g, group)
        PARTIAL.write_text(json.dumps(results, indent=1))
    # recovery rates per group, macro and micro
    rec = {}
    for group, gres in results["groups"].items():
        for h in HORIZONS:
            row = gres[str(h)]
            for kind, key in (("macro", "macro_auprc"), ("micro", "auprc")):
                denom = row["history"][key] - row["static"][key]
                rec.setdefault(group, {}).setdefault(str(h), {})[kind] = (
                    float((row["mean_only"][key] - row["static"][key]) / denom)
                    if denom != 0 else None)
    results["recovery"] = rec
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print("WROTE results.json")


if __name__ == "__main__":
    main()
