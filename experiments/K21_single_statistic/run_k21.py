"""K21: single-quantity conditions on the strong base (K06(b)).

K18 dropped one real-history statistic at a time and the gain barely
moved (statistics substitute for each other). K21 goes the other way:
how much of the full gain comes back when only one quantity (or one
group) of the real history is given to the sidecar?

Conditions (features = static(5) + the listed history columns):
  (a) only_mean     history col 1 (mean)
  (b) only_min      history col 3 (minimum)
  (c) only_frac35   history col 7 (fraction <= 3.5 A)
  (d) shape_only    history cols 2 and 5 (std, last-first); no
                    position statistic
  (e) full          static + history(10): exactly K06(b)'s real-history
                    condition. The six K09/K12 statistics are six of the
                    ten history features; the other four (max, last-prev,
                    frac<=4.5, frac increasing) belong to the full set.
  (supplementary) real6: static + the six statistics only
                    (cols 0,1,2,3,5,7), to measure the gap between the
                    6-statistic set and K06(b)'s 10-feature condition.

history(10) = [first, mean, std, min, max, last-first, last-prev,
frac<=3.5, frac<=4.5, frac(diff>0)] of the real 11-frame series
(0..1 ns), as defined in K06(b)'s score_b.py.

Base, learner, metrics and data are identical to K18 / eval_b.py:
frozen T1I2006 + HistGradientBoosting sidecar (160 iters, lr 0.065,
15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained on the
8 calibration proteins, evaluated on 100 test proteins x 3 replicas
(163,074 rows) from K06(b)'s saved features. Gains are AUPRC
improvements over the static sidecar (macro primary, micro secondary);
recovery = gain(condition) / gain(full).

Results are checkpointed per horizon in results_k21_partial.json.
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
PARTIAL = HERE / "results_k21_partial.json"
orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

COLS = {"only_mean": [1], "only_min": [3], "only_frac35": [7],
        "shape_only": [2, 5], "real6": [0, 1, 2, 3, 5, 7]}


def load(proteins):
    rows = []
    for p in proteins:
        with np.load(OUTD / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({k: z[f"r{r}_{k}"] for k in
                             ("static", "history", "frozen", "label", "gen8", "gen6")})
    return rows


def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


CONDS = {"static": lambda r: r["static"],
         "full": lambda r: np.column_stack([r["static"], r["history"]])}
for name, cols in COLS.items():
    CONDS[name] = (lambda c: (lambda r: np.column_stack(
        [r["static"], r["history"][:, c]])))(cols)


def main():
    cal, test = load(CAL), load(TEST)
    y_cal = np.concatenate([r["label"] for r in cal])
    y_test = np.concatenate([r["label"] for r in test])
    frozen_test = np.concatenate([r["frozen"] for r in test])
    spans, cur = [], 0
    for p in TEST:
        with np.load(OUTD / f"{p}.npz") as z:
            n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n))
        cur += n
    if PARTIAL.is_file():
        results = json.loads(PARTIAL.read_text())
    else:
        results = {"n_cal": int(len(y_cal)), "n_test": int(len(y_test)), "by_horizon": {}}
    for j, h in enumerate(HORIZONS):
        row = results["by_horizon"].get(str(h), {})
        if "frozen" not in row:
            row["frozen"] = metrics(y_test[:, j], frozen_test)
            fm = [average_precision_score(y_test[s:e, j], frozen_test[s:e])
                  for _, s, e in spans if 0 < y_test[s:e, j].sum() < e - s]
            row["frozen"]["macro_auprc"] = float(np.mean(fm))
        for name, fn in CONDS.items():
            if name in row:
                continue
            Xtr = np.concatenate([fn(r) for r in cal])
            Xte = np.concatenate([fn(r) for r in test])
            pred = learner(Xtr, y_cal[:, j]).predict_proba(Xte)[:, 1]
            m = metrics(y_test[:, j], pred)
            macro = [average_precision_score(y_test[s:e, j], pred[s:e])
                     for _, s, e in spans if 0 < y_test[s:e, j].sum() < e - s]
            m["macro_auprc"] = float(np.mean(macro))
            row[name] = m
            print(h, name, {k: round(v, 4) for k, v in m.items()}, flush=True)
        results["by_horizon"][str(h)] = row
        PARTIAL.write_text(json.dumps(results, indent=1))
    recovery = {}
    for h in HORIZONS:
        row = results["by_horizon"][str(h)]
        entry = {}
        for kind, key in (("macro", "macro_auprc"), ("micro", "auprc")):
            g_full = row["full"][key] - row["static"][key]
            entry[kind] = {"gain_full": float(g_full), "recovery": {
                name: float((row[name][key] - row["static"][key]) / g_full)
                for name in COLS}}
        recovery[str(h)] = entry
    results["recovery"] = recovery
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print("WROTE results.json")


if __name__ == "__main__":
    main()
