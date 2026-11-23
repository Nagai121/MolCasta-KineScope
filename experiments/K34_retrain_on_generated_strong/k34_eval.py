"""K34: retrained-on-generated correction model on the strong base.

Note on what is already measured: K06(b)'s eval_b.py trained the DocA
correction model per condition on its own features, so its gen8/gen6
conditions ARE the retrained-on-generated conditions ordered here.
This script re-runs them from the saved K06(b) feature files to verify
the numbers, and adds the half of the K10 analogy that is missing on
the strong base: the frozen cross-application.
  (a) history   : train real-10, eval real-10   (K06(b) history)
  (c) gen8      : train gen8, eval gen8         (K06(b) gen8, re-verified)
  (c6) gen6     : train gen6, eval gen6         (K06(b) gen6, re-verified)
  (b') frozen6  : train real-6, eval gen6       (NEW: readout mismatch)
  (h6) history6 : train real-6, eval real-6     (NEW control: how much
                  the 6-stat readout carries on the real side)
The 6 stats are history columns [0,1,2,3,5,7] = first, mean, std, min,
last-first, frac<=3.5 -- exactly the gen6 definition (score_b.py).
Train = calibration 8 proteins, test = 100 proteins x 3 replicas;
the two sets are disjoint (same split as K06(b))."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HORIZONS = (1, 5, 10, 30, 50, 90)
SEED = 20261121
OUTD = Path("/home/hatch/workspace/k06b/run/k06b_out")
orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]
SIX = [0, 1, 2, 3, 5, 7]

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

def main():
    cal, test = load(CAL), load(TEST)
    y_cal = np.concatenate([r["label"] for r in cal])
    y_test = np.concatenate([r["label"] for r in test])
    frozen_test = np.concatenate([r["frozen"] for r in test])
    spans, cur = [], 0
    for p in TEST:
        with np.load(OUTD / f"{p}.npz") as z:
            n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n
    # condition feature builders: (train rows -> X, test rows -> X)
    def st(r): return r["static"]
    def hi(r): return np.column_stack([r["static"], r["history"]])
    def g8(r): return np.column_stack([r["static"], r["gen8"]])
    def g6(r): return np.column_stack([r["static"], r["gen6"]])
    def h6(r): return np.column_stack([r["static"], r["history"][:, SIX]])
    CONDS = {"static": (st, st), "history": (hi, hi), "gen8": (g8, g8), "gen6": (g6, g6),
             "frozen6": (h6, g6), "history6": (h6, h6)}
    results = {"n_cal": int(len(y_cal)), "n_test": int(len(y_test)),
               "calibration_proteins": CAL, "test_proteins": TEST,
               "train_test_disjoint": len(set(CAL) & set(TEST)) == 0,
               "by_horizon": {}}
    for j, h in enumerate(HORIZONS):
        row = {"frozen": metrics(y_test[:, j], frozen_test)}
        for name, (ftr, fte) in CONDS.items():
            Xtr = np.concatenate([ftr(r) for r in cal])
            Xte = np.concatenate([fte(r) for r in test])
            pred = learner(Xtr, y_cal[:, j]).predict_proba(Xte)[:, 1]
            row[name] = metrics(y_test[:, j], pred)
            macro = [average_precision_score(y_test[s:e, j], pred[s:e])
                     for _, s, e in spans if 0 < y_test[s:e, j].sum() < e - s]
            row[name]["macro_auprc"] = float(np.mean(macro))
            row[name]["_pred"] = pred
        frozen_macro = [average_precision_score(y_test[s:e, j], frozen_test[s:e])
                        for _, s, e in spans if 0 < y_test[s:e, j].sum() < e - s]
        row["frozen"]["macro_auprc"] = float(np.mean(frozen_macro))
        # better-protein counts vs the static condition, per protein
        st_pred = row["static"]["_pred"]
        for name in CONDS:
            row[name]["macro_better_proteins"] = int(sum(
                average_precision_score(y_test[s:e, j], row[name]["_pred"][s:e]) >
                average_precision_score(y_test[s:e, j], st_pred[s:e])
                for _, s, e in spans if 0 < y_test[s:e, j].sum() < e - s))
            del row[name]["_pred"]
        # recovery vs the real-history macro gain over static
        base = row["static"]["macro_auprc"]; real = row["history"]["macro_auprc"]
        for name in ("gen8", "gen6", "frozen6", "history6"):
            row[name]["macro_recovery"] = float((row[name]["macro_auprc"] - base) / (real - base))
        results["by_horizon"][str(h)] = row
        print(h, {k: round(v["macro_auprc"], 4) for k, v in row.items() if "macro_auprc" in v},
              "recovery", {n: round(row[n]["macro_recovery"], 3) for n in ("gen8", "gen6", "frozen6", "history6")},
              flush=True)
    # verification against K06(b)
    k06b = json.loads(Path("/home/hatch/workspace/k06b/run/k06b_results.json").read_text())
    worst = 0.0
    for h in HORIZONS:
        for cond in ("static", "history", "gen8", "gen6", "frozen"):
            for m in ("auprc", "auc", "brier", "logloss"):
                worst = max(worst, abs(results["by_horizon"][str(h)][cond][m] - k06b["by_horizon"][str(h)][cond][m]))
    results["verification_vs_k06b_max_abs_diff"] = float(worst)
    print("max abs diff vs K06(b) recorded values:", round(worst, 6))
    Path("/home/hatch/workspace/k34/k34_results.json").write_text(json.dumps(results, indent=1))
    print("WROTE k34_results.json")

if __name__ == "__main__":
    main()
