"""K06(b) evaluation: frozen T1I2006 + DocA sidecar learner.
Conditions (sidecar features): static(5) / static+real history(15) /
static+generated8(13) / static+generated6(11); plus frozen alone.
Trained on the 8 calibration proteins, evaluated pooled + macro
per-protein on the 100 test proteins."""
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

CONDS = {
    "static": lambda r: r["static"],
    "history": lambda r: np.column_stack([r["static"], r["history"]]),
    "gen8": lambda r: np.column_stack([r["static"], r["gen8"]]),
    "gen6": lambda r: np.column_stack([r["static"], r["gen6"]]),
}

def main():
    cal, test = load(CAL), load(TEST)
    y_cal = np.concatenate([r["label"] for r in cal])
    y_test = np.concatenate([r["label"] for r in test])
    frozen_test = np.concatenate([r["frozen"] for r in test])
    results = {"n_cal": int(len(y_cal)), "n_test": int(len(y_test)), "by_horizon": {}}
    # per-protein row spans for macro
    spans, cur = [], 0
    for p in TEST:
        with np.load(OUTD / f"{p}.npz") as z:
            n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n
    for j, h in enumerate(HORIZONS):
        row = {"frozen": metrics(y_test[:, j], frozen_test)}
        for name, fn in CONDS.items():
            Xtr = np.concatenate([fn(r) for r in cal])
            Xte = np.concatenate([fn(r) for r in test])
            pred = learner(Xtr, y_cal[:, j]).predict_proba(Xte)[:, 1]
            row[name] = metrics(y_test[:, j], pred)
            macro = [average_precision_score(y_test[s:e, j], pred[s:e])
                     for _, s, e in spans if 0 < y_test[s:e, j].sum() < e - s]
            row[name]["macro_auprc"] = float(np.mean(macro))
        frozen_macro = [average_precision_score(y_test[s:e, j], frozen_test[s:e])
                        for _, s, e in spans if 0 < y_test[s:e, j].sum() < e - s]
        row["frozen"]["macro_auprc"] = float(np.mean(frozen_macro))
        results["by_horizon"][str(h)] = row
        print(h, {k: round(v["auprc"], 4) for k, v in row.items()}, flush=True)
    Path("/home/hatch/workspace/k06b/run/k06b_results.json").write_text(json.dumps(results, indent=1))
    print("WROTE k06b_results.json")

if __name__ == "__main__":
    main()
