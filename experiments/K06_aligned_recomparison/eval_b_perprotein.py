import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
HORIZONS = (1, 5, 10, 30, 50, 90)
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
                                          random_state=20261121).fit(X, y)

CONDS = {"static": lambda r: r["static"],
         "history": lambda r: np.column_stack([r["static"], r["history"]]),
         "gen8": lambda r: np.column_stack([r["static"], r["gen8"]]),
         "gen6": lambda r: np.column_stack([r["static"], r["gen6"]])}
cal, test = load(CAL), load(TEST)
y_cal = np.concatenate([r["label"] for r in cal])
y_test = np.concatenate([r["label"] for r in test])
spans, cur = [], 0
for p in TEST:
    with np.load(OUTD / f"{p}.npz") as z:
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
    spans.append((p, cur, cur + n)); cur += n
out = {}
for j, h in enumerate(HORIZONS):
    preds = {}
    for name, fn in CONDS.items():
        Xtr = np.concatenate([fn(r) for r in cal])
        Xte = np.concatenate([fn(r) for r in test])
        preds[name] = learner(Xtr, y_cal[:, j]).predict_proba(Xte)[:, 1]
    per = {}
    for name, pred in preds.items():
        per[name] = [float(average_precision_score(y_test[s:e, j], pred[s:e]))
                     for _, s, e in spans if 0 < y_test[s:e, j].sum() < e - s]
    row = {}
    for name in ("history", "gen8", "gen6"):
        better = sum(1 for a, b in zip(per[name], per["static"]) if a > b + 1e-12)
        worse = sum(1 for a, b in zip(per[name], per["static"]) if a < b - 1e-12)
        row[name] = {"better_proteins": better, "worse_proteins": worse,
                     "mean_gain_vs_static": float(np.mean(np.array(per[name]) - np.array(per["static"])))}
    out[str(h)] = row
    print(h, row, flush=True)
Path("/home/hatch/workspace/k06b/run/k06b_perprotein.json").write_text(json.dumps(out, indent=1))
print("WROTE k06b_perprotein.json")
