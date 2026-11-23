"""K17 prediction comparison under Doc D conditions (contact):
static only / + real history / + generated history (proposals 11-14,
branch-mean 9-point series -> the same 6 history statistics),
HistGradientBoosting retrained per condition and horizon.
Recovery = (AUPRC_gen - AUPRC_static) / (AUPRC_real - AUPRC_static)."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
OUT = Path("/home/hatch/workspace/k17/out")

def feats(series, bound=4.5):
    return np.stack([series[0], series.mean(0), series.std(0), series.min(0),
                     series[-1] - series[0], (series <= bound).mean(0)], axis=1).astype(np.float32)

def load(proteins):
    rows = []
    for p in proteins:
        for r in (1, 2, 3):
            with np.load(OUT / f"{p}_R{r}.npz") as z:
                rows.append({kk: z[kk] for kk in z.files})
    return rows

def stack(rows, key):
    return np.concatenate([r[key] for r in rows])

def main():
    train, test = load(PROTEINS[:4]), load(PROTEINS[4:])
    Xtr, Xte = stack(train, "X"), stack(test, "X")
    ytr, yte = stack(train, "y"), stack(test, "y")
    blocks_tr = {"static": Xtr, "real": np.column_stack([Xtr, feats(stack(train, "hist").T)])}
    blocks_te = {"static": Xte, "real": np.column_stack([Xte, feats(stack(test, "hist").T)])}
    for n in (11, 12, 13, 14):
        gtr = np.concatenate([r[f"gen{n}"].mean(axis=0) for r in train], axis=1)  # (9, N_total)
        gte = np.concatenate([r[f"gen{n}"].mean(axis=0) for r in test], axis=1)
        blocks_tr[f"gen{n}"] = np.column_stack([Xtr, feats(gtr)])
        blocks_te[f"gen{n}"] = np.column_stack([Xte, feats(gte)])
    results = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "by_horizon": {}}
    for j, h in enumerate(HORIZONS):
        row = {}
        for name in blocks_tr:
            m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.075, max_leaf_nodes=31,
                                               min_samples_leaf=15, l2_regularization=0.02,
                                               random_state=42)
            m.fit(blocks_tr[name], ytr[:, j])
            p = m.predict_proba(blocks_te[name])[:, 1]
            row[name] = dict(auprc=float(average_precision_score(yte[:, j], p)),
                             auc=float(roc_auc_score(yte[:, j], p)),
                             brier=float(brier_score_loss(yte[:, j], p)))
        base, real = row["static"]["auprc"], row["real"]["auprc"]
        for n in (11, 12, 13, 14):
            row[f"gen{n}"]["recovery"] = float((row[f"gen{n}"]["auprc"] - base) / (real - base)) \
                if real != base else None
        results["by_horizon"][str(h)] = row
        print(h, {kk: round(v["auprc"], 4) for kk, v in row.items()},
              "recovery", {n: None if row[f"gen{n}"]["recovery"] is None else round(row[f"gen{n}"]["recovery"], 3)
                           for n in (11, 12, 13, 14)}, flush=True)
    Path("/home/hatch/workspace/k17/k17_results.json").write_text(json.dumps(results, indent=1))
    print("WROTE k17_results.json")

if __name__ == "__main__":
    main()
