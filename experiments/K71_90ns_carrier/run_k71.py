"""K71: which single statistic carries the 90 ns guidance? (HB formation, D conditions).

Data-D conditions, all candidates (no distance-band split).
Training is unchanged (4 training proteins, all candidates); evaluation is
pooled (1mix_A + 2qt4_A) and per test protein, same split as K57/K68.
Conditions: static only, static + full 6 history stats, and static + one
statistic alone for each of the 6 history statistics
(first, mean, std, min, last-first, fraction <= 3.5 A) -- the K21-style
"single statistic" conditions, but on the weak (D) base instead of the
strong K06(b) base. Horizons 90 ns (main) and 1 ns (reference).

Control: pooled static_only / full_hist6 / mean_only must match K68's
results.json to 4 decimals (1 ns: static 0.3637, full 0.4815, mean 0.4865;
90 ns: static 0.3035, full 0.4246, mean 0.3970 approx).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss)

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).parent
K68_RESULTS = Path(__file__).parent.parent / "K68_mean_carrier_by_protein" / "results.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (90, 1)
STAT_NAMES = ("first", "mean", "std", "min", "last_minus_first", "frac_le3.5")


def load_protein(p):
    parts = []
    for r in (1, 2, 3):
        z = np.load(CACHE / f"{p}_R{r}.npz")
        parts.append({k: z[k] for k in z.files})
    X = np.concatenate([q["X"] for q in parts])
    H = np.concatenate([q["hist"].T for q in parts])  # (n, 11)
    Y = np.concatenate([q["y"] for q in parts])
    return X, H, Y


def hist6(series, bound=3.5):
    return np.stack([series[:, 0], series.mean(1), series.std(1),
                     series.min(1), series[:, -1] - series[:, 0],
                     (series <= bound).mean(1)], axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "auprc": float(average_precision_score(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }


def main():
    Xtr = {p: load_protein(p) for p in TRAIN_P}
    Xte = {p: load_protein(p) for p in TEST_P}
    Xtr_all = np.concatenate([Xtr[p][0] for p in TRAIN_P])
    Htr_all = np.concatenate([Xtr[p][1] for p in TRAIN_P])
    Ytr_all = np.concatenate([Xtr[p][2] for p in TRAIN_P])
    Ftr_all = hist6(Htr_all)
    k68 = json.loads(K68_RESULTS.read_text())
    conds = [("static_only", None, None),
             ("full_hist6", Ftr_all, lambda H: hist6(H))]
    for i, name in enumerate(STAT_NAMES):
        conds.append((f"only_{name}", Ftr_all[:, [i]],
                      (lambda j: lambda H: hist6(H)[:, [j]])(i)))
    results = {"conditions": {}, "control_check": {}, "recovery": {}}
    for name, ftr_all, fte_fn in conds:
        results["conditions"][name] = {"pooled": {}, "per_protein": {}}
        for h in HORIZONS:
            hi = (1, 5, 10, 30, 50, 90).index(h)
            ytr = Ytr_all[:, hi]
            preds = {}
            for p in TEST_P:
                Xp, Hp, Yp = Xte[p]
                if ftr_all is None:
                    pr = predict(Xtr_all, ytr, Xp)
                else:
                    fte = fte_fn(Hp)
                    pr = predict(np.concatenate([Xtr_all, ftr_all], axis=1), ytr,
                                 np.concatenate([Xp, fte], axis=1))
                preds[p] = (Yp[:, hi], pr)
            y_pool = np.concatenate([preds[p][0] for p in TEST_P])
            p_pool = np.concatenate([preds[p][1] for p in TEST_P])
            results["conditions"][name]["pooled"][str(h)] = metrics(y_pool, p_pool)
            for p in TEST_P:
                y, pr = preds[p]
                results["conditions"][name]["per_protein"].setdefault(p, {})[str(h)] = metrics(y, pr)
            print(name, h, "pooled auprc",
                  round(results["conditions"][name]["pooled"][str(h)]["auprc"], 4),
                  flush=True)
    # control: static/full/mean vs K68 pooled
    k68_map = {"static_only": "static_only", "full_hist6": "full_hist6",
               "only_mean": "mean_only"}
    for name, k68name in k68_map.items():
        for h in HORIZONS:
            d = max(abs(results["conditions"][name]["pooled"][str(h)][k]
                        - k68["conditions"][k68name]["pooled"][str(h)][k])
                    for k in ("auprc", "auc", "brier", "logloss"))
            results["control_check"].setdefault(name, {})[str(h)] = {
                "max_abs_diff_vs_K68_pooled": d}
            print("control", name, h, "maxdiff", f"{d:.2e}", flush=True)
    # gain and recovery per horizon, pooled and per protein
    for h in HORIZONS:
        s = results["conditions"]["static_only"]["pooled"][str(h)]["auprc"]
        f = results["conditions"]["full_hist6"]["pooled"][str(h)]["auprc"]
        gain = f - s
        rec = {}
        for name in results["conditions"]:
            a = results["conditions"][name]["pooled"][str(h)]["auprc"]
            per = {}
            for p in TEST_P:
                sp = results["conditions"]["static_only"]["per_protein"][p][str(h)]["auprc"]
                fp = results["conditions"]["full_hist6"]["per_protein"][p][str(h)]["auprc"]
                ap = results["conditions"][name]["per_protein"][p][str(h)]["auprc"]
                gp = fp - sp
                per[p] = (ap - sp) / gp if gp else None
            rec[name] = {"recovery_pooled": (a - s) / gain if gain else None,
                         "per_protein": per}
        results["recovery"][str(h)] = {"gain_pooled": gain, "by_condition": rec}
        print(h, f"gain_pool {gain:+.4f}",
              {n: round(rec[n]["recovery_pooled"], 3) for n in rec}, flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
