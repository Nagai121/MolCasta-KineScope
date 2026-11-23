"""K18: leave-one-statistic-out ablation of the real history on the
strong base (K06(b) conditions: frozen T1I2006 + sidecar learner,
8 calibration proteins, 100 test proteins x 3 replicas).

The full condition is exactly K06(b)'s real-history condition:
static(5) + history(10), where history(10) = [first, mean, std, min, max,
last-first, last-prev, frac<=3.5, frac<=4.5, frac(diff>0)] of the real
11-frame series (0..1 ns). The six statistics named in K09/K12 map to
history columns: first=0, mean=1, std=2, min=3, last-first=5,
frac<=3.5=7. Each ablation removes that single column (leaving static +
the other 9 history features) and retrains the sidecar per horizon.

So "necessity" is judged inside the full real-history feature set: the
four history features outside the six (max, last-prev, frac<=4.5,
frac increasing) stay in every ablation and may substitute for a dropped
statistic. This is deliberate: the full condition must reproduce K06(b).

Learner and metrics are copied verbatim from eval_b.py
(HistGradientBoosting, 160 iters, lr 0.065, 15 leaves,
min_samples_leaf 20, L2 0.15, seed 20261121). Gains are AUPRC
improvements over the static sidecar (macro per-protein primary, micro
pooled secondary); drop share = (gain_full - gain_ablated) / gain_full.

Data: K06(b)'s saved per-protein features
(~/workspace/k06b/run/k06b_out/, 108 proteins). No re-acquisition, no
checkpoint loading. Results are checkpointed per condition.
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
PARTIAL = HERE / "results_k18_partial.json"
orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# statistic name -> history column dropped
DROP = {"drop_mean": 1, "drop_frac35": 7, "drop_min": 3,
        "drop_std": 2, "drop_lastminusfirst": 5, "drop_first": 0}


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


def features(r, drop_col=None):
    if drop_col is None:
        return np.column_stack([r["static"], r["history"]])
    keep = [c for c in range(r["history"].shape[1]) if c != drop_col]
    return np.column_stack([r["static"], r["history"][:, keep]])


CONDS = {"static": lambda r: r["static"], "full": lambda r: features(r)}
for name, col in DROP.items():
    CONDS[name] = (lambda c: (lambda r: features(r, c)))(col)


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
    # gains over static and drop shares of the full gain
    drop = {}
    for h in HORIZONS:
        row = results["by_horizon"][str(h)]
        entry = {}
        for kind, key in (("macro", "macro_auprc"), ("micro", "auprc")):
            g_full = row["full"][key] - row["static"][key]
            entry[kind] = {"gain_full": float(g_full), "drop_share": {
                name: float((g_full - (row[name][key] - row["static"][key])) / g_full)
                for name in DROP}}
        drop[str(h)] = entry
    results["drop"] = drop
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print("WROTE results.json")


if __name__ == "__main__":
    main()
