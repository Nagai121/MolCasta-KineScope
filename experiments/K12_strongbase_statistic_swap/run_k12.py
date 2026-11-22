"""K12: strong-base (K06(b), frozen T1I2006 + sidecar) statistic swap.

Reuse K06(b)'s saved per-protein features (~/workspace/k06b/run/k06b_out/,
108 proteins x 3 replicas): static(5), history(10), frozen, label(6),
gen8(8), gen6(6). No trajectory re-acquisition, no checkpoint loading.

Definitions (from score_b.py in the K06(b) run directory):
  history 10 = [series[0], mean, std, min, max, last-first, last-prev,
                frac<=3.5, frac<=4.5, frac(diff>0)] of the real 11-frame
                series (0..1 ns, 100 ps spacing).
  gen6 6 = [first, mean, std, min, last-first, frac<=3.5] of the generated
           5-point series (origin distance + 0.1/0.3/0.5/1.0 ns ahead).
Real 6-stat values are taken from history as:
  first <- history[0], mean <- history[1], std <- history[2],
  min <- history[3], last-first <- history[5], frac<=3.5 <- history[7].
Caveat (recorded in README): the real stats come from an 11-frame
backward (history) series while gen6 comes from a 5-point forward
(generated) series, so "first" and "last-first" also change their time
meaning on swap, exactly as in K09. Only "mean" is direction-free.

Learner / metrics / macro definition are copied verbatim from
experiments/K06_aligned_recomparison/eval_b.py (HistGB, 160 iters,
lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121).
Conditions are retrained per horizon. Results are checkpointed per
condition so a restart resumes without redoing finished fits.
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
PARTIAL = HERE / "results_k12_partial.json"
orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# gen6 column -> history column holding the same real statistic
REAL6_FROM_HIST = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 7}
SWAP_NAMES = {0: "swap_first", 1: "swap_mean", 2: "swap_std",
              3: "swap_min", 4: "swap_lastminusfirst", 5: "swap_frac35"}


def load(proteins):
    rows = []
    for p in proteins:
        with np.load(OUTD / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({k: z[f"r{r}_{k}"] for k in
                             ("static", "history", "frozen", "label", "gen8", "gen6")})
    return rows


def real6(r):
    """D-style 6 stats of the real 11-frame series, from history(10)."""
    h = r["history"]
    return np.column_stack([h[:, REAL6_FROM_HIST[i]] for i in range(6)]).astype(np.float32)


def swapped(r, col):
    g = r["gen6"].copy()
    g[:, col] = real6(r)[:, col]
    return np.column_stack([r["static"], g])


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
for col, name in SWAP_NAMES.items():
    CONDS[name] = (lambda c: (lambda r: swapped(r, c)))(col)


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
    # recovery rates: (swap gain - gen6 gain) / (history gain - gen6 gain),
    # gains = macro AUPRC improvement over static (primary), and micro too.
    rec = {}
    for h in HORIZONS:
        row = results["by_horizon"][str(h)]
        for kind, key in (("macro", "macro_auprc"), ("micro", "auprc")):
            g_hist = row["history"][key] - row["static"][key]
            g_gen = row["gen6"][key] - row["static"][key]
            denom = g_hist - g_gen
            rec.setdefault(str(h), {})[kind] = {
                name: float((row[name][key] - row["static"][key] - g_gen) / denom)
                for name in SWAP_NAMES.values()}
    results["recovery"] = rec
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print("WROTE results.json")


if __name__ == "__main__":
    main()
