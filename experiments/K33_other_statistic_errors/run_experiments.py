"""K33: how much gain do the non-mean statistic errors cost?
(strong base, hydrogen bonds)

K22/K24/K28 degraded only the mean, and K12 found restoring the mean
alone recovers 0.933 of the gain at 1 ns, yet the generated features as
a whole fall below the static correction (K12b). This experiment asks
whether the remaining damage lives in the non-mean statistics.

Method: K06(b) conditions (frozen T1I2006 + correction model, 8
calibration proteins for training, 100 unseen proteins x 3 replicas for
evaluation, 163,074 candidates). The correction model is trained on
real history (static + real history10) and NOT retrained per condition;
only the test features are modified. In the real history features, the
mean stays real and one statistic at a time -- and all three together --
is replaced by the generated value:
  std_only:   real history[2] (std)          <- gen6[2]
  min_only:   real history[3] (min)          <- gen6[3]
  frac35_only: real history[7] (frac<=3.5)   <- gen6[5]
  combined3:  all three at once
Column mapping follows K12 (gen6 = [first, mean, std, min, last-first,
frac<=3.5]; real history10 = [first, mean, std, min, max, last-first,
last-second-last, frac<=3.5, frac<=4.5, frac(diff>0)]).
Reference conditions (clearly marked): frac45 uses gen8[5]
(frac<=4.0, a different threshold -- no generated frac<=4.5 exists),
and combined4 = combined3 + frac45.

References: static correction, full real history (a10), gen6/gen8 from
K06(b) k06b_results.json. The (a10) model is retrained here and checked
against K06(b)'s recorded values (must reproduce exactly).

Pre-fixed reading: if any single replacement drops the gain to the
generated conditions' level, that statistic's error is the carrier of
the remaining damage. If even all three together do not, the non-mean
statistic errors do not explain it.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED_MODEL = 20261121

# (real history col, generated array key, generated col)
CONDITIONS = {
    "std_only": [(2, "gen6", 2)],
    "min_only": [(3, "gen6", 3)],
    "frac35_only": [(7, "gen6", 5)],
    "combined3": [(2, "gen6", 2), (3, "gen6", 3), (7, "gen6", 5)],
    "frac45_ref": [(8, "gen8", 5)],
    "combined4_ref": [(2, "gen6", 2), (3, "gen6", 3), (7, "gen6", 5), (8, "gen8", 5)],
}

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            saved = {k: z[k] for k in z.files}
        for r in (1, 2, 3):
            rows.append({"static": saved[f"r{r}_static"], "history": saved[f"r{r}_history"],
                         "gen6": saved[f"r{r}_gen6"], "gen8": saved[f"r{r}_gen8"],
                         "label": saved[f"r{r}_label"]})
    return rows

def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED_MODEL).fit(X, y)

def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

cal = load(CAL); test = load(TEST)
Str = np.concatenate([r["static"] for r in cal]); Ste = np.concatenate([r["static"] for r in test])
Htr = np.concatenate([r["history"] for r in cal]); Hte = np.concatenate([r["history"] for r in test])
G6te = np.concatenate([r["gen6"] for r in test]); G8te = np.concatenate([r["gen8"] for r in test])
Ytr = np.concatenate([r["label"] for r in cal]); Yte = np.concatenate([r["label"] for r in test])
print("n_cal", len(Ytr), "n_test", len(Yte), flush=True)

spans, cur = [], 0
for p in TEST:
    with np.load(SAVED / f"{p}.npz") as z:
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
    spans.append((p, cur, cur + n)); cur += n

def macro_auprc(y, p, j):
    vals = [average_precision_score(y[s:e, j], p[s:e])
            for _, s, e in spans if 0 < y[s:e, j].sum() < e - s]
    return float(np.mean(vals))

GEN = {"gen6": G6te, "gen8": G8te}
results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())

# train the real-history model once per horizon, verify reproduction of K06(b)
models = {}
for j, h in enumerate(HORIZONS):
    m = learner(np.concatenate([Str, Htr], axis=1), Ytr[:, j])
    models[h] = m
    p = m.predict_proba(np.concatenate([Ste, Hte], axis=1))[:, 1]
    row = metrics(Yte[:, j], p); row["macro_auprc"] = macro_auprc(Yte, p, j)
    ref = k06b["by_horizon"][str(h)]["history"]
    print("reproduce a10", h, "auprc", round(row["auprc"], 6), "ref", round(ref["auprc"], 6),
          "diff", round(abs(row["auprc"] - ref["auprc"]), 6), flush=True)
    results.setdefault("real_history", {})[str(h)] = row
    PARTIAL.write_text(json.dumps(results, indent=2))

for name, repl in CONDITIONS.items():
    results.setdefault(name, {})
    Hmod = Hte.copy()
    for col, gkey, gcol in repl:
        Hmod[:, col] = GEN[gkey][:, gcol]
    Xte = np.concatenate([Ste, Hmod], axis=1)
    for j, h in enumerate(HORIZONS):
        if str(h) in results[name]:
            print(name, h, "already done, skipped", flush=True); continue
        p = models[h].predict_proba(Xte)[:, 1]
        row = metrics(Yte[:, j], p)
        row["macro_auprc"] = macro_auprc(Yte, p, j)
        results[name][str(h)] = row
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        PARTIAL.write_text(json.dumps(results, indent=2))

summary = {}
for h in HORIZONS:
    s = k06b["by_horizon"][str(h)]["static"]
    row = {"static_auprc": s["auprc"], "static_macro_auprc": s["macro_auprc"],
           "a10_gain_pooled": float(k06b["by_horizon"][str(h)]["history"]["auprc"] - s["auprc"]),
           "a10_gain_macro": float(k06b["by_horizon"][str(h)]["history"]["macro_auprc"] - s["macro_auprc"]),
           "gen6_gain_pooled": float(k06b["by_horizon"][str(h)]["gen6"]["auprc"] - s["auprc"]),
           "gen8_gain_pooled": float(k06b["by_horizon"][str(h)]["gen8"]["auprc"] - s["auprc"]),
           "gen6_gain_macro": float(k06b["by_horizon"][str(h)]["gen6"]["macro_auprc"] - s["macro_auprc"]),
           "gen8_gain_macro": float(k06b["by_horizon"][str(h)]["gen8"]["macro_auprc"] - s["macro_auprc"])}
    for name in CONDITIONS:
        r = results[name][str(h)]
        row[f"{name}_gain_pooled"] = float(r["auprc"] - s["auprc"])
        row[f"{name}_gain_macro"] = float(r["macro_auprc"] - s["macro_auprc"])
        row[f"{name}_retention_pooled"] = float((r["auprc"] - s["auprc"]) / row["a10_gain_pooled"])
    summary[str(h)] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for name in list(CONDITIONS) + ["real_history"]:
        for h in HORIZONS:
            r = results[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
