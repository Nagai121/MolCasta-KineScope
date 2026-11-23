"""K24 step 2 (reproduction): does shrinking the per-pair differences of
the real means by the generated side's ratio reproduce the generation
failure? (strong base, hydrogen bonds)

Step 1 (measure_shrinkage.py, same candidates/definitions as K05/K08)
found the generated per-pair means are compressed relative to the real
ones: SD ratio alpha = 0.834, correlation 0.836 (regression slope 0.697).

Method: K06(b) conditions (frozen T1I2006 + correction model, 8
calibration proteins for training, 100 unseen proteins x 3 replicas for
evaluation, 163,074 candidates; real 11-frame series re-extracted for
K22, verified against K06(b)'s saved history statistics with max diff
0.0). Each candidate's whole real series is shifted in parallel so its
mean becomes center + alpha * (real mean - center), where center is the
mean of the per-candidate real means over the CALIBRATION candidates
only (no test-set statistics), applied to both calibration and test.
The series shape is unchanged; only level-dependent statistics move.
The correction model is retrained per condition (learner identical to
K06(b) eval_b.py).

Conditions:
  shrunk_sd    (main): alpha = SD ratio from step 1 (0.8340)
  shrunk_slope (reference): alpha = regression slope from step 1 (0.6972),
               the stronger of the two natural shrinkage factors
References: static correction, full real history (a10), gen6/gen8 from
K06(b) k06b_results.json; K22's noise-matched gains quoted from K22's
results.json for comparison.

Pre-fixed reading: if the shrunk real history's gain falls to the
generated conditions' level, the generation failure is explained by
shrinkage of the per-pair differences. If the gain stays above, shrinkage
does not explain it either.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/k22_cache")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED_MODEL = 20261121
MEAS = json.loads((OUT / "measure_results.json").read_text())["all"]
CONDITIONS = {"shrunk_sd": float(MEAS["sd_ratio_alpha"]),
              "shrunk_slope": float(MEAS["regression_slope_gen_on_real"])}

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            saved = {k: z[k] for k in z.files}
        with np.load(SERIES / f"{p}.npz") as z:
            series = {k: z[k] for k in z.files}
        for r in (1, 2, 3):
            rows.append({"static": saved[f"r{r}_static"], "label": saved[f"r{r}_label"],
                         "series": series[f"r{r}_series"]})
    return rows

def history10(series):
    s = series.T
    return np.column_stack((s[0], s.mean(axis=0), s.std(axis=0), s.min(axis=0),
                            s.max(axis=0), s[-1] - s[0], s[-1] - s[-2],
                            (s <= 3.5).mean(axis=0), (s <= 4.5).mean(axis=0),
                            (np.diff(s, axis=0) > 0).mean(axis=0))).astype(np.float32)

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
Ytr = np.concatenate([r["label"] for r in cal]); Yte = np.concatenate([r["label"] for r in test])
SRtr = np.concatenate([r["series"] for r in cal]); SRte = np.concatenate([r["series"] for r in test])
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

Mtr = SRtr.mean(axis=1).astype(np.float64); Mte = SRte.mean(axis=1).astype(np.float64)
CENTER = float(Mtr.mean())
print("center (calibration mean of per-candidate means) =", CENTER, flush=True)

results = {"_center_calibration": CENTER, "_alphas": CONDITIONS, "_step1_all": MEAS}
if PARTIAL.exists():
    results.update(json.loads(PARTIAL.read_text()))
    print("resumed", flush=True)

for name, alpha in CONDITIONS.items():
    results.setdefault(name, {})
    d_tr = ((alpha - 1.0) * (Mtr - CENTER)).astype(np.float32)
    d_te = ((alpha - 1.0) * (Mte - CENTER)).astype(np.float32)
    Htr = history10(SRtr + d_tr[:, None]); Hte = history10(SRte + d_te[:, None])
    Xtr = np.concatenate([Str, Htr], axis=1); Xte = np.concatenate([Ste, Hte], axis=1)
    results[f"_shift_stats_{name}"] = {
        "alpha": alpha, "cal_shift_mean": float(d_tr.mean()), "cal_shift_sd": float(d_tr.std()),
        "test_shift_mean": float(d_te.mean()), "test_shift_sd": float(d_te.std()),
        "shrunk_mean_sd_cal": float((CENTER + alpha * (Mtr - CENTER)).std()),
        "real_mean_sd_cal": float(Mtr.std())}
    for j, h in enumerate(HORIZONS):
        if str(h) in results[name]:
            print(name, h, "already done, skipped", flush=True); continue
        m = learner(Xtr, Ytr[:, j])
        p = m.predict_proba(Xte)[:, 1]
        row = metrics(Yte[:, j], p)
        row["macro_auprc"] = macro_auprc(Yte, p, j)
        results[name][str(h)] = row
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        PARTIAL.write_text(json.dumps(results, indent=2))

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())
k22 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K22_noise_matched_mean/results.json").read_text())
summary = {}
for h in HORIZONS:
    s = k06b["by_horizon"][str(h)]["static"]
    row = {"static_auprc": s["auprc"], "static_macro_auprc": s["macro_auprc"],
           "a10_gain_pooled": float(k06b["by_horizon"][str(h)]["history"]["auprc"] - s["auprc"]),
           "a10_gain_macro": float(k06b["by_horizon"][str(h)]["history"]["macro_auprc"] - s["macro_auprc"]),
           "gen6_gain_pooled": float(k06b["by_horizon"][str(h)]["gen6"]["auprc"] - s["auprc"]),
           "gen8_gain_pooled": float(k06b["by_horizon"][str(h)]["gen8"]["auprc"] - s["auprc"]),
           "gen6_gain_macro": float(k06b["by_horizon"][str(h)]["gen6"]["macro_auprc"] - s["macro_auprc"]),
           "gen8_gain_macro": float(k06b["by_horizon"][str(h)]["gen8"]["macro_auprc"] - s["macro_auprc"]),
           "k22_noise_gain_pooled": float(k22["_summary"][str(h)]["shifted_gain_pooled_mean"])}
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
    for name in CONDITIONS:
        for h in HORIZONS:
            r = results[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
