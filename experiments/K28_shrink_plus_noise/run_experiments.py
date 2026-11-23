"""K28: can the generated mean error be reproduced by shrinkage + noise?
(strong base, hydrogen bonds)

K22 (independent noise only) left 0.43-0.64 of the gain; K24 (shrinkage
only) left almost all of it. But the generated mean may be "shrunk real
mean + independent noise". This experiment reproduces that combination
on the real series.

Fit (fit_regression.py, K05/K08 candidates, same definitions as K24):
  generated mean = a + b * real mean + residual
  a = 1.3829, b = 0.6972 (K24's slope), residual SD = 0.4028.
Method: K06(b) conditions (frozen T1I2006 + correction model, 8
calibration proteins for training, 100 unseen proteins x 3 replicas for
evaluation, 163,074 candidates; real 11-frame series re-extracted for
K22, verified against K06(b)'s saved history statistics with max diff
0.0). Each candidate's whole real series is shifted in parallel so its
mean becomes a + b * real mean + noise, where noise is drawn per
candidate independently with the residual SD (0.4028), seeds 101/202/303.
The shift is applied to both calibration and test; the correction model
is retrained per condition (learner identical to K06(b) eval_b.py).

References: static correction, full real history (a10), gen6/gen8 from
K06(b) k06b_results.json; K22's noise-matched and K24's shrunk gains
quoted from their results.json for comparison.

Pre-fixed reading: if the combined real history's gain falls to the
generated conditions' level, the generation failure is explained by
the mean error's structure (shrinkage + noise). If the gain stays above,
the generated mean error is not a linear shrinkage plus independent
noise; it is tied to the candidate's state or future in some other way.
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
NOISE_SEEDS = (101, 202, 303)
FIT = json.loads((OUT / "fit_results.json").read_text())["all"]
A, B, SIGMA = float(FIT["a"]), float(FIT["b"]), float(FIT["residual_sd"])

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
print("n_cal", len(Ytr), "n_test", len(Yte), "a", A, "b", B, "sigma", SIGMA, flush=True)

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

results = {"_fit_all": FIT}
if PARTIAL.exists():
    results.update(json.loads(PARTIAL.read_text()))
    print("resumed", flush=True)

for seed in NOISE_SEEDS:
    name = f"shrink_plus_noise_seed{seed}"
    results.setdefault(name, {})
    rng = np.random.default_rng(seed)
    e_tr = rng.normal(0.0, SIGMA, len(SRtr))
    e_te = rng.normal(0.0, SIGMA, len(SRte))
    d_tr = (A + (B - 1.0) * Mtr + e_tr).astype(np.float32)
    d_te = (A + (B - 1.0) * Mte + e_te).astype(np.float32)
    Htr = history10(SRtr + d_tr[:, None]); Hte = history10(SRte + d_te[:, None])
    Xtr = np.concatenate([Str, Htr], axis=1); Xte = np.concatenate([Ste, Hte], axis=1)
    results[f"_shift_stats_seed{seed}"] = {
        "a": A, "b": B, "sigma": SIGMA,
        "cal_shift_mean": float(d_tr.mean()), "cal_shift_sd": float(d_tr.std()),
        "test_shift_mean": float(d_te.mean()), "test_shift_sd": float(d_te.std()),
        "realized_noise_sd_cal": float(e_tr.std()), "realized_noise_sd_test": float(e_te.std())}
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
k24 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K24_shrinkage_matched/results.json").read_text())
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
           "k22_noise_gain_pooled": float(k22["_summary"][str(h)]["shifted_gain_pooled_mean"]),
           "k24_shrunk_sd_gain_pooled": float(k24["_summary"][str(h)]["shrunk_sd_gain_pooled"]),
           "k24_shrunk_slope_gain_pooled": float(k24["_summary"][str(h)]["shrunk_slope_gain_pooled"])}
    gains = []
    for seed in NOISE_SEEDS:
        r = results[f"shrink_plus_noise_seed{seed}"][str(h)]
        row[f"seed{seed}_gain_pooled"] = float(r["auprc"] - s["auprc"])
        row[f"seed{seed}_gain_macro"] = float(r["macro_auprc"] - s["macro_auprc"])
        gains.append(r["auprc"] - s["auprc"])
    row["combined_gain_pooled_mean"] = float(np.mean(gains))
    row["combined_gain_pooled_sd_across_seeds"] = float(np.std(gains))
    row["combined_gain_pooled_min"] = float(np.min(gains))
    row["combined_gain_pooled_max"] = float(np.max(gains))
    row["combined_retention_pooled"] = float(np.mean(gains) / row["a10_gain_pooled"])
    summary[str(h)] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for seed in NOISE_SEEDS:
        name = f"shrink_plus_noise_seed{seed}"
        for h in HORIZONS:
            r = results[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
