"""K82: is the weak 1 ns -> 50 ns transfer caused by reliance on the
frozen body prediction? (strong base, hydrogen bonds)

K56 found that a readout trained at 1 ns transfers weakly to 50 ns
(recovery 0.608 pooled / 0.62 macro), while the reverse direction is
strong (K79: 1.026 / 0.996). K80 showed the difference is not the
relied-on history statistic: the 1 ns readout relies most on the mean
at 50 ns too, and the mean is a carrier at 50 ns (K77). The remaining
candidate is the frozen body prediction (static feature 0), which K61
showed is needed for the long-horizon readout: dropping it keeps only
0.38 of the 50 ns gain (and -0.12 at 90 ns), while dropping the
distance or kind groups keeps 0.83-1.07.

Method: K06(b) conditions (frozen T1I2006 + correction model,
HistGradientBoosting max_iter=160, lr .065, max_leaf_nodes 15,
min_samples_leaf 20, L2 .15, seed 20261121; 8 calibration proteins for
training, 100 unseen proteins x 3 replicas for evaluation, 163,074
candidates). Two variants of the correction inputs:
  - "with":    static 5 features + real history 10 features (K56/K79
    conditions; static col 0 is the frozen body prediction, verified
    to equal the saved r*_frozen array up to float32 rounding).
  - "without": static cols 1..4 + real history 10 features (the frozen
    body prediction column dropped, K61's drop_frozen inputs).
For each variant, fit static-only and history corrections trained at
1 ns and at 50 ns, and evaluate each at 1 ns and 50 ns (4 train->eval
combinations per variant, 16 fits in total).

Gain definition (fixed before running): within each variant, the gain
at evaluation horizon e is (history AUPRC(e) - static-only AUPRC(e)),
where the static-only condition is trained at e and uses the same
variant's inputs (i.e. the "without" gain is measured against the
"without" static-only baseline, so the two variants' gains are
like-for-like within variant). Transfer recovery = transferred gain /
same-eval-horizon-trained gain, computed within each variant.
As a cross-check against K61 (whose drop_frozen gain was measured
against the FULL static-only baseline), the K61-style gain is also
recorded: (without-history AUPRC(e) - with-static-only AUPRC(e)).

Pre-fixed reading: if dropping the frozen body prediction raises the
1 -> 50 ns transfer recovery from 0.608 to 0.8 or more, the weakness
disappears and is caused by the reliance on the frozen body
prediction. If the weakness remains (recovery below 0.8), it is
caused by something else.
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
EVAL = (1, 50)
SEED_MODEL = 20261121

assert not (set(CAL) & set(TEST))
print("no overlap: cal", len(CAL), "test", len(TEST), flush=True)

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({"static": z[f"r{r}_static"], "history": z[f"r{r}_history"],
                             "label": z[f"r{r}_label"]})
    return rows

def learner(X, y):
    return HistGradientBoostingClassifier(learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED_MODEL, max_iter=160).fit(X, y)

def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

cal = load(CAL); test = load(TEST)
Str = np.concatenate([r["static"] for r in cal]).astype(np.float64)
Ste = np.concatenate([r["static"] for r in test]).astype(np.float64)
Htr = np.concatenate([r["history"] for r in cal]).astype(np.float64)
Hte = np.concatenate([r["history"] for r in test]).astype(np.float64)
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

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)

# Variants: ("with", static cols 0..4) and ("without", static cols 1..4).
VARIANTS = {"with": [0, 1, 2, 3, 4], "without": [1, 2, 3, 4]}
for variant, cols in VARIANTS.items():
    Str_v = Str[:, cols]; Ste_v = Ste[:, cols]
    for feat_name, Xtr, Xte in (("static", Str_v, Ste_v),
                                ("history", np.column_stack([Str_v, Htr]),
                                 np.column_stack([Ste_v, Hte]))):
        for h_train in (1, 50):
            key = f"{variant}_{feat_name}_train{h_train}"
            results.setdefault(key, {})
            if len(results[key]) == len(EVAL):
                print(key, "already done, skipped", flush=True); continue
            t = HORIZONS.index(h_train)
            m = learner(Xtr, Ytr[:, t])
            for h_eval in EVAL:
                if str(h_eval) in results[key]:
                    continue
                e = HORIZONS.index(h_eval)
                p = m.predict_proba(Xte)[:, 1]
                row = metrics(Yte[:, e], p)
                row["macro_auprc"] = macro_auprc(Yte, p, e)
                results[key][str(h_eval)] = row
                print(key, "->", h_eval, {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in row.items()}, flush=True)
                PARTIAL.write_text(json.dumps(results, indent=2))

# References for cross-checks.
k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())
k56 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K56_readout_transfer_across_horizons/results.json").read_text())
k61 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K61_static_groups_for_long_horizons/results.json").read_text())
k79 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K79_readout_transfer_from_50ns/results.json").read_text())

print("--- verify 'with' matched fits against K06(b) ---", flush=True)
for h in EVAL:
    for feat_name in ("static", "history"):
        ref = k06b["by_horizon"][str(h)][feat_name]
        got = results[f"with_{feat_name}_train{h}"][str(h)]
        print("verify", feat_name, h, "auprc diff", round(abs(got["auprc"] - ref["auprc"]), 6),
              "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

def gain(variant, feat, h_train, h_eval, pooled=True):
    s = results[f"{variant}_static_train{h_eval}"][str(h_eval)]
    a = results[f"{variant}_{feat}_train{h_train}"][str(h_eval)]
    k = "auprc" if pooled else "macro_auprc"
    return float(a[k] - s[k])

summary = {}
for variant in VARIANTS:
    for pooled in (True, False):
        tag = "pooled" if pooled else "macro"
        # matched gains and transfer gains, all within this variant
        g11 = gain(variant, "history", 1, 1, pooled)
        g50 = gain(variant, "history", 50, 50, pooled)
        g1_50 = gain(variant, "history", 1, 50, pooled)
        g50_1 = gain(variant, "history", 50, 1, pooled)
        summary[f"{variant}_{tag}"] = {
            "gain_1_to_1": g11, "gain_50_to_50": g50,
            "gain_1_to_50": g1_50, "gain_50_to_1": g50_1,
            "recovery_1_to_50": g1_50 / g50 if g50 else None,
            "recovery_50_to_1": g50_1 / g11 if g11 else None,
        }
        print(variant, tag, "gains 1->1", round(g11, 4), "50->50", round(g50, 4),
              "1->50", round(g1_50, 4), "50->1", round(g50_1, 4),
              "| recovery 1->50", round(summary[f"{variant}_{tag}"]["recovery_1_to_50"], 4),
              "50->1", round(summary[f"{variant}_{tag}"]["recovery_50_to_1"], 4), flush=True)

# K61-style cross-check for the "without" variant: gain measured
# against the FULL static-only baseline (as K61 did), not the
# "without" static-only baseline used in the primary definition.
k61_check = {}
for h in EVAL:
    s_full = results[f"with_static_train{h}"][str(h)]
    a_drop = results[f"without_history_train{h}"][str(h)]
    k61_check[str(h)] = {
        "k61_style_gain_pooled": float(a_drop["auprc"] - s_full["auprc"]),
        "k61_style_gain_macro": float(a_drop["macro_auprc"] - s_full["macro_auprc"]),
    }
    print("K61-style gain (drop_frozen history vs full static)", h, "ns pooled",
          round(k61_check[str(h)]["k61_style_gain_pooled"], 4), "macro",
          round(k61_check[str(h)]["k61_style_gain_macro"], 4), flush=True)

results["_summary"] = summary
results["_k61_style_check"] = k61_check
results["_references"] = {
    "K06b": {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history")} for h in EVAL},
    "K56_recovery_1_to_50": {"pooled": k56["_summary"]["50"]["train1_recovery_pooled"],
                             "macro": k56["_summary"]["50"]["train1_recovery_macro"]},
    "K79_recovery_50_to_1": {"pooled": k79["_summary"]["1"]["recovery_pooled"],
                             "macro": k79["_summary"]["1"]["recovery_macro"]},
    # K61's drop_frozen gain, measured against the FULL static-only
    # baseline (its definition), for comparison with the K61-style
    # check above. K61's own table: 1 ns +0.0635, 50 ns +0.0133.
    "K61_drop_frozen_gain_pooled": {
        str(h): float(k61["drop_frozen"][str(h)]["auprc"] - k61["static"][str(h)]["auprc"])
        for h in EVAL},
    "K61_drop_frozen_gain_macro": {
        str(h): float(k61["drop_frozen"][str(h)]["macro_auprc"] - k61["static"][str(h)]["macro_auprc"])
        for h in EVAL},
}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "condition", "train_horizon_ns", "eval_horizon_ns", "n", "positives",
                "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for variant in VARIANTS:
        for feat_name in ("static", "history"):
            for t in (1, 50):
                for e in EVAL:
                    r = results[f"{variant}_{feat_name}_train{t}"][str(e)]
                    w.writerow([variant, feat_name, t, e, r["n"], r["positives"], r["auc"],
                                r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
