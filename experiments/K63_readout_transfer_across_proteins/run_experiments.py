"""K63: does a readout trained on one protein work on another protein?
(strong base, hydrogen bonds)

So far the correction model has been trained on several proteins. This
experiment asks whether a readout trained on a single protein also works
on another protein.

Method: K06(b) conditions (frozen T1I2006 + correction model,
HistGradientBoosting max_iter=160, lr .065, max_leaf_nodes 15,
min_samples_leaf 20, L2 .15, seed 20261121). 1mix_A and 2qt4_A are the
weak base's test proteins and are NOT among the K06(b) 108 proteins, so
their K06(b)-style features (static 5 + real history10 + labels) were
extracted for this experiment with extract_features.py, following
score_b.py exactly (same frames, same frozen model, same formulas).
  - transfer A: train on 1mix_A only (3 replicas), evaluate on 2qt4_A
  - transfer B: train on 2qt4_A only, evaluate on 1mix_A
  - reference (as assigned): train on BOTH proteins, evaluate on each
    protein. NOTE: this reference is in-sample for the evaluated protein
    (the model saw that protein's candidates in training), so its gain
    likely overstates the available gain and the recovery against it is
    conservative.
  - clean secondary reference: the K06(b) standard model, trained on the
    8 calibration proteins, evaluated on each protein (fully
    out-of-sample; neither protein is in the calibration set).
  - extra anchor: the both-trained model evaluated on the 100-protein
    K06(b) test set (out-of-sample).
The gain of a condition is its AUPRC minus the static-only condition's
AUPRC trained on the SAME training set and evaluated on the same protein
(pooled over the protein's 3 replicas; per-replica AUPRCs are reported
as the spread check). Recovery = transfer gain / reference gain.

Pre-fixed reading: if the gain on another protein recovers 80% or more
of the reference gain, the mean readout is the same across proteins. If
it does not, the readout differs per protein.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K63 = Path("/home/hatch/workspace/kinescope/work/k63_cache")
SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED_MODEL = 20261121
PROTEINS = ("1mix_A", "2qt4_A")

assert not (set(CAL) & set(TEST))

def load_protein(path, protein):
    rows = []
    with np.load(path / f"{protein}.npz") as z:
        for r in (1, 2, 3):
            rows.append({"static": z[f"r{r}_static"], "history": z[f"r{r}_history"],
                         "label": z[f"r{r}_label"]})
    return rows

def load_many(path, proteins):
    rows = []
    for p in proteins:
        rows.extend(load_protein(path, p))
    return rows

def stack(rows):
    S = np.concatenate([r["static"] for r in rows]).astype(np.float64)
    H = np.concatenate([r["history"] for r in rows]).astype(np.float64)
    Y = np.concatenate([r["label"] for r in rows])
    return S, H, Y

def learner(X, y):
    return HistGradientBoostingClassifier(learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED_MODEL, max_iter=160).fit(X, y)

def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

k63 = {p: load_protein(K63, p) for p in PROTEINS}
cal_rows = load_many(SAVED, CAL)
test_rows = load_many(SAVED, TEST)
print("1mix_A candidates:", sum(len(r["label"]) for r in k63["1mix_A"]),
      "| 2qt4_A candidates:", sum(len(r["label"]) for r in k63["2qt4_A"]),
      "| cal:", sum(len(r["label"]) for r in cal_rows),
      "| test100:", sum(len(r["label"]) for r in test_rows), flush=True)

# Training sets: each protein alone, both pooled, and the 8 cal proteins.
# Evaluation sets: each protein alone, and the 100-protein test set.
TRAIN_SETS = {"1mix_A": k63["1mix_A"], "2qt4_A": k63["2qt4_A"],
              "both": k63["1mix_A"] + k63["2qt4_A"], "cal8": cal_rows}
EVAL_SETS = {"1mix_A": k63["1mix_A"], "2qt4_A": k63["2qt4_A"], "test100": test_rows}
# Per-replica spans inside each eval protein's stacked arrays (for the
# per-replica spread check). test100 has no per-protein spans here.
EVAL_SPANS = {}
for name, rows in EVAL_SETS.items():
    spans, cur = [], 0
    for r in rows:
        n = len(r["label"]); spans.append((cur, cur + n)); cur += n
    EVAL_SPANS[name] = spans

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())

for train_name, train_rows in TRAIN_SETS.items():
    Str, Htr, Ytr = stack(train_rows)
    for eval_name, eval_rows in EVAL_SETS.items():
        Ste, Hte, Yte = stack(eval_rows)
        for feat_name, Xtr, Xte in (("static", Str, Ste),
                                    ("history", np.column_stack([Str, Htr]), np.column_stack([Ste, Hte]))):
            key = f"{feat_name}_train[{train_name}]_eval[{eval_name}]"
            results.setdefault(key, {})
            for j, h in enumerate(HORIZONS):
                if str(h) in results[key]:
                    print(key, h, "already done, skipped", flush=True); continue
                m = learner(Xtr, Ytr[:, j])
                p = m.predict_proba(Xte)[:, 1]
                row = metrics(Yte[:, j], p)
                spans = EVAL_SPANS[eval_name]
                row["per_replica_auprc"] = [float(average_precision_score(Yte[s:e, j], p[s:e]))
                                            for s, e in spans]
                results[key][str(h)] = row
                print(key, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k != "per_replica_auprc"},
                      "per_rep", [round(v, 3) for v in row["per_replica_auprc"]], flush=True)
                PARTIAL.write_text(json.dumps(results, indent=2))

# Sanity check: the cal8-trained model on the 100-protein test set must
# reproduce K06(b) exactly (same code path as K52/K56).
for h in HORIZONS:
    for feat_name, k in (("static", "static"), ("history", "history")):
        ref = k06b["by_horizon"][str(h)][k]
        got = results[f"{feat_name}_train[cal8]_eval[test100]"][str(h)]
        print("verify cal8->test100 vs K06b", feat_name, h, "auprc diff",
              round(abs(got["auprc"] - ref["auprc"]), 6), flush=True)

def gain(train_name, eval_name, h):
    s = results[f"static_train[{train_name}]_eval[{eval_name}]"][str(h)]
    a = results[f"history_train[{train_name}]_eval[{eval_name}]"][str(h)]
    return float(a["auprc"] - s["auprc"])

summary = {}
for eval_name in ("1mix_A", "2qt4_A"):
    other = "2qt4_A" if eval_name == "1mix_A" else "1mix_A"
    row = {}
    for h in HORIZONS:
        g_transfer = gain(other, eval_name, h)
        g_both = gain("both", eval_name, h)
        g_cal8 = gain("cal8", eval_name, h)
        row[str(h)] = {
            "transfer_gain": g_transfer, "both_gain": g_both, "cal8_gain": g_cal8,
            "recovery_vs_both": g_transfer / g_both if g_both else None,
            "recovery_vs_cal8": g_transfer / g_cal8 if g_cal8 else None,
            "transfer_history_auprc": results[f"history_train[{other}]_eval[{eval_name}]"][str(h)]["auprc"],
            "transfer_static_auprc": results[f"static_train[{other}]_eval[{eval_name}]"][str(h)]["auprc"],
            "both_history_auprc": results[f"history_train[both]_eval[{eval_name}]"][str(h)]["auprc"],
            "both_static_auprc": results[f"static_train[both]_eval[{eval_name}]"][str(h)]["auprc"],
            "cal8_history_auprc": results[f"history_train[cal8]_eval[{eval_name}]"][str(h)]["auprc"],
            "cal8_static_auprc": results[f"static_train[cal8]_eval[{eval_name}]"][str(h)]["auprc"],
        }
    summary[eval_name] = row
# Extra anchor: both-trained model on the 100-protein test set.
row = {}
for h in HORIZONS:
    row[str(h)] = {"both_gain_on_test100": gain("both", "test100", h),
                   "cal8_gain_on_test100": gain("cal8", "test100", h)}
summary["test100_anchor"] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train", "eval", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss", "per_replica_auprc"])
    for key, per_h in results.items():
        if key.startswith("_"):
            continue
        feat, rest = key.split("_train[")
        train_name, eval_name = rest[:-1].split("]_eval[")
        for h in HORIZONS:
            r = per_h[str(h)]
            w.writerow([feat, train_name, eval_name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["brier"], r["logloss"],
                        ";".join(f"{v:.4f}" for v in r["per_replica_auprc"])])
print("saved to", OUT)