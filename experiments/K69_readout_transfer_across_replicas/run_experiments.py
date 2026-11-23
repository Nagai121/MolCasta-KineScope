"""K69: does a readout trained on one replica work on another replica?
(strong base, hydrogen bonds)

K63 found that transfer of a readout trained on one protein is unstable.
This experiment asks the same for replicas.

Method: K06(b) conditions (frozen T1I2006 + correction model,
HistGradientBoosting max_iter=160, lr .065, max_leaf_nodes 15,
min_samples_leaf 20, L2 .15, seed 20261121). Proteins: the 100 K06(b)
test proteins; train and eval use the SAME proteins, split by replica.
  - transfer: train on replica 1 only, evaluate on replica 2, replica 3,
    and replicas 2+3 pooled
  - reference (as assigned): train on ALL replicas, evaluate on replica
    2, replica 3, and replicas 2+3 pooled. NOTE: this reference is
    in-sample for the evaluated replicas (the model saw those candidates
    in training), so its gain likely overstates the available gain and
    the recovery against it is conservative.
  - clean references: train on replicas 1+2, evaluate on replica 3; and
    train on replicas 1+3, evaluate on replica 2 (the evaluated replica
    is unseen in both).
The gain of a condition is its AUPRC minus the static-only condition's
AUPRC trained on the SAME training set and evaluated on the same replica
set (pooled over the 100 proteins, and per-protein macro). Recovery =
transfer gain / reference gain. As a sanity check, the K06(b) standard
model (trained on the 8 calibration proteins, all replicas) evaluated on
the 100 test proteins (all replicas) must reproduce K06(b) exactly.

Pre-fixed reading: if the gain on another replica recovers 80% or more
of the reference gain, the mean readout is the same across replicas. If
it does not, the readout differs per replica.
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

assert not (set(CAL) & set(TEST))

def load_replica(proteins, replica):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            rows.append({"protein": p, "static": z[f"r{replica}_static"],
                         "history": z[f"r{replica}_history"], "label": z[f"r{replica}_label"]})
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

R1 = load_replica(TEST, 1); R2 = load_replica(TEST, 2); R3 = load_replica(TEST, 3)
CAL_ROWS = []
for p in CAL:
    with np.load(SAVED / f"{p}.npz") as z:
        for r in (1, 2, 3):
            CAL_ROWS.append({"protein": p, "static": z[f"r{r}_static"],
                             "history": z[f"r{r}_history"], "label": z[f"r{r}_label"]})
print("R1:", sum(len(r["label"]) for r in R1), "R2:", sum(len(r["label"]) for r in R2),
      "R3:", sum(len(r["label"]) for r in R3), "cal8:", sum(len(r["label"]) for r in CAL_ROWS), flush=True)

TRAIN_SETS = {"R1": R1, "ALL": R1 + R2 + R3, "R1R2": R1 + R2, "R1R3": R1 + R3, "CAL8": CAL_ROWS}
EVAL_SETS = {"R2": R2, "R3": R3, "R2R3": R2 + R3, "ALL": R1 + R2 + R3}
# (train, eval) pairs to run. CAL8 is only evaluated on ALL (sanity check).
PAIRS = [(t, e) for t in ("R1", "ALL", "R1R2", "R1R3") for e in ("R2", "R3", "R2R3")]
PAIRS += [("CAL8", "ALL")]

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())

def macro_auprc(rows, Y, p, j):
    spans, cur = [], 0
    for r in rows:
        n = len(r["label"]); spans.append((cur, cur + n)); cur += n
    vals = [average_precision_score(Y[s:e, j], p[s:e])
            for s, e in spans if 0 < Y[s:e, j].sum() < e - s]
    return float(np.mean(vals))

for train_name, eval_name in PAIRS:
    train_rows, eval_rows = TRAIN_SETS[train_name], EVAL_SETS[eval_name]
    Str, Htr, Ytr = stack(train_rows)
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
            row["macro_auprc"] = macro_auprc(eval_rows, Yte, p, j)
            results[key][str(h)] = row
            print(key, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))

for h in HORIZONS:
    for feat_name, k in (("static", "static"), ("history", "history")):
        ref = k06b["by_horizon"][str(h)][k]
        got = results[f"{feat_name}_train[CAL8]_eval[ALL]"][str(h)]
        print("verify CAL8->ALL vs K06b", feat_name, h, "auprc diff",
              round(abs(got["auprc"] - ref["auprc"]), 6),
              "macro diff", round(abs(got["macro_auprc"] - ref["macro_auprc"]), 6), flush=True)

def gain(train_name, eval_name, h, macro=False):
    s = results[f"static_train[{train_name}]_eval[{eval_name}]"][str(h)]
    a = results[f"history_train[{train_name}]_eval[{eval_name}]"][str(h)]
    key = "macro_auprc" if macro else "auprc"
    return float(a[key] - s[key])

summary = {}
for eval_name in ("R2", "R3", "R2R3"):
    row = {}
    for h in HORIZONS:
        g_tr = gain("R1", eval_name, h)
        g_all = gain("ALL", eval_name, h)
        entry = {"transfer_gain": g_tr, "all_gain": g_all,
                 "recovery_vs_all": g_tr / g_all if g_all else None,
                 "transfer_gain_macro": gain("R1", eval_name, h, macro=True),
                 "all_gain_macro": gain("ALL", eval_name, h, macro=True)}
        entry["recovery_vs_all_macro"] = (entry["transfer_gain_macro"] / entry["all_gain_macro"]
                                          if entry["all_gain_macro"] else None)
        if eval_name == "R3":
            g_clean = gain("R1R2", "R3", h)
            entry["clean_gain"] = g_clean
            entry["recovery_vs_clean"] = g_tr / g_clean if g_clean else None
            entry["clean_gain_macro"] = gain("R1R2", "R3", h, macro=True)
            entry["recovery_vs_clean_macro"] = (entry["transfer_gain_macro"] / entry["clean_gain_macro"]
                                                if entry["clean_gain_macro"] else None)
        if eval_name == "R2":
            g_clean = gain("R1R3", "R2", h)
            entry["clean_gain"] = g_clean
            entry["recovery_vs_clean"] = g_tr / g_clean if g_clean else None
            entry["clean_gain_macro"] = gain("R1R3", "R2", h, macro=True)
            entry["recovery_vs_clean_macro"] = (entry["transfer_gain_macro"] / entry["clean_gain_macro"]
                                                if entry["clean_gain_macro"] else None)
        row[str(h)] = entry
    summary[eval_name] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train", "eval", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for key, per_h in results.items():
        if key.startswith("_"):
            continue
        feat, rest = key.split("_train[")
        train_name, eval_name = rest[:-1].split("]_eval[")
        for h in HORIZONS:
            r = per_h[str(h)]
            w.writerow([feat, train_name, eval_name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
