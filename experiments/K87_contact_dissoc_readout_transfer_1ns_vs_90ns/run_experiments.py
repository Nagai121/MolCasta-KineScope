"""K87: does the contact-dissociation readout transfer across horizons?
(K50 conditions, weak base)

K75 did this for hydrogen-bond dissociation (strong base) and the
readout transferred cleanly (recovery 0.87-1.09 over the full 6x6
matrix). This experiment asks the same for contact dissociation.

Method: K50 conditions (weak base, Doc D). Model: HistGradientBoosting
(max_iter=120, lr .075, max_leaf_nodes 31, min_samples_leaf 15, L2 .02,
seed 42), retrained per train horizon. Train = 1bgf_A, 1ifg_A, 1j8r_A,
2b0a_A (6,383 candidates); test = 1mix_A + 2qt4_A pooled (2,915).
Static: 5 features (origin min distance, CA-CA distance, log1p
sequence separation, heavy-atom counts of residues i and j). History:
6 statistics [first, mean, std, min, last-first, frac <= 4.5 A] from
the 11-frame minimum-distance series (0..1 ns). Horizons 1/5/10/30/50/
90 ns. The features are horizon-independent, so one model trained at
horizon h1 is evaluated against every horizon h2's labels; the full
6x6 matrix is computed.
  - primary: train@1 -> eval@90, and train@90 -> eval@1
  - reference: the matched-horizon condition's gain (train@h -> eval@h)
The gain of a condition is its AUPRC minus the static-only condition's
AUPRC trained at the SAME train horizon and evaluated at the same eval
horizon (pooled over the two test proteins, and per-protein macro over
the two). Recovery = cross-horizon gain / matched-horizon gain.
Verification: the matched-horizon conditions must reproduce K50's
recorded values exactly (static 0.3105 / full 0.5246 at 1 ns, etc.).

Pre-fixed reading: if the recovery is 0.8 or more in BOTH directions
(1 -> 90 and 90 -> 1), the contact-dissociation readout also transfers
across horizons. If only one direction is weak, check whether it is the
same shape as hydrogen-bond formation's 1 ns -> 50 ns weakness (K56:
transfer INTO one horizon is weak from every train horizon) using the
full matrix, and localize it with the static-only transfer in the same
matrix (the weak base has no frozen-model feature to drop, so the
frozen-drop check does not apply here; the static-only transfer across
horizons is the corresponding check).
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.5


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def stack(parts):
    X = np.concatenate([p["X"] for p in parts]).astype(np.float64)
    H = hist6(np.concatenate([p["hist"] for p in parts]))
    Y = np.concatenate([p["y"] for p in parts])
    return X, H, Y


def learner(X, y):
    if np.unique(y).size < 2:
        return None
    return HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                          min_samples_leaf=15, l2_regularization=.02,
                                          random_state=42).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def macro_auprc(parts, Y, p, j):
    spans, cur = [], 0
    # macro over the two test proteins (each protein's 3 replicas pooled)
    per_protein = {}
    for part in parts:
        n = len(part["y"])
        per_protein.setdefault(part["protein"], []).append((cur, cur + n))
        cur += n
    vals = []
    for prot, ss in per_protein.items():
        s, e = ss[0][0], ss[-1][1]
        if 0 < Y[s:e, j].sum() < e - s:
            vals.append(average_precision_score(Y[s:e, j], p[s:e]))
    return float(np.mean(vals))


train_parts = load(TRAIN_P)
test_parts = load(TEST_P)
for part, p in zip(train_parts, [f"{q}_R{r}" for q in TRAIN_P for r in (1, 2, 3)]):
    part["protein"] = p.split("_R")[0]
for part, p in zip(test_parts, [f"{q}_R{r}" for q in TEST_P for r in (1, 2, 3)]):
    part["protein"] = p.split("_R")[0]
Str, Htr, Ytr = stack(train_parts)
Ste, Hte, Yte = stack(test_parts)
print("train n:", len(Ytr), "test n:", len(Yte), flush=True)
print("test positives:", [int(Yte[:, j].sum()) for j in range(6)], flush=True)

k50 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K50_contact_dissociation_carrier/results.json").read_text())

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)

for j1, h1 in enumerate(HORIZONS):
    for feat_name, Xtr, Xte in (("static", Str, Ste),
                                ("history", np.column_stack([Str, Htr]), np.column_stack([Ste, Hte]))):
        key = f"{feat_name}_train[{h1}]"
        results.setdefault(key, {})
        if all(str(h2) in results[key] for h2 in HORIZONS):
            print(key, "already done, skipped", flush=True)
            continue
        m = learner(Xtr, Ytr[:, j1])
        for j2, h2 in enumerate(HORIZONS):
            if str(h2) in results[key]:
                continue
            p = np.full(len(Yte), float(Ytr[:, j1].mean())) if m is None else m.predict_proba(Xte)[:, 1]
            row = metrics(Yte[:, j2], p)
            row["macro_auprc"] = macro_auprc(test_parts, Yte, p, j2)
            results[key][str(h2)] = row
            print(key, "eval", h2, {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in row.items()}, flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))

print("verify matched vs K50 (should be 0.0):")
for h in HORIZONS:
    for feat_name, k50_key in (("static", "static_only"), ("history", "full")):
        ref = k50["dissociation"][k50_key][str(h)]
        got = results[f"{feat_name}_train[{h}]"][str(h)]
        print(f" {feat_name} {h} auprc diff {abs(got['auprc'] - ref['auprc']):.6f}", flush=True)


# Gain definition (K56's): the gain of a history condition trained at h1
# and evaluated at h2 is its AUPRC minus the AUPRC of the static-only
# condition trained at the EVALUATION horizon h2 (the best static model
# for that horizon). The reference gain is the matched-horizon gain
# (train@h2 -> eval@h2). (Corrected 2026-11-23: the first version used
# the static condition trained at the TRAIN horizon h1, which measures
# something else when the static baseline itself transfers poorly.)
def gain(h1, h2, macro=False):
    s = results[f"static_train[{h2}]"][str(h2)]
    a = results[f"history_train[{h1}]"][str(h2)]
    key = "macro_auprc" if macro else "auprc"
    return float(a[key] - s[key])


summary = {}
for h2 in HORIZONS:
    row = {}
    for h1 in HORIZONS:
        g = gain(h1, h2)
        g_ref = gain(h2, h2)
        gm = gain(h1, h2, macro=True)
        gm_ref = gain(h2, h2, macro=True)
        row[str(h1)] = {"transfer_gain": g, "matched_gain": g_ref,
                        "recovery_vs_matched": g / g_ref if g_ref else None,
                        "transfer_gain_macro": gm, "matched_gain_macro": gm_ref,
                        "recovery_vs_matched_macro": gm / gm_ref if gm_ref else None}
    summary[str(h2)] = row
print("summary (rows=eval horizon, cols=train horizon):", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K50"] = k50["dissociation"]

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train_horizon_ns", "eval_horizon_ns", "n", "positives",
                "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for key, per_h in results.items():
        if key.startswith("_"):
            continue
        feat, h1 = key.split("_train[")
        h1 = h1[:-1]
        for h2 in HORIZONS:
            r = per_h[str(h2)]
            w.writerow([feat, h1, h2, r["n"], r["positives"], r["auc"], r["auprc"],
                        r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
