"""K85: does a readout trained at one horizon work at another horizon?
(salt-bridge formation, K15 conditions)

K56 (hydrogen bonds, strong base) found a readout trained at 1 ns
transfers weakly to 50 ns (recovery 0.608), while the reverse direction
(50 ns -> 1 ns) is strong (1.026 in K79; 1.33 to 90 ns). K82 showed the
1 ns -> 50 ns weakness disappears when the frozen body's prediction is
dropped. K83 found contact formation transfers both ways (1 -> 50:
0.960, 50 -> 1: 1.082). This experiment asks the same for salt-bridge
formation, using the K15 expanded set.

Method: K15 conditions. Train 58 proteins (2,557 candidates), test 30
proteins (1,170 candidates), split frozen in K15 split.json. Salt bridge
= min distance between charged heavy atoms (LYS NZ / ARG NE,NH1,NH2 vs
ASP OD1,OD2 / GLU OE1,OE2); candidate: origin (frame 10) distance in
(4.0, 10.0] A, sequence separation >= 2; label: distance <= 4.0 A at
the horizon. Static 5 features, history = 6 statistics of the 0-1 ns
distance series (first, mean, std, min, last-first, fraction <= 4.0 A).
HistGBM (max_iter=120, lr .075, max_leaf_nodes=31, min_samples_leaf=15,
L2 .02, seed 42), refit per condition. Horizons 1/5/10/30/50/90 ns.

Features are horizon-independent, so a model trained at one horizon can
be evaluated against another horizon's labels (same as K56). The
static-only condition is also trained per horizon. Gain at evaluation
horizon e = (transferred history AUPRC(e) - static AUPRC(e)); the
reference is the matched-horizon gain at e. Recovery = transferred gain
/ matched gain. Matched-horizon fits are verified against K15
(static AUPRC 0.2402/0.2513/0.2034/0.1582/0.2295/0.2136 and history
0.3842/0.3301/0.2627/0.2466/0.2569/0.2736).

Because positives are few (93-149 per horizon over 30 test proteins),
both the pooled value and the per-protein macro average are reported.
Macro AUPRC averages only proteins with both classes present at that
horizon; the number of proteins used is recorded. Salt-bridge formation
has no frozen body (K15: static 5 features + history 6 statistics), so
K82's "drop the frozen-body prediction" condition does not apply.

Pre-fixed reading: if both directions recover 0.8 or more, the
salt-bridge readout transfers across horizons. If only one direction is
weak, say so honestly, and note that K82 does not apply and that K31
found the mean alone recovers only 0.34-1.10 of salt-bridge guidance
depending on horizon, i.e. the carrier varies by horizon.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
REPO = Path("/home/hatch/workspace/kinescope/repo")
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.0

_SPLIT = json.loads((REPO / "experiments/K15_saltbridge_expansion/split.json").read_text())
TRAIN_P = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"])
TEST_P = tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
            parts.append((p, {k: z[k] for k in z.files}))
    return parts


def cat(parts, key):
    return np.concatenate([d[key] for _, d in parts])


def hist6(series, bound):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                          min_samples_leaf=15, l2_regularization=.02,
                                          random_state=42).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def macro_auprc(y, p, prot):
    vals = []
    for name in np.unique(prot):
        m = prot == name
        yy, pp = y[m], p[m]
        if 0 < yy.sum() < len(yy):
            vals.append(float(average_precision_score(yy, pp)))
    return (float(np.mean(vals)) if vals else None), len(vals)


train, test = load(TRAIN_P), load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = hist6(cat(train, "hist"), BOUND), hist6(cat(test, "hist"), BOUND)
Ytr, Yte = cat(train, "y"), cat(test, "y")
prot_te = np.concatenate([[p] * len(d["y"]) for p, d in test])
print("train", Xtr.shape, "test", Xte.shape, "test positives", Yte.sum(0), flush=True)

k15 = json.loads((REPO / "experiments/K15_saltbridge_expansion/results.json").read_text())
k15_static = {h: k15["full"]["static_only"][str(h)]["auprc"] for h in HORIZONS}
k15_hist = {h: k15["full"]["base_hist_D6"][str(h)]["auprc"] for h in HORIZONS}

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)
results.setdefault("static", {})
results.setdefault("history", {})

# Train static and history models at every horizon; evaluate each at
# every horizon (full 6x6 transfer matrix, as in K56/K83).
for feat_name, Xtr_f, Xte_f in (("static", Xtr, Xte),
                                ("history", np.concatenate([Xtr, Htr], axis=1),
                                 np.concatenate([Xte, Hte], axis=1))):
    for h_train in HORIZONS:
        key = f"{feat_name}_train{h_train}"
        results[feat_name].setdefault(key, {})
        t = HORIZONS.index(h_train)
        m = learner(Xtr_f, Ytr[:, t])
        for h_eval in HORIZONS:
            if str(h_eval) in results[feat_name][key]:
                continue
            e = HORIZONS.index(h_eval)
            p = m.predict_proba(Xte_f)[:, 1]
            row = metrics(Yte[:, e], p)
            mac, n_mac = macro_auprc(Yte[:, e], p, prot_te)
            row["auprc_macro"] = mac
            row["n_proteins_macro"] = n_mac
            results[feat_name][key][str(h_eval)] = row
            print(key, "->", h_eval, "auprc", round(row["auprc"], 4),
                  "macro", None if mac is None else round(mac, 4), flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))

# Verify matched-horizon fits against K15.
for h in HORIZONS:
    for feat_name, ref in (("static", k15_static), ("history", k15_hist)):
        got = results[feat_name][f"{feat_name}_train{h}"][str(h)]["auprc"]
        print("verify matched", feat_name, h, "auprc diff", round(abs(got - ref[h]), 6), flush=True)

# Gains and recovery ratios: rows = evaluation horizon, columns = training horizon.
summary = {}
for e in HORIZONS:
    s = results["static"][f"static_train{e}"][str(e)]["auprc"]
    s_mac = results["static"][f"static_train{e}"][str(e)]["auprc_macro"]
    matched = results["history"][f"history_train{e}"][str(e)]["auprc"]
    matched_mac = results["history"][f"history_train{e}"][str(e)]["auprc_macro"]
    matched_gain = matched - s
    matched_gain_mac = matched_mac - s_mac
    row = {"static_auprc": s, "static_auprc_macro": s_mac,
           "matched_history_auprc": matched, "matched_history_auprc_macro": matched_mac,
           "matched_gain": matched_gain, "matched_gain_macro": matched_gain_mac,
           "recovery": {}, "recovery_macro": {}}
    for t in HORIZONS:
        tr = results["history"][f"history_train{t}"][str(e)]["auprc"]
        tr_mac = results["history"][f"history_train{t}"][str(e)]["auprc_macro"]
        g = tr - s
        g_mac = tr_mac - s_mac
        row["recovery"][str(t)] = (g / matched_gain) if matched_gain else None
        row["recovery_macro"][str(t)] = (g_mac / matched_gain_mac) if matched_gain_mac else None
        print("eval", e, "train", t, "transfer gain", round(g, 4),
              "recovery", round(row["recovery"][str(t)], 4),
              "macro recovery", None if row["recovery_macro"][str(t)] is None
              else round(row["recovery_macro"][str(t)], 4), flush=True)
    summary[str(e)] = row

results["_summary"] = summary
results["_references_K15"] = {"static_auprc": k15_static, "history_auprc": k15_hist}
results["_split"] = dict(train_proteins=list(TRAIN_P), test_proteins=list(TEST_P),
                         n_train=int(len(Xtr)), n_test=int(len(Xte)),
                         test_positives=[int(v) for v in Yte.sum(0)],
                         train_positives=[int(v) for v in Ytr.sum(0)])

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train_horizon_ns", "eval_horizon_ns", "n", "positives",
                "auc", "auprc", "auprc_macro", "n_proteins_macro", "brier", "logloss"])
    for feat_name in ("static", "history"):
        for t in HORIZONS:
            for e in HORIZONS:
                r = results[feat_name][f"{feat_name}_train{t}"][str(e)]
                w.writerow([feat_name, t, e, r["n"], r["positives"], r["auc"], r["auprc"],
                            r["auprc_macro"], r["n_proteins_macro"], r["brier"], r["logloss"]])
print("saved to", OUT)
