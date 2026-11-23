"""K83: does a readout trained at one horizon work at another horizon?
(contact formation, D conditions)

K56 (hydrogen bonds, strong base) found that a readout trained at 1 ns
transfers weakly to 50 ns (recovery 0.608), while the reverse direction
(50 ns -> 1 ns) is strong (1.026 in K79; 1.33 to 90 ns). K82 showed the
1 ns -> 50 ns weakness disappears when the frozen body's prediction is
dropped from the correction model. Contact formation has no frozen body
(D conditions: static 5 features + history 6 statistics), so K82's
explanation cannot apply to it. This experiment asks whether the same
asymmetric transfer pattern appears for contact formation anyway.

Method: D conditions (K13 base). ATLAS analysis 6 proteins x 3 replicas,
train 1bgf_A/1ifg_A/1j8r_A/2b0a_A (15,926 candidates), test 1mix_A+2qt4_A
(7,384 candidates). Contact = residue pair, CA distance <= 12 A, sequence
separation >= 2; distance = min heavy-atom distance; candidate: origin
(frame 10) distance > 4.5 A; label: distance <= 4.5 A at the horizon.
Static 5 features, history = 6 statistics of the 0-1 ns distance series
(first, mean, std, min, last-first, fraction <= 4.5 A). HistGBM
(max_iter=120, lr .075, max_leaf_nodes=31, min_samples_leaf=15,
L2 .02, seed 42), refit per condition. Horizons 1/5/10/30/50/90 ns.

Features are horizon-independent, so a model trained at one horizon can
be evaluated against another horizon's labels (same as K56). The
static-only condition is also trained per horizon. Gain at evaluation
horizon e = (transferred history AUPRC(e) - static AUPRC(e)); the
reference is the matched-horizon gain at e. Recovery = transferred gain
/ matched gain. Matched-horizon fits are verified against K13/K16
(static AUPRC 0.3398/0.2870/0.3204/0.2868/0.2604/0.2872 and history
0.5669/0.5192/0.5234/0.4386/0.4638/0.4722).

Pre-fixed reading: if both directions recover 0.8 or more, the contact
readout transfers across horizons. If only one direction is weak, say so
honestly, and note that K82's frozen-body explanation does not apply
(contact has no frozen body) and that K27 found the mean alone recovers
only 0.65-0.71 of contact guidance, i.e. the carrier is more spread out
than for hydrogen bonds.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
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
            z = np.load(CACHE / f"{p}_R{r}_contact.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


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


train, test = load(TRAIN_P), load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = hist6(cat(train, "hist"), BOUND), hist6(cat(test, "hist"), BOUND)
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, "test positives", Yte.sum(0), flush=True)

k13 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K13_other_phenomena_real/results.json").read_text())
k13_static = {h: k13["contact"]["static_only"][str(h)]["auprc"] for h in HORIZONS}
k13_hist = {h: k13["contact"]["base_hist_D6"][str(h)]["auprc"] for h in HORIZONS}

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed", flush=True)
results.setdefault("static", {})
results.setdefault("history", {})

# Train static and history models at every horizon; evaluate each at
# every horizon (full 6x6 transfer matrix, as in K56).
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
            results[feat_name][key][str(h_eval)] = row
            print(key, "->", h_eval, "auprc", round(row["auprc"], 4), flush=True)
            PARTIAL.write_text(json.dumps(results, indent=2))

# Verify matched-horizon fits against K13.
for h in HORIZONS:
    for feat_name, ref in (("static", k13_static), ("history", k13_hist)):
        got = results[feat_name][f"{feat_name}_train{h}"][str(h)]["auprc"]
        print("verify matched", feat_name, h, "auprc diff", round(abs(got - ref[h]), 6), flush=True)

# Gains and recovery ratios: rows = evaluation horizon, columns = training horizon.
summary = {}
for e in HORIZONS:
    s = results["static"][f"static_train{e}"][str(e)]["auprc"]
    matched = results["history"][f"history_train{e}"][str(e)]["auprc"]
    matched_gain = matched - s
    row = {"static_auprc": s, "matched_history_auprc": matched,
           "matched_gain": matched_gain, "recovery": {}}
    for t in HORIZONS:
        tr = results["history"][f"history_train{t}"][str(e)]["auprc"]
        g = tr - s
        row["recovery"][str(t)] = (g / matched_gain) if matched_gain else None
        print("eval", e, "train", t, "transfer gain", round(g, 4),
              "recovery", round(row["recovery"][str(t)], 4), flush=True)
    summary[str(e)] = row

results["_summary"] = summary
results["_references_K13"] = {"static_auprc": k13_static, "history_auprc": k13_hist}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "train_horizon_ns", "eval_horizon_ns", "n", "positives",
                "auc", "auprc", "brier", "logloss"])
    for feat_name in ("static", "history"):
        for t in HORIZONS:
            for e in HORIZONS:
                r = results[feat_name][f"{feat_name}_train{t}"][str(e)]
                w.writerow([feat_name, t, e, r["n"], r["positives"], r["auc"], r["auprc"],
                            r["brier"], r["logloss"]])
print("saved to", OUT)
