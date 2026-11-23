"""K80: what do the 1 ns and 50 ns readouts rely on? (strong base, H-bonds)

K56 found the 1 ns -> 50 ns transfer weak (recovery 0.608) while the
reverse (K79: 50 ns -> 1 ns / 90 ns) is strong (1.026 / 1.330). K77
showed that at 50 ns the carriers are mean (0.933) and fraction
(0.888), so the weakness is not a missing carrier at 50 ns. This
experiment compares what the two readouts rely on, by dropping one
history statistic at a time (K18 style) and retraining.

Design (all K06(b) conditions: frozen T1I2006 + HistGradientBoosting
sidecar, 160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15,
seed 20261121; 8 calibration proteins train, 100 test proteins x 3
replicas evaluate; saved K06(b) features, no trajectory reload):
  train/eval combos: (1,1), (1,50), (50,50), and (50,1) as reference.
  For each combo, conditions: full (static 5 + history 10) and full
  minus one of the six statistics (drop the column, K18 style):
  first(0), mean(1), std(2), min(3), last-first(5), frac<=3.5(7).
  The other four history features (max, last-prev, frac<=4.5,
  frac(diff>0)) stay in every condition.
  Static-only is trained at the *evaluation* horizon (K56/K79 rule)
  and used as the gain baseline at that horizon.

Gain(combo, cond) = AUPRC(cond, eval) - AUPRC(static@eval, eval).
Residual rate = Gain(combo, drop X) / Gain(combo, full).
A statistic is "relied on" if dropping it lowers the gain a lot
(low residual rate). Because statistics substitute for each other
(K18: dropping any one statistic loses at most ~12% of the gain),
a small loss means "substitutable", not "uninformative"; read this
together with the K21/K73/K77 "add one statistic" results.

Pre-fixed reading: if the 1 ns readout relies, at 50 ns, on a
statistic that is not a carrier at 50 ns, that difference explains
the weakness. If both readouts rely on the same statistics, the
weakness has another cause.

Cross-checks: the (1,1) and (50,50) drop losses should match K18's
table (train=eval); the (1,50) full gain / 0.608 recovery should
match K56; the (50,1) full gain / 1.026 recovery should match K79.
Results are checkpointed per condition in results_k80_partial.json.
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
PARTIAL = HERE / "results_k80_partial.json"
orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# train/eval horizon pairs: (train, eval)
COMBOS = [(1, 1), (1, 50), (50, 50), (50, 1)]
DROPS = {"drop_first": 0, "drop_mean": 1, "drop_std": 2,
         "drop_min": 3, "drop_lastfirst": 5, "drop_frac35": 7}
STAT_NAMES = {"drop_first": "first", "drop_mean": "mean", "drop_std": "std",
              "drop_min": "min", "drop_lastfirst": "last-first",
              "drop_frac35": "frac<=3.5"}


def load(proteins):
    rows = []
    for p in proteins:
        with np.load(OUTD / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({k: z[f"r{r}_{k}"] for k in
                             ("static", "history", "frozen", "label")})
    return rows


def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def build_X(rows, drop_col):
    keep = [i for i in range(10) if i != drop_col] if drop_col is not None else list(range(10))
    return np.concatenate([np.column_stack([r["static"], r["history"][:, keep]]) for r in rows])


def main():
    cal, test = load(CAL), load(TEST)
    y_cal = np.concatenate([r["label"] for r in cal])
    y_test = np.concatenate([r["label"] for r in test])
    spans, cur = [], 0
    for p in TEST:
        with np.load(OUTD / f"{p}.npz") as z:
            n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n))
        cur += n
    if PARTIAL.is_file():
        results = json.loads(PARTIAL.read_text())
    else:
        results = {"n_cal": int(len(y_cal)), "n_test": int(len(y_test)),
                   "calibration_proteins": CAL, "test_proteins": TEST,
                   "combos": {}}
    X_static_tr = np.concatenate([r["static"] for r in cal])
    X_static_te = np.concatenate([r["static"] for r in test])
    for h_train, h_eval in COMBOS:
        key = f"train{h_train}_eval{h_eval}"
        combo = results["combos"].get(key, {})
        jt, je = HORIZONS.index(h_train), HORIZONS.index(h_eval)
        # static-only, trained at the evaluation horizon (K56/K79 rule)
        if "static" not in combo:
            pred = learner(X_static_tr, y_cal[:, je]).predict_proba(X_static_te)[:, 1]
            m = metrics(y_test[:, je], pred)
            macro = [average_precision_score(y_test[s:e, je], pred[s:e])
                     for _, s, e in spans if 0 < y_test[s:e, je].sum() < e - s]
            m["macro_auprc"] = float(np.mean(macro))
            combo["static"] = m
            print(key, "static", {k: round(v, 4) for k, v in m.items()}, flush=True)
        for name, drop_col in [("full", None)] + list(DROPS.items()):
            if name in combo:
                continue
            Xtr = build_X(cal, drop_col)
            Xte = build_X(test, drop_col)
            pred = learner(Xtr, y_cal[:, jt]).predict_proba(Xte)[:, 1]
            m = metrics(y_test[:, je], pred)
            macro = [average_precision_score(y_test[s:e, je], pred[s:e])
                     for _, s, e in spans if 0 < y_test[s:e, je].sum() < e - s]
            m["macro_auprc"] = float(np.mean(macro))
            combo[name] = m
            print(key, name, {k: round(v, 4) for k, v in m.items()}, flush=True)
        results["combos"][key] = combo
        PARTIAL.write_text(json.dumps(results, indent=1))

    # Gains and residual rates per combo.
    summary = {}
    for h_train, h_eval in COMBOS:
        key = f"train{h_train}_eval{h_eval}"
        combo = results["combos"][key]
        entry = {}
        for kind, kkey in (("macro", "macro_auprc"), ("micro", "auprc")):
            g_full = combo["full"][kkey] - combo["static"][kkey]
            entry[kind] = {"gain_full": float(g_full),
                           "gain_static_auprc": float(combo["static"][kkey]),
                           "full_auprc": float(combo["full"][kkey]),
                           "residual": {name: float((combo[name][kkey] - combo["static"][kkey]) / g_full)
                                        for name in DROPS},
                           "loss_pct": {name: float(100 * (1 - (combo[name][kkey] - combo["static"][kkey]) / g_full))
                                        for name in DROPS}}
        summary[key] = entry
    results["summary"] = summary

    # Cross-checks against K18 (drop losses, train=eval), K56, K79.
    k18 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K18_strongbase_ablation/results.json").read_text())
    k56 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K56_readout_transfer_across_horizons/results.json").read_text())
    k79 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K79_readout_transfer_from_50ns/results.json").read_text())
    checks = {}
    # K18: loss fraction when dropping a statistic, train=eval horizon.
    # K18 key for last-first is drop_lastminusfirst; K80 uses drop_lastfirst.
    k18_key = {"drop_first": "drop_first", "drop_mean": "drop_mean",
               "drop_std": "drop_std", "drop_min": "drop_min",
               "drop_lastfirst": "drop_lastminusfirst", "drop_frac35": "drop_frac35"}
    for h in (1, 50):
        key = f"train{h}_eval{h}"
        checks[f"k18_train{h}"] = {}
        for name in DROPS:
            kk = k18_key[name]
            checks[f"k18_train{h}"][name] = {
                "k80_loss_pct_macro": summary[key]["macro"]["loss_pct"][name],
                "k18_loss_pct_macro": 100 * k18["drop"][str(h)]["macro"]["drop_share"][kk],
                "k80_loss_pct_micro": summary[key]["micro"]["loss_pct"][name],
                "k18_loss_pct_micro": 100 * k18["drop"][str(h)]["micro"]["drop_share"][kk],
            }
    # K56: 1->50 transfer recovery (pooled/macro).
    s50 = results["combos"]["train1_eval50"]["static"]
    f50 = results["combos"]["train1_eval50"]["full"]
    f50_matched = results["combos"]["train50_eval50"]["full"]
    checks["k56_train1_eval50"] = {
        "k80_recovery_pooled": float((f50["auprc"] - s50["auprc"]) / (f50_matched["auprc"] - s50["auprc"])),
        "k80_recovery_macro": float((f50["macro_auprc"] - s50["macro_auprc"]) / (f50_matched["macro_auprc"] - s50["macro_auprc"])),
        "k56_recovery_pooled": k56["_summary"]["50"]["train1_recovery_pooled"],
        "k56_recovery_macro": k56["_summary"]["50"]["train1_recovery_macro"],
    }
    # K79: 50->1 transfer recovery.
    s1 = results["combos"]["train50_eval1"]["static"]
    f1 = results["combos"]["train50_eval1"]["full"]
    f1_matched = results["combos"]["train1_eval1"]["full"]
    checks["k79_train50_eval1"] = {
        "k80_recovery_pooled": float((f1["auprc"] - s1["auprc"]) / (f1_matched["auprc"] - s1["auprc"])),
        "k80_recovery_macro": float((f1["macro_auprc"] - s1["macro_auprc"]) / (f1_matched["macro_auprc"] - s1["macro_auprc"])),
        "k79_recovery_pooled": k79["_summary"]["1"]["recovery_pooled"],
        "k79_recovery_macro": k79["_summary"]["1"]["recovery_macro"],
    }
    results["checks"] = checks
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print("WROTE results.json")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
