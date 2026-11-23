"""K86 follow-up: why is the 90 -> 1 ns transfer weak (recovery 0.336)
while 1 -> 90 ns is strong (0.894)? All gains use the K56 definition
(static reference trained at the EVAL horizon).

Diagnostics:
  1. Mean-only transfer: does the asymmetry live in the mean readout
     itself, or in the other history statistics?
  2. Label overlap: are 1 ns positives (mostly) a subset of 90 ns
     positives, while 90 ns positives include many 1 ns negatives
     (slow dissociators)? Then a 90 ns-trained ranking puts slow
     dissociators high, which hurts at 1 ns, while a 1 ns-trained
     ranking still works at 90 ns.
  3. Prediction correlation between the history@1 and history@90
     models on the test set.
  4. Static readout transfer: does the asymmetry also exist without
     history features?
  5. Bootstrap 95% CIs for the corrected recoveries (pooled + macro),
     resampling test candidates.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
REPO = Path("/home/hatch/workspace/kinescope/repo/experiments")
HERE = REPO / "K86_saltbridge_dissoc_readout_transfer_1ns_vs_90ns"
_SPLIT = json.loads((REPO / "K15_saltbridge_expansion" / "split.json").read_text())
TRAIN_P = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"])
TEST_P = tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])
BOUND = 4.0
HORIZONS = (1, 5, 10, 30, 50, 90)
rng = np.random.default_rng(20261123)


def load(prots, track=False):
    parts, pl = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt_diss.npz")
            parts.append({k: z[k] for k in z.files})
            pl.append(np.full(len(z["y"]), p, dtype=object))
    return parts, (np.concatenate(pl) if track else None)


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def fit(X, y):
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                       max_leaf_nodes=31, min_samples_leaf=15,
                                       l2_regularization=.02, random_state=42)
    m.fit(X, y)
    return m


train_parts, _ = load(TRAIN_P)
test_parts, test_prot = load(TEST_P, track=True)
Xtr = cat(train_parts, "X"); Xte = cat(test_parts, "X")
Ftr = hist6(cat(train_parts, "hist")); Fte = hist6(cat(test_parts, "hist"))
Ytr = cat(train_parts, "y"); Yte = cat(test_parts, "y")
print("train positives per horizon:", Ytr.sum(0), "of", len(Ytr), flush=True)
print("test positives per horizon:", Yte.sum(0), "of", len(Yte), flush=True)

# --- label overlap (test, and train for reference) ---
for tag, Y in (("train", Ytr), ("test", Yte)):
    p1, p90 = Y[:, 0].astype(bool), Y[:, 5].astype(bool)
    print(f"{tag} label overlap: P(90pos|1pos)={p90[p1].mean():.3f} "
          f"P(1pos|90pos)={p1[p90].mean():.3f} "
          f"1pos&90neg={int((p1 & ~p90).sum())} 1neg&90pos={int((~p1 & p90).sum())}",
          flush=True)

# --- refit static / full / mean-only at 1 and 90 ns, save predictions ---
preds = {}
for h1, j1 in ((1, 0), (90, 5)):
    for name, Xtr_f, Xte_f in (
            ("static", Xtr, Xte),
            ("history", np.concatenate([Xtr, Ftr], axis=1), np.concatenate([Xte, Fte], axis=1)),
            ("mean_only", np.concatenate([Xtr, Ftr[:, [1]]], axis=1),
             np.concatenate([Xte, Fte[:, [1]]], axis=1))):
        m = fit(Xtr_f, Ytr[:, j1])
        preds[f"{name}_train[{h1}]"] = m.predict_proba(Xte_f)[:, 1]
np.savez(HERE / "k86_predictions.npz", **preds, Yte=Yte, test_prot=test_prot.astype(str))
print("predictions saved", flush=True)


def ap(feat, h1, h2, Y=Yte, P=None):
    P = preds if P is None else P
    j2 = HORIZONS.index(h2)
    return average_precision_score(Y[:, j2], P[f"{feat}_train[{h1}]"])


def macro_ap(feat, h1, h2, Y=Yte, P=None, PL=None):
    P = preds if P is None else P
    PL = test_prot if PL is None else PL
    j2 = HORIZONS.index(h2)
    vals = []
    for prot in np.unique(PL):
        m = PL == prot
        yj = Y[m, j2]
        if 0 < yj.sum() < m.sum():
            vals.append(average_precision_score(yj, P[f"{feat}_train[{h1}]"][m]))
    return float(np.mean(vals)) if vals else None


def gain(feat, h1, h2, macro=False):
    # K56: static reference trained at the EVAL horizon h2.
    f = macro_ap if macro else ap
    return f(feat, h1, h2) - f("static", h2, h2)


def recovery(feat, h1, h2, macro=False):
    return gain(feat, h1, h2, macro) / gain(feat, h2, h2, macro)


for feat in ("history", "mean_only"):
    for macro in (False, True):
        tag = "macro" if macro else "pooled"
        print(f"{feat} {tag}: 1->90 recovery {recovery(feat, 1, 90, macro):.3f} "
              f"(transfer {gain(feat, 1, 90, macro):.4f} / matched {gain(feat, 90, 90, macro):.4f}) | "
              f"90->1 recovery {recovery(feat, 90, 1, macro):.3f} "
              f"(transfer {gain(feat, 90, 1, macro):.4f} / matched {gain(feat, 1, 1, macro):.4f})",
              flush=True)

# --- prediction correlations ---
for pair in (("history_train[1]", "history_train[90]"),
             ("static_train[1]", "static_train[90]"),
             ("mean_only_train[1]", "mean_only_train[90]")):
    a, b = preds[pair[0]], preds[pair[1]]
    print(f"pred corr {pair[0]} vs {pair[1]}: pearson {np.corrcoef(a, b)[0, 1]:.3f} "
          f"spearman {spearmanr(a, b).statistic:.3f}", flush=True)

# --- absolute AUPRCs of each model at each horizon ---
print("absolute AUPRC (test pooled):")
for feat in ("static", "history", "mean_only"):
    print(f" {feat}: train@1 -> eval@1 {ap(feat, 1, 1):.4f} eval@90 {ap(feat, 1, 90):.4f} | "
          f"train@90 -> eval@1 {ap(feat, 90, 1):.4f} eval@90 {ap(feat, 90, 90):.4f}", flush=True)

# --- bootstrap CIs for the corrected recoveries (history, pooled + macro) ---
N = len(Yte)
B = 2000
rec = {"1->90 pooled": [], "90->1 pooled": [], "1->90 macro": [], "90->1 macro": [],
       "1->90 pooled mean_only": [], "90->1 pooled mean_only": []}
for b in range(B):
    idx = rng.integers(0, N, N)
    Yb, Pb = Yte[idx], test_prot[idx]
    Pr = {k: v[idx] for k, v in preds.items()}

    def g(feat, h1, h2, macro=False):
        f = (lambda ft, a, c: macro_ap(ft, a, c, Y=Yb, P=Pr, PL=Pb)) if macro else \
            (lambda ft, a, c: ap(ft, a, c, Y=Yb, P=Pr))
        return f(feat, h1, h2) - f("static", h2, h2)

    rec["1->90 pooled"].append(g("history", 1, 90) / g("history", 90, 90))
    rec["90->1 pooled"].append(g("history", 90, 1) / g("history", 1, 1))
    rec["1->90 macro"].append(g("history", 1, 90, True) / g("history", 90, 90, True))
    rec["90->1 macro"].append(g("history", 90, 1, True) / g("history", 1, 1, True))
    rec["1->90 pooled mean_only"].append(g("mean_only", 1, 90) / g("mean_only", 90, 90))
    rec["90->1 pooled mean_only"].append(g("mean_only", 90, 1) / g("mean_only", 1, 1))
    if (b + 1) % 500 == 0:
        print("bootstrap", b + 1, flush=True)

out = {}
for k, v in rec.items():
    v = np.array(v, dtype=float)
    v = v[np.isfinite(v)]
    out[k] = dict(mean=float(v.mean()), sd=float(v.std()),
                  lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                  frac_above_0_8=float((v >= 0.8).mean()), n=len(v))
    print(f"bootstrap {k}: mean {out[k]['mean']:.3f} sd {out[k]['sd']:.3f} "
          f"95% CI [{out[k]['lo']:.3f}, {out[k]['hi']:.3f}] "
          f"P(recovery>=0.8)={out[k]['frac_above_0_8']:.3f}", flush=True)
(HERE / "k86_bootstrap.json").write_text(json.dumps(out, indent=2))

# --- per-protein gains for the weak direction (90 -> 1), pooled per protein ---
print("\nper-protein gains, train@90 -> eval@1 (history - static@1):")
rows = []
for prot in np.unique(test_prot):
    m = test_prot == prot
    yj = Yte[m, 0]
    if 0 < yj.sum() < m.sum():
        g = (average_precision_score(yj, preds["history_train[90]"][m])
             - average_precision_score(yj, preds["static_train[1]"][m]))
        rows.append((prot, int(m.sum()), int(yj.sum()), round(float(g), 3)))
rows.sort(key=lambda t: t[3])
print(f" n_proteins={len(rows)} median={np.median([r[3] for r in rows]):.3f} "
      f"mean={np.mean([r[3] for r in rows]):.3f}")
print("  lowest:", rows[:5])
print("  highest:", rows[-5:])
print("saved", HERE / "k86_bootstrap.json")
