"""K58: contact DISSOCIATION version of K33 -- how much guidance do
the errors of the non-mean statistics cost?

K54 (hydrogen-bond dissociation, residue-level generation) found
that the std error alone is the main carrier at every horizon
(idea 14). K58 measures the same split for contact dissociation
under K50's conditions (Doc D).

Method (K33's, adapted to Doc D):
- Model: HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
  min_samples_leaf 15, L2 0.02, seed 42), trained per horizon on the
  REAL history (static + hist6) of the 4 train proteins, never
  retrained for the swap conditions. Only the test features change.
- Real history features are the 6 statistics of the 11-frame minimum
  heavy-atom distance series (0..1 ns): [first, mean, std, min,
  last-first, frac <= 4.5 A]. The mean is kept real; std / min /
  frac<=4.5 are replaced one at a time, and all three together, with
  the values computed from the GENERATED series.
- Generated series: residue-level generation, proposals 11-14
  (K11/K17/K20 protocol; see gen_k58.py). Each idea gives 4 branches
  x 9 points (origin + 8 steps). The generated statistics are
  computed over the BRANCH-MEAN 9-point series, the same convention
  as K48/K54. Ideas 14 (closest to the real series, see the quality
  check) and 11 (quietest) are measured as primary and cross-check,
  the same pair as K48/K54. Alignment: dissociation candidates are
  matched to generated pairs by (res_i, res_j) per trajectory; every
  candidate must match. (K41's generated series cannot be used here:
  they cover only formation candidates, origin > 4.5 A, while
  dissociation candidates have origin <= 4.5 A.)
- Reference conditions: static_only, base_hist (real hist6), and gen6
  (all 6 statistics from the generated series, per idea).
- Asymmetry to note (same as K33/K48/K54): the real statistics are
  computed over the BACKWARD 11-frame history (0..1 ns), while the
  generated statistics are computed over the FORWARD 9-point series
  (1..2 ns). The swap therefore also changes backward to forward,
  not only real to generated.
- Retention = (condition - static) / (base_hist - static) in AUPRC.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
GEN_DIR = Path("/home/hatch/workspace/kinescope/work/k58_gen_residue")
K50 = Path("/home/hatch/workspace/kinescope/repo/experiments/K50_contact_dissociation_carrier/results.json")
HERE = Path(__file__).resolve().parent
PARTIAL = HERE / "results_k58_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.5


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def load_gen(prot, rep, res_i, res_j):
    """Generated 6 statistics per idea for the given candidates,
    aligned by (res_i, res_j). Returns {idea: (N, 6)}.
    Each idea's statistics are computed over the branch-mean 9-point
    series (origin + 8 steps), the same convention as K48/K54."""
    z = np.load(GEN_DIR / f"{prot}_R{rep}.npz")
    key = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(z["res_i"], z["res_j"]))}
    idx = np.array([key[(int(u), int(v))] for u, v in zip(res_i, res_j)])
    assert len(idx) == len(res_i), (prot, rep, len(idx), len(res_i))
    out = {}
    for num in (11, 12, 13, 14):
        gen9 = z[f"gen{num}"][:, :, idx].mean(axis=0)  # (9, N) branch-mean
        out[num] = hist6(gen9.T)
    return out


def load(prots):
    xs, hs, ys = [], [], []
    gs = {num: [] for num in (11, 12, 13, 14)}
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
            xs.append(z["X"].astype(np.float64))
            hs.append(z["hist"].astype(np.float64))  # K50 cache stores (N, 11)
            ys.append(z["y"].astype(np.float64))
            g = load_gen(p, r, z["res_i"], z["res_j"])
            for num in gs:
                gs[num].append(g[num])
    return (np.concatenate(xs), np.concatenate(hs), np.concatenate(ys),
            {num: np.concatenate(v) for num, v in gs.items()})


def main():
    Xtr, Htr, Ytr, Gtr = load(TRAIN_P)
    Xte, Hte, Yte, Gte = load(TEST_P)
    Ftr, Fte = hist6(Htr), hist6(Hte)
    print("train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)

    results = {}
    if PARTIAL.exists():
        results = json.loads(PARTIAL.read_text())
        print("resumed:", sorted(results), flush=True)

    def eval_on(fte_block, tag):
        out = {}
        for hi, h in enumerate(HORIZONS):
            p = predict(np.concatenate([Xtr, Ftr], axis=1), Ytr[:, hi],
                        np.concatenate([Xte, fte_block], axis=1))
            out[str(h)] = metrics(Yte[:, hi], p)
        print(tag, "AUPRC", {h: round(out[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
        return out

    def eval_static():
        out = {}
        for hi, h in enumerate(HORIZONS):
            p = predict(Xtr, Ytr[:, hi], Xte)
            out[str(h)] = metrics(Yte[:, hi], p)
        print("static_only", "AUPRC", {h: round(out[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
        return out

    if "static_only" not in results:
        results["static_only"] = eval_static()
        PARTIAL.write_text(json.dumps(results, indent=2))
    if "base_hist" not in results:
        results["base_hist"] = eval_on(Fte, "base_hist")
        PARTIAL.write_text(json.dumps(results, indent=2))
    # swap conditions, measured for idea 14 (closest to real, K20)
    # and idea 11 (quietest) as a cross-check, the same pair as K48/K54.
    for idea in (14, 11):
        G = Gte[idea]
        for name, repl in (("std_only", [(2, 2)]), ("min_only", [(3, 3)]),
                           ("frac45_only", [(5, 5)]),
                           ("combined3", [(2, 2), (3, 3), (5, 5)])):
            tag = f"{name}_i{idea}"
            if tag in results:
                print(tag, "already done, skipped", flush=True)
                continue
            Fswap = Fte.copy()
            for real_col, gen_col in repl:
                Fswap[:, real_col] = G[:, gen_col]
            results[tag] = eval_on(Fswap, tag)
            PARTIAL.write_text(json.dumps(results, indent=2))
        tag = f"gen6_i{idea}"
        if tag not in results:
            results[tag] = eval_on(G, tag)
            PARTIAL.write_text(json.dumps(results, indent=2))

    # --- derived: retention of the real-history gain ---
    retention = {}
    for name in [f"{n}_i{i}" for i in (14, 11)
                 for n in ("std_only", "min_only", "frac45_only", "combined3", "gen6")]:
        retention[name] = {}
        for h in HORIZONS:
            s = results["static_only"][str(h)]["auprc"]
            b = results["base_hist"][str(h)]["auprc"]
            c = results[name][str(h)]["auprc"]
            retention[name][str(h)] = float((c - s) / (b - s))
    results["_retention"] = retention
    k50 = json.loads(K50.read_text())["dissociation"]
    results["_checks_vs_K50"] = {
        "static_max_abs_diff": max(
            abs(results["static_only"][str(h)]["auprc"] - k50["static_only"][str(h)]["auprc"])
            for h in HORIZONS),
        "base_max_abs_diff": max(
            abs(results["base_hist"][str(h)]["auprc"] - k50["full"][str(h)]["auprc"])
            for h in HORIZONS),
    }
    print("checks vs K50:", results["_checks_vs_K50"], flush=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    rows = []
    for name, by_h in results.items():
        if name.startswith("_"):
            continue
        for h, m in by_h.items():
            rows.append(dict(condition=name, horizon_ns=h, **m))
    with open(HERE / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved to", HERE)


if __name__ == "__main__":
    main()
