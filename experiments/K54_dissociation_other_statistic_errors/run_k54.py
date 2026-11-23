"""K54: hydrogen-bond DISSOCIATION -- is the error shape of the
non-mean statistics the same as formation? (Doc D conditions)

K33 (formation, strong base) found the shape depends on the horizon:
at 1 ns replacing std/min/frac<=3.5 together still leaves 0.65 of the
gain; at 5/30 ns the three together drop it to the generated level;
at 90 ns std alone erases it. K54 measures the same split for
dissociation under K46's conditions.

Method (K33's, adapted to K46's dissociation base):
- Model: HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
  min_samples_leaf 15, L2 0.02, seed 42), trained per horizon on the
  REAL history (static + hist6) of the 4 train proteins, never
  retrained for the swap conditions. Only the test features change.
- Real history features are the 6 statistics of the 11-frame
  dissociation distance series (0..1 ns): [first, mean, std, min,
  last-first, frac <= 3.5 A]. The mean is kept real; std / min /
  frac<=3.5 are replaced one at a time, and all three together, with
  the values computed from the GENERATED series.
- Generated series: proposal-22 (K06(a)'s generation), regenerated
  for the dissociation candidates (see gen_k54.py). gen5 = [origin
  distance, generated forward distances at 0.1/0.3/0.5/1.0 ns];
  gen6 = the same 6 statistics computed over those 5 points.
  Alignment: dissociation candidates are matched to generated pairs
  by (donor_atom, acceptor_atom) per trajectory; every candidate must
  match (the generator keeps pairs with origin distance <= 6.0 A,
  which includes all dissociation candidates at <= 3.5 A).
- Reference conditions: static_only, base_hist (real hist6), and gen6
  (all 6 statistics from the generated series).
- Asymmetry to note (same as K33): the real statistics are computed
  over the BACKWARD 11-frame history (0..1 ns), while the generated
  statistics are computed over the FORWARD 5-point series (1..2 ns).
  The swap therefore also changes backward to forward, not only real
  to generated.
- Pipeline check: static_only/base_hist must reproduce K46's recorded
  values (static 0.4160 -> full 0.5945 at 1 ns).
- Retention = (condition - static) / (base_hist - static) in AUPRC.
- Threading: run with OMP_NUM_THREADS=1 (K41's lesson: default
  threading on this VM is slowed by contention with parallel jobs).
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
GEN_DIR = Path("/home/hatch/workspace/kinescope/work/k54_gen22")
HERE = Path(__file__).resolve().parent
PARTIAL = HERE / "results_k54_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 3.5


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def load_gen(prot, rep, donor, acceptor, origin):
    """Generated 6 statistics for the given candidates, aligned by
    (donor_atom, acceptor_atom). Returns (N, 6) float32."""
    z = np.load(GEN_DIR / f"atlas-{prot}-analysis-R{rep}.npz")
    pair = z["pair"]  # (M, 3): donor atom, acceptor atom, origin distance
    gen = z["gen_dist"]  # (M, 4): forward distances at 0.1/0.3/0.5/1.0 ns
    key = {(int(p[0]), int(p[1])): k for k, p in enumerate(pair)}
    idx = np.array([key[(int(u), int(v))] for u, v in zip(donor, acceptor)])
    assert len(idx) == len(donor), (prot, rep, len(idx), len(donor))
    gen5 = np.column_stack([origin, gen[idx]]).T  # (5, N)
    return hist6(gen5.T)


def load(prots):
    xs, hs, ys, gs = [], [], [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            xs.append(z["X"].astype(np.float64))
            hs.append(z["hist"].astype(np.float64))
            ys.append(z["y"].astype(np.float64))
            gs.append(load_gen(p, r, z["donor_atom"], z["acceptor_atom"], z["hist"][-1]))
    return (np.concatenate(xs), np.concatenate(hs), np.concatenate(ys),
            np.concatenate(gs))


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
        if tag in results:
            print(tag, "already done, skipped", flush=True)
            return results[tag]
        out = {}
        for hi, h in enumerate(HORIZONS):
            p = predict(np.concatenate([Xtr, Ftr], axis=1), Ytr[:, hi],
                        np.concatenate([Xte, fte_block], axis=1))
            out[str(h)] = metrics(Yte[:, hi], p)
        print(tag, "AUPRC", {h: round(out[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
        results[tag] = out
        PARTIAL.write_text(json.dumps(results, indent=2))
        return out

    def eval_static(tag="static_only"):
        if tag in results:
            print(tag, "already done, skipped", flush=True)
            return results[tag]
        out = {}
        for hi, h in enumerate(HORIZONS):
            p = predict(Xtr, Ytr[:, hi], Xte)
            out[str(h)] = metrics(Yte[:, hi], p)
        print(tag, "AUPRC", {h: round(out[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
        results[tag] = out
        PARTIAL.write_text(json.dumps(results, indent=2))
        return out

    eval_static()
    eval_on(Fte, "base_hist")
    for name, repl in (("std_only", [(2, 2)]), ("min_only", [(3, 3)]),
                       ("frac35_only", [(5, 5)]),
                       ("combined3", [(2, 2), (3, 3), (5, 5)])):
        Fswap = Fte.copy()
        for real_col, gen_col in repl:
            Fswap[:, real_col] = Gte[:, gen_col]
        eval_on(Fswap, name)
    eval_on(Gte, "gen6")

    retention = {}
    for name in list(results):
        if name in ("static_only", "base_hist"):
            continue
        retention[name] = {}
        for h in HORIZONS:
            s = results["static_only"][str(h)]["auprc"]
            b = results["base_hist"][str(h)]["auprc"]
            c = results[name][str(h)]["auprc"]
            retention[name][str(h)] = float((c - s) / (b - s))
    results["_retention"] = retention
    k46 = json.loads((HERE.parent / "K46_dissociation_carrier" / "results.json").read_text())
    results["_checks_vs_K46"] = {
        "static_max_abs_diff": max(
            abs(results["static_only"][str(h)]["auprc"] - k46["dissociation"]["static_only"][str(h)]["auprc"])
            for h in HORIZONS),
        "base_max_abs_diff": max(
            abs(results["base_hist"][str(h)]["auprc"] - k46["dissociation"]["full_hist6"][str(h)]["auprc"])
            for h in HORIZONS),
    }
    print("checks vs K46:", results["_checks_vs_K46"], flush=True)
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
