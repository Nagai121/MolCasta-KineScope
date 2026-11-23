"""K48: salt-bridge version of K33/K41 -- how much guidance do the errors
of the non-mean statistics cost?

K33 (hydrogen bonds, strong base) and K41 (contact, Doc D) found
different shapes: at 1 ns the three statistics together leave 0.65 of
the gain for hydrogen bonds but erase it for contact (where the
below-threshold fraction alone is the carrier). K48 measures the same
split for salt-bridge formation on K15's expanded set, to align the
three phenomena.

Method (K33's, adapted to the K15 expanded set):
- Model: HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
  min_samples_leaf 15, L2 0.02, seed 42), trained per horizon on the
  REAL history (static + hist6) of the 58 train proteins, never
  retrained for the swap conditions. Only the test features change.
- Real history features are the 6 statistics of the 11-frame minimum
  distance series (0..1 ns): [first, mean, std, min, last-first,
  frac <= 4.0 A]. The mean is kept real; std / min / frac<=4.0 are
  replaced one at a time, and all three together, with the values
  computed from the GENERATED series.
- Generated series: K20's residue-level generation (ideas 11-14),
  branch-mean 9-point series (origin + 8 steps) per candidate,
  ~/workspace/k20/out/. Generated statistics are computed over those
  9 points the same way. All four ideas are measured (K20 found them
  nearly identical in mean error, K38).
- Reference conditions: static_only, base_hist (real hist6), and gen6
  (all 6 statistics from the generated series, per idea).
- Unlike K41, there is no subset restriction: K20 generated series
  exist for all 3,727 candidates (train 2,557 + test 1,170).
- Train/test split: K15's frozen split.json (train = original 4 +
  54 additions, test = original 2 + 28 additions).
- Pipeline check: static_only/base_hist must reproduce K15's recorded
  values (static 0.2402 -> full 0.3842 at 1 ns).
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

K20 = Path("/home/hatch/workspace/k20/out")
K15 = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
HERE = Path(__file__).resolve().parent
PARTIAL = HERE / "results_k48_partial.json"
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
_SPLIT = json.loads((K15 / "split.json").read_text())
TRAIN_P = ORIG_TRAIN + tuple(_SPLIT["train_additions"])
TEST_P = ORIG_TEST + tuple(_SPLIT["test_additions"])
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.0


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def load_k20(prots):
    """Load K20 files for the given proteins; returns X, hist, y in
    K20's file order (protein order, then replica order)."""
    xs, hs, ys = [], [], []
    gen = {n: [] for n in (11, 12, 13, 14)}
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K20 / f"{p}_R{r}.npz")
            if len(z["hist"]) == 0:
                continue
            xs.append(z["X"].astype(np.float64))
            hs.append(z["hist"].astype(np.float64))
            ys.append(z["y"].astype(np.float64))
            for n in gen:
                gen[n].append(z[f"gen{n}"].astype(np.float64).mean(axis=0))  # (9, N)
    return (np.concatenate(xs), np.concatenate(hs), np.concatenate(ys),
            {n: np.concatenate(v, axis=1) for n, v in gen.items()})  # (9, N)


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
    Xtr, Htr, Ytr, Gtr = load_k20(TRAIN_P)
    Xte, Hte, Yte, Gte = load_k20(TEST_P)
    Ftr, Fte = hist6(Htr), hist6(Hte)
    Gte6 = {n: hist6(Gte[n].T) for n in Gte}  # (N, 6) per idea; Gte[n] is (9, N)
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

    # swap conditions, measured for idea 14 (closest to real, K20/K38)
    # and idea 11 (quietest) as a cross-check on the combined condition.
    for idea in (14, 11):
        G = Gte6[idea]
        swaps = {
            f"std_only_i{idea}": [(2, 2)],
            f"min_only_i{idea}": [(3, 3)],
            f"frac40_only_i{idea}": [(5, 5)],
            f"combined3_i{idea}": [(2, 2), (3, 3), (5, 5)],
        }
        for name, repl in swaps.items():
            if name in results:
                print(name, "already done, skipped", flush=True)
                continue
            Fswap = Fte.copy()
            for real_col, gen_col in repl:
                Fswap[:, real_col] = G[:, gen_col]
            eval_on(Fswap, name)
        eval_on(G, f"gen6_i{idea}")

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
    k15 = json.loads((K15 / "results.json").read_text())["full"]
    results["_checks_vs_K15"] = {
        "static_max_abs_diff": max(
            abs(results["static_only"][str(h)]["auprc"] - k15["static_only"][str(h)]["auprc"])
            for h in HORIZONS),
        "base_max_abs_diff": max(
            abs(results["base_hist"][str(h)]["auprc"] - k15["base_hist_D6"][str(h)]["auprc"])
            for h in HORIZONS),
    }
    print("checks vs K15:", results["_checks_vs_K15"], flush=True)
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
