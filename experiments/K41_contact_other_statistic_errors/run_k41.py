"""K41: contact version of K33 -- how much guidance do the errors of the
non-mean statistics cost?

K33 (hydrogen bonds, strong base) found the shape depends on the horizon:
at 1 ns the three statistics together still leave 0.65 of the gain, at
5/30 ns they drop it to the generated level, and at 90 ns std alone
erases it. K41 measures the same split for contact formation under Doc D.

Method (K33's, adapted to Doc D):
- Model: HistGradientBoosting (120 iters, lr 0.075, 31 leaves,
  min_samples_leaf 15, L2 0.02, seed 42), trained per horizon on the
  REAL history (static + hist6) of the 4 train proteins, never retrained
  for the swap conditions. Only the test features change.
- Real history features are the 6 statistics of the 11-frame minimum
  distance series (0..1 ns): [first, mean, std, min, last-first,
  frac <= 4.5 A]. The mean is kept real; std / min / frac<=4.5 are
  replaced one at a time, and all three together, with the values
  computed from the GENERATED series.
- Generated series: K14's contact series (proposal 22, 5 points:
  origin + 0.1/0.3/0.5/1.0 ns). Generated statistics are computed over
  those 5 points the same way (std, min, frac <= 4.5).
- Reference conditions: static_only, base_hist (real hist6), and gen6
  (all 6 statistics from the generated series).
- Subset restriction: K14's generated series exist only for the 4,968
  feedable candidates (21% of K13's 23,310; representative pair is a
  donor-acceptor pair). The swap conditions can only be evaluated on
  the feedable TEST candidates, so the references are recomputed on
  that same subset. As a pipeline check, static_only/base_hist are
  also evaluated on the FULL test set and compared with K16's recorded
  values (static 0.3398 -> full 0.5669 at 1 ns).
- Alignment: K14's `min` (5-point minimum series, feedable subset) is a
  subsequence of `min_all` (all candidates), whose origin column matches
  the K13 cache's hist[-1] exactly (max diff 0.0) in the same
  protein/replica order. Each generated row is matched to its cache row
  by the full 5-value minimum series (all 4,968 rows matched uniquely;
  nonfinite = 0). The alignment is recomputed here, not read from /tmp.
- Retention = (condition - static) / (base_hist - static) in AUPRC,
  computed on the feedable test subset.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
K14 = Path("/home/hatch/workspace/kinescope/repo/experiments/K14_other_phenomena_generated/k14_contact.npz")
K14_META = Path("/home/hatch/workspace/k14/out/k14_meta.json")
HERE = Path(__file__).resolve().parent
PARTIAL = HERE / "results_k41_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
ALL_P = TRAIN_P + TEST_P
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


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def gen6(series, bound=BOUND):
    return hist6(series, bound)


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


def align_gen_to_cache():
    """Match each K14 generated row to its K13 cache row (index into the
    concatenated 6-protein cache in ALL_P order)."""
    with np.load(K14) as z:
        gen = z["gen"].astype(np.float64)
        mn = z["min"].astype(np.float64)
        min_all = z["min_all"].astype(np.float64)
    meta = json.loads(K14_META.read_text())
    feed = [meta["contact"]["per_traj"][f"atlas-{p}-analysis-R{r}"][1]
            for p in ALL_P for r in (1, 2, 3)]
    blocks, off = [], 0
    for p in ALL_P:
        for r in (1, 2, 3):
            n = len(np.load(CACHE / f"{p}_R{r}_contact.npz")["hist"])
            blocks.append((off, off + n))
            off += n
    idx = np.full(len(mn), -1, dtype=int)
    g = 0
    for (b0, b1), f in zip(blocks, feed):
        sub = min_all[b0:b1]
        keys = {tuple(np.round(row, 5)): i for i, row in enumerate(sub)}
        assert len(keys) == len(sub), "duplicate min rows in block"
        for j in range(f):
            k = tuple(np.round(mn[g + j], 5))
            assert k in keys, "unmatched min row"
            idx[g + j] = b0 + keys[k]
        g += f
    assert (idx >= 0).all() and len(set(idx.tolist())) == len(idx)
    assert np.abs(np.load(K14)["rep"][:, 0].astype(np.float64) - gen[:, 0]).max() < 1e-4
    return gen, idx


def main():
    train, test = load(TRAIN_P), load(TEST_P)
    Xtr, Xte = cat(train, "X"), cat(test, "X")
    Htr, Hte = cat(train, "hist"), cat(test, "hist")
    Ytr, Yte = cat(train, "y"), cat(test, "y")
    Ftr, Fte = hist6(Htr), hist6(Hte)
    print("train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)

    gen, idx = align_gen_to_cache()
    # cache rows in ALL_P order; test rows start after the 4 train proteins
    n_train_rows = sum(len(np.load(CACHE / f"{p}_R{r}_contact.npz")["hist"])
                       for p in TRAIN_P for r in (1, 2, 3))
    test_feed_local = idx[idx >= n_train_rows] - n_train_rows
    assert len(test_feed_local) == len(set(test_feed_local.tolist()))
    print("feedable test candidates:", len(test_feed_local), "of", len(Xte), flush=True)
    Gte = gen6(gen[idx >= n_train_rows])  # generated statistics for feedable test rows

    results = {}
    if PARTIAL.exists():
        results = json.loads(PARTIAL.read_text())
        print("resumed:", sorted(results), flush=True)

    def eval_on(mask_rows, fte_block, tag):
        """Train per horizon on real history; evaluate on test rows
        `mask_rows` (indices into the test set) with history block
        `fte_block` (same rows). Returns {horizon: metrics}."""
        out = {}
        for hi, h in enumerate(HORIZONS):
            p = predict(np.concatenate([Xtr, Ftr], axis=1), Ytr[:, hi],
                        np.concatenate([Xte[mask_rows], fte_block], axis=1))
            out[str(h)] = metrics(Yte[mask_rows, hi], p)
        print(tag, "AUPRC", {h: round(out[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
        return out

    def eval_static(mask_rows, tag):
        out = {}
        for hi, h in enumerate(HORIZONS):
            p = predict(Xtr, Ytr[:, hi], Xte[mask_rows])
            out[str(h)] = metrics(Yte[mask_rows, hi], p)
        print(tag, "AUPRC", {h: round(out[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
        return out

    all_rows = np.arange(len(Xte))
    # --- pipeline check on the FULL test set (must reproduce K16) ---
    if "static_only_fulltest" not in results:
        results["static_only_fulltest"] = eval_static(all_rows, "static_only_fulltest")
        PARTIAL.write_text(json.dumps(results, indent=2))
    if "base_hist_fulltest" not in results:
        results["base_hist_fulltest"] = eval_on(all_rows, Fte, "base_hist_fulltest")
        PARTIAL.write_text(json.dumps(results, indent=2))

    # --- measurement on the feedable test subset ---
    Fsub = Fte[test_feed_local]
    if "static_only" not in results:
        results["static_only"] = eval_static(test_feed_local, "static_only (feedable subset)")
        PARTIAL.write_text(json.dumps(results, indent=2))
    if "base_hist" not in results:
        results["base_hist"] = eval_on(test_feed_local, Fsub, "base_hist (feedable subset)")
        PARTIAL.write_text(json.dumps(results, indent=2))

    swaps = {
        "std_only": [(2, 2)],
        "min_only": [(3, 3)],
        "frac45_only": [(5, 5)],
        "combined3": [(2, 2), (3, 3), (5, 5)],
    }
    for name, repl in swaps.items():
        if name in results:
            print(name, "already done, skipped", flush=True)
            continue
        Fswap = Fsub.copy()
        for real_col, gen_col in repl:
            Fswap[:, real_col] = Gte[:, gen_col]
        results[name] = eval_on(test_feed_local, Fswap, name)
        PARTIAL.write_text(json.dumps(results, indent=2))
    if "gen6" not in results:
        results["gen6"] = eval_on(test_feed_local, Gte, "gen6")
        PARTIAL.write_text(json.dumps(results, indent=2))

    # --- derived: retention of the real-history gain (feedable subset) ---
    retention = {}
    for name in list(swaps) + ["gen6"]:
        retention[name] = {}
        for h in HORIZONS:
            s = results["static_only"][str(h)]["auprc"]
            b = results["base_hist"][str(h)]["auprc"]
            c = results[name][str(h)]["auprc"]
            retention[name][str(h)] = float((c - s) / (b - s))
    results["_retention"] = retention
    k16 = json.loads((HERE.parent / "K16_contact_channel" / "results.json").read_text())
    results["_checks_vs_K16"] = {
        "static_fulltest_max_abs_diff": max(
            abs(results["static_only_fulltest"][str(h)]["auprc"] - k16["static_only"][str(h)]["auprc"])
            for h in HORIZONS),
        "base_fulltest_max_abs_diff": max(
            abs(results["base_hist_fulltest"][str(h)]["auprc"] - k16["base_hist"][str(h)]["auprc"])
            for h in HORIZONS),
    }
    results["_subset"] = dict(n_feedable_test=int(len(test_feed_local)), n_test=int(len(Xte)),
                              test_positives_subset=[int(v) for v in Yte[test_feed_local].sum(0)])
    print("checks vs K16:", results["_checks_vs_K16"], flush=True)
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
