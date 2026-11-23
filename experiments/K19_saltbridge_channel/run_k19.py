"""K19: salt-bridge formation channel split on the K15 expanded set.
Part 1: ablate one of the 6 real-history statistics at a time, two variants
  (a) drop the column (K18 style: necessity within the 6-stat block),
  (b) replace it by the train-set mean of that statistic (K16 style:
      constantize; the statistic carries no per-candidate information).
  Train and test both get the same treatment; retrain per condition.
Part 2: history swaps, input-only (K03/K16 style): train on full real
  history, test history only is swapped.
  (a) other-pair: within same protein+replica, random permutation (seed 123).
  (b) other-replica: same (res_i, res_j) pair from another replica
      (R1<-R2, R2<-R3, R3<-R1); only where that pair is a candidate there.
K09-style swap (generated -> real) does not apply: salt bridge has no
generated series. Partial results are checkpointed to
results_k19_partial.json after each condition.
OMP_NUM_THREADS should be set to 2 by the caller.
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).parent
CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15 = Path(__file__).parent.parent / "K15_saltbridge_expansion"
_SPLIT = json.loads((K15 / "split.json").read_text())
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
TRAIN_P = ORIG_TRAIN + tuple(_SPLIT["train_additions"])
TEST_P = ORIG_TEST + tuple(_SPLIT["test_additions"])
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.0
STAT_NAMES = ["first", "mean", "std", "min", "last_minus_first", "frac_le4"]


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)


def load(prots):
    X, H, Y, meta = [], [], [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
            X.append(z["X"]); H.append(z["hist"]); Y.append(z["y"])
            meta.extend([(p, r, int(a), int(b)) for a, b in zip(z["res_i"], z["res_j"])])
    return (np.concatenate(X), np.concatenate(H), np.concatenate(Y), meta)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    out = dict(n=int(len(y)), positives=int(y.sum()),
               prevalence=float(y.mean()) if len(y) else None,
               brier=float(brier_score_loss(y, p)),
               logloss=float(log_loss(y, p, labels=[0, 1])))
    if 0 < y.sum() < len(y):
        out["auprc"] = float(average_precision_score(y, p)); out["auc"] = float(roc_auc_score(y, p))
    else:
        out["auprc"] = None; out["auc"] = None
    return out


def eval_cond(name, ftr, fte, Xtr, Xte, Ytr, Yte, results):
    if name in results:
        return results[name]
    res = {}
    for hi, h in enumerate(HORIZONS):
        ytr, yte = Ytr[:, hi], Yte[:, hi]
        if ftr is None:
            p = predict(Xtr, ytr, Xte)
        else:
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr, np.concatenate([Xte, fte], axis=1))
        res[str(h)] = metrics(yte, p)
    results[name] = res
    (HERE / "results_k19_partial.json").write_text(json.dumps(results, indent=2))
    print(name, "AUPRC", {h: round(res[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
    return res


def main():
    Xtr, Htr, Ytr, _ = load(TRAIN_P)
    Xte, Hte, Yte, meta_te = load(TEST_P)
    print("train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)
    Ftr, Fte = hist6(Htr), hist6(Hte)
    partial = HERE / "results_k19_partial.json"
    results = json.loads(partial.read_text()) if partial.exists() else {}

    eval_cond("static_only", None, None, Xtr, Xte, Ytr, Yte, results)
    eval_cond("base_hist_D6", Ftr, Fte, Xtr, Xte, Ytr, Yte, results)

    # Part 1: drop-column (K18) and constantize (K16) per statistic.
    const = Ftr.mean(axis=0)  # train-set means only; test info not used
    print("train-mean constants:", {s: round(float(v), 4) for s, v in zip(STAT_NAMES, const)}, flush=True)
    for i, s in enumerate(STAT_NAMES):
        keep = [j for j in range(6) if j != i]
        eval_cond(f"drop_{s}", Ftr[:, keep], Fte[:, keep], Xtr, Xte, Ytr, Yte, results)
        Ctr, Cte = Ftr.copy(), Fte.copy()
        Ctr[:, i] = const[i]; Cte[:, i] = const[i]
        eval_cond(f"const_{s}", Ctr, Cte, Xtr, Xte, Ytr, Yte, results)

    # Part 2: input-only swaps (train on full real history).
    # (a) other-pair within same protein+replica, seed 123 (K03).
    rng = np.random.default_rng(123)
    Hswap = Hte.copy()
    groups = {}
    for i, (p, r, _, _) in enumerate(meta_te):
        groups.setdefault((p, r), []).append(i)
    for idx in groups.values():
        idx = np.array(idx)
        Hswap[idx] = Hswap[idx][rng.permutation(len(idx))]
    eval_cond("swap_other_pair", hist6(Htr), hist6(Hswap), Xtr, Xte, Ytr, Yte, results)

    # (b) other-replica, same (res_i, res_j), R1<-R2, R2<-R3, R3<-R1.
    by_pr = {}
    for i, (p, r, a, b) in enumerate(meta_te):
        by_pr[(p, r, a, b)] = i
    Hrep = Hte.copy()
    swapped = np.zeros(len(Hte), dtype=bool)
    nxt = {1: 2, 2: 3, 3: 1}
    for i, (p, r, a, b) in enumerate(meta_te):
        j = by_pr.get((p, nxt[r], a, b))
        if j is not None:
            Hrep[i] = Hte[j]; swapped[i] = True
    print("swap_replica coverage:", int(swapped.sum()), "/", len(Hte), flush=True)
    eval_cond("swap_other_replica", hist6(Htr), hist6(Hrep), Xtr, Xte, Ytr, Yte, results)
    # subset: only rows whose history was actually swapped.
    sub_res = {}
    for hi, h in enumerate(HORIZONS):
        ytr, yte = Ytr[:, hi], Yte[swapped, hi]
        p = predict(np.concatenate([Xtr, Ftr], axis=1), ytr,
                    np.concatenate([Xte[swapped], hist6(Hrep[swapped])], axis=1))
        sub_res[str(h)] = metrics(yte, p)
    results["swap_other_replica_subset"] = sub_res
    # base on the same subset for comparison
    sub_base = {}
    for hi, h in enumerate(HORIZONS):
        ytr, yte = Ytr[:, hi], Yte[swapped, hi]
        p = predict(np.concatenate([Xtr, Ftr], axis=1), ytr,
                    np.concatenate([Xte[swapped], Fte[swapped]], axis=1))
        sub_base[str(h)] = metrics(yte, p)
    results["base_hist_D6_subset"] = sub_base
    (HERE / "results_k19_partial.json").write_text(json.dumps(results, indent=2))
    print("swap_replica_subset AUPRC",
          {h: round(sub_res[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
    print("base_subset AUPRC",
          {h: round(sub_base[str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)

    # retention vs base (AUPRC)
    ret = {}
    for name in results:
        if name in ("retention_vs_base", "swap_other_replica_subset", "base_hist_D6_subset"):
            continue
        ret[name] = {}
        for h in HORIZONS:
            s = results["static_only"][str(h)]["auprc"]; b = results["base_hist_D6"][str(h)]["auprc"]
            c = results[name][str(h)]["auprc"]
            ret[name][str(h)] = None if (s is None or b is None or c is None or abs(b - s) < 1e-9) else float((c - s) / (b - s))
    results["retention_vs_base"] = ret
    results["_info"] = dict(train_proteins=list(TRAIN_P), test_proteins=list(TEST_P),
                            n_train=int(len(Xtr)), n_test=int(len(Xte)),
                            test_positives=[int(v) for v in Yte.sum(0)],
                            train_mean_constants={s: float(v) for s, v in zip(STAT_NAMES, const)},
                            swap_replica_coverage=int(swapped.sum()))
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    rows = []
    for name, by_h in results.items():
        if name in ("retention_vs_base", "_info"):
            continue
        for h, m in by_h.items():
            rows.append(dict(condition=name, horizon_ns=h, **m))
    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("saved")


if __name__ == "__main__":
    main()
