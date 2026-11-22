"""K09b: 複数統計量の同時差し替え

K09 の追実験。基準の実5点を K06(a)/K10 と同じフレーム [0, 3, 5, 7, 10]
（時刻 0, 0.3, 0.5, 0.7, 1.0 ns）にそろえる。K09 の実5点はフレーム
[0, 1, 3, 5, 10] だったため、K09 の単独差し替えの値とは基準が違う。
生成系列は K05 の第22案の5点のまま。

条件（学習・テスト両方に同じ処理、条件ごとに学習し直す。土台と同一）:
  static_only, real_history_11pt（参照）, real5b（S+H5、フレーム[0,3,5,7,10]、対照）,
  gen5（S+G5、基準）,
  swap_mean_b（平均だけを実5bの値に）、swap_frac_b（3.5 A以下割合だけを実5bの値に）,
  swap_mean_frac（平均＋3.5 A以下割合を同時に実5bの値に）,
  swap_all（6統計量すべてを実5bの値に。実5bと同一になるはずで、一致の確認にもなる）.
回復率 = (条件の AUPRC - 生成5点の AUPRC) / (実5bの AUPRC - 生成5点の AUPRC)
"""
import os
# Thread thrash guard: the cell has ~1.5 shared CPUs and other jobs run HistGBM
# concurrently; the default 11 OpenMP threads spin and make each fit ~60x slower
# in CPU time. Limit threads before numpy/sklearn are imported.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
REAL5B_FRAMES = [0, 3, 5, 7, 10]  # 時刻 0, 0.3, 0.5, 0.7, 1.0 ns（K06(a)/K10 と同じ）
STAT_NAMES = ["first", "mean", "std", "min", "last_minus_first", "frac_le_3.5"]
IDX = {n: j for j, n in enumerate(STAT_NAMES)}

def load_gen5():
    z = np.load(GEN)
    tids, pair, gen = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
    assert gen5.shape == (14582, 5), gen5.shape
    return {(str(t), int(pair[i, 0]), int(pair[i, 1])): gen5[i]
            for i, t in enumerate(tids)}

GEN5 = load_gen5()

def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            g = np.stack([GEN5[(tid, int(da), int(aa))]
                          for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["gen5"] = g
            parts.append(d)
    return parts

def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)

def hist6(series, bound=3.5):
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
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

train = load(TRAIN_P); test = load(TEST_P)
Xtr = cat(train, "X"); Xte = cat(test, "X")
Htr = cat(train, "hist"); Hte = cat(test, "hist")
Gtr = cat(train, "gen5"); Gte = cat(test, "gen5")
R5b_tr = Htr[:, REAL5B_FRAMES]; R5b_te = Hte[:, REAL5B_FRAMES]
print("train", Xtr.shape, Htr.shape, Gtr.shape, "test", Xte.shape, Hte.shape, Gte.shape, flush=True)
print("gen5 first vs hist last: max abs diff train %.2e test %.2e" % (
    np.abs(Gtr[:, 0] - Htr[:, -1]).max(), np.abs(Gte[:, 0] - Hte[:, -1]).max()), flush=True)
print("real5b last vs hist last: max abs diff %.2e" % np.abs(R5b_te[:, -1] - Hte[:, -1]).max(), flush=True)

gen_tr, gen_te = hist6(Gtr), hist6(Gte)
real5b_tr, real5b_te = hist6(R5b_tr), hist6(R5b_te)
real11_tr, real11_te = hist6(Htr), hist6(Hte)

def swapped(idxs):
    ftr = gen_tr.copy(); fte = gen_te.copy()
    for j in idxs:
        ftr[:, j] = real5b_tr[:, j]; fte[:, j] = real5b_te[:, j]
    return (ftr, fte)

CONDITIONS = {
    "static_only": None,
    "real_history_11pt": (real11_tr, real11_te),
    "real5b": (real5b_tr, real5b_te),
    "gen5": (gen_tr, gen_te),
    "swap_mean_b": swapped([IDX["mean"]]),
    "swap_frac_b": swapped([IDX["frac_le_3.5"]]),
    "swap_mean_frac": swapped([IDX["mean"], IDX["frac_le_3.5"]]),
    "swap_all": swapped(list(range(6))),
}

PARTIAL = OUT / "results_k09b_partial.json"
results = json.loads(PARTIAL.read_text())["conditions"] if PARTIAL.exists() else {}
done = {n for n, v in results.items() if len(v) == len(HORIZONS)}
print("resuming: already complete:", sorted(done), flush=True)

def save_incremental(recovery_so_far):
    payload = {"conditions": results, "auprc_recovery_vs_gen5_to_real5b": recovery_so_far,
               "horizons": list(HORIZONS), "real5b_frames": REAL5B_FRAMES,
               "stat_names": STAT_NAMES,
               "notes": "gen5 is forward (origin +0.1/0.3/0.5/1.0 ns); real5b is history frames 0,3,5,7,10 (0/0.3/0.5/0.7/1.0 ns, ending at origin), same as K06(a)/K10. static_only/real_history_11pt/real5b/gen5 restored from the K06(a)/K09 full-precision results (cross-checked against the interrupted K09b run log to 4dp); swap conditions computed in this run."}
    (OUT / "results_k09b.json").write_text(json.dumps(payload, indent=2))
    with open(OUT / "results_k09b.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss", "auprc_recovery_vs_gen5_to_real5b"])
        for cname in CONDITIONS:
            if cname not in results:
                continue
            for h in HORIZONS:
                r = results[cname][str(h)]
                rec = recovery_so_far.get(cname, {}).get(str(h), "") if cname.startswith("swap_") else ""
                w.writerow([cname, h, r["n"], r["positives"], r["auc"], r["auprc"], r["brier"], r["logloss"], rec])

recovery = {}
for name, feats in CONDITIONS.items():
    if name in done:
        print("skip (already complete):", name, flush=True)
        continue
    results[name] = {}
    for hi, h in enumerate(HORIZONS):
        ytr = np.concatenate([p["y"] for p in train])[:, hi]
        yte = np.concatenate([p["y"] for p in test])[:, hi]
        if feats is None:
            p = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = feats
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr, np.concatenate([Xte, fte], axis=1))
        results[name][str(h)] = metrics(yte, p)
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[name][str(h)].items()}, flush=True)
    if name.startswith("swap_"):
        recovery[name] = {}
        for h in HORIZONS:
            denom = results["real5b"][str(h)]["auprc"] - results["gen5"][str(h)]["auprc"]
            recovery[name][str(h)] = float((results[name][str(h)]["auprc"] - results["gen5"][str(h)]["auprc"]) / denom) if denom != 0 else float("nan")
    save_incremental(recovery)

recovery = {}
for name in CONDITIONS:
    if name.startswith("swap_"):
        recovery[name] = {}
        for h in HORIZONS:
            denom = results["real5b"][str(h)]["auprc"] - results["gen5"][str(h)]["auprc"]
            recovery[name][str(h)] = float((results[name][str(h)]["auprc"] - results["gen5"][str(h)]["auprc"]) / denom) if denom != 0 else float("nan")
        print("recovery", name, {h: round(recovery[name][str(h)], 4) for h in HORIZONS}, flush=True)

payload = {"conditions": results, "auprc_recovery_vs_gen5_to_real5b": recovery,
           "horizons": list(HORIZONS), "real5b_frames": REAL5B_FRAMES,
           "stat_names": STAT_NAMES,
           "notes": "gen5 is forward (origin +0.1/0.3/0.5/1.0 ns); real5b is history frames 0,3,5,7,10 (0/0.3/0.5/0.7/1.0 ns, ending at origin), same as K06(a)/K10."}
(OUT / "results_k09b.json").write_text(json.dumps(payload, indent=2))
with open(OUT / "results_k09b.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss", "auprc_recovery_vs_gen5_to_real5b"])
    for name in CONDITIONS:
        for h in HORIZONS:
            r = results[name][str(h)]
            rec = recovery.get(name, {}).get(str(h), "") if name.startswith("swap_") else ""
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["brier"], r["logloss"], rec])
print("saved to", OUT)
