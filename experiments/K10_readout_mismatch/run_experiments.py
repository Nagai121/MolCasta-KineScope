"""K10: 読み出しのずれの切り分け (train/eval history source cross)

土台は資料D条件 (base_reproduction / K07 / K08 と同一):
 ATLAS 6 proteins x3 replicas, train 1bgf_A,1ifg_A,1j8r_A,2b0a_A /
 test 1mix_A+2qt4_A pooled, origin 1 ns, HB formation,
 static features 7, HistGradientBoosting (120, lr .075, leaf31,
 min_samples_leaf15, L2 .02, seed42), horizons 1/5/10/30/50/90 ns.

History features: SAME 6 statistics (hist6) from 5-point series for both sources.
 - real5: backward real history frames [0,3,5,7,10] of 0-1 ns 11 frames
          (offsets -1.0,-0.7,-0.5,-0.3,0 ns)  == K06(a) S+H5
 - gen5:  origin distance + proposal-22 generated distances at
          +0.1/+0.3/+0.5/+1.0 ns (K05 generated_series.npz) == K06(a) S+G6

Conditions (static block unchanged; only history source at train/eval differs):
 (a) train real5 -> eval real5   (control, should equal K06(a) S+H5)
 (b) train real5 -> eval gen5
 (c) train gen5  -> eval gen5   (should equal K06(a) S+G6)
 (d) train gen5  -> eval real5
 plus static_only reference.
No retraining/recalibration in cross conditions: the model fitted on the
train source is applied as-is to the eval source.
"""
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
REAL5_IDX = [0, 3, 5, 7, 10]

def load_gen_series():
    z = np.load(GEN)
    tids, pair, gen = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])  # (N,5)
    m = {}
    for i, t in enumerate(tids):
        m[(str(t), int(pair[i, 0]), int(pair[i, 1]))] = gen5[i]
    assert len(m) == len(tids) == 14582, (len(m), len(tids))
    return m

GEN_MAP = load_gen_series()

def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            gen5 = np.stack([GEN_MAP[(tid, int(da), int(aa))]
                             for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["gen5"] = gen5
            # origin check: gen5[:,0] vs real history last frame (frame10 = origin)
            err = np.abs(gen5[:, 0] - d["hist"].T[:, -1]).max()
            assert err < 1e-4, (tid, err)
            parts.append(d)
    return parts

def cat(parts, key):
    arrs = [p[key].T if key == "hist" else p[key] for p in parts]
    return np.concatenate(arrs)

def hist6(series, bound=3.5):
    # series: (n, T)
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)

def metrics(y, p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

train = load(TRAIN_P); test = load(TEST_P)
Xtr = cat(train, "X"); Xte = cat(test, "X")
Htr = cat(train, "hist"); Hte = cat(test, "hist")          # (n,11)
Gtr = cat(train, "gen5"); Gte = cat(test, "gen5")          # (n,5)
Ytr = cat(train, "y"); Yte = cat(test, "y")
print("train", Xtr.shape, Htr.shape, Gtr.shape, "test", Xte.shape, Hte.shape, Gte.shape,
      "test pos", Yte.sum(0), flush=True)

FEATS = {
    "real5": (hist6(Htr[:, REAL5_IDX]), hist6(Hte[:, REAL5_IDX])),
    "gen5": (hist6(Gtr), hist6(Gte)),
}
# diagnostics: correlation / mean diff between real5 and gen5 per candidate
diag = {}
for nm, (a, b) in (("train", (Htr[:, REAL5_IDX], Gtr)), ("test", (Hte[:, REAL5_IDX], Gte))):
    diag[nm] = {"mean_real5": float(a.mean()), "mean_gen5": float(b.mean()),
                "mean_diff_gen_minus_real": float((b - a).mean()),
                "corr_series_values": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                "corr_candidate_means": float(np.corrcoef(a.mean(1), b.mean(1))[0, 1])}
print("diag", diag, flush=True)

CONDITIONS = {
    "static_only": None,
    "a_train_real_eval_real": ("real5", "real5"),
    "b_train_real_eval_gen": ("real5", "gen5"),
    "c_train_gen_eval_gen": ("gen5", "gen5"),
    "d_train_gen_eval_real": ("gen5", "real5"),
}

results = {}
for name, src in CONDITIONS.items():
    results[name] = {}
    for hi, h in enumerate(HORIZONS):
        ytr, yte = Ytr[:, hi], Yte[:, hi]
        if src is None:
            xtr_, xte_ = Xtr, Xte
        else:
            tr_src, ev_src = src
            xtr_ = np.concatenate([Xtr, FEATS[tr_src][0]], axis=1)
            xte_ = np.concatenate([Xte, FEATS[ev_src][1]], axis=1)
        m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                           min_samples_leaf=15, l2_regularization=.02, random_state=42)
        m.fit(xtr_, ytr)
        p = m.predict_proba(xte_)[:, 1]
        results[name][str(h)] = metrics(yte, p)
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[name][str(h)].items()}, flush=True)

# K06(a) reference check
k06a = json.loads((OUT.parent / "K06_aligned_recomparison" / "k06a_results.json").read_text())
checks = {
    "a_vs_K06a_S+H5_auprc_max_abs_diff": max(abs(results["a_train_real_eval_real"][str(h)]["auprc"] - k06a["S+H5"][f"{h}ns"]["auprc"]) for h in HORIZONS),
    "c_vs_K06a_S+G6_auprc_max_abs_diff": max(abs(results["c_train_gen_eval_gen"][str(h)]["auprc"] - k06a["S+G6"][f"{h}ns"]["auprc"]) for h in HORIZONS),
    "a_vs_K06a_S+H5_auc_max_abs_diff": max(abs(results["a_train_real_eval_real"][str(h)]["auc"] - k06a["S+H5"][f"{h}ns"]["auc"]) for h in HORIZONS),
    "c_vs_K06a_S+G6_auc_max_abs_diff": max(abs(results["c_train_gen_eval_gen"][str(h)]["auc"] - k06a["S+G6"][f"{h}ns"]["auc"]) for h in HORIZONS),
}
print("checks vs K06a:", checks, flush=True)

payload = {"conditions": results, "k06a_reference_checks": checks,
           "series_diagnostics": diag, "horizons": list(HORIZONS),
           "real5_frame_idx": REAL5_IDX,
           "note": "(a)=K06a S+H5, (c)=K06a S+G6 expected; cross conditions apply train-source model as-is"}
(OUT / "results.json").write_text(json.dumps(payload, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss"])
    for name in CONDITIONS:
        for h in HORIZONS:
            r = results[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
