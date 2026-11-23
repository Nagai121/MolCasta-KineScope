"""K43: is the generated mean the same thing as a static prediction of
the real mean?

Part 1 (K06(b) candidates, the instruction's main setting): refit K39's
two models (OLS + HistGB, static-5 -> real mean) on the calibration 8
proteins only, predict on the test 100, and compare the generated mean
(gen6[:,1]) with that prediction (Pearson/Spearman corr, diff std).
Note: K39's fit uses K06(b)'s static 5 features (frozen prob, origin
distance, 3 base descriptors), NOT DocD's 7 -- the DocD-7 features do
not exist for the K06(b) candidate set. The instruction's "(K39's fit)"
is the operative specification; the DocD-7 form is measured in part 2.

Part 2 (DocD candidates, where both feature sets exist): the same
comparison with (a') the DocD-7 fit (K23's settings) and (b) the K40
rich-99 fit, both trained on the DocD train 4 proteins and evaluated
on the DocD test 2 proteins. Generated mean from K05's series.
Pre-fixed reading: corr ~0.99 and small diff -> the generator is
effectively a static predictor; corr < 0.9 -> the generated mean has a
component beyond the static prediction."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

OUTD = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
K40OUT = Path("/home/hatch/workspace/k40/out")
GEN = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")

def pearson(x, y): return float(np.corrcoef(x, y)[0, 1])
def spearman(x, y): return float(spearmanr(x, y).statistic)
def cmp_pair(name, gen, pred, real):
    d = gen - pred
    return dict(n=int(len(gen)), pearson=pearson(gen, pred), spearman=spearman(gen, pred),
                diff_mean=float(d.mean()), diff_std=float(d.std()), diff_mae=float(np.abs(d).mean()),
                corr_gen_real=pearson(gen, real), corr_pred_real=pearson(pred, real),
                gen_std=float(gen.std()), pred_std=float(pred.std()), real_std=float(real.std()))

def gbm_docd():
    return HistGradientBoostingRegressor(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                         min_samples_leaf=15, l2_regularization=.02, random_state=42)
def gbm_doca():
    return HistGradientBoostingRegressor(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                         min_samples_leaf=20, l2_regularization=.15, random_state=20261121)

def load_k06b(proteins):
    S, H, G6 = [], [], []
    for p in proteins:
        with np.load(OUTD / f"{p}.npz") as z:
            for r in (1, 2, 3):
                S.append(z[f"r{r}_static"]); H.append(z[f"r{r}_history"]); G6.append(z[f"r{r}_gen6"])
    return np.concatenate(S), np.concatenate(H), np.concatenate(G6)

def main():
    res = {"part1_k06b": {}, "part2_docd": {}}
    # ---- part 1 ----
    Scal, Hcal, Gcal = load_k06b(CAL)
    Ste, Hte, Gte = load_k06b(TEST)
    real_cal, real_te = Hcal[:, 1], Hte[:, 1]
    gen_te = Gte[:, 1]
    print("part1 n_cal", len(real_cal), "n_test", len(real_te), flush=True)
    for name, model in (("ols", LinearRegression()), ("gbm", gbm_doca())):
        model.fit(Scal, real_cal)
        pred = model.predict(Ste)
        res["part1_k06b"][name] = cmp_pair(name, gen_te, pred, real_te)
        print("part1", name, {k: round(v, 4) for k, v in res["part1_k06b"][name].items() if k != "n"}, flush=True)
    # K39 reproduction check: R^2 of the real-mean fit and of the gen-mean fit on test
    from sklearn.metrics import r2_score
    for name, model in (("ols", LinearRegression()), ("gbm", gbm_doca())):
        model.fit(Scal, real_cal)
        print("  K39 check", name, "R2 real-mean fit (test):",
              round(r2_score(real_te, model.predict(Ste)), 4), flush=True)
        model2 = LinearRegression() if name == "ols" else gbm_doca()
        model2.fit(Scal, Gcal[:, 1])
        print("  K39 check", name, "R2 gen-mean fit (test):",
              round(r2_score(gen_te, model2.predict(Ste)), 4), flush=True)
    # ---- part 2 ----
    z = np.load(GEN)
    gen5 = np.column_stack([z["pair"][:, 2].astype(np.float64), z["gen"].astype(np.float64)])
    gen_means = {(str(t), int(z["pair"][i, 0]), int(z["pair"][i, 1])): float(gen5[i].mean())
                 for i, t in enumerate(z["tid"])}
    def load_docd(prots, want_rich):
        X7s, Rs, Ys, Gs = [], [], [], []
        for p in prots:
            for r in (1, 2, 3):
                with np.load(CACHE / f"{p}_R{r}.npz") as c:
                    X7s.append(c["X"]); Ys.append(c["hist"].mean(axis=0))
                    tid = f"atlas-{p}-analysis-R{r}"
                    Gs.append(np.array([gen_means[(tid, int(da), int(aa))]
                                        for da, aa in zip(c["donor_atom"], c["acceptor_atom"])]))
                if want_rich:
                    with np.load(K40OUT / f"{p}_R{r}.npz") as k:
                        Rs.append(k["Xrich"])
        out = [np.concatenate(X7s), np.concatenate(Ys), np.concatenate(Gs)]
        if want_rich: out.append(np.concatenate(Rs))
        return out
    X7tr, ytr, gtr = load_docd(TRAIN_P, False)
    X7te, yte, gte = load_docd(TEST_P, False)
    _, _, _, Rtr = load_docd(TRAIN_P, True)
    _, _, _, Rte = load_docd(TEST_P, True)
    print("part2 n_train", len(ytr), "n_test", len(yte), flush=True)
    for name, (Xtr_, Xte_) in (("docd7_ols", (X7tr, X7te)), ("docd7_gbm", (X7tr, X7te)),
                              ("rich99_ols", (Rtr, Rte)), ("rich99_gbm", (Rtr, Rte))):
        model = (make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
                 if name.endswith("ols") else gbm_docd())
        model.fit(Xtr_, ytr)
        pred = model.predict(Xte_)
        key = name.replace("_ols", "").replace("_gbm", "")
        res["part2_docd"][name] = cmp_pair(name, gte, pred, yte)
        print("part2", name, {k: round(v, 4) for k, v in res["part2_docd"][name].items() if k != "n"}, flush=True)
    Path("/home/hatch/workspace/k43/k43_results.json").write_text(json.dumps(res, indent=1))
    print("WROTE k43_results.json")

if __name__ == "__main__":
    main()
