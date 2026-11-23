"""K29: is the generated mean error related to the label or the future?
(weak base, data D candidates -- the same candidates as K05/K08/K28)

Per candidate: real mean m_r (11-frame history), generated mean m_g
(K05 5-point series), error e = m_g - m_r; labels y (6 horizons);
future distances d_h at frames 20/60/110/310/510/910 (extract_future.py,
verified: frame 0/10 match the cache exactly, and (d_h <= 3.5) reproduces
the labels with agreement 1.0000).

Measurements, for the generated error and for the K28 synthetic error
(e_syn = a + (b-1)*m_r + noise, a = 1.3829, b = 0.6972, noise SD 0.4028,
seeds 101/202/303):
  (a) relation with the label, per horizon: corr(e, y_h), mean(e|y=1) -
      mean(e|y=0), AUC of e predicting y_h
  (b) relation with the real mean's deviation from the overall mean
      (dev = m_r - center): corr(e, dev) and corr(|e|, |dev|)
  (c) relation with the future distance, per horizon: corr(e, d_h) and
      corr(e, d_h - m_r)
Context: corr(m_r, y_h) and corr(m_r, d_h) -- the signal the real mean
itself carries -- are reported alongside, since the synthetic error
inherits a label relation through the shrinkage term (b-1)*m_r.
"""
import numpy as np, json
from pathlib import Path
from sklearn.metrics import roc_auc_score

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
FUT = Path("/home/hatch/workspace/kinescope/work/k29_cache")
GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
A, B, SIGMA = 1.3829, 0.6972, 0.4028
SEEDS = (101, 202, 303)

z = np.load(GEN)
tids, pair, gen = z["tid"], z["pair"], z["gen"]
gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
GEN_MEAN = {(str(t), int(pair[i, 0]), int(pair[i, 1])): float(gen5[i].mean())
            for i, t in enumerate(tids)}

mr, mg, ys, dh = [], [], [], []
for p in TRAIN_P + TEST_P:
    for r in (1, 2, 3):
        d = np.load(CACHE / f"{p}_R{r}.npz")
        tid = f"atlas-{p}-analysis-R{r}"
        mg.append(np.array([GEN_MEAN[(tid, int(da), int(aa))]
                            for da, aa in zip(d["donor_atom"], d["acceptor_atom"])]))
        mr.append(d["hist"].mean(axis=0).astype(np.float64))
        ys.append(d["y"].astype(np.float64))  # (N, 6)
        dh.append(np.load(FUT / f"{p}_R{r}_future.npz")["future"].astype(np.float64).T)  # (N, 6)
mr = np.concatenate(mr); mg = np.concatenate(mg)
Y = np.concatenate(ys); D = np.concatenate(dh)
print("n =", len(mr), flush=True)
CENTER = float(mr.mean())
dev = mr - CENTER
e_gen = mg - mr

def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])

def per_error(e, tag):
    out = {}
    out["error_mean"] = float(e.mean()); out["error_sd"] = float(e.std())
    out["corr_with_real_mean"] = corr(e, mr)
    out["b_corr_with_deviation"] = corr(e, dev)
    out["b_corr_abs_error_with_abs_deviation"] = corr(np.abs(e), np.abs(dev))
    for j, h in enumerate(HORIZONS):
        y = Y[:, j]
        out[f"a_corr_with_label_{h}ns"] = corr(e, y)
        out[f"a_mean_error_pos_minus_neg_{h}ns"] = float(e[y == 1].mean() - e[y == 0].mean())
        out[f"a_auc_error_predicts_label_{h}ns"] = float(roc_auc_score(y, e))
        out[f"c_corr_with_future_distance_{h}ns"] = corr(e, D[:, j])
        out[f"c_corr_with_future_minus_mean_{h}ns"] = corr(e, D[:, j] - mr)
    return out

results = {"n": int(len(mr)), "center": CENTER,
           "context_corr_real_mean_with_label": {f"{h}ns": corr(mr, Y[:, j]) for j, h in enumerate(HORIZONS)},
           "context_corr_real_mean_with_future_distance": {f"{h}ns": corr(mr, D[:, j]) for j, h in enumerate(HORIZONS)},
           "generated": per_error(e_gen, "generated")}
print("generated:", json.dumps(results["generated"], indent=1), flush=True)

syn = {}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    e_syn = A + (B - 1.0) * mr + rng.normal(0.0, SIGMA, len(mr))
    syn[f"seed{seed}"] = per_error(e_syn, f"synthetic_seed{seed}")
results["synthetic"] = syn
# average across seeds for the comparison table
avg = {}
for k in syn["seed101"]:
    vals = [syn[f"seed{s}"][k] for s in SEEDS]
    avg[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
results["synthetic_mean_across_seeds"] = avg
print("synthetic (mean across seeds):", json.dumps({k: round(v["mean"], 4) for k, v in avg.items()}, indent=1), flush=True)
(OUT / "measure_results.json").write_text(json.dumps(results, indent=2))
print("saved to", OUT)

# --- supplementary: how much label/future signal does each mean carry? ---
# (appended after the main save; recomputes from the same in-memory arrays)
def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])
extra = {}
for j, h in enumerate(HORIZONS):
    extra[f"corr_gen_mean_with_label_{h}ns"] = corr(mg, Y[:, j])
    extra[f"corr_gen_mean_with_future_{h}ns"] = corr(mg, D[:, j])
    extra[f"gen_mean_pos_minus_neg_{h}ns"] = float(mg[Y[:, j] == 1].mean() - mg[Y[:, j] == 0].mean())
    extra[f"real_mean_pos_minus_neg_{h}ns"] = float(mr[Y[:, j] == 1].mean() - mr[Y[:, j] == 0].mean())
syn_extra = {}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    m_syn = A + B * mr + rng.normal(0.0, SIGMA, len(mr))
    for j, h in enumerate(HORIZONS):
        syn_extra.setdefault(f"corr_syn_mean_with_label_{h}ns", []).append(corr(m_syn, Y[:, j]))
        syn_extra.setdefault(f"corr_syn_mean_with_future_{h}ns", []).append(corr(m_syn, D[:, j]))
        syn_extra.setdefault(f"syn_mean_pos_minus_neg_{h}ns", []).append(
            float(m_syn[Y[:, j] == 1].mean() - m_syn[Y[:, j] == 0].mean()))
for k, v in syn_extra.items():
    extra[k] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
results["mean_signal"] = extra
(OUT / "measure_results.json").write_text(json.dumps(results, indent=2))
print("mean_signal:")
for k, v in extra.items():
    print(" ", k, v if isinstance(v, dict) else round(v, 4))
