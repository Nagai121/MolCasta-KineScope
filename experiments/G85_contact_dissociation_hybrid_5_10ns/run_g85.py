"""G85: contact dissociation mixing at 5 ns and 10 ns (D conditions).

Same design as G79 (which measured 1 ns: mixed - prefix = -0.040,
no help), extended to the 5 ns and 10 ns horizons. Prefix 0.1 ns mean
plus the method-22 generated mean rebuilt in G18, as two separate
features. Conditions (static features + ...), all on the same
candidate set:

- static_only
- full_hist (static + real 6 statistics of the 0-1 ns history)
- prefix_mean_0.1ns (static + prefix mean, frames [0,1])
- gen_mean_only (static + generated mean)
- mixed (static + prefix mean + generated mean, separate features)

Candidate set: the MATCHED subset only, i.e. candidates whose residue
pair has at least one G18 generated pair (G79's count_overlap.py:
70% of train candidates, 76% of test candidates). The base
(static/full/prefix-only) is re-measured on that same subset, so the
recovery denominator is the matched-subset full gain. The full-set
base is verified separately against G71 (5 ns: static 0.3022 /
full 0.5528 / gain +0.2506 / prefix 0.1 ns recovery 0.802;
10 ns: static 0.2772 / full 0.4794 / gain +0.2022 / prefix 0.1 ns
recovery 0.578).

Generated mean per candidate: as in G79 (pointwise minimum over the
residue pair's generated pairs' 4-point series, then mean over the
4 points). Limitation, recorded: the measurement covers only the
76% of test candidates that have a generated mean, and the generated
series cover donor-acceptor pairs only.

Model: HistGradientBoosting (max_iter=120, lr .075, max_leaf_nodes 31,
min_samples_leaf 15, L2 .02, seed 42), retrained per condition and
horizon. Distances use the per-frame box, axis-wise (already applied
at extraction; the cache is reused as-is). History bound for the 6
statistics is 4.5 A (K50).

Pre-fixed reading (same as G79/formation): mixed recovery minus
prefix-only recovery > +0.05 means mixing helps at that horizon;
otherwise it does not.
"""
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).resolve().parent
G18 = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
K50 = Path("/home/hatch/workspace/kinescope/work/k50_cache")
ATOMS = Path("/tmp/g79_atoms/trajectories")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")


def hist6(series, bound):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


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
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def load_with_gen(prots):
    """Load k50 cache plus the per-candidate generated mean.

    Returns X, H, Y, matched mask, gen_mean array (NaN where unmatched).
    Order: protein order, then replica 1-3, as in the cache files.
    """
    Xs, Hs, Ys, matched, gen_mean = [], [], [], [], []
    for p in prots:
        a = pd.read_parquet(ATOMS / f"atlas-{p}-analysis-R1" / "atoms.parquet")
        res_idx = a["residue_index"].to_numpy()
        for r in (1, 2, 3):
            g = np.load(G18 / f"atlas-{p}-analysis-R{r}.npz")
            dres = res_idx[g["pair"][:, 0].astype(int)]
            ares = res_idx[g["pair"][:, 1].astype(int)]
            series = {}
            for di, ai, gd in zip(dres.tolist(), ares.tolist(), g["gen_dist"]):
                key = (min(di, ai), max(di, ai))
                if key in series:
                    series[key] = np.minimum(series[key], gd)
                else:
                    series[key] = gd.astype(np.float64)
            k = np.load(K50 / f"{p}_R{r}_contact_diss.npz")
            Xs.append(k["X"].astype(np.float64))
            Hs.append(k["hist"])
            Ys.append(k["y"])
            m, gm = [], []
            for ri, rj in zip(k["res_i"].tolist(), k["res_j"].tolist()):
                key = (min(ri, rj), max(ri, rj))
                if key in series:
                    m.append(True)
                    gm.append(float(series[key].mean()))
                else:
                    m.append(False)
                    gm.append(float("nan"))
            matched.append(np.array(m))
            gen_mean.append(np.array(gm))
    return (np.concatenate(Xs), np.concatenate(Hs), np.concatenate(Ys),
            np.concatenate(matched), np.concatenate(gen_mean))


HORIZONS = {5: 1, 10: 2}  # horizon_ns -> y column index
# horizon_ns -> (static AUPRC, full AUPRC, prefix 0.1ns recovery) from G71
G71_REF = {5: (0.3022, 0.5528, 0.802), 10: (0.2772, 0.4794, 0.578)}

Xtr, Htr, Ytr, mtr, gmtr = load_with_gen(TRAIN_P)
Xte, Hte, Yte, mte, gmte = load_with_gen(TEST_P)
print(f"train n: {len(Ytr)} matched {int(mtr.sum())} "
      f"test n: {len(Yte)} matched {int(mte.sum())}", flush=True)
pf_tr = Htr[:, [0, 1]].mean(1)[:, None]
pf_te = Hte[:, [0, 1]].mean(1)[:, None]

results = {"train_n_matched": int(mtr.sum()), "test_n_matched": int(mte.sum()),
           "train_n_full": int(len(Ytr)), "test_n_full": int(len(Yte)),
           "horizons": {}}
rows = []
for hns, hi in HORIZONS.items():
    ytr_full, yte_full = Ytr[:, hi], Yte[:, hi]

    # --- verification on the FULL set (G71 base + prefix recovery) ---
    p_static = predict(Xtr, ytr_full, Xte)
    p_full = predict(np.column_stack([Xtr, hist6(Htr, 4.5)]), ytr_full,
                     np.column_stack([Xte, hist6(Hte, 4.5)]))
    p_pref = predict(np.column_stack([Xtr, pf_tr]), ytr_full,
                     np.column_stack([Xte, pf_te]))
    s_f, f_f = metrics(yte_full, p_static)["auprc"], metrics(yte_full, p_full)["auprc"]
    rec_f = (metrics(yte_full, p_pref)["auprc"] - s_f) / (f_f - s_f)
    ref_s, ref_f, ref_rec = G71_REF[hns]
    print(f"{hns} ns FULL SET verify vs G71: static {s_f:.4f} (recorded {ref_s}) "
          f"full {f_f:.4f} (recorded {ref_f}) gain {f_f - s_f:+.4f} "
          f"prefix 0.1ns recovery {rec_f:.3f} (recorded {ref_rec})", flush=True)
    assert abs(s_f - ref_s) < 1e-4 and abs(f_f - ref_f) < 1e-4
    assert abs(rec_f - ref_rec) < 0.005

    # --- measurement on the MATCHED subset ---
    Xtr_m, Xte_m = Xtr[mtr], Xte[mte]
    Htr_m, Hte_m = Htr[mtr], Hte[mte]
    ytr, yte = ytr_full[mtr], yte_full[mte]
    pf_tr_m, pf_te_m = pf_tr[mtr], pf_te[mte]
    gm_tr_m = gmtr[mtr][:, None]
    gm_te_m = gmte[mte][:, None]
    conds = {
        "static_only": (Xtr_m, Xte_m),
        "full_hist": (np.column_stack([Xtr_m, hist6(Htr_m, 4.5)]),
                      np.column_stack([Xte_m, hist6(Hte_m, 4.5)])),
        "prefix_mean_0.1ns": (np.column_stack([Xtr_m, pf_tr_m]),
                              np.column_stack([Xte_m, pf_te_m])),
        "gen_mean_only": (np.column_stack([Xtr_m, gm_tr_m]),
                          np.column_stack([Xte_m, gm_te_m])),
        "mixed": (np.column_stack([Xtr_m, pf_tr_m, gm_tr_m]),
                  np.column_stack([Xte_m, pf_te_m, gm_te_m])),
    }
    entry = {"horizon_ns": hns, "test_n_matched": int(mte.sum()),
             "test_positives_matched": int(yte.sum()),
             "test_positives_full": int(yte_full.sum()),
             "full_set_verify": {"static_auprc": float(s_f), "full_auprc": float(f_f),
                                 "prefix_0.1ns_recovery": float(rec_f)},
             "conditions": {}}
    for name, (xtr, xte) in conds.items():
        p = predict(xtr, ytr, xte)
        m = metrics(yte, p)
        entry["conditions"][name] = m
        print(f"{hns} ns matched {name}: AUPRC {m['auprc']:.4f} AUC {m['auc']:.4f}", flush=True)
        rows.append({"horizon_ns": hns, "condition": name, **m})
    s = entry["conditions"]["static_only"]["auprc"]
    f = entry["conditions"]["full_hist"]["auprc"]
    entry["matched_base_gain"] = float(f - s)
    for name in ("prefix_mean_0.1ns", "gen_mean_only", "mixed"):
        pm = entry["conditions"][name]["auprc"]
        entry["conditions"][name]["gain"] = float(pm - s)
        entry["conditions"][name]["recovery"] = float((pm - s) / (f - s))
        print(f"{hns} ns matched {name}: gain {pm - s:+.4f} recovery {(pm - s) / (f - s):.3f}",
              flush=True)
    entry["mixed_minus_prefix"] = float(
        entry["conditions"]["mixed"]["recovery"]
        - entry["conditions"]["prefix_mean_0.1ns"]["recovery"])
    print(f"{hns} ns mixed - prefix recovery diff: {entry['mixed_minus_prefix']:+.3f} "
          f"(line +0.05)", flush=True)
    results["horizons"][str(hns)] = entry

(HERE / "results.json").write_text(json.dumps(results, indent=2))
with (HERE / "results.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["horizon_ns", "condition", "n", "positives", "auc",
                                       "auprc", "brier", "logloss", "gain", "recovery"])
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in
                    ["horizon_ns", "condition", "n", "positives", "auc", "auprc", "brier",
                     "logloss", "gain", "recovery"]})
np.savez_compressed(HERE / "matched_gen_mean.npz",
                    train_matched=mtr, test_matched=mte,
                    train_gen_mean=gmtr, test_gen_mean=gmte)
print("saved to", HERE)
