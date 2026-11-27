"""L2-13: L63-L68 (salt formation, K15 base, 6 statistics) re-measured with
the history window placed immediately before the origin frame (frames
(10 - L/0.1)..10), all six horizons in one run.

Base, candidates, learner and readings are identical to L63-L68; only the
window placement changes (L used frames 0..k). At length 1.0 ns the window
is frames 0..10, identical to L, so the static-only / full AUPRCs must
reproduce the L63-L68 records (tol 5e-4), the mean-only recovery at 1.0 ns
must reproduce the value remeasured in L63-L68 (tol 5e-4), and the
6-statistics recovery at 1.0 ns must be 1.0 (tol 1e-6); the run stops
before the main phase if any horizon fails.

The 6 statistics of a window are [first frame of the window, mean, std,
min, frame 10 minus first frame of the window, fraction <= 4.0 A].
Recovery = (AUPRC at length - static only) / (full-history gain),
pooled and macro (mean over proteins), as in L63-L68.
Bootstrap intervals (protein-level, 10,000 draws, seed 42, percentile)
are computed for the 6-statistics curve of each horizon.
Per-horizon results are checkpointed to ~/workspace/l2/l2_13_checkpoint.json
so an interrupted run resumes where it stopped.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
SPLIT = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json")
OUT = Path(__file__).resolve().parent
CKPT = Path("/home/hatch/workspace/l2/l2_13_checkpoint.json")
SEED = 42
BOOT_SEED = 42
N_BOOT = 10000
HORIZONS = [(1, 0), (5, 1), (10, 2), (30, 3), (50, 4), (90, 5)]
BOUND = 4.0
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
NF = {L: int(round(L / 0.1)) + 1 for L in LENGTHS}
FRAMES = {L: list(range(11 - NF[L], 11)) for L in LENGTHS}
COND6 = {L: f"suffix_stats6_{L:.1f}ns" for L in LENGTHS}
CONDMEAN = {L: f"suffix_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = (["static_only", "full_history"] + [COND6[L] for L in LENGTHS]
              + [CONDMEAN[L] for L in LENGTHS])
BOOT_CONDITIONS = ["static_only", "full_history"] + [COND6[L] for L in LENGTHS]

REFS = {'1': {'static': 0.2402, 'full': 0.3842, 'n_train': 2557, 'n_test': 1170, 'positives': 93, 'mean_rec_1_0': 0.8890425378317479}, '5': {'static': 0.2513, 'full': 0.3301, 'n_train': 2557, 'n_test': 1170, 'positives': 129, 'mean_rec_1_0': 0.8759579717176592}, '10': {'static': 0.2034, 'full': 0.2627, 'n_train': 2557, 'n_test': 1170, 'positives': 131, 'mean_rec_1_0': 0.34118679953999487}, '30': {'static': 0.1582, 'full': 0.2466, 'n_train': 2557, 'n_test': 1170, 'positives': 133, 'mean_rec_1_0': 1.0955824949194528}, '50': {'static': 0.2295, 'full': 0.2569, 'n_train': 2557, 'n_test': 1170, 'positives': 137, 'mean_rec_1_0': 0.7995533111932827}, '90': {'static': 0.2136, 'full': 0.2736, 'n_train': 2557, 'n_test': 1170, 'positives': 149, 'mean_rec_1_0': 0.4588813714993671}}
L_READINGS = {'1': {'min08': 0.8, 'min09': 0.8, 'min08_macro': 0.5, 'min09_macro': 0.5, 'largest_step': [0.7, 0.8, 0.40965257932052124], 'stats6_rec_curve': {'0.0': 0.571222607685573, '0.1': 0.68469702028845, '0.2': 0.33422879866341, '0.3': 0.5721012679224908, '0.4': 0.43232026975118004, '0.5': 0.7163963113887016, '0.6': 0.5321837434156557, '0.7': 0.5109466138548524, '0.8': 0.9205991931753736, '0.9': 0.9731120938524912, '1.0': 1.0}}, '5': {'min08': 0.3, 'min09': 0.9, 'min08_macro': 0.0, 'min09_macro': 0.1, 'largest_step': [0.8, 0.9, 0.4796480068172838], 'stats6_rec_curve': {'0.0': 0.5953276092659866, '0.1': 0.29857929512778275, '0.2': 0.7558138536540125, '0.3': 0.8743336923355192, '0.4': 0.8579781471165411, '0.5': 0.6547690611583864, '0.6': 0.48648871751681083, '0.7': 0.8134407004510301, '0.8': 0.6472865773304312, '0.9': 1.126934584147715, '1.0': 1.0}}, '10': {'min08': 0.0, 'min09': 0.0, 'min08_macro': 0.0, 'min09_macro': 0.0, 'largest_step': [0.4, 0.5, 0.5925029834775292], 'stats6_rec_curve': {'0.0': 1.2092388112788883, '0.1': 0.3521663727503818, '0.2': 0.4614955289474361, '0.3': 0.9450115674355296, '0.4': 0.7056707963372452, '0.5': 1.2981737798147743, '0.6': 0.9176335066230125, '0.7': 0.6815821417112464, '0.8': 0.8441677188850469, '0.9': 1.2637994270953379, '1.0': 1.0}}, '30': {'min08': 0.3, 'min09': 0.6, 'min08_macro': 0.0, 'min09_macro': 0.0, 'largest_step': [0.4, 0.5, 0.3736256020989711], 'stats6_rec_curve': {'0.0': 0.6251570362570426, '0.1': 0.4908283626406841, '0.2': 0.683523961834241, '0.3': 0.8570961459960255, '0.4': 0.5230959670115511, '0.5': 0.8967215691105223, '0.6': 1.1095873076855651, '0.7': 0.8100878883347065, '0.8': 0.7970428256580815, '0.9': 0.7976956695059513, '1.0': 1.0}}, '50': {'min08': 0.0, 'min09': 0.0, 'min08_macro': 0.0, 'min09_macro': 0.0, 'largest_step': [0.1, 0.2, 0.6512032443396087], 'stats6_rec_curve': {'0.0': 1.2182058274110759, '0.1': 1.7109738498381066, '0.2': 2.3621770941777154, '0.3': 1.7283748082216086, '0.4': 1.91405810309454, '0.5': 1.7364011516225013, '0.6': 1.910715804345789, '0.7': 1.2916483524930624, '0.8': 1.1506463477071414, '0.9': 0.5755026050071906, '1.0': 1.0}}, '90': {'min08': 0.0, 'min09': 0.2, 'min08_macro': 0.0, 'min09_macro': 0.2, 'largest_step': [0.1, 0.2, 0.8496074802131068], 'stats6_rec_curve': {'0.0': 0.8455475237966913, '0.1': 0.18672815380996277, '0.2': 1.0363356340230696, '0.3': 0.8657486414202525, '0.4': 0.7635169639337764, '0.5': 1.2085516018082263, '0.6': 0.8063468480215149, '0.7': 1.1510454356325215, '0.8': 1.064396040436835, '0.9': 0.8156021628864448, '1.0': 1.0}}}

sp = json.loads(SPLIT.read_text())
TRAIN_P = sp["orig_train"] + sp["train_additions"]
TEST_P = sp["orig_test"] + sp["test_additions"]


def load(proteins):
    X, H, Y = [], [], []
    spans = []
    cur = 0
    for p in proteins:
        xs, hs, ys = [], [], []
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
            xs.append(z["X"]); hs.append(z["hist"]); ys.append(z["y"])
        x = np.concatenate(xs); h = np.concatenate(hs); y = np.concatenate(ys)
        X.append(x); H.append(h); Y.append(y)
        spans.append((p, cur, cur + len(y))); cur += len(y)
    return (np.concatenate(X), np.concatenate(H), np.concatenate(Y), spans)


def hist6(series):
    return np.column_stack([
        series[:, 0], series.mean(1), series.std(1), series.min(1),
        series[:, -1] - series[:, 0], (series <= BOUND).mean(1)])


def learner(X, y):
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=0.02,
        random_state=SEED).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def ap_numpy(y, score):
    order = np.argsort(-score, kind="stable")
    ys = y[order].astype(np.float64)
    ss = score[order]
    ends = np.flatnonzero(np.r_[ss[:-1] != ss[1:], True])
    tp = np.cumsum(ys)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall_step = np.diff(np.r_[0.0, tp]) / tp[-1]
    return float(np.sum(recall_step * precision))


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def build_feats(Xtr, Htr, Xte, Hte):
    feats = {}
    feats["static_only"] = (Xtr, Xte)
    feats["full_history"] = (np.column_stack([Xtr, hist6(Htr)]),
                             np.column_stack([Xte, hist6(Hte)]))
    for L in LENGTHS:
        nf = NF[L]
        feats[COND6[L]] = (
            np.column_stack([Xtr, hist6(Htr[:, 11 - nf:])]),
            np.column_stack([Xte, hist6(Hte[:, 11 - nf:])]))
        feats[CONDMEAN[L]] = (
            np.column_stack([Xtr, Htr[:, 11 - nf:].mean(1, keepdims=True).astype(np.float32)]),
            np.column_stack([Xte, Hte[:, 11 - nf:].mean(1, keepdims=True).astype(np.float32)]))
    return feats


def main():
    t0 = time.time()
    Xtr, Htr, Ytr, _ = load(TRAIN_P)
    Xte, Hte, Yte, spans = load(TEST_P)
    prot_of_row = np.concatenate([np.full(e - s, i) for i, (_, s, e) in enumerate(spans)])
    print("train", len(Ytr), "test", len(Yte), flush=True)

    # ---------- premise phase: static / full / stats6@1.0 / mean@1.0, all horizons ----------
    checks = []
    pre = {}
    for hns, hidx in HORIZONS:
        y_tr = Ytr[:, hidx]
        y_te = Yte[:, hidx]
        feats = build_feats(Xtr, Htr, Xte, Hte)
        ref = REFS[str(hns)]
        preds = {}
        for cname in ("static_only", "full_history", COND6[1.0], CONDMEAN[1.0]):
            xtr, xte = feats[cname]
            preds[cname] = learner(xtr, y_tr).predict_proba(xte)[:, 1]
        pre[hns] = preds
        a_s = float(average_precision_score(y_te, preds["static_only"]))
        a_f = float(average_precision_score(y_te, preds["full_history"]))
        gain = a_f - a_s
        mean_rec10 = float((average_precision_score(y_te, preds[CONDMEAN[1.0]]) - a_s) / gain)
        s6_rec10 = float((average_precision_score(y_te, preds[COND6[1.0]]) - a_s) / gain)

        def chk(name, got, reference, tol):
            ok = abs(got - reference) < tol if tol > 0 else got == reference
            checks.append(dict(horizon_ns=hns, check=name, got=float(got),
                               reference=float(reference), tol=tol, ok=bool(ok)))
            print(f"premise {hns}ns {name}: got {got} ref {reference} tol {tol} "
                  f"-> {'OK' if ok else 'FAIL'}", flush=True)
            return ok

        chk("n_train", len(y_tr), ref["n_train"], 0)
        chk("n_test", len(y_te), ref["n_test"], 0)
        chk("positives", int(y_te.sum()), ref["positives"], 0)
        chk("static_auprc", a_s, ref["static"], 5e-4)
        chk("full_auprc", a_f, ref["full"], 5e-4)
        chk("mean_recovery_at_1.0ns", mean_rec10, ref["mean_rec_1_0"], 5e-4)
        chk("stats6_recovery_at_1.0ns_is_1", s6_rec10, 1.0, 1e-6)
    all_ok = all(c["ok"] for c in checks)
    (OUT / "premise_check.json").write_text(json.dumps(
        dict(checks=checks, all_ok=bool(all_ok),
             note="Length 1.0 ns uses frames 0..10, identical to L63-L68; "
                  "static/full AUPRCs must match the L records (tol 5e-4), the "
                  "mean-only recovery at 1.0 ns must match the value remeasured "
                  "in L63-L68 (tol 5e-4), and the 6-statistics recovery at 1.0 ns "
                  "must be 1.0 (tol 1e-6)."),
        indent=2, ensure_ascii=False))
    if not all_ok:
        print("PREMISE FAILED - stopping before the main phase", flush=True)
        sys.exit(2)
    print(f"premise OK ({time.time() - t0:.0f}s)", flush=True)

    # ---------- main phase (checkpointed per horizon) ----------
    done = {}
    if CKPT.exists():
        done = json.loads(CKPT.read_text())
        print("checkpoint: horizons already done:", sorted(done.keys()), flush=True)

    blocks = {}
    all_preds = {}
    for hns, hidx in HORIZONS:
        if str(hns) in done:
            blocks[str(hns)] = done[str(hns)]["block"]
            all_preds[hns] = {c: np.array(v, dtype=np.float32)
                              for c, v in done[str(hns)]["preds"].items()}
            print(f"{hns}ns: restored from checkpoint", flush=True)
            continue
        y_tr = Ytr[:, hidx]
        y_te = Yte[:, hidx]
        feats = build_feats(Xtr, Htr, Xte, Hte)
        preds = dict(pre[hns])
        results = {}
        per_protein_ap = {c: {} for c in CONDITIONS}
        for cname in CONDITIONS:
            if cname not in preds:
                xtr, xte = feats[cname]
                preds[cname] = learner(xtr, y_tr).predict_proba(xte)[:, 1]
            pred = preds[cname]
            m = metrics(y_te, pred)
            aps = []
            for p, s, e in spans:
                yy = y_te[s:e]
                if 0 < yy.sum() < e - s:
                    ap = float(average_precision_score(yy, pred[s:e]))
                    aps.append(ap)
                    per_protein_ap[cname][p] = ap
                else:
                    per_protein_ap[cname][p] = None
            m["macro_auprc"] = float(np.mean(aps)) if aps else None
            m["n_proteins_macro"] = len(aps)
            results[cname] = m
        print(f"{hns}ns: fits done ({time.time() - t0:.0f}s)", flush=True)

        gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
        gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

        def rec(cname, kind):
            if kind == "pooled":
                return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
            return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

        recovery = {c: dict(pooled=float(rec(c, "pooled")), macro=float(rec(c, "macro")))
                    for c in CONDITIONS if c != "static_only"}

        def increments_for(condmap, kind):
            inc = {}
            prev = None
            for L in LENGTHS:
                r = recovery[condmap[L]][kind]
                if prev is not None:
                    inc[f"{L:.1f}"] = float(r - prev)
                prev = r
            return inc

        inc6 = dict(pooled=increments_for(COND6, "pooled"), macro=increments_for(COND6, "macro"))
        incmean = dict(pooled=increments_for(CONDMEAN, "pooled"), macro=increments_for(CONDMEAN, "macro"))

        def min_length(condmap, kind, thr):
            for L in LENGTHS:
                if recovery[condmap[L]][kind] >= thr:
                    return L
            return None

        def largest_step(condmap, kind):
            inc = increments_for(condmap, kind)
            key = max(inc, key=lambda k: inc[k])
            Lto = float(key)
            return dict(from_ns=round(Lto - 0.1, 1), to_ns=Lto, increment=inc[key])

        comparison = {}
        for thr, key in ((0.8, "0_8"), (0.9, "0_9")):
            ml6 = min_length(COND6, "pooled", thr)
            mlm = min_length(CONDMEAN, "pooled", thr)
            comparison[f"min_length_recovery_{key}"] = dict(
                stats6_ns=ml6, mean_ns=mlm,
                shrink_ns=(None if (ml6 is None or mlm is None) else float(mlm - ml6)))
        comparison["recovery_diff_stats6_minus_mean_pooled"] = {
            f"{L:.1f}": float(recovery[COND6[L]]["pooled"] - recovery[CONDMEAN[L]]["pooled"])
            for L in LENGTHS}

        per_protein = {}
        for p, s, e in spans:
            yy = y_te[s:e]
            entry = dict(n=int(e - s), positives=int(yy.sum()))
            ap_s = per_protein_ap["static_only"][p]
            ap_f = per_protein_ap["full_history"][p]
            entry["auprc_static"] = ap_s
            entry["auprc_full"] = ap_f
            entry["full_gain"] = (ap_f - ap_s) if (ap_s is not None and ap_f is not None) else None
            entry["auprc_suffix_stats6"] = {}
            entry["recovery_suffix_stats6"] = {}
            entry["auprc_suffix_mean"] = {}
            entry["recovery_suffix_mean"] = {}
            for L in LENGTHS:
                ap6 = per_protein_ap[COND6[L]][p]
                apm = per_protein_ap[CONDMEAN[L]][p]
                entry["auprc_suffix_stats6"][f"{L:.1f}"] = ap6
                entry["auprc_suffix_mean"][f"{L:.1f}"] = apm
                if ap6 is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                    entry["recovery_suffix_stats6"][f"{L:.1f}"] = float((ap6 - ap_s) / entry["full_gain"])
                else:
                    entry["recovery_suffix_stats6"][f"{L:.1f}"] = None
                if apm is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                    entry["recovery_suffix_mean"][f"{L:.1f}"] = float((apm - ap_s) / entry["full_gain"])
                else:
                    entry["recovery_suffix_mean"][f"{L:.1f}"] = None
            per_protein[p] = entry

        pp_summary = {}
        valid_pp = [p for p, en in per_protein.items() if en["recovery_suffix_stats6"]["1.0"] is not None]
        for L in LENGTHS:
            vals = np.array([per_protein[p]["recovery_suffix_stats6"][f"{L:.1f}"] for p in valid_pp])
            pp_summary[f"{L:.1f}"] = dict(
                n_proteins=int(len(vals)), median=float(np.median(vals)),
                q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)),
                frac_ge_0_8=float(np.mean(vals >= 0.8)))
        pp_inc_summary = {}
        for i, L in enumerate(LENGTHS[1:], start=1):
            Lp = LENGTHS[i - 1]
            vals = np.array([per_protein[p]["recovery_suffix_stats6"][f"{L:.1f}"]
                             - per_protein[p]["recovery_suffix_stats6"][f"{Lp:.1f}"]
                             for p in valid_pp])
            pp_inc_summary[f"{L:.1f}"] = dict(
                median=float(np.median(vals)), q25=float(np.percentile(vals, 25)),
                q75=float(np.percentile(vals, 75)), frac_positive=float(np.mean(vals > 0)))

        # ---- bootstrap (stats6 curve), same construction as L63 ----
        for cname in BOOT_CONDITIONS:
            ap_np = ap_numpy(y_te, preds[cname].astype(np.float64))
            assert abs(ap_np - results[cname]["auprc"]) < 1e-9, (cname, ap_np, results[cname]["auprc"])
        boot_prep = {}
        for cname in BOOT_CONDITIONS:
            score = preds[cname].astype(np.float64)
            order = np.argsort(-score, kind="stable")
            ss = score[order]
            starts = np.flatnonzero(np.r_[True, ss[:-1] != ss[1:]])
            boot_prep[cname] = dict(y_sorted=y_te[order].astype(np.float64),
                                    prot_sorted=prot_of_row[order], starts=starts)

        def pooled_ap_boot(prep, mult):
            w = mult[prep["prot_sorted"]]
            wp = w * prep["y_sorted"]
            tp_g = np.add.reduceat(wp, prep["starts"])
            tot_g = np.add.reduceat(w, prep["starts"])
            if tp_g.sum() <= 0:
                return None
            cum_tp = np.cumsum(tp_g)
            cum_tot = np.cumsum(tot_g)
            precision = np.divide(cum_tp, cum_tot, out=np.zeros_like(cum_tp), where=cum_tot > 0)
            return float(np.sum((tp_g / cum_tp[-1]) * precision))

        defined = [p for p, _, _ in spans if per_protein_ap["static_only"][p] is not None]
        didx = [i for i, (p, _, _) in enumerate(spans) if p in defined]
        draw_to_defined = np.full(len(spans), -1, dtype=int)
        for pos, i in enumerate(didx):
            draw_to_defined[i] = pos
        ap_mat = np.array([[per_protein_ap[c][p] for c in BOOT_CONDITIONS] for p in defined])
        cond_idx = {c: j for j, c in enumerate(BOOT_CONDITIONS)}

        rng = np.random.default_rng(BOOT_SEED)
        nP = len(spans)
        boot_rec = {kind: {f"{L:.1f}": [] for L in LENGTHS} for kind in ("pooled", "macro")}
        n_bad = dict(pooled=0, macro=0)
        for b in range(N_BOOT):
            draw = rng.integers(0, nP, nP)
            mult = np.bincount(draw, minlength=nP).astype(np.float64)
            ap_s = pooled_ap_boot(boot_prep["static_only"], mult)
            ap_f = pooled_ap_boot(boot_prep["full_history"], mult)
            if ap_s is None or ap_f is None or ap_f - ap_s <= 0:
                n_bad["pooled"] += 1
            else:
                g = ap_f - ap_s
                for L in LENGTHS:
                    ap = pooled_ap_boot(boot_prep[COND6[L]], mult)
                    boot_rec["pooled"][f"{L:.1f}"].append((ap - ap_s) / g)
            dpos = draw_to_defined[draw]
            wsel = dpos[dpos >= 0]
            if len(wsel) == 0:
                n_bad["macro"] += 1
            else:
                m = ap_mat[wsel].mean(axis=0)
                gm = m[cond_idx["full_history"]] - m[cond_idx["static_only"]]
                if gm <= 0:
                    n_bad["macro"] += 1
                else:
                    for L in LENGTHS:
                        boot_rec["macro"][f"{L:.1f}"].append(
                            (m[cond_idx[COND6[L]]] - m[cond_idx["static_only"]]) / gm)
            if (b + 1) % 2500 == 0:
                print(f"{hns}ns bootstrap {b + 1}/{N_BOOT} ({time.time() - t0:.0f}s)", flush=True)

        bootstrap = {}
        for kind in ("pooled", "macro"):
            bootstrap[kind] = {}
            for L in LENGTHS:
                arr = np.array(boot_rec[kind][f"{L:.1f}"])
                bootstrap[kind][f"{L:.1f}"] = dict(
                    n_boot=int(len(arr)), seed=BOOT_SEED,
                    point_estimate=recovery[COND6[L]][kind],
                    ci_lower=float(np.percentile(arr, 2.5)),
                    ci_upper=float(np.percentile(arr, 97.5)),
                    boot_mean=float(arr.mean()), boot_std=float(arr.std()))
        print(f"{hns}ns bootstrap done ({time.time() - t0:.0f}s), bad gains: {n_bad}", flush=True)

        step = largest_step(COND6, "pooled")
        Lfrom, Lto = step["from_ns"], step["to_ns"]
        step_vals = []
        for p, entry in per_protein.items():
            a = entry["recovery_suffix_stats6"].get(f"{Lfrom:.1f}")
            bb = entry["recovery_suffix_stats6"].get(f"{Lto:.1f}")
            if a is not None and bb is not None:
                step_vals.append(bb - a)
        step_vals = np.array(step_vals)
        step_per_protein = dict(
            from_ns=Lfrom, to_ns=Lto, n_proteins=int(len(step_vals)),
            median=float(np.median(step_vals)), q25=float(np.percentile(step_vals, 25)),
            q75=float(np.percentile(step_vals, 75)),
            min=float(step_vals.min()), max=float(step_vals.max()),
            frac_positive=float(np.mean(step_vals > 0)),
            frac_ge_0p1=float(np.mean(step_vals >= 0.1)))

        readings = dict(
            min_length_recovery_0_8=dict(pooled=min_length(COND6, "pooled", 0.8), macro=min_length(COND6, "macro", 0.8)),
            min_length_recovery_0_9=dict(pooled=min_length(COND6, "pooled", 0.9), macro=min_length(COND6, "macro", 0.9)),
            largest_step=dict(pooled=step, macro=largest_step(COND6, "macro")),
            step_per_protein=step_per_protein)

        lref = L_READINGS[str(hns)]
        block = dict(
            horizon_ns=hns,
            references=dict(static_auprc=REFS[str(hns)]["static"], full_auprc=REFS[str(hns)]["full"],
                            mean_recovery_at_1_0ns=REFS[str(hns)]["mean_rec_1_0"],
                            source="L63-L68 records (K15 / K2-135 values and the mean-only recovery remeasured there)"),
            metrics=results,
            base_gain=dict(pooled=float(gain_pooled), macro=float(gain_macro)),
            recovery_vs_full_history_gain=recovery,
            **{"recovery_increment_per_0.1ns": dict(stats6=inc6, mean=incmean)},
            readings=readings,
            comparison_stats6_vs_mean_only=comparison,
            comparison_vs_L=dict(
                recorded_readings=lref,
                min_length_0_8_L_pooled=lref["min08"], min_length_0_9_L_pooled=lref["min09"],
                min_length_0_8_L_macro=lref["min08_macro"], min_length_0_9_L_macro=lref["min09_macro"],
                min_length_0_8_L2_pooled=readings["min_length_recovery_0_8"]["pooled"],
                min_length_0_9_L2_pooled=readings["min_length_recovery_0_9"]["pooled"],
                min_length_0_8_L2_macro=readings["min_length_recovery_0_8"]["macro"],
                min_length_0_9_L2_macro=readings["min_length_recovery_0_9"]["macro"]),
            bootstrap=dict(
                n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
                resampling_unit="protein (30 test proteins, resampled with replacement)",
                note="Bootstrap intervals are computed for the 6-statistics curve. Within each resample, static-only, full-history and suffix-stats6 AUPRCs are recomputed from the saved per-candidate predictions and the gain and recovery are recomputed inside the sample. Models are not refit. Pooled AUPRC uses a numpy implementation of sklearn average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every bootstrapped condition.",
                n_draws_with_nonpositive_gain=n_bad, results=bootstrap),
            per_protein=per_protein,
            per_protein_summary=pp_summary,
            per_protein_increment_summary=pp_inc_summary)
        blocks[str(hns)] = block
        all_preds[hns] = preds
        done[str(hns)] = dict(block=block,
                             preds={c: preds[c].astype(np.float32).tolist() for c in CONDITIONS})
        CKPT.write_text(json.dumps(sanitize(done), ensure_ascii=False))
        print(f"{hns}ns: block saved to checkpoint ({time.time() - t0:.0f}s)", flush=True)
        print(f"{hns}ns readings: min08 pooled {readings['min_length_recovery_0_8']['pooled']} "
              f"macro {readings['min_length_recovery_0_8']['macro']}, "
              f"min09 pooled {readings['min_length_recovery_0_9']['pooled']} "
              f"macro {readings['min_length_recovery_0_9']['macro']}, "
              f"largest {step['from_ns']}->{step['to_ns']} +{step['increment']:.4f}", flush=True)

    out = {
        "label": "Salt formation (K15 base), guidance recovery vs short-MD length, window placed immediately before the origin frame, 6 statistics (L2 re-measurement of L63-L68)",
        "experiment": "L2-13",
        "phenomenon": "salt-bridge formation (K15 base, k15_cache)",
        "length_definition": {
            "unit": "covered length (ns); window = the last n_frames of the 0-1 ns history, ending at the origin frame 10 (E01 Protocol A placement)",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=NF[L], frames=FRAMES[L]) for L in LENGTHS},
        },
        "base": "K15 base: train 58 / test 30 proteins (split.json orig + additions), static 5 features, HistGBM (120 iters, lr 0.075, 31 leaves, min_samples_leaf 15, L2 0.02, seed 42) retrained per condition; history features per covered length = 6 statistics of the window (first frame of the window, mean, std, min, frame 10 minus first frame of the window, fraction <= 4.0 A); full_history = static + the same 6 statistics over all 11 frames; recovery denominator = full-history gain, as in L63-L68",
        "premise_checks": "see premise_check.json (all_ok required before the main phase)",
        "horizons": blocks,
    }
    (OUT / "results.json").write_text(json.dumps(sanitize(out), indent=2, ensure_ascii=False))

    lines = ["horizon_ns,covered_ns,n_frames,"
             "stats6_auprc_pooled,stats6_auprc_macro,stats6_recovery_pooled,stats6_recovery_macro,"
             "stats6_recovery_pooled_ci_lower,stats6_recovery_pooled_ci_upper,stats6_recovery_macro_ci_lower,stats6_recovery_macro_ci_upper,"
             "stats6_increment_pooled,stats6_increment_macro,"
             "mean_auprc_pooled,mean_auprc_macro,mean_recovery_pooled,mean_recovery_macro,mean_increment_pooled,mean_increment_macro"]
    for hns, _ in HORIZONS:
        blk = blocks[str(hns)]
        res = blk["metrics"]
        recov = blk["recovery_vs_full_history_gain"]
        inc6 = blk["recovery_increment_per_0.1ns"]["stats6"]
        incmean = blk["recovery_increment_per_0.1ns"]["mean"]
        boot = blk["bootstrap"]["results"]
        for L in LENGTHS:
            c6 = COND6[L]
            cm = CONDMEAN[L]
            m6 = res[c6]
            mm = res[cm]
            lines.append(",".join([
                str(hns), f"{L:.1f}", str(NF[L]),
                f"{m6['auprc']:.6f}", f"{m6['macro_auprc']:.6f}",
                f"{recov[c6]['pooled']:.6f}", f"{recov[c6]['macro']:.6f}",
                f"{boot['pooled'][f'{L:.1f}']['ci_lower']:.6f}",
                f"{boot['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
                f"{boot['macro'][f'{L:.1f}']['ci_lower']:.6f}",
                f"{boot['macro'][f'{L:.1f}']['ci_upper']:.6f}",
                f"{inc6['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
                f"{inc6['macro'].get(f'{L:.1f}', float('nan')):.6f}",
                f"{mm['auprc']:.6f}", f"{mm['macro_auprc']:.6f}",
                f"{recov[cm]['pooled']:.6f}", f"{recov[cm]['macro']:.6f}",
                f"{incmean['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
                f"{incmean['macro'].get(f'{L:.1f}', float('nan')):.6f}"]))
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    plines = ["horizon_ns,protein,n,positives,auprc_static,auprc_full,full_gain,"
              + ",".join(f"stats6_auprc_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"stats6_recovery_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"mean_auprc_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"mean_recovery_{L:.1f}ns" for L in LENGTHS)]
    for hns, _ in HORIZONS:
        for p, entry in blocks[str(hns)]["per_protein"].items():
            row = [str(hns), p, str(entry["n"]), str(entry["positives"]),
                   "" if entry["auprc_static"] is None else f"{entry['auprc_static']:.6f}",
                   "" if entry["auprc_full"] is None else f"{entry['auprc_full']:.6f}",
                   "" if entry["full_gain"] is None else f"{entry['full_gain']:.6f}"]
            row += ["" if entry["auprc_suffix_stats6"][f"{L:.1f}"] is None else f"{entry['auprc_suffix_stats6'][f'{L:.1f}']:.6f}" for L in LENGTHS]
            row += ["" if entry["recovery_suffix_stats6"][f"{L:.1f}"] is None else f"{entry['recovery_suffix_stats6'][f'{L:.1f}']:.6f}" for L in LENGTHS]
            row += ["" if entry["auprc_suffix_mean"][f"{L:.1f}"] is None else f"{entry['auprc_suffix_mean'][f'{L:.1f}']:.6f}" for L in LENGTHS]
            row += ["" if entry["recovery_suffix_mean"][f"{L:.1f}"] is None else f"{entry['recovery_suffix_mean'][f'{L:.1f}']:.6f}" for L in LENGTHS]
            plines.append(",".join(row))
    (OUT / "per_protein.csv").write_text("\n".join(plines) + "\n")

    npz = dict(proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]))
    for hns, hidx in HORIZONS:
        npz[f"h{hns}_label"] = Yte[:, hidx].astype(np.int8)
        for c in CONDITIONS:
            npz[f"h{hns}_{c}"] = all_preds[hns][c].astype(np.float32)
    np.savez_compressed(OUT / "test_predictions.npz", **npz)
    CKPT.unlink(missing_ok=True)
    print(f"saved ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
