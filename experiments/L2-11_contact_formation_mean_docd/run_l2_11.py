"""L2-11: L15-L20 (contact formation, DocD, mean only) re-measured with the
history window placed immediately before the origin frame (frames
(10 - L/0.1)..10), all six horizons in one run.

Base, candidates, learner and readings are identical to L15-L20; only the
window placement changes (L used frames 0..k). At length 1.0 ns the window
is frames 0..10, identical to L, so the static-only / full / mean-at-1.0ns
AUPRCs must reproduce the L15-L20 records (premise check, tol 5e-4);
the run stops before the main phase if any horizon fails.

Recovery = (AUPRC at length - static only) / (full-history gain),
the full model being static + 6 statistics over frames 0..10 (as in L).
"""
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
FILE_SUFFIX = "contact"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = [(1, 0), (5, 1), (10, 2), (30, 3), (50, 4), (90, 5)]
BOUND = 4.5
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
NF = {L: int(round(L / 0.1)) + 1 for L in LENGTHS}
FRAMES = {L: list(range(11 - NF[L], 11)) for L in LENGTHS}

REFS = {'1': {'static': 0.33977026863870574, 'full': 0.5668799691676633, 'mean_full': 0.5008301534366115, 'positives': 219, 'n_test': 7384}, '5': {'static': 0.2869855867812501, 'full': 0.5191989452346799, 'mean_full': 0.4478561990956109, 'positives': 214, 'n_test': 7384}, '10': {'static': 0.320384938464356, 'full': 0.5233798114377418, 'mean_full': 0.45162809820314465, 'positives': 226, 'n_test': 7384}, '30': {'static': 0.28681357061495694, 'full': 0.4386046069734208, 'mean_full': 0.4371572874384626, 'positives': 247, 'n_test': 7384}, '50': {'static': 0.26038453431471853, 'full': 0.46379209531119275, 'mean_full': 0.3935261394770647, 'positives': 250, 'n_test': 7384}, '90': {'static': 0.2872126453371935, 'full': 0.47216777524969433, 'mean_full': 0.4065937715486499, 'positives': 257, 'n_test': 7384}}
L_READINGS = {'1': {'min08': 0.9, 'min09': 0.9, 'largest_step': [0.4, 0.5, 0.16190828057499862], 'recovery_curve': {'0.0': 0.42879569341995666, '0.1': 0.4322888542323157, '0.2': 0.5120239278692977, '0.3': 0.5894011010657195, '0.4': 0.6115298532505383, '0.5': 0.7734381338255369, '0.6': 0.676952068044254, '0.7': 0.7826584861502195, '0.8': 0.7617761955088159, '0.9': 0.9149619279306862, '1.0': 0.7091721948590651}}, '5': {'min08': 0.8, 'min09': None, 'largest_step': [0.0, 0.1, 0.1533527071824744], 'recovery_curve': {'0.0': 0.4875793130448822, '0.1': 0.6409320202273566, '0.2': 0.7842246482790599, '0.3': 0.7712630692009723, '0.4': 0.782001894562789, '0.5': 0.7674718377851363, '0.6': 0.726436617489972, '0.7': 0.7817402840193677, '0.8': 0.8491141736306982, '0.9': 0.7841313190930616, '1.0': 0.6927707061548024}}, '10': {'min08': None, 'min09': None, 'largest_step': [0.1, 0.2, 0.13916133948636233], 'recovery_curve': {'0.0': 0.4669235428940985, '0.1': 0.43395423983701764, '0.2': 0.57311557932338, '0.3': 0.5779545458292918, '0.4': 0.684317258711813, '0.5': 0.7272192629050488, '0.6': 0.7005894393423735, '0.7': 0.5885112787511427, '0.8': 0.6614952428954306, '0.9': 0.6413186890035943, '1.0': 0.6465343573283037}}, '30': {'min08': 0.2, 'min09': 0.4, 'largest_step': [0.1, 0.2, 0.2611122273209372], 'recovery_curve': {'0.0': 0.4995925846108251, '0.1': 0.577910831639767, '0.2': 0.8390230589607042, '0.3': 0.8635523721259094, '0.4': 0.9791672855068535, '0.5': 1.0075907364858951, '0.6': 0.9463211736542483, '0.7': 0.8975125101035644, '0.8': 1.0543904749682922, '0.9': 1.0785421204035643, '1.0': 0.9904650526824241}}, '50': {'min08': None, 'min09': None, 'largest_step': [0.1, 0.2, 0.24518698683994777], 'recovery_curve': {'0.0': 0.7469674514385245, '0.1': 0.5009540709019091, '0.2': 0.7461410577418569, '0.3': 0.6951111743681312, '0.4': 0.7336808777434309, '0.5': 0.633301517987569, '0.6': 0.5754166065291316, '0.7': 0.6288566011340294, '0.8': 0.6739416382307953, '0.9': 0.7480554184500102, '1.0': 0.6545558312095094}}, '90': {'min08': None, 'min09': None, 'largest_step': [0.1, 0.2, 0.1784901179374807], 'recovery_curve': {'0.0': 0.5016120633000837, '0.1': 0.5018826802930115, '0.2': 0.6803727982304922, '0.3': 0.643833764481737, '0.4': 0.6795730767271367, '0.5': 0.6651057807833809, '0.6': 0.6819534303390224, '0.7': 0.6266134543371418, '0.8': 0.6385381515940136, '0.9': 0.7379443195006, '1.0': 0.6454599354337109}}}


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_{FILE_SUFFIX}.npz")
            d = {k: z[k] for k in z.files}
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
    return np.stack(
        [series[:, 0], series.mean(1), series.std(1), series.min(1),
         series[:, -1] - series[:, 0], (series <= bound).mean(1)],
        axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def auprc(y, p):
    return float(average_precision_score(y, p))


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


def main():
    t0 = time.time()
    train, _ = load(TRAIN_P)
    test, test_protein = load(TEST_P)
    Xtr, Xte = cat(train, "X"), cat(test, "X")
    Htr, Hte = cat(train, "hist"), cat(test, "hist")
    Ytr, Yte = cat(train, "y"), cat(test, "y")
    prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
    print("train", Xtr.shape, "test", Xte.shape, flush=True)

    f6tr, f6te = hist6(Htr), hist6(Hte)
    mean_full_tr = Htr.mean(1)[:, None].astype(np.float32)
    mean_full_te = Hte.mean(1)[:, None].astype(np.float32)

    # ---------- premise phase: static / full / mean at 1.0 ns, all horizons ----------
    checks = []
    stored = {}
    for hns, hidx in HORIZONS:
        ytr, yte = Ytr[:, hidx], Yte[:, hidx]
        p_static = predict(Xtr, ytr, Xte)
        p_full = predict(np.concatenate([Xtr, f6tr], axis=1), ytr,
                         np.concatenate([Xte, f6te], axis=1))
        p_mean10 = predict(np.concatenate([Xtr, mean_full_tr], axis=1), ytr,
                           np.concatenate([Xte, mean_full_te], axis=1))
        stored[hns] = dict(ytr=ytr, yte=yte, static=p_static, full=p_full, mean10=p_mean10)
        ref = REFS[str(hns)]
        got = dict(n=int(len(yte)), positives=int(yte.sum()),
                   static=auprc(yte, p_static), full=auprc(yte, p_full),
                   mean_full=auprc(yte, p_mean10))
        for name, tol in (("static", 5e-4), ("full", 5e-4), ("mean_full", 5e-4)):
            ok = abs(got[name] - ref[name]) < tol
            checks.append(dict(horizon_ns=hns, check=f"{name}_auprc", got=got[name],
                               reference=ref[name], tol=tol, ok=bool(ok)))
            print(f"premise {hns}ns {name}: got {got[name]:.6f} ref {ref[name]:.6f} "
                  f"-> {'OK' if ok else 'FAIL'}", flush=True)
        for name in ("n", "positives"):
            ok = got[name] == ref[name if name == "positives" else "n_test"]
            checks.append(dict(horizon_ns=hns, check=name, got=got[name],
                               reference=ref[name if name == "positives" else "n_test"],
                               tol=0, ok=bool(ok)))
    all_ok = all(c["ok"] for c in checks)
    (OUT / "premise_check.json").write_text(json.dumps(
        dict(checks=checks, all_ok=bool(all_ok),
             note="Length 1.0 ns uses frames 0..10, identical to L15-L20; "
                  "static/full/mean-at-1.0ns AUPRCs must match the L records."),
        indent=2, ensure_ascii=False))
    if not all_ok:
        print("PREMISE FAILED - stopping before the main phase", flush=True)
        sys.exit(2)
    print(f"premise OK ({time.time() - t0:.0f}s)", flush=True)

    # ---------- main phase ----------
    out_horizons = {}
    all_preds = {}
    for hns, hidx in HORIZONS:
        st = stored[hns]
        ytr, yte = st["ytr"], st["yte"]
        preds = {"static_only": st["static"], "full_hist6": st["full"]}
        results = {"static_only": metrics(yte, st["static"]),
                   "full_hist6": metrics(yte, st["full"])}
        for L in LENGTHS:
            nf = NF[L]
            key = f"suffix_mean_{L:.1f}ns"
            if L == 1.0:
                preds[key] = st["mean10"]
            else:
                mtr = Htr[:, 11 - nf:].mean(1)[:, None].astype(np.float32)
                mte = Hte[:, 11 - nf:].mean(1)[:, None].astype(np.float32)
                preds[key] = predict(np.concatenate([Xtr, mtr], axis=1), ytr,
                                     np.concatenate([Xte, mte], axis=1))
            results[key] = metrics(yte, preds[key])
        base_gain = results["full_hist6"]["auprc"] - results["static_only"]["auprc"]

        recovery = {}
        for L in LENGTHS:
            recovery[L] = float((results[f"suffix_mean_{L:.1f}ns"]["auprc"]
                                 - results["static_only"]["auprc"]) / base_gain)
        increments = {}
        prev = None
        for L in LENGTHS:
            if prev is not None:
                increments[L] = recovery[L] - recovery[prev]
            prev = L

        def min_length(thr):
            for L in LENGTHS:
                if recovery[L] >= thr:
                    return L
            return None

        largest_L = max(increments, key=lambda L: increments[L])
        largest = dict(from_ns=round(largest_L - 0.1, 1), to_ns=largest_L,
                       increment=increments[largest_L])

        per_protein = {}
        for L in LENGTHS:
            key = f"suffix_mean_{L:.1f}ns"
            row = {}
            for prot in TEST_P:
                idx = prot_idx[prot]
                a_s = auprc(yte[idx], preds["static_only"][idx])
                a_f = auprc(yte[idx], preds["full_hist6"][idx])
                a_p = auprc(yte[idx], preds[key][idx])
                g = a_f - a_s
                row[prot] = dict(n=int(len(idx)), positives=int(yte[idx].sum()),
                                 auprc_static=a_s, auprc_full=a_f, auprc_suffix=a_p,
                                 full_gain=g,
                                 recovery=(a_p - a_s) / g if g else float("nan"))
            per_protein[L] = row
        intervals = {}
        for L in LENGTHS:
            three = [recovery[L]] + [per_protein[L][p]["recovery"] for p in TEST_P]
            intervals[L] = dict(pooled=recovery[L],
                                per_protein={p: per_protein[L][p]["recovery"] for p in TEST_P},
                                ci_lower=float(min(three)), ci_upper=float(max(three)))
        step_pp = {}
        for prot in TEST_P:
            a = per_protein[largest["from_ns"]][prot]["recovery"]
            b = per_protein[largest["to_ns"]][prot]["recovery"]
            step_pp[prot] = dict(before=a, after=b, increment=b - a)

        lref = L_READINGS[str(hns)]
        out_horizons[str(hns)] = dict(
            horizon_ns=hns,
            metrics=results,
            base_gain=base_gain,
            recovery={f"{L:.1f}": recovery[L] for L in LENGTHS},
            increments={f"{L:.1f}": increments[L] for L in sorted(increments)},
            readings=dict(
                min_length_recovery_0_8_pooled=min_length(0.8),
                min_length_recovery_0_9_pooled=min_length(0.9),
                largest_increment=largest,
                step_per_protein=step_pp,
                intervals_3value_min_max={f"{L:.1f}": intervals[L] for L in LENGTHS}),
            per_protein={f"{L:.1f}": per_protein[L] for L in LENGTHS},
            comparison_vs_L=dict(
                recorded=L_READINGS[str(hns)],
                min_length_0_8_L=lref["min08"], min_length_0_9_L=lref["min09"],
                min_length_0_8_L2=min_length(0.8), min_length_0_9_L2=min_length(0.9)))
        for cname, p in preds.items():
            all_preds[f"h{hns}_{cname}"] = p.astype(np.float32)
        all_preds[f"h{hns}_y"] = yte.astype(np.int8)
        print(f"{hns}ns: recovery " + " ".join(f"{recovery[L]:.3f}" for L in LENGTHS),
              f"| min08 {min_length(0.8)} min09 {min_length(0.9)} "
              f"largest {largest['from_ns']}->{largest['to_ns']} +{largest['increment']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = {
        "label": "Contact formation (DocD), guidance recovery vs short-MD length, window placed immediately before the origin frame (L2 re-measurement of L15-L20)",
        "experiment": "L2-11",
        "phenomenon": "contact formation (Doc D, k13_cache)",
        "length_definition": {
            "unit": "covered length (ns); window = the last n_frames of the 0-1 ns history, ending at the origin frame 10 (E01 Protocol A placement)",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=NF[L], frames=FRAMES[L])
                        for L in LENGTHS},
        },
        "base": "DocD contact formation: train 1bgf_A/1ifg_A/1j8r_A/2b0a_A, test 1mix_A+2qt4_A, static 5 features, HistGBM (120 iters, lr 0.075, 31 leaves, min_samples_leaf 15, L2 0.02, seed 42) retrained per condition; full = static + 6 statistics over frames 0..10 (bound 4.5 A); recovery denominator = full-history gain, as in L15-L20",
        "premise_checks": "see premise_check.json (all_ok required before the main phase)",
        "horizons": out_horizons,
    }
    (OUT / "results.json").write_text(json.dumps(sanitize(out), indent=2, ensure_ascii=False))

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_ns", "covered_ns", "n_frames", "frames", "test_positives",
                    "auprc_static", "auprc_full", "auprc_suffix_mean", "auc_suffix_mean",
                    "brier_suffix_mean", "logloss_suffix_mean", "full_gain",
                    "recovery_pooled", "recovery_1mix_A", "recovery_2qt4_A",
                    "interval_lower", "interval_upper", "increment_from_previous_0.1ns"])
        for hns, _ in HORIZONS:
            blk = out_horizons[str(hns)]
            for L in LENGTHS:
                m = blk["metrics"][f"suffix_mean_{L:.1f}ns"]
                iv = blk["readings"]["intervals_3value_min_max"][f"{L:.1f}"]
                w.writerow([hns, L, NF[L], f"{FRAMES[L][0]}-{FRAMES[L][-1]}",
                            m["positives"], blk["metrics"]["static_only"]["auprc"],
                            blk["metrics"]["full_hist6"]["auprc"], m["auprc"], m["auc"],
                            m["brier"], m["logloss"], blk["base_gain"],
                            blk["recovery"][f"{L:.1f}"],
                            blk["per_protein"][f"{L:.1f}"]["1mix_A"]["recovery"],
                            blk["per_protein"][f"{L:.1f}"]["2qt4_A"]["recovery"],
                            iv["ci_lower"], iv["ci_upper"],
                            blk["increments"].get(f"{L:.1f}", "")])
    np.savez_compressed(OUT / "test_predictions.npz", test_protein=test_protein, **all_preds)
    print(f"saved ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
