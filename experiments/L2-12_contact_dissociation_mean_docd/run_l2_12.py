"""L2-12: L21-L26 (contact dissociation, DocD, mean only) re-measured with the
history window placed immediately before the origin frame (frames
(10 - L/0.1)..10), all six horizons in one run.

Base, candidates, learner and readings are identical to L21-L26; only the
window placement changes (L used frames 0..k). At length 1.0 ns the window
is frames 0..10, identical to L, so the static-only / full / mean-at-1.0ns
AUPRCs must reproduce the L21-L26 records (premise check, tol 5e-4);
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

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
FILE_SUFFIX = "contact_diss"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = [(1, 0), (5, 1), (10, 2), (30, 3), (50, 4), (90, 5)]
BOUND = 4.5
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
NF = {L: int(round(L / 0.1)) + 1 for L in LENGTHS}
FRAMES = {L: list(range(11 - NF[L], 11)) for L in LENGTHS}

REFS = {'1': {'static': 0.3104787071230086, 'full': 0.5246376068451989, 'mean_full': 0.5035794140182117, 'positives': 207, 'n_test': 2915}, '5': {'static': 0.3022152865040793, 'full': 0.5528218101436452, 'mean_full': 0.5253145099519625, 'positives': 211, 'n_test': 2915}, '10': {'static': 0.27719749486883505, 'full': 0.4794242789590083, 'mean_full': 0.46304808390300317, 'positives': 228, 'n_test': 2915}, '30': {'static': 0.2756011407090635, 'full': 0.512394342139979, 'mean_full': 0.4714394928303689, 'positives': 246, 'n_test': 2915}, '50': {'static': 0.2927131222356843, 'full': 0.4533068909153429, 'mean_full': 0.45140623235577654, 'positives': 267, 'n_test': 2915}, '90': {'static': 0.28863705922548766, 'full': 0.4750163886541065, 'mean_full': 0.4404262521624885, 'positives': 294, 'n_test': 2915}}
L_READINGS = {'1': {'min08': 0.1, 'min09': 0.1, 'largest_step': [0.0, 0.1, 0.3723463028556572], 'recovery_curve': {'0.0': 0.5464644453798264, '0.1': 0.9188107482354836, '0.2': 0.90555732475862, '0.3': 0.9167338580375038, '0.4': 0.8647767783572929, '0.5': 0.9448371334780769, '0.6': 0.9739656653108495, '0.7': 1.013710293354126, '0.8': 0.9649251394966037, '0.9': 0.9149940255402376, '1.0': 0.9016702417956753}}, '5': {'min08': 0.1, 'min09': None, 'largest_step': [0.0, 0.1, 0.16110776386631864], 'recovery_curve': {'0.0': 0.6411825784914895, '0.1': 0.8022903423578082, '0.2': 0.6197282915519632, '0.3': 0.7249165405175404, '0.4': 0.7661270614002759, '0.5': 0.6773725263288088, '0.6': 0.8146724639357023, '0.7': 0.8638990153544781, '0.8': 0.8285434838243433, '0.9': 0.8441283093683526, '1.0': 0.8902370944211933}}, '10': {'min08': 0.5, 'min09': 0.6, 'largest_step': [0.8, 0.9, 0.1649379213294515], 'recovery_curve': {'0.0': 0.7218049708875812, '0.1': 0.5777857704999325, '0.2': 0.588377720507266, '0.3': 0.7123559734148683, '0.4': 0.728758150317707, '0.5': 0.8020609506070854, '0.6': 0.9256088153165569, '0.7': 0.9131877924661608, '0.8': 0.7860791453266002, '0.9': 0.9510170666560517, '1.0': 0.9190206424451522}}, '30': {'min08': 0.3, 'min09': None, 'largest_step': [0.0, 0.1, 0.06996486930284584], 'recovery_curve': {'0.0': 0.7263971818382361, '0.1': 0.7963620511410819, '0.2': 0.795308289784015, '0.3': 0.8060573165081821, '0.4': 0.8295124578688389, '0.5': 0.8274939396562517, '0.6': 0.8443754831464032, '0.7': 0.8946229089153388, '0.8': 0.8355652123119707, '0.9': 0.840027381878393, '1.0': 0.8270438126511892}}, '50': {'min08': 0.5, 'min09': 0.9, 'largest_step': [0.4, 0.5, 0.1422961430086067], 'recovery_curve': {'0.0': 0.6213801747296722, '0.1': 0.6117013525019642, '0.2': 0.6654178500891325, '0.3': 0.7067456261867938, '0.4': 0.6847437119802837, '0.5': 0.8270398549888904, '0.6': 0.7768628184690435, '0.7': 0.735089160250689, '0.8': 0.8681918816991328, '0.9': 0.9516587358040421, '1.0': 0.988164805053192}}, '90': {'min08': 0.1, 'min09': 0.1, 'largest_step': [0.0, 0.1, 0.14368778594590725], 'recovery_curve': {'0.0': 0.7909258386192057, '0.1': 0.934613624565113, '0.2': 0.8609737487111764, '0.3': 0.901902630412348, '0.4': 0.9917370333246973, '0.5': 0.9295589455768505, '0.6': 0.871742630734573, '0.7': 0.9712902635361592, '0.8': 0.8725183731717435, '0.9': 0.9166564617103847, '1.0': 0.814410017475325}}}


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
             note="Length 1.0 ns uses frames 0..10, identical to L21-L26; "
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
        "label": "Contact dissociation (DocD), guidance recovery vs short-MD length, window placed immediately before the origin frame (L2 re-measurement of L21-L26)",
        "experiment": "L2-12",
        "phenomenon": "contact formation (Doc D, k50_cache)",
        "length_definition": {
            "unit": "covered length (ns); window = the last n_frames of the 0-1 ns history, ending at the origin frame 10 (E01 Protocol A placement)",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=NF[L], frames=FRAMES[L])
                        for L in LENGTHS},
        },
        "base": "DocD contact dissociation: train 1bgf_A/1ifg_A/1j8r_A/2b0a_A, test 1mix_A+2qt4_A, static 5 features, HistGBM (120 iters, lr 0.075, 31 leaves, min_samples_leaf 15, L2 0.02, seed 42) retrained per condition; full = static + 6 statistics over frames 0..10 (bound 4.5 A); recovery denominator = full-history gain, as in L21-L26",
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
