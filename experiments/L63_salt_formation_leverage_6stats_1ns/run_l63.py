"""L63: L01/L02 と同じ曲がりの測り方を統計量6種で測る (塩橋の形成、K15 の土台、1 ns先)。

L27〜L38 (平均だけ) の続き。覆う長さ 0〜1.0 ns (0.1 ns 刻み、11通り) で、
切り取り (フレーム 0〜(L/0.1)) から計算した統計量6種 (最初、平均、標準偏差、
最小、最後−最初、4.0 Å以下割合) を履歴の特徴量としたときの利得の
回復率を測る。同じ実行の中で平均だけの曲がりも測り直し、対応する
L27〜L38 の実験 (L27) の記録と一致するか確かめる。

土台は K15 の条件: 学習58・テスト30タンパク質、静的特徴5個、
HistGradientBoosting (120回、学習率 0.075、葉 31、葉あたり最小 15、L2 0.02、
種 42) を条件ごとに学習し直す。全部入りは静的＋履歴6統計量 (11フレーム全部)。
覆う長さ 1.0 ns の統計量6種の条件は全部入りと同一の特徴量になるため、
回復率は 1.0 になるはずで、そのことも確認する。

回復率 = (その長さの AUPRC − 静的のみ) / (全部入りの利得)。
全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る。

読み方 (結果を見る前に固定):
1. 回復率 0.8 と 0.9 に届く最小の覆う長さを統合値 (pooled) で読む (macro も併記)。
2. 段の判定は、隣り合う長さの間の増え幅がいちばん大きい場所と大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、パーセンタイル法)。
   区間は統計量6種の曲がりについて計算する (平均だけの区間は L27 の記録にある)。
4. 段の前後ではタンパク質ごとの値も記録する。
あわせて、平均だけの曲がり (同じ実行で測り直したもの) と比べて、
0.8 と 0.9 に必要な覆う長さが縮んだかどうかを記録する。

前提の確認 (premise): 静的のみ・全部入りの AUPRC が既存の記録
(K15 results.json / K2-135 (1 ns: static 0.2402, full 0.3842)) と 5e-4 以内で一致すること、および測り直した平均だけの
回復率 (全体) が L27 の results.json の値と 5e-4 以内で一致すること。
合わなければ止める。
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
SEED = 42
BOOT_SEED = 42
N_BOOT = 10000
HORIZON_IDX = 0
HORIZON_NS = 1
BOUND = 4.0
REF_STATIC = 0.2402
REF_FULL = 0.3842
REF_POSITIVES = 93
REF_N_TRAIN = 2557
REF_N_TEST = 1170
REF_MEAN_RECOVERY_POOLED = [0.571222607685573, 0.6466910664079819, 0.6156060837255383, 0.9627127965196343, 0.5450626973121004, 0.6594661892907897, 0.663107271907787, 0.9013518628670175, 0.8519770640029041, 0.9720738430978302, 0.8890425378317479]

LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(int(round(L / 0.1)) + 1)) for L in LENGTHS}
COND6 = {L: f"prefix_stats6_{L:.1f}ns" for L in LENGTHS}
CONDMEAN = {L: f"prefix_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = (["static_only", "full_history"] + [COND6[L] for L in LENGTHS]
              + [CONDMEAN[L] for L in LENGTHS])
BOOT_CONDITIONS = ["static_only", "full_history"] + [COND6[L] for L in LENGTHS]

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


def main():
    t0 = time.time()
    Xtr, Htr, Ytr, _ = load(TRAIN_P)
    Xte, Hte, Yte, spans = load(TEST_P)
    y_tr = Ytr[:, HORIZON_IDX]
    y_te = Yte[:, HORIZON_IDX]
    print("train", len(y_tr), "test", len(y_te), "test pos", int(y_te.sum()), flush=True)
    assert len(y_tr) == REF_N_TRAIN and len(y_te) == REF_N_TEST, (len(y_tr), len(y_te))
    assert int(y_te.sum()) == REF_POSITIVES, int(y_te.sum())
    prot_of_row = np.concatenate([np.full(e - s, i) for i, (_, s, e) in enumerate(spans)])

    feats = {}
    feats["static_only"] = (Xtr, Xte)
    feats["full_history"] = (np.column_stack([Xtr, hist6(Htr)]),
                             np.column_stack([Xte, hist6(Hte)]))
    for L in LENGTHS:
        k = int(round(L / 0.1)) + 1
        feats[COND6[L]] = (
            np.column_stack([Xtr, hist6(Htr[:, :k])]),
            np.column_stack([Xte, hist6(Hte[:, :k])]))
        feats[CONDMEAN[L]] = (
            np.column_stack([Xtr, Htr[:, :k].mean(1, keepdims=True).astype(np.float32)]),
            np.column_stack([Xte, Hte[:, :k].mean(1, keepdims=True).astype(np.float32)]))

    preds, results = {}, {}
    per_protein_ap = {c: {} for c in CONDITIONS}
    for cname in CONDITIONS:
        xtr, xte = feats[cname]
        pred = learner(xtr, y_tr).predict_proba(xte)[:, 1]
        preds[cname] = pred
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
        print(cname, "pooled %.4f macro %s (%.0fs)" % (
            m["auprc"], ("%.4f" % m["macro_auprc"]) if m["macro_auprc"] is not None else "NA",
            time.time() - t0), flush=True)

    # ---- premise: base reproduction + mean-only cross-check ----
    checks = []

    def chk(name, got, ref, tol):
        ok = abs(got - ref) < tol
        checks.append(dict(check=name, got=float(got), reference=float(ref), tol=tol, ok=bool(ok)))
        print("check", name, "got %.6f ref %.6f tol %g -> %s" % (got, ref, tol, ok), flush=True)
        return ok

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]
    ok = True
    ok &= chk("static_pooled", results["static_only"]["auprc"], REF_STATIC, 5e-4)
    ok &= chk("full_pooled", results["full_history"]["auprc"], REF_FULL, 5e-4)

    def rec(cname, kind):
        if kind == "pooled":
            return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

    mean_recovery_remeasured = [float(rec(CONDMEAN[L], "pooled")) for L in LENGTHS]
    max_mean_diff = 0.0
    for i, L in enumerate(LENGTHS):
        d = abs(mean_recovery_remeasured[i] - REF_MEAN_RECOVERY_POOLED[i])
        max_mean_diff = max(max_mean_diff, d)
        ok &= chk(f"mean_recovery_{L:.1f}ns_vs_L27",
                  mean_recovery_remeasured[i], REF_MEAN_RECOVERY_POOLED[i], 5e-4)
    rec6_at_full = float(rec(COND6[1.0], "pooled"))
    ok &= chk("stats6_recovery_at_1.0ns_is_1", rec6_at_full, 1.0, 1e-6)
    (OUT / "premise_check.json").write_text(json.dumps(
        dict(checks=checks, all_ok=bool(ok),
             mean_recovery_max_abs_diff=float(max_mean_diff)), indent=2, ensure_ascii=False))
    if not ok:
        print("PREMISE FAILED", flush=True)
        sys.exit(2)
    print("premise OK (%.0fs), mean cross-check max diff %.2e" % (time.time() - t0, max_mean_diff), flush=True)

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

    # ---- per-protein (stats6 primary, mean also recorded) ----
    per_protein = {}
    for p, s, e in spans:
        yy = y_te[s:e]
        entry = dict(n=int(e - s), positives=int(yy.sum()))
        ap_s = per_protein_ap["static_only"][p]
        ap_f = per_protein_ap["full_history"][p]
        entry["auprc_static"] = ap_s
        entry["auprc_full"] = ap_f
        entry["full_gain"] = (ap_f - ap_s) if (ap_s is not None and ap_f is not None) else None
        entry["auprc_prefix_stats6"] = {}
        entry["recovery_prefix_stats6"] = {}
        entry["auprc_prefix_mean"] = {}
        entry["recovery_prefix_mean"] = {}
        for L in LENGTHS:
            ap6 = per_protein_ap[COND6[L]][p]
            apm = per_protein_ap[CONDMEAN[L]][p]
            entry["auprc_prefix_stats6"][f"{L:.1f}"] = ap6
            entry["auprc_prefix_mean"][f"{L:.1f}"] = apm
            if ap6 is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                entry["recovery_prefix_stats6"][f"{L:.1f}"] = float((ap6 - ap_s) / entry["full_gain"])
            else:
                entry["recovery_prefix_stats6"][f"{L:.1f}"] = None
            if apm is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                entry["recovery_prefix_mean"][f"{L:.1f}"] = float((apm - ap_s) / entry["full_gain"])
            else:
                entry["recovery_prefix_mean"][f"{L:.1f}"] = None
        per_protein[p] = entry

    pp_summary = {}
    valid_pp = [p for p, en in per_protein.items() if en["recovery_prefix_stats6"]["1.0"] is not None]
    for L in LENGTHS:
        vals = np.array([per_protein[p]["recovery_prefix_stats6"][f"{L:.1f}"] for p in valid_pp])
        pp_summary[f"{L:.1f}"] = dict(
            n_proteins=int(len(vals)), median=float(np.median(vals)),
            q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)),
            frac_ge_0_8=float(np.mean(vals >= 0.8)))
    pp_inc_summary = {}
    for i, L in enumerate(LENGTHS[1:], start=1):
        Lp = LENGTHS[i - 1]
        vals = []
        for p in valid_pp:
            a = per_protein[p]["recovery_prefix_stats6"][f"{Lp:.1f}"]
            b = per_protein[p]["recovery_prefix_stats6"][f"{L:.1f}"]
            vals.append(b - a)
        vals = np.array(vals)
        pp_inc_summary[f"{L:.1f}"] = dict(
            median=float(np.median(vals)), q25=float(np.percentile(vals, 25)),
            q75=float(np.percentile(vals, 75)), frac_positive=float(np.mean(vals > 0)))

    # ---- bootstrap (stats6 curve) ----
    for cname in BOOT_CONDITIONS:
        ap_np = ap_numpy(y_te, preds[cname].astype(np.float64))
        assert abs(ap_np - results[cname]["auprc"]) < 1e-9, (cname, ap_np, results[cname]["auprc"])
    print("numpy AP verified vs sklearn", flush=True)

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
            print("bootstrap %d/%d (%.0fs)" % (b + 1, N_BOOT, time.time() - t0), flush=True)

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
    print("bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad), flush=True)

    step = largest_step(COND6, "pooled")
    Lfrom, Lto = step["from_ns"], step["to_ns"]
    step_vals = []
    for p, entry in per_protein.items():
        a = entry["recovery_prefix_stats6"].get(f"{Lfrom:.1f}")
        bb = entry["recovery_prefix_stats6"].get(f"{Lto:.1f}")
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

    out = {
        "label": "salt formation 1ns, guidance recovery vs short-MD covered length with 6 prefix statistics, K15 base",
        "experiment": "L63",
        "horizon_ns": HORIZON_NS,
        "length_definition": {
            "unit": "covered length (ns), same definition as L01/L02; prefix = frames 0..(L/0.1) of the 0-1 ns history, 100 ps spacing",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=len(FRAMES[L]), frames=FRAMES[L]) for L in LENGTHS},
        },
        "base": {
            "name": "K15 base: train 58 / test 30 proteins, static 5 features, HistGBM (120 iters, lr 0.075, 31 leaves, min_samples_leaf 15, L2 0.02, seed 42) retrained per condition; history features per covered length = 6 statistics of the prefix (first, mean, std, min, last-first, fraction <= 4.0 A); full_history = static + the same 6 statistics over all 11 frames",
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "test_positives": int(y_te.sum()),
        },
        "premise_checks": checks,
        "mean_only_crosscheck": dict(
            reference_experiment="L27",
            remeasured_recovery_pooled=mean_recovery_remeasured,
            recorded_recovery_pooled=REF_MEAN_RECOVERY_POOLED,
            max_abs_diff=float(max_mean_diff), tol=5e-4),
        "references": dict(static_auprc=REF_STATIC, full_auprc=REF_FULL,
                           gain=REF_FULL - REF_STATIC, source="K15 results.json / K2-135 (1 ns: static 0.2402, full 0.3842)"),
        "metrics": results,
        "base_gain": dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        "recovery_vs_full_history_gain": recovery,
        "recovery_increment_per_0.1ns": dict(stats6=inc6, mean=incmean),
        "readings": readings,
        "comparison_stats6_vs_mean_only": comparison,
        "bootstrap": dict(
            n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
            resampling_unit="protein (30 test proteins, resampled with replacement)",
            note="Bootstrap intervals are computed for the 6-statistics curve. Within each resample, static-only, full-history and prefix-stats6 AUPRCs are recomputed from the saved per-candidate predictions and the gain and recovery are recomputed inside the sample. Models are not refit. Pooled AUPRC uses a numpy implementation of sklearn average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every bootstrapped condition. Intervals for the mean-only curve are in L27.",
            n_draws_with_nonpositive_gain=n_bad, results=bootstrap),
        "per_protein": per_protein,
        "per_protein_summary": pp_summary,
        "per_protein_increment_summary": pp_inc_summary,
    }
    (OUT / "results.json").write_text(json.dumps(sanitize(out), indent=2, ensure_ascii=False))

    lines = ["covered_ns,n_frames,"
             "stats6_auprc_pooled,stats6_auprc_macro,stats6_recovery_pooled,stats6_recovery_macro,"
             "stats6_recovery_pooled_ci_lower,stats6_recovery_pooled_ci_upper,stats6_recovery_macro_ci_lower,stats6_recovery_macro_ci_upper,"
             "stats6_increment_pooled,stats6_increment_macro,"
             "mean_auprc_pooled,mean_auprc_macro,mean_recovery_pooled,mean_recovery_macro,mean_increment_pooled,mean_increment_macro"]
    for L in LENGTHS:
        c6 = COND6[L]
        cm = CONDMEAN[L]
        m6 = results[c6]
        mm = results[cm]
        lines.append(",".join([
            f"{L:.1f}", str(len(FRAMES[L])),
            f"{m6['auprc']:.6f}", f"{m6['macro_auprc']:.6f}",
            f"{recovery[c6]['pooled']:.6f}", f"{recovery[c6]['macro']:.6f}",
            f"{bootstrap['pooled'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{bootstrap['macro'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['macro'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{inc6['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{inc6['macro'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{mm['auprc']:.6f}", f"{mm['macro_auprc']:.6f}",
            f"{recovery[cm]['pooled']:.6f}", f"{recovery[cm]['macro']:.6f}",
            f"{incmean['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{incmean['macro'].get(f'{L:.1f}', float('nan')):.6f}"]))
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    plines = ["protein,n,positives,auprc_static,auprc_full,full_gain,"
              + ",".join(f"stats6_auprc_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"stats6_recovery_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"mean_auprc_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"mean_recovery_{L:.1f}ns" for L in LENGTHS)]
    for p, entry in per_protein.items():
        row = [p, str(entry["n"]), str(entry["positives"]),
               "" if entry["auprc_static"] is None else f"{entry['auprc_static']:.6f}",
               "" if entry["auprc_full"] is None else f"{entry['auprc_full']:.6f}",
               "" if entry["full_gain"] is None else f"{entry['full_gain']:.6f}"]
        row += ["" if entry["auprc_prefix_stats6"][f"{L:.1f}"] is None else f"{entry['auprc_prefix_stats6'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        row += ["" if entry["recovery_prefix_stats6"][f"{L:.1f}"] is None else f"{entry['recovery_prefix_stats6'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        row += ["" if entry["auprc_prefix_mean"][f"{L:.1f}"] is None else f"{entry['auprc_prefix_mean'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        row += ["" if entry["recovery_prefix_mean"][f"{L:.1f}"] is None else f"{entry['recovery_prefix_mean'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        plines.append(",".join(row))
    (OUT / "per_protein.csv").write_text("\n".join(plines) + "\n")

    np.savez_compressed(
        OUT / "test_predictions.npz",
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]),
        label=y_te.astype(np.int8),
        **{c: preds[c].astype(np.float32) for c in CONDITIONS})
    print("saved (%.0fs total)" % (time.time() - t0), flush=True)
    print(json.dumps(readings, indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(comparison, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
