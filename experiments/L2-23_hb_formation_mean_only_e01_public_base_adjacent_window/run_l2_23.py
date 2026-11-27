"""L2-23: E01 の Protocol A を全11長さに広げた測り直し (平均だけ)。

E01 (experiments/E01_true_short_md_guidance) は、公開特徴量20個 (static20) を
土台に、水素結合の形成について、起点フレーム10の直前に置いた区間の
平均だけの回復率を長さ 0.1 / 0.2 / 0.3 / 0.5 / 1.0 ns の5点で測った。
L2-23 では同じ土台・同じ候補・同じ評価方法 (タンパク質で分ける5分割の
交差検証、検証の外に出した予測で測る) のまま、長さを L 系列と同じ
0, 0.1, …, 1.0 ns の11点すべてで測る。

データ: E01 の抽出済みバンドルはこの計算機にないため、E01 の抽出スクリプト
(extract_e01_e02.py) の定義をそのまま写して ATLAS の解析版軌道から作り直した
(~/workspace/kinescope/work/e01_cache/)。候補数はタンパク質×レプリカごとに
K06(b) の保存データ (形成) と K2-18 の入力バンドル (解離) と一致し、
全体でも E01 の記録 (163074 件) と一致することを確認済み。
使うのは E01 の特徴量の先頭20列 (公開特徴量) だけ。

条件:
- static20: 公開特徴量20個のみ。
- full: static20 + 区間 [0..10] 全体の統計量6種 (E01 の hist_stats と同じ定義)。
- window_mean_{L}: static20 + フレーム (10 − L/0.1) 〜 10 の平均 (1列)。
回復率 = (その条件の AUPRC − static20) / (full の利得)。
全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る。

読み方 (L と同じ):
1. 回復率が 0.8 と 0.9 に届く最小の覆う長さを pooled で読む (macro も併記)。
2. 隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、その大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、
   パーセンタイル法)。標本の中でも static20・full・各長さの AUPRC を
   測り直し、利得と回復率をその標本の中で計算し直す。

前提確認 (これが崩れたら、その時点で止めて統括へ報告する):
- 候補の総数が E01 の記録 (163074 件) と一致すること。
- 各予測幅で、static20 と full の AUPRC (pooled) が E01 の記録と
  5e-3 以内で一致すること (E01 は別の計算機・別の sklearn 版で実行されたため、
  L 系列内の照合より少し広い許容幅にした)。
- 各予測幅で、長さ 1.0 ns の平均だけの回復率 (pooled) が E01 の記録と
  0.03 以内で一致すること。

前提確認の経緯 (2026-11-28): 初回の実行では回復率の許容幅も 5e-3 にしてあり、
予測幅 1 ns で回復率が 0.9611 (E01 の記録 0.9422) となって止まった。
統括の指示で原因を調べた結果 (premise_investigation.md)、データ (候補数と値) と
定義 (候補・特徴・統計量・窓・モデル設定・分割) は E01 と同じで、ずれは
評価手順そのものが持つ行の順番への敏感さだった (順番を変えるだけで回復率は
0.939〜0.965 の幅 0.026 で動き、E01 の値はその中に入る)。このため Echizen の
判断で、前提確認は AUPRC の一致と E01 の回復率が再現のばらつきの範囲内に
あることで満たされたものとし、並べ方 (ファイル名の辞書順) と数の型 (float32) を
E01 の手順にそろえて本測定を進めることになった。回復率の許容幅 0.03 は、
予測幅 1 ns で実測したばらつきの幅 0.026 を覆う値である。

使い方: run_l2_23.py <horizon_ns> (1/5/10/30/50/90 のどれか1つ)。
予測幅ごとに results.json へ追記し、oof_predictions_<幅>ns.npz を保存する。
全予測幅がそろうと results.csv と per_protein.csv を書く。
実行はローカルVM。sklearn はスレッド数 1 で動かす。
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

MODE = "hb_form"
CACHE = Path("/home/hatch/workspace/kinescope/work/e01_cache")
RESULTS_100 = Path("/home/hatch/workspace/k06b/results_100.json")
E01_RESULTS = Path("/home/hatch/workspace/kinescope/repo/experiments/"
                   "E01_true_short_md_guidance/e01_short_md_results.json")
OUT = Path(__file__).resolve().parent
N_EXPECTED = 163074
CLF_KW = dict(max_iter=160, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=25, l2_regularization=0.15, random_state=20261126)
BOOT_SEED = 42
N_BOOT = 10000
HORIZONS = [1, 5, 10, 30, 50, 90]
HIDX = {h: i for i, h in enumerate(HORIZONS)}
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(10 - int(round(L / 0.1)), 11)) for L in LENGTHS}
COND_MEAN = {L: f"window_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = ["static_only", "full_history"] + [COND_MEAN[L] for L in LENGTHS]


def hist_stats(S, thr):
    # E01 の hist_stats と同じ定義・同じ型 (float32)
    S = np.asarray(S, dtype=np.float32)
    return np.column_stack([S[:, 0], S.mean(1), S.std(1), S.min(1),
                            S[:, -1] - S[:, 0], (S <= thr).mean(1)]).astype(np.float32)


def load_all():
    # 並べ方は E01 にそろえる: ファイル名の辞書順 (E01 は sorted(CACHE.glob(...)))
    proteins = sorted(p.stem for p in CACHE.glob("*.npz"))
    Xs, Ss, Ys, Gs = [], [], [], []
    for p in proteins:
        z = np.load(CACHE / f"{p}.npz")
        for r in (1, 2, 3):
            Xs.append(z[f"r{r}_{MODE}_X"].astype(np.float32))
            Ss.append(z[f"r{r}_{MODE}_series"].astype(np.float32))
            Ys.append(z[f"r{r}_{MODE}_y"].astype(np.int64))
            Gs.append(np.full(len(Xs[-1]), p))
    X = np.vstack(Xs)
    S = np.vstack(Ss)
    Y = np.vstack(Ys)
    G = np.concatenate(Gs)
    assert X.shape[0] == N_EXPECTED, (X.shape, N_EXPECTED)
    assert X.shape[1] == 20 and S.shape[1] == 11 and Y.shape[1] == 6
    return X, S, Y, G, proteins


def oof_predict(Xf, y, groups):
    pred = np.zeros(len(y), dtype=np.float64)
    gkf = GroupKFold(5)
    for tr, te in gkf.split(Xf, y, groups):
        m = HistGradientBoostingClassifier(**CLF_KW).fit(Xf[tr], y[tr])
        pred[te] = m.predict_proba(Xf[te])[:, 1]
    return pred


def macro_ap(y, pred, groups, proteins):
    aps = {}
    for p in proteins:
        m = groups == p
        yy = y[m]
        if 0 < yy.sum() < len(yy):
            aps[p] = float(average_precision_score(yy, pred[m]))
    return aps


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


def load_results():
    p = OUT / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {
        "label": "L2-23: HB formation recovery vs adjacent-window length (window ends at origin frame 10), mean only, E01 public base (static20, 5-fold protein-group CV, OOF). Full-length version of E01 Protocol A.",
        "window_definition": {
            "lengths": {f"{L:.1f}": dict(covered_ns=L, frames=FRAMES[L]) for L in LENGTHS},
            "note": "Covered length L ns uses frames (10 - L/0.1) .. 10 of the 0-1 ns history (100 ps spacing), ending at the origin frame 10 with no time gap (E01 Protocol A placement).",
        },
        "base": {
            "name": "E01 public base: static20 (100% public 20 features, no frozen model), HistGradientBoosting (160 iters, lr 0.06, 31 leaves, min_samples_leaf 25, L2 0.15, seed 20261126), 5-fold GroupKFold by protein, out-of-fold predictions",
            "n_candidates": N_EXPECTED,
            "data_note": "E01's extracted bundles are not on this machine; the data was re-extracted from the ATLAS analysis trajectories with extract_e01_e02.py's definitions copied verbatim (work/e01_extract.py). Per protein x replica candidate counts match K06(b) (formation) and the K2-18 bundle (dissociation); totals match E01's records.",
        },
        "premise_checks": {},
        "by_horizon": {},
    }


def save_results(res):
    (OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


def run_horizon(h_ns, X, S, Y, G, proteins, e01_ref):
    hi = HIDX[h_ns]
    t0 = time.time()
    y = Y[:, hi]
    pmap = {p: i for i, p in enumerate(proteins)}
    prot_of_row = np.array([pmap[g] for g in G])
    spans = []
    for i, p in enumerate(proteins):
        idx = np.where(prot_of_row == i)[0]
        spans.append((p, idx[0], idx[-1] + 1))
    print(f"=== horizon {h_ns} ns: n {len(y)} pos {int(y.sum())} ===", flush=True)

    feats = {"static_only": X,
             "full_history": np.hstack([X, hist_stats(S, 3.5)])}
    for L in LENGTHS:
        idx = FRAMES[L]
        feats[COND_MEAN[L]] = np.hstack([X, S[:, idx].mean(axis=1, keepdims=True)])

    preds, results, per_protein_ap = {}, {}, {}
    for cname in CONDITIONS:
        pred = oof_predict(feats[cname], y, G)
        preds[cname] = pred
        aps = macro_ap(y, pred, G, proteins)
        per_protein_ap[cname] = {p: aps.get(p) for p in proteins}
        results[cname] = dict(
            auprc=float(average_precision_score(y, pred)),
            macro_auprc=float(np.mean(list(aps.values()))),
            n_proteins_macro=len(aps))
        print(cname, "pooled %.4f macro %.4f (%.0fs)"
              % (results[cname]["auprc"], results[cname]["macro_auprc"], time.time() - t0), flush=True)

    np.savez_compressed(
        OUT / f"oof_predictions_{h_ns}ns.npz",
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]),
        label=y.astype(np.int8),
        **{c: preds[c].astype(np.float32) for c in CONDITIONS})

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

    def rec(cname, kind):
        if kind == "pooled":
            return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

    ref = e01_ref[str(h_ns)]["A_true_1.0ns_mean"]
    checks = []

    def chk(name, got, reference, tol):
        ok = abs(got - reference) < tol
        checks.append(dict(check=name, got=float(got), reference=float(reference), tol=tol, ok=bool(ok)))
        print("check", name, "got %.4f ref %.4f tol %g -> %s" % (got, reference, tol, ok), flush=True)
        return ok

    ok = True
    ok &= chk("static_pooled_vs_E01", results["static_only"]["auprc"], ref["ap_static"], 5e-3)
    ok &= chk("full_pooled_vs_E01", results["full_history"]["auprc"], ref["ap_full"], 5e-3)
    ok &= chk("window_mean_1.0_recovery_pooled_vs_E01", rec(COND_MEAN[1.0], "pooled"), ref["recovery_pooled"], 0.03)

    res = load_results()
    res["premise_checks"][str(h_ns)] = checks
    save_results(res)
    if not ok:
        print("PREMISE FAILED at horizon", h_ns, "- stopping as instructed.", flush=True)
        sys.exit(2)
    print("premise OK for horizon", h_ns, flush=True)

    recovery = {c: dict(pooled=float(rec(c, "pooled")), macro=float(rec(c, "macro")))
                for c in CONDITIONS if c != "static_only"}
    increments = {}
    for kind in ("pooled", "macro"):
        inc, prev = {}, None
        for L in LENGTHS:
            r = recovery[COND_MEAN[L]][kind]
            if prev is not None:
                inc[f"{L:.1f}"] = float(r - prev)
            prev = r
        increments[kind] = inc

    def min_length(kind, thr):
        for L in LENGTHS:
            if recovery[COND_MEAN[L]][kind] >= thr:
                return L
        return None

    def largest_step(kind):
        inc = increments[kind]
        key = max(inc, key=lambda k: inc[k])
        Lto = float(key)
        return dict(from_ns=Lto - 0.1, to_ns=Lto, increment=inc[key])

    per_protein = {}
    for p, s, e in spans:
        yy = y[s:e]
        entry = dict(n=int(e - s), positives=int(yy.sum()))
        ap_s = per_protein_ap["static_only"][p]
        ap_f = per_protein_ap["full_history"][p]
        entry["auprc_static"] = ap_s
        entry["auprc_full"] = ap_f
        entry["full_gain"] = (ap_f - ap_s) if (ap_s is not None and ap_f is not None) else None
        entry["auprc_window"] = {}
        entry["recovery_window"] = {}
        for L in LENGTHS:
            ap = per_protein_ap[COND_MEAN[L]][p]
            entry["auprc_window"][f"{L:.1f}"] = ap
            if ap is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                entry["recovery_window"][f"{L:.1f}"] = float((ap - ap_s) / entry["full_gain"])
            else:
                entry["recovery_window"][f"{L:.1f}"] = None
        per_protein[p] = entry

    # ---- ブートストラップ (タンパク質単位、L と同じ段取り) ----
    for cname in CONDITIONS:
        ap_np = ap_numpy(y, preds[cname].astype(np.float64))
        assert abs(ap_np - results[cname]["auprc"]) < 1e-9, (cname, ap_np, results[cname]["auprc"])
    print("numpy AP verified vs sklearn for all conditions", flush=True)

    boot_prep = {}
    for cname in CONDITIONS:
        score = preds[cname].astype(np.float64)
        order = np.argsort(-score, kind="stable")
        ss = score[order]
        starts = np.flatnonzero(np.r_[True, ss[:-1] != ss[1:]])
        boot_prep[cname] = dict(y_sorted=y[order].astype(np.float64),
                                prot_sorted=prot_of_row[order], starts=starts)

    def pooled_ap_boot(prep, mult):
        w = mult[prep["prot_sorted"]]
        wp = w * prep["y_sorted"]
        tp_g = np.add.reduceat(wp, prep["starts"])
        tot_g = np.add.reduceat(w, prep["starts"])
        cum_tp = np.cumsum(tp_g)
        cum_tot = np.cumsum(tot_g)
        precision = np.divide(cum_tp, cum_tot, out=np.zeros_like(cum_tp), where=cum_tot > 0)
        return float(np.sum((tp_g / cum_tp[-1]) * precision))

    defined = [p for p in proteins if per_protein_ap["static_only"][p] is not None]
    draw_to_defined = np.full(len(proteins), -1, dtype=int)
    for i, p in enumerate(proteins):
        if p in defined:
            draw_to_defined[i] = defined.index(p)
    ap_mat = np.array([[per_protein_ap[c][p] for c in CONDITIONS] for p in defined])
    cond_idx = {c: j for j, c in enumerate(CONDITIONS)}

    rng = np.random.default_rng(BOOT_SEED)
    nP = len(proteins)
    boot_rec = {kind: {f"{L:.1f}": [] for L in LENGTHS} for kind in ("pooled", "macro")}
    n_bad_gain = dict(pooled=0, macro=0)
    for b in range(N_BOOT):
        draw = rng.integers(0, nP, nP)
        mult = np.bincount(draw, minlength=nP).astype(np.float64)
        ap_s = pooled_ap_boot(boot_prep["static_only"], mult)
        ap_f = pooled_ap_boot(boot_prep["full_history"], mult)
        g = ap_f - ap_s
        if g <= 0:
            n_bad_gain["pooled"] += 1
        else:
            for L in LENGTHS:
                ap = pooled_ap_boot(boot_prep[COND_MEAN[L]], mult)
                boot_rec["pooled"][f"{L:.1f}"].append((ap - ap_s) / g)
        dpos = draw_to_defined[draw]
        wsel = dpos[dpos >= 0]
        if len(wsel) == 0:
            n_bad_gain["macro"] += 1
        else:
            m = ap_mat[wsel].mean(axis=0)
            gm = m[cond_idx["full_history"]] - m[cond_idx["static_only"]]
            if gm <= 0:
                n_bad_gain["macro"] += 1
            else:
                for L in LENGTHS:
                    boot_rec["macro"][f"{L:.1f}"].append(
                        (m[cond_idx[COND_MEAN[L]]] - m[cond_idx["static_only"]]) / gm)
        if (b + 1) % 2000 == 0:
            print("bootstrap %d/%d (%.0fs)" % (b + 1, N_BOOT, time.time() - t0), flush=True)

    bootstrap = {}
    for kind in ("pooled", "macro"):
        bootstrap[kind] = {}
        for L in LENGTHS:
            arr = np.array(boot_rec[kind][f"{L:.1f}"])
            if len(arr) == 0 or np.isnan(arr).any():
                raise RuntimeError(f"bootstrap {kind} {L}: invalid values")
            bootstrap[kind][f"{L:.1f}"] = dict(
                n_boot=int(len(arr)), seed=BOOT_SEED,
                point_estimate=recovery[COND_MEAN[L]][kind],
                ci_lower=float(np.percentile(arr, 2.5)),
                ci_upper=float(np.percentile(arr, 97.5)),
                boot_mean=float(arr.mean()), boot_std=float(arr.std()))
    print("bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)

    step = largest_step("pooled")
    Lfrom, Lto = step["from_ns"], step["to_ns"]
    step_inc_protein = []
    for p, entry in per_protein.items():
        r_from = entry["recovery_window"].get(f"{Lfrom:.1f}")
        r_to = entry["recovery_window"].get(f"{Lto:.1f}")
        if r_from is not None and r_to is not None:
            step_inc_protein.append(r_to - r_from)
    step_inc_protein = np.array(step_inc_protein)
    step_per_protein = dict(
        from_ns=Lfrom, to_ns=Lto, n_proteins=int(len(step_inc_protein)),
        median=float(np.median(step_inc_protein)),
        q25=float(np.percentile(step_inc_protein, 25)),
        q75=float(np.percentile(step_inc_protein, 75)),
        min=float(step_inc_protein.min()), max=float(step_inc_protein.max()),
        frac_positive=float(np.mean(step_inc_protein > 0)),
        frac_ge_0p1=float(np.mean(step_inc_protein >= 0.1)))

    readings = dict(
        min_length_recovery_0_8=dict(pooled=min_length("pooled", 0.8), macro=min_length("macro", 0.8)),
        min_length_recovery_0_9=dict(pooled=min_length("pooled", 0.9), macro=min_length("macro", 0.9)),
        largest_step=dict(pooled=step, macro=largest_step("macro")),
        step_per_protein=step_per_protein)

    res = load_results()
    res["by_horizon"][str(h_ns)] = dict(
        horizon_ns=h_ns,
        metrics=results,
        base_gain=dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        recovery_vs_full_history_gain=recovery,
        recovery_increment_per_0p1ns=increments,
        readings=readings,
        bootstrap=dict(n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
                       resampling_unit="protein (100 test proteins, resampled with replacement)",
                       n_draws_with_nonpositive_gain=n_bad_gain,
                       results=bootstrap),
        per_protein=per_protein)
    save_results(res)
    print(f"horizon {h_ns} ns saved (%.0fs)" % (time.time() - t0), flush=True)
    print(json.dumps(readings, indent=2, ensure_ascii=False), flush=True)
    return True


def write_csvs(res):
    lines = ["horizon_ns,covered_ns,n_frames,auprc_pooled,auprc_macro,recovery_pooled,recovery_macro,"
             "recovery_pooled_ci_lower,recovery_pooled_ci_upper,recovery_macro_ci_lower,recovery_macro_ci_upper,"
             "increment_pooled,increment_macro"]
    for h_ns in HORIZONS:
        hz = res["by_horizon"][str(h_ns)]
        for L in LENGTHS:
            c = COND_MEAN[L]
            m = hz["metrics"][c]
            r = hz["recovery_vs_full_history_gain"][c]
            b = hz["bootstrap"]["results"]
            inc = hz["recovery_increment_per_0p1ns"]
            lines.append(",".join([
                str(h_ns), f"{L:.1f}", str(len(FRAMES[L])),
                f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}",
                f"{r['pooled']:.6f}", f"{r['macro']:.6f}",
                f"{b['pooled'][f'{L:.1f}']['ci_lower']:.6f}", f"{b['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
                f"{b['macro'][f'{L:.1f}']['ci_lower']:.6f}", f"{b['macro'][f'{L:.1f}']['ci_upper']:.6f}",
                f"{inc['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
                f"{inc['macro'].get(f'{L:.1f}', float('nan')):.6f}"]))
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    plines = ["horizon_ns,protein,n,positives,auprc_static,auprc_full,full_gain,"
              + ",".join(f"auprc_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"recovery_{L:.1f}ns" for L in LENGTHS)]
    for h_ns in HORIZONS:
        hz = res["by_horizon"][str(h_ns)]
        for p, entry in hz["per_protein"].items():
            row = [str(h_ns), p, str(entry["n"]), str(entry["positives"]),
                   "" if entry["auprc_static"] is None else f"{entry['auprc_static']:.6f}",
                   "" if entry["auprc_full"] is None else f"{entry['auprc_full']:.6f}",
                   "" if entry["full_gain"] is None else f"{entry['full_gain']:.6f}"]
            row += ["" if entry["auprc_window"][f"{L:.1f}"] is None else f"{entry['auprc_window'][f'{L:.1f}']:.6f}" for L in LENGTHS]
            row += ["" if entry["recovery_window"][f"{L:.1f}"] is None else f"{entry['recovery_window'][f'{L:.1f}']:.6f}" for L in LENGTHS]
            plines.append(",".join(row))
    (OUT / "per_protein.csv").write_text("\n".join(plines) + "\n")
    print("wrote results.csv / per_protein.csv", flush=True)


def main():
    h_ns = int(sys.argv[1])
    assert h_ns in HORIZONS, h_ns
    res = load_results()
    if str(h_ns) in res["by_horizon"]:
        print(f"horizon {h_ns} ns already done; skipping", flush=True)
        return
    X, S, Y, G, proteins = load_all()
    e01 = json.loads(E01_RESULTS.read_text())
    e01_ref = e01["protocol_a_and_c"][MODE]["by_horizon"]
    assert e01["protocol_a_and_c"][MODE]["n_candidates"] == N_EXPECTED
    run_horizon(h_ns, X, S, Y, G, proteins, e01_ref)
    res = load_results()
    if all(str(h) in res["by_horizon"] for h in HORIZONS):
        write_csvs(res)
        print("ALL HORIZONS DONE", flush=True)


if __name__ == "__main__":
    main()
