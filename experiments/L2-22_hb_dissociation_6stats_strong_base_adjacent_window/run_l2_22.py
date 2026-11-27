"""L2-22: L45〜L50 の曲がりを、区間を起点フレーム10の直前に置き直して測り直す。

L2系列 (L01〜L74 の全面改訂): L では履歴の区間を先頭からの区間 [0..k] として
起点フレーム10と組にしていたため、短い長さでは起点との間に時間の空きがあった。
E01 で、この空きが回復率を水増ししていたことが分かった。L2 では区間を
起点フレーム10の直前に置く (E01 の Protocol A と同じ置き方):
- 覆う長さ L ns の区間はフレーム (10 − L/0.1) 〜 10 (起点に接する)。
  0 ns -> [10] (1フレーム)、0.1 ns -> [9,10]、…、1.0 ns -> [0..10] (11フレーム全部)。
- 長さ 1.0 ns の点では区間が [0..10] になり、L と完全に同じ条件になる。

土台・候補・予測器は対応する L 実験 (L45〜L50) と完全に同じ:
- 水素結合の解離の強い土台 (K75 の条件)。入力は K2-18 の入力バンドル
  (較正 6,144件、テスト未見100タンパク質×3レプリカ 76,342件)。
  凍結した本体 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、seed 20261121)。
- 統計量6種は実験Dの定義 (最初・平均・標準偏差 (ddof=0)・最小・最後−最初・
  3.5 Å以下だった割合) を区間に適用する。「最初」は区間の最初のフレーム
  (フレーム 10−k)、「最後」はフレーム10。
- 条件は静的のみ・全部入り (要約値10個)・区間の平均だけ (11長さ)・
  区間の統計量6種 (11長さ) で、すべて条件ごとに学習し直す。
- 回復率 = (その長さの AUPRC − 静的のみ) / (全部入りの利得)。
  全体 (pooled) とタンパク質ごとの平均 (macro、100タンパク質) の両方で測る。
  macro300 (タンパク質×レプリカの300グループの平均) は土台の照合用に記録する。

読み方 (L と同じ):
1. 回復率が 0.8 と 0.9 に届く最小の覆う長さを pooled で読む (macro も併記)。
2. 隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、その大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、パーセンタイル法)。
   標本の中でも静的のみ・全部入り・各長さの AUPRC を測り直し、利得と回復率を
   その標本の中で計算し直す (L45 と同じ段取り)。
4. 増え幅最大の場所の前後ではタンパク質ごとの値も記録する。

前提確認 (これが崩れたら、その時点で止めて統括へ報告する):
- 統計量の照合: 較正+テストの全候補で、11フレームの系列から計算した統計量6種
  (要約値10個の部分集合の位置 [0,1,2,3,5,7]) が、保存されている履歴特徴10個の
  該当列と 1e-4 未満で一致すること (L45 と同じ)。
- 各予測幅で、静的のみ・全部入りの AUPRC (pooled / macro / macro300) が
  対応する L 実験の記録と 5e-4 以内で一致すること。
- 各予測幅で、長さ 1.0 ns の区間の平均だけ・統計量6種の回復率 (pooled / macro) が
  対応する L 実験の記録 (prefix 1.0 ns の回復率) と 5e-3 以内で一致すること。

使い方: run_l2_22.py <horizon_ns> (1/5/10/30/50/90 のどれか1つ)。
予測幅ごとに results.json へ追記し、test_predictions_<幅>ns.npz を保存する。
全予測幅がそろうと results.csv と per_protein.csv を書く。
実行はローカルVM (L45 と同じ段取り)。sklearn はスレッド数 1 で動かす。
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

BUNDLE = Path("/home/hatch/workspace/kinescope/repo/experiments/"
              "K2-18_hb_dissociation_half_means_strong_base/input_bundle.npz")
OUT = Path(__file__).resolve().parent
SEED = 20261121
BOOT_SEED = 42
N_BOOT = 10000
THRESH_HB = 3.5
HORIZONS = [1, 5, 10, 30, 50, 90]
HIDX = {h: i for i, h in enumerate(HORIZONS)}

LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(10 - int(round(L / 0.1)), 11)) for L in LENGTHS}
COND_MEAN = {L: f"window_mean_{L:.1f}ns" for L in LENGTHS}
COND_6 = {L: f"window_6stats_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = (["static_only", "full_history"]
              + [COND_MEAN[L] for L in LENGTHS]
              + [COND_6[L] for L in LENGTHS])
FAMS = (("mean_only", COND_MEAN), ("stats6", COND_6))

# 対応する L 実験の記録 (前提確認の照合先。各 results.json から抜き出した値)
REF = {
    1: dict(static_pooled=0.5060719789337638, static_macro=0.5221953617635298, static_macro300=0.5329610220169362,
            full_pooled=0.6141881772536423, full_macro=0.6246928969663523, full_macro300=0.63447293727364,
            mean10_pooled=0.926155649915475, mean10_macro=0.9224354722472313,
            stats6_10_pooled=0.9877733753306517, stats6_10_macro=0.9813767461743494,
            source='L45_hb_dissociation_leverage_strong_base_6stats_1ns'),
    5: dict(static_pooled=0.56647149431462, static_macro=0.5815884088272546, static_macro300=0.592053311735092,
            full_pooled=0.6512784693885272, full_macro=0.6627908539202774, full_macro300=0.6685482082555011,
            mean10_pooled=0.9970508361332638, mean10_macro=0.9950319514084117,
            stats6_10_pooled=1.0247172134287164, stats6_10_macro=1.0209789311777009,
            source='L46_hb_dissociation_leverage_strong_base_6stats_5ns'),
    10: dict(static_pooled=0.5730374597160662, static_macro=0.5930146819040956, static_macro300=0.6007067388905855,
            full_pooled=0.6501699958685653, full_macro=0.6627838521428493, full_macro300=0.6691533803056509,
            mean10_pooled=1.0980239897732669, mean10_macro=1.0927753878902422,
            stats6_10_pooled=0.9824850886742159, stats6_10_macro=1.008547967027001,
            source='L47_hb_dissociation_leverage_strong_base_6stats_10ns'),
    30: dict(static_pooled=0.6042006063817402, static_macro=0.6183467339145851, static_macro300=0.6261232797884183,
            full_pooled=0.6662744859124756, full_macro=0.6765522087639578, full_macro300=0.6828108163732711,
            mean10_pooled=0.9611547552356284, mean10_macro=0.9765642080903888,
            stats6_10_pooled=0.9546302536594504, stats6_10_macro=0.9566004162129028,
            source='L48_hb_dissociation_leverage_strong_base_6stats_30ns'),
    50: dict(static_pooled=0.6001246773212514, static_macro=0.6170905639166389, static_macro300=0.6246338431343973,
            full_pooled=0.6567897206087568, full_macro=0.6686901160958323, full_macro300=0.6758481525279876,
            mean10_pooled=0.96652055914262, mean10_macro=0.9756633064959299,
            stats6_10_pooled=1.00145536473656, stats6_10_macro=0.992935048097141,
            source='L49_hb_dissociation_leverage_strong_base_6stats_50ns'),
    90: dict(static_pooled=0.6272181976515258, static_macro=0.6428612916233888, static_macro300=0.6494454142877857,
            full_pooled=0.6757709725433758, full_macro=0.6875240961722523, full_macro300=0.6930815633092315,
            mean10_pooled=0.9576996432034974, mean10_macro=0.9731561076163336,
            stats6_10_pooled=0.9807574972951021, stats6_10_macro=0.9795883708698964,
            source='L50_hb_dissociation_leverage_strong_base_6stats_90ns'),
}


def six_stats(S):
    """実験Dの features_from_history を部分系列 S (候補数 × k フレーム) に適用する。"""
    return np.stack([S[:, 0], S.mean(1), S.std(1), S.min(1),
                     S[:, -1] - S[:, 0], (S <= THRESH_HB).mean(1)],
                    axis=1).astype(np.float32)


def ten_stats(S):
    """履歴特徴10個を11フレームの系列から再計算する (premise の照合用)。"""
    return np.stack([S[:, 0], S.mean(1), S.std(1), S.min(1), S.max(1),
                     S[:, -1] - S[:, 0], S[:, -1] - S[:, -2],
                     (S <= 3.5).mean(1), (S <= 4.5).mean(1),
                     (np.diff(S, axis=1) > 0).mean(1)], axis=1)


def check_history_stats(cal, test):
    colmax = np.zeros(10)
    n_checked = 0
    for d in (cal, test):
        calc = ten_stats(d["series"])
        diff = np.abs(d["hist"] - calc)
        colmax = np.maximum(colmax, diff.max(axis=0))
        n_checked += len(d["series"])
    worst6 = float(colmax[[0, 1, 2, 3, 5, 7]].max())
    return dict(n_candidates_checked=int(n_checked),
                max_abs_diff_per_column_10stats=[float(x) for x in colmax],
                max_abs_diff_6stats_subset=worst6,
                tol=1e-4, ok=bool(worst6 < 1e-4))


def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
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


def results_path():
    return OUT / "results.json"


def load_results():
    if results_path().exists():
        return json.loads(results_path().read_text())
    return {
        "label": "L2-22: HB dissociation, recovery vs adjacent-window length (window ends at origin frame 10), 6 history statistics (experiment-D definition), strong base (K75 conditions, K2-18 input bundle). Re-measurement of L45-L50 with the interval placed immediately before the origin (E01 Protocol A placement).",
        "window_definition": {
            "lengths": {f"{L:.1f}": dict(covered_ns=L, frames=FRAMES[L]) for L in LENGTHS},
            "note": "Covered length L ns uses frames (10 - L/0.1) .. 10 of the 0-1 ns history (100 ps spacing), i.e. the interval ends at the origin frame 10 with no time gap. At 1.0 ns the window is frames [0..10], identical to the L-series prefix.",
        },
        "statistics6_definition": {
            "order": ["first value (at frame 10-k)", "mean", "std (ddof=0)", "min",
                      "last - first (frame 10 minus frame 10-k)", "fraction of frames <= 3.5 A"],
            "threshold_A": THRESH_HB,
        },
        "base": {
            "name": "HB dissociation strong base (K75 conditions): frozen model + HistGBM correction (160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), K2-18 input bundle (calibration 6144, test 100 unseen proteins x 3 replicas = 76342)",
            "static_features": 5, "history_features_full": 10,
        },
        "premise_checks": {},
        "by_horizon": {},
    }


def save_results(res):
    results_path().write_text(json.dumps(res, indent=2, ensure_ascii=False))


def run_horizon(h_ns, cal, test, spans, prot_of_row, group_idx, proteins):
    hi = HIDX[h_ns]
    t0 = time.time()
    y_cal = cal["label"][:, hi].astype(int)
    y_test = test["label"][:, hi].astype(int)
    print(f"=== horizon {h_ns} ns: cal {len(y_cal)} test {len(y_test)} pos {int(y_test.sum())} ===", flush=True)

    feats = {}
    feats["static_only"] = (cal["static"], test["static"])
    feats["full_history"] = (np.column_stack([cal["static"], cal["hist"]]),
                             np.column_stack([test["static"], test["hist"]]))
    for L in LENGTHS:
        idx = FRAMES[L]
        feats[COND_MEAN[L]] = (
            np.column_stack([cal["static"], cal["series"][:, idx].mean(axis=1)[:, None]]),
            np.column_stack([test["static"], test["series"][:, idx].mean(axis=1)[:, None]]))
        feats[COND_6[L]] = (
            np.column_stack([cal["static"], six_stats(cal["series"][:, idx])]),
            np.column_stack([test["static"], six_stats(test["series"][:, idx])]))

    preds, results, per_protein_ap = {}, {}, {c: {} for c in CONDITIONS}
    for cname in CONDITIONS:
        xtr, xte = feats[cname]
        pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
        preds[cname] = pred
        m = metrics(y_test, pred)
        aps = []
        for p, s, e in spans:
            yy = y_test[s:e]
            if 0 < yy.sum() < e - s:
                ap = float(average_precision_score(yy, pred[s:e]))
                aps.append(ap)
                per_protein_ap[cname][p] = ap
            else:
                per_protein_ap[cname][p] = None
        m["macro_auprc"] = float(np.mean(aps))
        m["n_proteins_macro"] = len(aps)
        gaps = []
        for idxs in group_idx:
            yy = y_test[idxs]
            if 0 < yy.sum() < len(idxs):
                gaps.append(float(average_precision_score(yy, pred[idxs])))
        m["macro300_auprc"] = float(np.mean(gaps))
        m["n_groups_macro300"] = len(gaps)
        results[cname] = m
        print(cname, "pooled %.4f macro %.4f macro300 %.4f (%.0fs)"
              % (m["auprc"], m["macro_auprc"], m["macro300_auprc"], time.time() - t0), flush=True)

    np.savez_compressed(
        OUT / f"test_predictions_{h_ns}ns.npz",
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]),
        label=y_test.astype(np.int8),
        **{c: preds[c].astype(np.float32) for c in CONDITIONS})

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

    def rec(cname, kind):
        if kind == "pooled":
            return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

    ref = REF[h_ns]
    checks = []

    def chk(name, got, reference, tol):
        ok = abs(got - reference) < tol
        checks.append(dict(check=name, got=float(got), reference=float(reference), tol=tol, ok=bool(ok)))
        print("check", name, "got %.4f ref %.4f tol %g -> %s" % (got, reference, tol, ok), flush=True)
        return ok

    ok = True
    ok &= chk("static_pooled_vs_L", results["static_only"]["auprc"], ref["static_pooled"], 5e-4)
    ok &= chk("static_macro_vs_L", results["static_only"]["macro_auprc"], ref["static_macro"], 5e-4)
    ok &= chk("static_macro300_vs_L", results["static_only"]["macro300_auprc"], ref["static_macro300"], 5e-4)
    ok &= chk("full_pooled_vs_L", results["full_history"]["auprc"], ref["full_pooled"], 5e-4)
    ok &= chk("full_macro_vs_L", results["full_history"]["macro_auprc"], ref["full_macro"], 5e-4)
    ok &= chk("full_macro300_vs_L", results["full_history"]["macro300_auprc"], ref["full_macro300"], 5e-4)
    ok &= chk("window_mean_1.0_recovery_pooled_vs_L_prefix", rec(COND_MEAN[1.0], "pooled"), ref["mean10_pooled"], 5e-3)
    ok &= chk("window_mean_1.0_recovery_macro_vs_L_prefix", rec(COND_MEAN[1.0], "macro"), ref["mean10_macro"], 5e-3)
    ok &= chk("window_6stats_1.0_recovery_pooled_vs_L_prefix", rec(COND_6[1.0], "pooled"), ref["stats6_10_pooled"], 5e-3)
    ok &= chk("window_6stats_1.0_recovery_macro_vs_L_prefix", rec(COND_6[1.0], "macro"), ref["stats6_10_macro"], 5e-3)

    res = load_results()
    res["premise_checks"][str(h_ns)] = checks
    save_results(res)
    if not ok:
        print("PREMISE FAILED at horizon", h_ns, "- stopping as instructed.", flush=True)
        sys.exit(2)
    print("premise OK for horizon", h_ns, flush=True)

    recovery = {c: dict(pooled=float(rec(c, "pooled")), macro=float(rec(c, "macro")))
                for c in CONDITIONS if c != "static_only"}

    def increments_for(condmap, kind):
        inc, prev = {}, None
        for L in LENGTHS:
            r = recovery[condmap[L]][kind]
            if prev is not None:
                inc[f"{L:.1f}"] = float(r - prev)
            prev = r
        return inc

    increments = {fam: {kind: increments_for(cm, kind) for kind in ("pooled", "macro")}
                  for fam, cm in FAMS}

    def min_length(condmap, kind, thr):
        for L in LENGTHS:
            if recovery[condmap[L]][kind] >= thr:
                return L
        return None

    def largest_step(fam, condmap, kind):
        inc = increments[fam][kind]
        key = max(inc, key=lambda k: inc[k])
        Lto = float(key)
        return dict(from_ns=Lto - 0.1, to_ns=Lto, increment=inc[key])

    def per_protein_for(condmap):
        pp = {}
        for p, s, e in spans:
            yy = y_test[s:e]
            entry = dict(n=int(e - s), positives=int(yy.sum()))
            ap_s = per_protein_ap["static_only"][p]
            ap_f = per_protein_ap["full_history"][p]
            entry["auprc_static"] = ap_s
            entry["auprc_full"] = ap_f
            entry["full_gain"] = (ap_f - ap_s) if (ap_s is not None and ap_f is not None) else None
            entry["auprc_window"] = {}
            entry["recovery_window"] = {}
            for L in LENGTHS:
                ap = per_protein_ap[condmap[L]][p]
                entry["auprc_window"][f"{L:.1f}"] = ap
                if ap is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                    entry["recovery_window"][f"{L:.1f}"] = float((ap - ap_s) / entry["full_gain"])
                else:
                    entry["recovery_window"][f"{L:.1f}"] = None
            pp[p] = entry
        return pp

    per_protein = {"stats6": per_protein_for(COND_6), "mean_only": per_protein_for(COND_MEAN)}

    # ---- ブートストラップ (L45 と同じ段取り) ----
    for cname in CONDITIONS:
        ap_np = ap_numpy(y_test, preds[cname].astype(np.float64))
        assert abs(ap_np - results[cname]["auprc"]) < 1e-9, (cname, ap_np, results[cname]["auprc"])
    print("numpy AP verified vs sklearn for all conditions", flush=True)

    boot_prep = {}
    for cname in CONDITIONS:
        score = preds[cname].astype(np.float64)
        order = np.argsort(-score, kind="stable")
        ss = score[order]
        starts = np.flatnonzero(np.r_[True, ss[:-1] != ss[1:]])
        boot_prep[cname] = dict(y_sorted=y_test[order].astype(np.float64),
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

    defined = [p for p, _, _ in spans if per_protein_ap["static_only"][p] is not None]
    draw_to_defined = np.full(len(proteins), -1, dtype=int)
    for i, p in enumerate(proteins):
        if p in defined:
            draw_to_defined[i] = defined.index(p)
    ap_mat = np.array([[per_protein_ap[c][p] for c in CONDITIONS] for p in defined])
    cond_idx = {c: j for j, c in enumerate(CONDITIONS)}

    rng = np.random.default_rng(BOOT_SEED)
    nP = len(proteins)
    boot_rec = {fam: {kind: {f"{L:.1f}": [] for L in LENGTHS} for kind in ("pooled", "macro")}
                for fam, _ in FAMS}
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
            for fam, cm in FAMS:
                for L in LENGTHS:
                    ap = pooled_ap_boot(boot_prep[cm[L]], mult)
                    boot_rec[fam]["pooled"][f"{L:.1f}"].append((ap - ap_s) / g)
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
                for fam, cm in FAMS:
                    for L in LENGTHS:
                        boot_rec[fam]["macro"][f"{L:.1f}"].append(
                            (m[cond_idx[cm[L]]] - m[cond_idx["static_only"]]) / gm)
        if (b + 1) % 2000 == 0:
            print("bootstrap %d/%d (%.0fs)" % (b + 1, N_BOOT, time.time() - t0), flush=True)

    bootstrap = {}
    for fam, cm in FAMS:
        bootstrap[fam] = {}
        for kind in ("pooled", "macro"):
            bootstrap[fam][kind] = {}
            for L in LENGTHS:
                arr = np.array(boot_rec[fam][kind][f"{L:.1f}"])
                if len(arr) == 0 or np.isnan(arr).any():
                    raise RuntimeError(f"bootstrap {fam} {kind} {L}: invalid values")
                bootstrap[fam][kind][f"{L:.1f}"] = dict(
                    n_boot=int(len(arr)), seed=BOOT_SEED,
                    point_estimate=recovery[cm[L]][kind],
                    ci_lower=float(np.percentile(arr, 2.5)),
                    ci_upper=float(np.percentile(arr, 97.5)),
                    boot_mean=float(arr.mean()), boot_std=float(arr.std()))
    print("bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)

    step = largest_step("stats6", COND_6, "pooled")
    Lfrom, Lto = step["from_ns"], step["to_ns"]
    step_inc_protein = []
    for p, entry in per_protein["stats6"].items():
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
        stats6=dict(
            min_length_recovery_0_8=dict(pooled=min_length(COND_6, "pooled", 0.8),
                                         macro=min_length(COND_6, "macro", 0.8)),
            min_length_recovery_0_9=dict(pooled=min_length(COND_6, "pooled", 0.9),
                                         macro=min_length(COND_6, "macro", 0.9)),
            largest_step=dict(pooled=step, macro=largest_step("stats6", COND_6, "macro")),
            step_per_protein=step_per_protein),
        mean_only=dict(
            min_length_recovery_0_8=dict(pooled=min_length(COND_MEAN, "pooled", 0.8),
                                         macro=min_length(COND_MEAN, "macro", 0.8)),
            min_length_recovery_0_9=dict(pooled=min_length(COND_MEAN, "pooled", 0.9),
                                         macro=min_length(COND_MEAN, "macro", 0.9)),
            largest_step=dict(pooled=largest_step("mean_only", COND_MEAN, "pooled"),
                              macro=largest_step("mean_only", COND_MEAN, "macro"))))

    diff_vs_mean = {f"{L:.1f}": dict(
        pooled=float(recovery[COND_6[L]]["pooled"] - recovery[COND_MEAN[L]]["pooled"]),
        macro=float(recovery[COND_6[L]]["macro"] - recovery[COND_MEAN[L]]["macro"]))
        for L in LENGTHS}

    res = load_results()
    res["by_horizon"][str(h_ns)] = dict(
        horizon_ns=h_ns,
        metrics=results,
        base_gain=dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        recovery_vs_full_history_gain=recovery,
        recovery_increment_per_0p1ns=increments,
        recovery_diff_stats6_minus_mean=diff_vs_mean,
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
    lines = ["horizon_ns,covered_ns,n_frames,family,auprc_pooled,auprc_macro,auprc_macro300,recovery_pooled,recovery_macro,"
             "recovery_pooled_ci_lower,recovery_pooled_ci_upper,recovery_macro_ci_lower,recovery_macro_ci_upper,"
             "increment_pooled,increment_macro,auc,brier,logloss"]
    for h_ns in HORIZONS:
        hz = res["by_horizon"][str(h_ns)]
        for fam, cm in FAMS:
            for L in LENGTHS:
                c = cm[L]
                m = hz["metrics"][c]
                r = hz["recovery_vs_full_history_gain"][c]
                b = hz["bootstrap"]["results"][fam]
                inc = hz["recovery_increment_per_0p1ns"][fam]
                lines.append(",".join([
                    str(h_ns), f"{L:.1f}", str(len(FRAMES[L])), fam,
                    f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}", f"{m['macro300_auprc']:.6f}",
                    f"{r['pooled']:.6f}", f"{r['macro']:.6f}",
                    f"{b['pooled'][f'{L:.1f}']['ci_lower']:.6f}", f"{b['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
                    f"{b['macro'][f'{L:.1f}']['ci_lower']:.6f}", f"{b['macro'][f'{L:.1f}']['ci_upper']:.6f}",
                    f"{inc['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
                    f"{inc['macro'].get(f'{L:.1f}', float('nan')):.6f}",
                    f"{m['auc']:.6f}", f"{m['brier']:.6f}", f"{m['logloss']:.6f}"]))
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    plines = ["horizon_ns,family,protein,n,positives,auprc_static,auprc_full,full_gain,"
              + ",".join(f"auprc_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"recovery_{L:.1f}ns" for L in LENGTHS)]
    for h_ns in HORIZONS:
        hz = res["by_horizon"][str(h_ns)]
        for fam in ("stats6", "mean_only"):
            for p, entry in hz["per_protein"][fam].items():
                row = [str(h_ns), fam, p, str(entry["n"]), str(entry["positives"]),
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
    b = np.load(BUNDLE)
    cal = dict(static=b["cal_static"].astype(np.float64),
               hist=b["cal_history"].astype(np.float64),
               series=b["cal_series"].astype(np.float64),
               label=b["cal_label"])
    test = dict(static=b["test_static"].astype(np.float64),
                hist=b["test_history"].astype(np.float64),
                series=b["test_series"].astype(np.float64),
                label=b["test_label"], protein=b["test_protein"], replica=b["test_replica"])
    res = load_results()
    if "history_stats_check" not in res:
        stat_check = check_history_stats(cal, test)
        print("history stats check:", stat_check, flush=True)
        res["history_stats_check"] = stat_check
        save_results(res)
        if not stat_check["ok"]:
            print("PREMISE FAILED: statistics definitions do not tie to the stored history features.", flush=True)
            sys.exit(2)
    if str(h_ns) in res["by_horizon"]:
        print(f"horizon {h_ns} ns already done; skipping", flush=True)
        return
    proteins = list(dict.fromkeys(test["protein"].tolist()))
    assert len(proteins) == 100, len(proteins)
    prot_idx = {p: np.where(test["protein"] == p)[0] for p in proteins}
    spans = [(p, prot_idx[p][0], prot_idx[p][-1] + 1) for p in proteins]
    for p, s, e in spans:
        assert (test["protein"][s:e] == p).all()
    prot_of_row = np.concatenate([np.full(len(prot_idx[p]), i) for i, p in enumerate(proteins)])
    group_idx = [np.where((test["protein"] == p) & (test["replica"] == r))[0]
                 for p in proteins for r in (1, 2, 3)]
    run_horizon(h_ns, cal, test, spans, prot_of_row, group_idx, proteins)
    res = load_results()
    if all(str(h) in res["by_horizon"] for h in HORIZONS):
        write_csvs(res)
        print("ALL HORIZONS DONE", flush=True)


if __name__ == "__main__":
    main()
