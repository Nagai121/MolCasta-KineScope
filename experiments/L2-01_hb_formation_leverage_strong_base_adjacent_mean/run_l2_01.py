"""L2-01: L02・L04〜L08 の曲がりを、区間を起点フレーム10の直前に置き直して測り直す
(K06(b) の条件 = 強い土台、水素結合の形成、平均だけ、予測幅 1/5/10/30/50/90 ns)。

L 系列 (L01〜L74) は履歴として先頭からの区間 [0..k] を起点フレーム10と組にしていたため、
短い長さでは区間の終わりと起点フレーム10の間に時間の空きがあった。
E01 で、この空きが回復率を水増ししていたことが分かった。
L2 系列では区間を起点フレーム10の直前に置き直す (E01 の Protocol A と同じ置き方)。
土台・候補・予測器の設定・読み方は、対応する L 実験と完全に同じ。

長さの定義 (フレーム数では L と同じ。区間の置き方だけが違う):
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。起点はフレーム10。
- 覆う長さ L ns の区間はフレーム (10 - L/0.1) 〜 10 (起点に接する最後の L/0.1+1 フレーム)。
  0 ns -> [10] (1フレーム)、0.1 ns -> [9,10]、…、1.0 ns -> [0..10] (11フレーム全部)。
- 長さ 1.0 ns の点では区間が [0..10] になり、L と同一条件になる。

条件 (すべて条件ごとに学習し直す):
- 静的のみ: 静的特徴5個。
- 全部入り (full_history): 静的特徴＋実履歴の要約値10個。回復率の分母 (実履歴の利得) の参照。
- 区間の平均だけ: 静的特徴＋区間内のペア間距離の平均だけ。覆う長さ 0.0〜1.0 ns の11通り。

土台 (K06(b) = L02〜L08 と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、seed 20261121)。
  較正タンパク質 8つで学習し、未見の100タンパク質×3レプリカ (163,074行) で評価する。
- 系列は G05 で作り直したもの (~/workspace/kinescope/work/g05_series/) を使う。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。
分母は同じ条件の 1.0 ns 全履歴の利得 (L の分母と一致するはず)。
全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る (L と同じまとめ方)。

読み方 (L と同一、結果を見る前に固定):
1. 回復率が 0.8 と 0.9 に最初に達する長さを、統合値 (pooled) で読む (macro も併記)。
2. 1刻みでの最大の増え幅の場所と大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、パーセンタイル法)。
   各標本の中でも静的のみ・全部入り・各長さの AUPRC を測り直し、
   利得と回復率をその標本の中で計算し直す (L02 と同じ段取り)。
4. 段の前後ではタンパク質ごとの値も記録する。

前提確認 (premise。これが崩れたら測定は無効として止める):
- 6予測幅すべてについて、長さ 1.0 ns の点 (区間が [0..10] で L と同一条件) で、
  静的のみ・全部入りの AUPRC (pooled / macro) が対応する L 実験
  (1 ns: L02、5 ns: L04、10 ns: L05、30 ns: L06、50 ns: L07、90 ns: L08) の
  results.json の記録値と 5e-4 以内で一致すること、
  利得 (pooled / macro) が 1e-3 以内で一致すること、
  1.0 ns の点の回復率 (pooled / macro) が 5e-3 以内で一致すること。
  照合値は各 L 実験の results.json から実行時に読み取る (写し間違いを防ぐため)。
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/g05_series")
REPO_EXP = Path("/home/hatch/workspace/kinescope/repo/experiments")
OUT = Path(__file__).resolve().parent
SEED = 20261121
BOOT_SEED = 42
N_BOOT = 10000
HORIZONS = [1, 5, 10, 30, 50, 90]  # ラベルの列の順
L_REF_DIR = {
    1: "L02_hb_formation_leverage_strong_base_1ns",
    5: "L04_hb_formation_leverage_strong_base_5ns",
    10: "L05_hb_formation_leverage_strong_base_10ns",
    30: "L06_hb_formation_leverage_strong_base_30ns",
    50: "L07_hb_formation_leverage_strong_base_50ns",
    90: "L08_hb_formation_leverage_strong_base_90ns",
}

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# L2 の長さの定義: 覆う長さ L ns -> フレーム (10 - L/0.1) 〜 10
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(10 - int(round(L / 0.1)), 11)) for L in LENGTHS}
COND_ADJ = {L: f"adjacent_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = ["static_only", "full_history"] + [COND_ADJ[L] for L in LENGTHS]


def load(proteins):
    static, hist, label, series = [], [], [], []
    for p in proteins:
        z = np.load(K06B / f"{p}.npz")
        s = np.load(SERIES / f"{p}.npz")
        for r in (1, 2, 3):
            static.append(z[f"r{r}_static"])
            hist.append(z[f"r{r}_history"])
            label.append(z[f"r{r}_label"])
            series.append(s[f"r{r}_series"])
    return dict(static=np.concatenate(static), hist=np.concatenate(hist),
                label=np.concatenate(label), series=np.concatenate(series, axis=1))


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
    """sklearn の average_precision_score と同じ段取り (同点はしきい値でまとめる)。"""
    order = np.argsort(-score, kind="stable")
    ys = y[order].astype(np.float64)
    ss = score[order]
    ends = np.flatnonzero(np.r_[ss[:-1] != ss[1:], True])
    tp = np.cumsum(ys)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall_step = np.diff(np.r_[0.0, tp]) / tp[-1]
    return float(np.sum(recall_step * precision))


def main():
    t0 = time.time()
    cal, test = load(CAL), load(TEST)
    print("cal", cal["label"].shape, "test", test["label"].shape, flush=True)

    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n
    assert cur == len(test["label"])
    prot_of_row = np.concatenate([np.full(e - s, i) for i, (_, s, e) in enumerate(spans)])

    # 特徴量は予測幅によらないので先に作る
    feats = {}
    feats["static_only"] = (cal["static"], test["static"])
    feats["full_history"] = (np.column_stack([cal["static"], cal["hist"]]),
                             np.column_stack([test["static"], test["hist"]]))
    for L in LENGTHS:
        idx = FRAMES[L]
        feats[COND_ADJ[L]] = (
            np.column_stack([cal["static"], cal["series"][idx].mean(axis=0)[:, None].astype(np.float32)]),
            np.column_stack([test["static"], test["series"][idx].mean(axis=0)[:, None].astype(np.float32)]))

    # ---------- 第1段階: 全予測幅の学習・予測と前提確認 ----------
    all_checks = []
    per_horizon = {}  # h -> dict(preds, results, per_protein_ap, y_test)
    for hi, h in enumerate(HORIZONS):
        y_cal = cal["label"][:, hi]
        y_test = test["label"][:, hi]
        preds, results = {}, {}
        per_protein_ap = {c: {} for c in CONDITIONS}
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
            results[cname] = m
        per_horizon[h] = dict(preds=preds, results=results,
                              per_protein_ap=per_protein_ap, y_test=y_test)
        print(f"[{h} ns] static pooled %.4f macro %.4f | full pooled %.4f macro %.4f (%.0fs)"
              % (results["static_only"]["auprc"], results["static_only"]["macro_auprc"],
                 results["full_history"]["auprc"], results["full_history"]["macro_auprc"],
                 time.time() - t0), flush=True)

        # 前提確認: 長さ 1.0 ns の点で対応する L 実験の記録と照合
        lref = json.loads((REPO_EXP / L_REF_DIR[h] / "results.json").read_text())
        lm = lref[f"metrics_{h}ns"]
        lg = lref[f"base_gain_{h}ns"]
        lrec = lref["recovery_vs_real_history_gain"]["prefix_mean_1.0ns"]
        gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
        gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]
        rec10_pooled = (results[COND_ADJ[1.0]]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        rec10_macro = (results[COND_ADJ[1.0]]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

        def chk(name, got, ref, tol):
            ok = abs(got - ref) < tol
            all_checks.append(dict(horizon_ns=h, check=name, got=float(got),
                                   reference=float(ref), tol=tol, ok=bool(ok),
                                   reference_source=f"experiments/{L_REF_DIR[h]}/results.json"))
            print(f"check [{h} ns]", name, "got %.4f ref %.4f tol %g -> %s" % (got, ref, tol, ok), flush=True)
            return ok

        chk("static_pooled_vs_L", results["static_only"]["auprc"], lm["static_only"]["auprc"], 5e-4)
        chk("static_macro_vs_L", results["static_only"]["macro_auprc"], lm["static_only"]["macro_auprc"], 5e-4)
        chk("full_pooled_vs_L", results["full_history"]["auprc"], lm["full_history"]["auprc"], 5e-4)
        chk("full_macro_vs_L", results["full_history"]["macro_auprc"], lm["full_history"]["macro_auprc"], 5e-4)
        chk("gain_pooled_vs_L", gain_pooled, lg["pooled"], 1e-3)
        chk("gain_macro_vs_L", gain_macro, lg["macro"], 1e-3)
        chk("recovery_1.0_pooled_vs_L", rec10_pooled, lrec["pooled"], 5e-3)
        chk("recovery_1.0_macro_vs_L", rec10_macro, lrec["macro"], 5e-3)

    premise_ok = all(c["ok"] for c in all_checks)
    (OUT / "premise_check.json").write_text(json.dumps(
        dict(checks=all_checks, all_ok=bool(premise_ok)), indent=2, ensure_ascii=False))
    if not premise_ok:
        print("PREMISE FAILED: 長さ 1.0 ns の点が L の記録と一致しないため、ここで止める。", flush=True)
        sys.exit(2)
    print("premise OK for all 6 horizons (%.0fs)" % (time.time() - t0), flush=True)

    # ---------- 第2段階: 予測幅ごとに回復率・読み・ブートストラップ ----------
    horizons_out = {}
    csv_lines = ["horizon_ns,covered_ns,n_frames,frames,auprc_pooled,auprc_macro,recovery_pooled,recovery_macro,"
                 "recovery_pooled_ci_lower,recovery_pooled_ci_upper,recovery_macro_ci_lower,recovery_macro_ci_upper,"
                 "increment_pooled,increment_macro,auc,brier,logloss,L_recovery_pooled,L_recovery_macro"]
    pp_lines = ["horizon_ns,protein,n,positives,auprc_static,auprc_full,full_gain,"
                + ",".join(f"auprc_{L:.1f}ns" for L in LENGTHS) + ","
                + ",".join(f"recovery_{L:.1f}ns" for L in LENGTHS)]
    npz_payload = dict(
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]))

    for hi, h in enumerate(HORIZONS):
        hp = per_horizon[h]
        preds, results, per_protein_ap, y_test = hp["preds"], hp["results"], hp["per_protein_ap"], hp["y_test"]
        lref = json.loads((REPO_EXP / L_REF_DIR[h] / "results.json").read_text())
        lrec_all = lref["recovery_vs_real_history_gain"]

        gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
        gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

        def rec(cname, kind):
            if kind == "pooled":
                return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
            return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

        recovery = {c: dict(pooled=float(rec(c, "pooled")), macro=float(rec(c, "macro")))
                    for c in CONDITIONS if c != "static_only"}
        increments = {}
        for kind in ("pooled", "macro"):
            increments[kind] = {}
            prev = None
            for L in LENGTHS:
                r = recovery[COND_ADJ[L]][kind]
                if prev is not None:
                    increments[kind][f"{L:.1f}"] = float(r - prev)
                prev = r

        def min_length(kind, thr):
            for L in LENGTHS:
                if recovery[COND_ADJ[L]][kind] >= thr:
                    return L
            return None

        def largest_step(kind):
            inc = increments[kind]
            key = max(inc, key=lambda k: inc[k])
            Lto = float(key)
            return dict(from_ns=Lto - 0.1, to_ns=Lto, increment=inc[key])

        # タンパク質ごとの回復率
        per_protein = {}
        for p, s, e in spans:
            yy = y_test[s:e]
            entry = dict(n=int(e - s), positives=int(yy.sum()))
            ap_s = per_protein_ap["static_only"][p]
            ap_f = per_protein_ap["full_history"][p]
            entry["auprc_static"] = ap_s
            entry["auprc_full"] = ap_f
            entry["full_gain"] = (ap_f - ap_s) if (ap_s is not None and ap_f is not None) else None
            entry["auprc_adjacent"] = {}
            entry["recovery_adjacent"] = {}
            for L in LENGTHS:
                ap = per_protein_ap[COND_ADJ[L]][p]
                entry["auprc_adjacent"][f"{L:.1f}"] = ap
                if ap is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                    entry["recovery_adjacent"][f"{L:.1f}"] = float((ap - ap_s) / entry["full_gain"])
                else:
                    entry["recovery_adjacent"][f"{L:.1f}"] = None
            per_protein[p] = entry

        # ブートストラップ (L02 と同じ段取り)
        for cname in CONDITIONS:
            ap_np = ap_numpy(y_test, preds[cname].astype(np.float64))
            assert abs(ap_np - results[cname]["auprc"]) < 1e-9, (h, cname, ap_np, results[cname]["auprc"])
        boot_prep = {}
        for cname in CONDITIONS:
            score = preds[cname].astype(np.float64)
            order = np.argsort(-score, kind="stable")
            ss = score[order]
            starts = np.flatnonzero(np.r_[True, ss[:-1] != ss[1:]])
            boot_prep[cname] = dict(
                y_sorted=y_test[order].astype(np.float64),
                prot_sorted=prot_of_row[order],
                starts=starts)

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
        ap_mat = np.array([[per_protein_ap[c][p] for c in CONDITIONS] for p in defined])
        cond_idx = {c: j for j, c in enumerate(CONDITIONS)}
        # draw (TEST の添字) -> defined リスト内の位置 (-1 は未定義)
        draw_to_defined = np.full(len(TEST), -1, dtype=int)
        pos = 0
        for i, (p, _, _) in enumerate(spans):
            if p in defined:
                draw_to_defined[i] = pos
                pos += 1

        rng = np.random.default_rng(BOOT_SEED)
        nP = len(TEST)
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
                    ap = pooled_ap_boot(boot_prep[COND_ADJ[L]], mult)
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
                            (m[cond_idx[COND_ADJ[L]]] - m[cond_idx["static_only"]]) / gm)
            if (b + 1) % 2000 == 0:
                print(f"[{h} ns] bootstrap %d/%d (%.0fs)" % (b + 1, N_BOOT, time.time() - t0), flush=True)

        bootstrap = {}
        for kind in ("pooled", "macro"):
            bootstrap[kind] = {}
            for L in LENGTHS:
                arr = np.array(boot_rec[kind][f"{L:.1f}"])
                if len(arr) == 0 or np.isnan(arr).any():
                    raise RuntimeError(f"bootstrap {h} {kind} {L}: invalid values (n={len(arr)}, nan={int(np.isnan(arr).sum())})")
                bootstrap[kind][f"{L:.1f}"] = dict(
                    n_boot=int(len(arr)), seed=BOOT_SEED,
                    point_estimate=recovery[COND_ADJ[L]][kind],
                    ci_lower=float(np.percentile(arr, 2.5)),
                    ci_upper=float(np.percentile(arr, 97.5)),
                    boot_mean=float(arr.mean()), boot_std=float(arr.std()))
        print(f"[{h} ns] bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)

        step = largest_step("pooled")
        Lfrom, Lto = step["from_ns"], step["to_ns"]
        step_inc_protein = []
        for p, entry in per_protein.items():
            r_from = entry["recovery_adjacent"].get(f"{Lfrom:.1f}")
            r_to = entry["recovery_adjacent"].get(f"{Lto:.1f}")
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

        # 対応する L 実験との比較 (同じ長さでの回復率の差)
        comparison_vs_L = {}
        for L in LENGTHS:
            lr = lrec_all[f"prefix_mean_{L:.1f}ns"]
            comparison_vs_L[f"{L:.1f}"] = dict(
                L_pooled=float(lr["pooled"]), L_macro=float(lr["macro"]),
                diff_pooled=float(recovery[COND_ADJ[L]]["pooled"] - lr["pooled"]),
                diff_macro=float(recovery[COND_ADJ[L]]["macro"] - lr["macro"]))
        l_readings = lref["readings"]
        comparison_readings = dict(
            L_min_length_0_8=l_readings["min_length_recovery_0_8"],
            L_min_length_0_9=l_readings["min_length_recovery_0_9"],
            L_largest_step=l_readings["largest_step"])

        horizons_out[str(h)] = dict(
            horizon_ns=h,
            reference_experiment=L_REF_DIR[h],
            metrics=results,
            base_gain=dict(pooled=float(gain_pooled), macro=float(gain_macro)),
            recovery_vs_real_history_gain=recovery,
            recovery_increment_per_0p1ns=increments,
            readings=readings,
            bootstrap=dict(n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
                           resampling_unit="protein (100 test proteins, resampled with replacement)",
                           n_draws_with_nonpositive_gain=n_bad_gain,
                           results=bootstrap),
            comparison_vs_L=dict(recovery_by_length=comparison_vs_L, readings=comparison_readings),
            per_protein=per_protein)

        for L in LENGTHS:
            c = COND_ADJ[L]
            m = results[c]
            csv_lines.append(",".join([
                str(h), f"{L:.1f}", str(len(FRAMES[L])),
                '"' + " ".join(str(f) for f in FRAMES[L]) + '"',
                f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}",
                f"{recovery[c]['pooled']:.6f}", f"{recovery[c]['macro']:.6f}",
                f"{bootstrap['pooled'][f'{L:.1f}']['ci_lower']:.6f}",
                f"{bootstrap['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
                f"{bootstrap['macro'][f'{L:.1f}']['ci_lower']:.6f}",
                f"{bootstrap['macro'][f'{L:.1f}']['ci_upper']:.6f}",
                f"{increments['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
                f"{increments['macro'].get(f'{L:.1f}', float('nan')):.6f}",
                f"{m['auc']:.6f}", f"{m['brier']:.6f}", f"{m['logloss']:.6f}",
                f"{comparison_vs_L[f'{L:.1f}']['L_pooled']:.6f}",
                f"{comparison_vs_L[f'{L:.1f}']['L_macro']:.6f}"]))
        for p, entry in per_protein.items():
            row = [str(h), p, str(entry["n"]), str(entry["positives"]),
                   "" if entry["auprc_static"] is None else f"{entry['auprc_static']:.6f}",
                   "" if entry["auprc_full"] is None else f"{entry['auprc_full']:.6f}",
                   "" if entry["full_gain"] is None else f"{entry['full_gain']:.6f}"]
            row += ["" if entry["auprc_adjacent"][f"{L:.1f}"] is None else f"{entry['auprc_adjacent'][f'{L:.1f}']:.6f}" for L in LENGTHS]
            row += ["" if entry["recovery_adjacent"][f"{L:.1f}"] is None else f"{entry['recovery_adjacent'][f'{L:.1f}']:.6f}" for L in LENGTHS]
            pp_lines.append(",".join(row))
        npz_payload[f"h{h}_label"] = y_test.astype(np.int8)
        for cname in CONDITIONS:
            npz_payload[f"h{h}_{cname}"] = preds[cname].astype(np.float32)
        print(f"[{h} ns] done: 0.8 pooled {readings['min_length_recovery_0_8']['pooled']}, "
              f"0.9 pooled {readings['min_length_recovery_0_9']['pooled']}, "
              f"largest step {step['from_ns']}->{step['to_ns']} {step['increment']:.3f} (%.0fs)"
              % (time.time() - t0), flush=True)

    out = {
        "label": "HB formation, guidance recovery vs adjacent-window covered length, strong base (K06b), horizons 1/5/10/30/50/90 ns",
        "series": "L2 (revision of L: window placed immediately before the origin frame 10, E01 Protocol A)",
        "length_definition": {
            "unit": "covered length (ns); frame count same as L01/L02, but the window is the last L/0.1+1 frames ending at the origin frame 10",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=len(FRAMES[L]), frames=FRAMES[L]) for L in LENGTHS},
            "note": "Length 0 = frame 10 only; 0.1 ns = frames 9-10; ...; 1.0 ns = frames 0-10 (identical to the L condition, used for the premise check). In L the window was frames 0..(L/0.1) paired with origin frame 10, leaving a time gap for short lengths.",
        },
        "base": {
            "name": "K06(b) strong base: frozen T1I2006 + HistGBM sidecar (160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained on 8 calibration proteins, evaluated on 100 unseen proteins x 3 replicas",
            "static_features": 5, "history_features": 10,
            "n_calibration": int(len(cal["label"])), "n_test": int(len(test["label"])),
        },
        "premise_checks": all_checks,
        "premise_all_ok": bool(premise_ok),
        "bootstrap_note": "Within each resample, static-only, full-history and adjacent-mean AUPRCs are recomputed and the gain and recovery are recomputed inside the sample. Models are not refit; per-candidate predicted probabilities are resampled. Pooled AUPRC uses a numpy implementation of sklearn's average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every condition and horizon. Macro AUPRC within a resample is the multiplicity-weighted mean of per-protein AUPRCs over proteins whose AUPRC is defined.",
        "horizons": horizons_out,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (OUT / "results.csv").write_text("\n".join(csv_lines) + "\n")
    (OUT / "per_protein.csv").write_text("\n".join(pp_lines) + "\n")
    np.savez_compressed(OUT / "test_predictions.npz", **npz_payload)
    print("saved results.json / results.csv / per_protein.csv / test_predictions.npz (%.0fs total)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
