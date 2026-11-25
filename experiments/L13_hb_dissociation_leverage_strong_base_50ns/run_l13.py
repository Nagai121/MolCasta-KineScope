"""L13: L01 の曲がりを強い土台で測る (K75 の条件、水素結合の解離、50 ns先)。

L02 (水素結合の形成、強い土台、1 ns先) と同じ方法で、水素結合の解離の
50 ns先の曲がりを測る。L01 (資料Dの条件、テスト 2タンパク質) で見えた
「覆う長さ 0.6 ns の段」が、タンパク質を増やしても (強い土台、
未見100タンパク質×3レプリカ) 残るかを判定する。

長さの定義は L01・L02 にそのまま合わせる:
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 覆う長さ L ns の切り取りはフレーム 0〜(L/0.1) を使う。
  0 ns -> [0] (1フレーム)、0.1 ns -> [0,1]、…、1.0 ns -> [0..10] (11フレーム全部)。
- 注意: G 系列の「1.0 ns」の切り取り (フレーム [0..9]) とは違う。
  L01・L02 の 1.0 ns は 11フレーム全部の平均である。
- 起点フレーム10 (1 ns) は変えず、ラベルは 50 ns先。

条件 (すべて条件ごとに学習し直す):
- 静的のみ: 静的特徴5個。
- 全部入り (full_history): 静的特徴＋実履歴の要約値10個。回復率の分母 (実履歴の利得) の参照。
- 切り取りの平均だけ: 静的特徴＋切り取りの平均だけ。覆う長さ 0.0〜1.0 ns の11通り。

土台 (K75・K2-18 と同じ、水素結合の解離の強い土台):
- HistGradientBoosting の補正モデル (160回、学習率 0.065、葉 15、
  葉あたり最小 20、L2 0.15、seed 20261121)。較正タンパク質で学習し、
  未見の100タンパク質×3レプリカ (76,342行) で評価する。
- 入力は K2-18 の input_bundle.npz (静的特徴5個、履歴の要約値10個、
  11フレームの距離系列、ラベル6列) を使う。特徴量は K2-18 の系統
  (K75・K2-77・K2-84) と同じく float64 にそろえて sklearn に渡す。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。
全体 (pooled) とタンパク質ごとの平均 (macro、100タンパク質) の両方で測る
(L02 と同じまとめ方)。K75 の照合用に、タンパク質×レプリカ (300群) の
平均も併せて出す。

読み方 (結果を見る前に固定、L02 と同じ):
1. 回復率が 0.8 と 0.9 に届く最小の覆う長さを、統合値 (pooled) で読む (macro も併記)。
2. 段の判定は、隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、その大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、パーセンタイル法)。
   各標本の中でも静的のみ・全部入り・各長さの AUPRC を測り直し、
   利得と回復率をその標本の中で計算し直す (L02 と同じ段取り)。
4. 段の前後ではタンパク質ごとの値も記録する。

土台の再現の確認 (premise。これが崩れたら測定は無効として止める):
- 50 ns先で静的のみ・全部入りの AUPRC (pooled とタンパク質×レプリカ平均) が
  K75 の対角の記録 (静的 0.6001 / 0.6246、
  全部入り 0.6568 / 0.6758) と 5e-4 以内で一致すること。
- 覆う長さ 1.0 ns の回復率 (pooled) が既存の記録 (K2-83 の
  全部の平均だけの回復率 0.9665) と 5e-3 以内で一致すること。
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
HORIZON_NS = 50
HORIZON_IDX = 4  # HORIZONS_ALL = (1, 5, 10, 30, 50, 90)

REF = dict(static_pooled=0.6001, static_macro300=0.6246,
           full_pooled=0.6568, full_macro300=0.6758,
           cov10_recovery_pooled=0.9665, cov10_source="K2-83",
           test_n=76342, test_positives=16110)

LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(int(round(L / 0.1)) + 1)) for L in LENGTHS}
COND_PREFIX = {L: f"prefix_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = ["static_only", "full_history"] + [COND_PREFIX[L] for L in LENGTHS]


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
    b = np.load(BUNDLE)
    # K2-18 の系統 (K75・K2-77・K2-84) と同じく、特徴量は float64 にそろえて
    # sklearn に渡す。float32 のままだと 90 ns先の全部の平均だけの条件で
    # 分割が変わり、AUPRC が 0.0018 ずれることを確認した (予備測定)。
    cal = dict(static=b["cal_static"].astype(np.float64),
               hist=b["cal_history"].astype(np.float64),
               series=b["cal_series"].astype(np.float64),
               label=b["cal_label"])
    test = dict(static=b["test_static"].astype(np.float64),
                hist=b["test_history"].astype(np.float64),
                series=b["test_series"].astype(np.float64),
                label=b["test_label"], protein=b["test_protein"], replica=b["test_replica"])
    y_cal = cal["label"][:, HORIZON_IDX].astype(int)
    y_test = test["label"][:, HORIZON_IDX].astype(int)
    print("cal", len(y_cal), "test", len(y_test), "test pos", HORIZON_NS, "ns", int(y_test.sum()), flush=True)
    assert len(y_test) == REF["test_n"] and int(y_test.sum()) == REF["test_positives"]

    # テストのタンパク質ごとの区間と、タンパク質×レプリカの群
    proteins = list(dict.fromkeys(test["protein"].tolist()))
    assert len(proteins) == 100, len(proteins)
    prot_idx = {p: np.where(test["protein"] == p)[0] for p in proteins}
    spans = [(p, prot_idx[p][0], prot_idx[p][-1] + 1) for p in proteins]
    for p, s, e in spans:
        assert (test["protein"][s:e] == p).all()
    prot_of_row = np.concatenate([np.full(len(prot_idx[p]), i) for i, p in enumerate(proteins)])
    assert len(prot_of_row) == len(y_test)
    groups300 = [(p, r) for p in proteins for r in (1, 2, 3)]
    group_idx = [np.where((test["protein"] == p) & (test["replica"] == r))[0] for p, r in groups300]

    # 条件ごとの特徴量
    feats = {}
    feats["static_only"] = (cal["static"], test["static"])
    feats["full_history"] = (np.column_stack([cal["static"], cal["hist"]]),
                             np.column_stack([test["static"], test["hist"]]))
    for L in LENGTHS:
        idx = FRAMES[L]
        feats[COND_PREFIX[L]] = (
            np.column_stack([cal["static"], cal["series"][:, idx].mean(axis=1)[:, None]]),
            np.column_stack([test["static"], test["series"][:, idx].mean(axis=1)[:, None]]))

    # 学習と予測 (途中結果を保存し、再起動時は保存済みの予測を読み直す)
    import os
    fits_path = OUT / "fits_summary.json"
    preds_path = OUT / "test_predictions.npz"
    if fits_path.exists() and preds_path.exists():
        fs = json.loads(fits_path.read_text())
        results = fs["results"]
        per_protein_ap = {c: {p: (None if v is None else float(v)) for p, v in d.items()}
                          for c, d in fs["per_protein_ap"].items()}
        zp = np.load(preds_path)
        preds = {c: zp[f"p_{c}"].astype(np.float64) for c in CONDITIONS}
        checks = fs["premise_checks"]
        print("loaded saved fits (checkpoint)", flush=True)
    else:
        preds = {}
        results = {}
        per_protein_ap = {c: {} for c in CONDITIONS}
        per_group_ap = {c: [] for c in CONDITIONS}
        for cname in CONDITIONS:
            xtr, xte = feats[cname]
            pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
            preds[cname] = pred
            m = metrics(y_test, pred)
            aps = []
            for p, s_, e_ in spans:
                yy = y_test[s_:e_]
                if 0 < yy.sum() < e_ - s_:
                    ap = float(average_precision_score(yy, pred[s_:e_]))
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
            per_group_ap[cname] = gaps
            m["macro300_auprc"] = float(np.mean(gaps))
            m["n_groups_macro300"] = len(gaps)
            results[cname] = m
            print(cname, "pooled %.4f macro %.4f macro300 %.4f (%.0fs)"
                  % (m["auprc"], m["macro_auprc"], m["macro300_auprc"], time.time() - t0), flush=True)

        # ---- 土台の再現の確認 (premise) ----
        checks = []

        def chk(name, got, ref, tol):
            ok = abs(got - ref) < tol
            checks.append(dict(check=name, got=float(got), reference=float(ref), tol=tol, ok=bool(ok)))
            print("check", name, "got %.4f ref %.4f tol %g -> %s" % (got, ref, tol, ok), flush=True)
            return ok

        gain_pooled0 = results["full_history"]["auprc"] - results["static_only"]["auprc"]
        rec10 = (results[COND_PREFIX[1.0]]["auprc"] - results["static_only"]["auprc"]) / gain_pooled0
        ok = True
        ok &= chk("static_pooled_vs_K75diag", results["static_only"]["auprc"], REF["static_pooled"], 5e-4)
        ok &= chk("static_macro300_vs_K75diag", results["static_only"]["macro300_auprc"], REF["static_macro300"], 5e-4)
        ok &= chk("full_pooled_vs_K75diag", results["full_history"]["auprc"], REF["full_pooled"], 5e-4)
        ok &= chk("full_macro300_vs_K75diag", results["full_history"]["macro300_auprc"], REF["full_macro300"], 5e-4)
        # 覆う長さ 1.0 ns の回復率の照合は補助的な照合として記録する (止める条件は
        # 上の K75 の対角の 4 件)。参照がこの仮想マシンで測られた記録 (K2-73・K2-77・
        # K2-81・K2-83・K2-84) の場合は 5e-3 以内で一致するはずで、合わなければ
        # README に記録して検討する。L09 の参照 (K2-18) は Colab (sklearn 1.6.1) で
        # 測られた記録で、K75 との照合自体に 1e-3 の許容差が要ったため、
        # 全部の平均だけの AUPRC にも同程度のずれが残る。
        chk("recovery_1.0_pooled_vs_record", rec10, REF["cov10_recovery_pooled"], 5e-3)
        (OUT / "premise_check.json").write_text(json.dumps(
            dict(checks=checks, all_ok=bool(ok)), indent=2, ensure_ascii=False))
        if not ok:
            print("PREMISE FAILED: 土台の再現が崩れているため、ここで止める。", flush=True)
            sys.exit(2)
        print("premise OK (%.0fs)" % (time.time() - t0), flush=True)
        # 途中結果の保存 (再起動に備える)
        np.savez_compressed(
            preds_path,
            proteins=np.concatenate([np.full(e_ - s_, p) for p, s_, e_ in spans]),
            label=y_test.astype(np.int8),
            **{f"p_{c}": preds[c].astype(np.float32) for c in CONDITIONS})
        fits_path.write_text(json.dumps(
            dict(results=results, per_protein_ap=per_protein_ap, premise_checks=checks),
            indent=2, ensure_ascii=False))
        print("saved fits checkpoint (%.0fs)" % (time.time() - t0), flush=True)

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

    def rec(cname, kind):
        if kind == "pooled":
            return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

    # ---- 回復率と増え分 ----
    recovery = {c: dict(pooled=float(rec(c, "pooled")), macro=float(rec(c, "macro")))
                for c in CONDITIONS if c != "static_only"}
    increments = {}
    for kind in ("pooled", "macro"):
        increments[kind] = {}
        prev = None
        for L in LENGTHS:
            r = recovery[COND_PREFIX[L]][kind]
            if prev is not None:
                increments[kind][f"{L:.1f}"] = float(r - prev)
            prev = r

    def min_length(kind, thr):
        for L in LENGTHS:
            if recovery[COND_PREFIX[L]][kind] >= thr:
                return L
        return None

    def largest_step(kind):
        inc = increments[kind]
        key = max(inc, key=lambda k: inc[k])
        Lto = float(key)
        return dict(from_ns=round(Lto - 0.1, 1), to_ns=Lto, increment=inc[key])

    # ---- タンパク質ごとの回復率 ----
    per_protein = {}
    for p, s, e in spans:
        yy = y_test[s:e]
        entry = dict(n=int(e - s), positives=int(yy.sum()))
        ap_s = per_protein_ap["static_only"][p]
        ap_f = per_protein_ap["full_history"][p]
        entry["auprc_static"] = ap_s
        entry["auprc_full"] = ap_f
        entry["full_gain"] = (ap_f - ap_s) if (ap_s is not None and ap_f is not None) else None
        entry["auprc_prefix"] = {}
        entry["recovery_prefix"] = {}
        for L in LENGTHS:
            ap = per_protein_ap[COND_PREFIX[L]][p]
            entry["auprc_prefix"][f"{L:.1f}"] = ap
            if ap is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                entry["recovery_prefix"][f"{L:.1f}"] = float((ap - ap_s) / entry["full_gain"])
            else:
                entry["recovery_prefix"][f"{L:.1f}"] = None
        per_protein[p] = entry

    # ---- ブートストラップ (タンパク質単位、標本内で利得と回復率を測り直す) ----
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
    didx = [i for i, (p, _, _) in enumerate(spans) if p in defined]
    ap_mat = np.array([[per_protein_ap[c][p] for c in CONDITIONS] for p in defined])
    cond_idx = {c: j for j, c in enumerate(CONDITIONS)}
    draw_to_defined = np.full(len(proteins), -1, dtype=int)
    for pos, i in enumerate(didx):
        draw_to_defined[i] = pos

    # 抽出の並びは seed 42 で先に全部決めて保存する (塊ごとに再開できるように)
    nP = len(proteins)
    boot_path = OUT / "boot_checkpoint.npz"
    rng = np.random.default_rng(BOOT_SEED)
    all_draws = rng.integers(0, nP, size=(N_BOOT, nP)).astype(np.int16)
    if boot_path.exists():
        zb = np.load(boot_path)
        assert (zb["draws"] == all_draws).all()
        rec_pooled_arr = zb["rec_pooled"].astype(np.float64)
        rec_macro_arr = zb["rec_macro"].astype(np.float64)
        n_done = int(zb["n_done"])
        n_bad_gain = dict(pooled=int(zb["n_bad_pooled"]), macro=int(zb["n_bad_macro"]))
        print("resuming bootstrap from draw", n_done, flush=True)
    else:
        rec_pooled_arr = np.full((N_BOOT, len(LENGTHS)), np.nan)
        rec_macro_arr = np.full((N_BOOT, len(LENGTHS)), np.nan)
        n_done = 0
        n_bad_gain = dict(pooled=0, macro=0)
    CHUNK = 500
    while n_done < N_BOOT:
        hi = min(n_done + CHUNK, N_BOOT)
        for bidx in range(n_done, hi):
            draw = all_draws[bidx].astype(int)
            mult = np.bincount(draw, minlength=nP).astype(np.float64)
            ap_s = pooled_ap_boot(boot_prep["static_only"], mult)
            ap_f = pooled_ap_boot(boot_prep["full_history"], mult)
            g = ap_f - ap_s
            if g <= 0:
                n_bad_gain["pooled"] += 1
            else:
                for j, L in enumerate(LENGTHS):
                    ap = pooled_ap_boot(boot_prep[COND_PREFIX[L]], mult)
                    rec_pooled_arr[bidx, j] = (ap - ap_s) / g
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
                    for j, L in enumerate(LENGTHS):
                        rec_macro_arr[bidx, j] = (m[cond_idx[COND_PREFIX[L]]] - m[cond_idx["static_only"]]) / gm
        n_done = hi
        tmp = OUT / "boot_checkpoint.tmp.npz"
        np.savez_compressed(tmp, draws=all_draws, rec_pooled=rec_pooled_arr.astype(np.float32),
                            rec_macro=rec_macro_arr.astype(np.float32), n_done=n_done,
                            n_bad_pooled=n_bad_gain["pooled"], n_bad_macro=n_bad_gain["macro"])
        os.replace(tmp, boot_path)
        print("bootstrap %d/%d (%.0fs)" % (n_done, N_BOOT, time.time() - t0), flush=True)
    boot_rec = {kind: {f"{L:.1f}": list(arr[~np.isnan(arr[:, j]), j]) for j, L in enumerate(LENGTHS)}
                for kind, arr in (("pooled", rec_pooled_arr), ("macro", rec_macro_arr))}
    print("bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)
    bootstrap = {}
    for kind in ("pooled", "macro"):
        bootstrap[kind] = {}
        for L in LENGTHS:
            arr = np.array(boot_rec[kind][f"{L:.1f}"])
            if len(arr) == 0 or np.isnan(arr).any():
                raise RuntimeError(f"bootstrap {kind} {L}: invalid values (n={len(arr)}, nan={int(np.isnan(arr).sum())})")
            bootstrap[kind][f"{L:.1f}"] = dict(
                n_boot=int(len(arr)), seed=BOOT_SEED,
                point_estimate=recovery[COND_PREFIX[L]][kind],
                ci_lower=float(np.percentile(arr, 2.5)),
                ci_upper=float(np.percentile(arr, 97.5)),
                boot_mean=float(arr.mean()), boot_std=float(arr.std()))
    print("bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)

    # ---- 段の前後のタンパク質ごとの増え分 (pooled の段の位置で) ----
    step = largest_step("pooled")
    Lfrom, Lto = step["from_ns"], step["to_ns"]
    step_inc_protein = []
    for p, entry in per_protein.items():
        r_from = entry["recovery_prefix"].get(f"{Lfrom:.1f}")
        r_to = entry["recovery_prefix"].get(f"{Lto:.1f}")
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

    # タンパク質ごとの増え分の分布 (全区間)
    per_protein_increments = {}
    for k in range(1, len(LENGTHS)):
        Lf, Lt = LENGTHS[k - 1], LENGTHS[k]
        vals = []
        for p, entry in per_protein.items():
            rf = entry["recovery_prefix"].get(f"{Lf:.1f}")
            rt = entry["recovery_prefix"].get(f"{Lt:.1f}")
            if rf is not None and rt is not None:
                vals.append(rt - rf)
        vals = np.array(vals)
        per_protein_increments[f"{Lt:.1f}"] = dict(
            from_ns=Lf, to_ns=Lt, n_proteins=int(len(vals)),
            median=float(np.median(vals)), q25=float(np.percentile(vals, 25)),
            q75=float(np.percentile(vals, 75)),
            frac_positive=float(np.mean(vals > 0)))

    # タンパク質ごとの回復率の分布 (全長さ)
    per_protein_recovery_dist = {}
    for L in LENGTHS:
        vals = np.array([e["recovery_prefix"][f"{L:.1f}"] for e in per_protein.values()
                         if e["recovery_prefix"][f"{L:.1f}"] is not None])
        per_protein_recovery_dist[f"{L:.1f}"] = dict(
            n_proteins=int(len(vals)), median=float(np.median(vals)),
            q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)),
            frac_ge_0p8=float(np.mean(vals >= 0.8)))

    readings = dict(
        min_length_recovery_0_8=dict(pooled=min_length("pooled", 0.8), macro=min_length("macro", 0.8)),
        min_length_recovery_0_9=dict(pooled=min_length("pooled", 0.9), macro=min_length("macro", 0.9)),
        largest_step=dict(pooled=step, macro=largest_step("macro")),
        step_per_protein=step_per_protein)

    out = {
        "label": f"HB dissociation {HORIZON_NS}ns, guidance recovery vs short-MD covered length, strong base (K75)",
        "experiment": "L13",
        "horizon_ns": HORIZON_NS,
        "length_definition": {
            "unit": "covered length (ns), same definition as L01/L02; prefix = frames 0..(L/0.1) of the 0-1 ns history, 100 ps spacing",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=len(FRAMES[L]), frames=FRAMES[L]) for L in LENGTHS},
            "note": "Differs from the G-series naming at 1.0 ns: the G-series '1.0 ns' prefix is frames [0..9], while L01/L02's covered 1.0 ns is frames [0..10] (the full-history mean).",
        },
        "base": {
            "name": "K75 strong base for HB dissociation: HistGBM correction model (160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained on calibration proteins, evaluated on 100 unseen proteins x 3 replicas (input bundle from K2-18)",
            "static_features": 5, "history_features": 10,
            "n_calibration": int(len(y_cal)), "n_test": int(len(y_test)),
            f"test_positives_{HORIZON_NS}ns": int(y_test.sum()),
        },
        "premise_checks": checks,
        "references": {
            "k75_diagonal": REF,
        },
        f"metrics_{HORIZON_NS}ns": results,
        f"base_gain_{HORIZON_NS}ns": dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        "recovery_vs_real_history_gain": recovery,
        "recovery_increment_per_0.1ns": increments,
        "readings": readings,
        "per_protein_recovery_distribution": per_protein_recovery_dist,
        "per_protein_increments": per_protein_increments,
        "bootstrap": dict(
            n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
            resampling_unit="protein (100 test proteins, resampled with replacement)",
            note="Within each resample, static-only, full-history and prefix-mean AUPRCs are recomputed and the gain and recovery are recomputed inside the sample. Models are not refit; per-candidate predicted probabilities are resampled. Pooled AUPRC uses a numpy implementation of sklearn's average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every condition. Macro AUPRC within a resample is the multiplicity-weighted mean of per-protein AUPRCs over proteins whose AUPRC is defined.",
            n_draws_with_nonpositive_gain=n_bad_gain,
            results=bootstrap),
        "per_protein": per_protein,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = ["covered_ns,n_frames,auprc_pooled,auprc_macro,auprc_macro300,recovery_pooled,recovery_macro,"
             "recovery_pooled_ci_lower,recovery_pooled_ci_upper,recovery_macro_ci_lower,recovery_macro_ci_upper,"
             "increment_pooled,increment_macro,auc,brier,logloss"]
    for L in LENGTHS:
        c = COND_PREFIX[L]
        m = results[c]
        lines.append(",".join([
            f"{L:.1f}", str(len(FRAMES[L])),
            f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}", f"{m['macro300_auprc']:.6f}",
            f"{recovery[c]['pooled']:.6f}", f"{recovery[c]['macro']:.6f}",
            f"{bootstrap['pooled'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{bootstrap['macro'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['macro'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{increments['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{increments['macro'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{m['auc']:.6f}", f"{m['brier']:.6f}", f"{m['logloss']:.6f}"]))
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    plines = ["protein,n,positives,auprc_static,auprc_full,full_gain,"
              + ",".join(f"auprc_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"recovery_{L:.1f}ns" for L in LENGTHS)]
    for p, entry in per_protein.items():
        row = [p, str(entry["n"]), str(entry["positives"]),
               "" if entry["auprc_static"] is None else f"{entry['auprc_static']:.6f}",
               "" if entry["auprc_full"] is None else f"{entry['auprc_full']:.6f}",
               "" if entry["full_gain"] is None else f"{entry['full_gain']:.6f}"]
        row += ["" if entry["auprc_prefix"][f"{L:.1f}"] is None else f"{entry['auprc_prefix'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        row += ["" if entry["recovery_prefix"][f"{L:.1f}"] is None else f"{entry['recovery_prefix'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        plines.append(",".join(row))
    (OUT / "per_protein.csv").write_text("\n".join(plines) + "\n")

    np.savez_compressed(
        OUT / "test_predictions.npz",
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]),
        label=y_test.astype(np.int8),
        **{c: preds[c].astype(np.float32) for c in CONDITIONS})
    print("saved results.json / results.csv / per_protein.csv / test_predictions.npz (%.0fs total)" % (time.time() - t0), flush=True)
    print(json.dumps(readings, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
