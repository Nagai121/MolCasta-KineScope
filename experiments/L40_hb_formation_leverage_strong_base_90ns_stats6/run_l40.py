"""L40: 統計量6種を使った場合の曲がりを予測幅 90 ns で測る (K06(b) の条件、水素結合の形成、90 ns先)。

L08 (強い土台、90 ns先、履歴は切り取りの平均だけ) で測った
「覆う長さに対する回復率の曲がり」を、履歴の平均だけではなく
統計量6種 (実験Dの定義) を使った場合に測り直す。
平均だけの場合 (L08) に比べて、必要な長さが縮むかを見る。

統計量6種の定義 (実験D、fixed_1ns_history_3phen_6h.py の features_from_history):
- 系列 (フレームごとの供与体・受容体間距離) に対して
  [最初の値, 平均, 標準偏差 (ddof=0), 最小値, 最後-最初, (系列 <= 3.5 Å) の割合]
  を float32 で並べたもの。水素結合の閾値は 3.5 Å (実験Dの THRESH["hb"]=3.5)。
- K06(b) の履歴特徴10個 (score_b.py の history) は
  [最初, 平均, 標準偏差, 最小, 最大, 最後-最初, 最後-直前,
   3.5 Å以下割合, 4.5 Å以下割合, 増加割合] で、
  統計量6種はその [0,1,2,3,5,7] 列と正確に一致する部分集合である。
  K06(b) の score_b.py には同じ6種を生成系列に適用する gen6 もあり、定義は共通。
- 本スクリプトの前提確認で、g05_series から計算した統計量6種 (全11フレーム) が
  k06b_out の保存値の該当列と 1e-4 以内で一致することを全タンパク質で確かめる。

長さの定義は L01・L02・L08 にそのまま合わせる:
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 覆う長さ L ns の切り取りはフレーム 0〜(L/0.1) を使う。
  0 ns -> [0] (1フレーム)、0.1 ns -> [0,1]、…、1.0 ns -> [0..10] (11フレーム全部)。
- 起点フレーム10 (1 ns) は変えず、ラベルは 90 ns先 (ラベルの列の添字 5、
  予測幅の並びは [1, 5, 10, 30, 50, 90] ns)。

条件 (すべて条件ごとに学習し直す):
- 静的のみ: 静的特徴5個。
- 全部入り (full_history): 静的特徴＋実履歴の要約値10個。回復率の分母 (実履歴の利得) の参照。
- 切り取りの平均だけ: 静的特徴＋切り取りの平均だけ。覆う長さ 0.0〜1.0 ns の11通り (L08 と同じ条件、前提確認用)。
- 切り取りの統計量6種: 静的特徴＋切り取りの統計量6種。覆う長さ 0.0〜1.0 ns の11通り (本測定)。

土台 (K06(b) と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、seed 20261121)。
  較正タンパク質 8つで学習し、未見の100タンパク質×3レプリカ (163,074行) で評価する。
- 系列は G05 で作り直したもの (~/workspace/kinescope/work/g05_series/) を使う。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。
全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る (G05 と同じまとめ方)。

読み方 (結果を見る前に固定):
1. 回復率が 0.8 と 0.9 に届く最小の覆う長さを、統合値 (pooled) で読む (macro も併記)。
   L08 (平均だけ) の答え (0.8 は全体・平均とも 0.2 ns、0.9 は全体・平均とも 0.4 ns) と
   比べて、縮んだかどうかを明記する。
2. 段の判定は、隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、その大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、パーセンタイル法)。
   各標本の中でも静的のみ・全部入り・各長さの AUPRC を測り直し、
   利得と回復率をその標本の中で計算し直す (G147・G148・G162 と同じ段取り)。
4. 段の前後ではタンパク質ごとの値も記録する。

土台の再現の確認 (premise。これが崩れたら測定は無効として止める):
- 90 ns先で静的のみ・全部入りの AUPRC (pooled / macro) が k06b の記録
  (~/workspace/k06b/results_100.json の by_horizon の "90":
  0.4418 / 0.4357、0.4683 / 0.4630) と 5e-4 以内で一致すること (L08 と同じ)。
- テスト陽性が k06b の記録の 13,068件と一致すること。
- 平均だけの条件の回復率が G81 (強い土台の切り取り、90 ns先) の記録と
  5e-3 以内で一致すること (L08 と同じ照合。統計量6種の平均が L08 と同じ
  入力になっていることの確認を兼ねる):
  覆う長さ 0.1 ns で 0.747 / 0.695、0.5 ns で 1.025 / 1.008、
  1.0 ns (= 全部の平均だけ) で 1.084 / 1.060。
- 統計量6種の定義の確認: g05_series の全11フレームから計算した統計量6種が、
  k06b_out の履歴特徴10個の [0,1,2,3,5,7] 列と較正・テスト全タンパク質で
  1e-4 以内で一致すること。
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
OUT = Path(__file__).resolve().parent
SEED = 20261121
BOOT_SEED = 42
N_BOOT = 10000
HORIZON_IDX = 5  # 90 ns先 (予測幅の並びは [1, 5, 10, 30, 50, 90])
HB_BOUND = 3.5  # 実験Dの THRESH["hb"]

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# L01 の長さの定義: 覆う長さ L ns -> フレーム 0..(L/0.1)
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(int(round(L / 0.1)) + 1)) for L in LENGTHS}
COND_MEAN = {L: f"prefix_mean_{L:.1f}ns" for L in LENGTHS}
COND_STATS6 = {L: f"prefix_stats6_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = (["static_only", "full_history"]
              + [COND_MEAN[L] for L in LENGTHS]
              + [COND_STATS6[L] for L in LENGTHS])


def stats6(series):
    """実験Dの features_from_history と同じ6種。series は (n_frames, n_candidates)。"""
    return np.column_stack([series[0], series.mean(axis=0), series.std(axis=0),
                            series.min(axis=0), series[-1] - series[0],
                            (series <= HB_BOUND).mean(axis=0)]).astype(np.float32)


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
    y_cal = cal["label"][:, HORIZON_IDX]
    y_test = test["label"][:, HORIZON_IDX]
    print("cal", len(y_cal), "test", len(y_test), "test pos 90ns", int(y_test.sum()), flush=True)

    # ---- 統計量6種の定義の確認 (g05_series の全11フレーム vs k06b_out の保存値) ----
    worst = 0.0
    for p in CAL + TEST:
        z = np.load(K06B / f"{p}.npz")
        s = np.load(SERIES / f"{p}.npz")
        for r in (1, 2, 3):
            st = stats6(s[f"r{r}_series"])
            ref = z[f"r{r}_history"][:, [0, 1, 2, 3, 5, 7]]
            worst = max(worst, float(np.abs(st - ref).max()))
    print("stats6 definition check: max |diff| vs k06b history cols [0,1,2,3,5,7] = %.3g" % worst, flush=True)

    # テストのタンパク質ごとの区間
    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n
    assert cur == len(y_test)
    prot_of_row = np.concatenate([np.full(e - s, i) for i, (_, s, e) in enumerate(spans)])

    # 条件ごとの特徴量
    feats = {}
    feats["static_only"] = (cal["static"], test["static"])
    feats["full_history"] = (np.column_stack([cal["static"], cal["hist"]]),
                             np.column_stack([test["static"], test["hist"]]))
    for L in LENGTHS:
        idx = FRAMES[L]
        feats[COND_MEAN[L]] = (
            np.column_stack([cal["static"], cal["series"][idx].mean(axis=0)[:, None].astype(np.float32)]),
            np.column_stack([test["static"], test["series"][idx].mean(axis=0)[:, None].astype(np.float32)]))
        feats[COND_STATS6[L]] = (
            np.column_stack([cal["static"], stats6(cal["series"][idx])]),
            np.column_stack([test["static"], stats6(test["series"][idx])]))

    # 学習と予測
    preds = {}
    results = {}
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
        print(cname, "pooled %.4f macro %.4f (%.0fs)" % (m["auprc"], m["macro_auprc"], time.time() - t0), flush=True)

    # ---- 土台の再現の確認 (premise) ----
    checks = []

    def chk(name, got, ref, tol):
        ok = abs(got - ref) < tol
        checks.append(dict(check=name, got=float(got), reference=float(ref), tol=tol, ok=bool(ok)))
        print("check", name, "got %.4f ref %.4f tol %g -> %s" % (got, ref, tol, ok), flush=True)
        return ok

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

    def rec(cname, kind):
        if kind == "pooled":
            return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

    ok = True
    ok &= chk("static_pooled_vs_k06b_90ns", results["static_only"]["auprc"], 0.4418, 5e-4)
    ok &= chk("static_macro_vs_k06b_90ns", results["static_only"]["macro_auprc"], 0.4357, 5e-4)
    ok &= chk("full_pooled_vs_k06b_90ns", results["full_history"]["auprc"], 0.4683, 5e-4)
    ok &= chk("full_macro_vs_k06b_90ns", results["full_history"]["macro_auprc"], 0.4630, 5e-4)
    ok &= chk("test_positives_vs_k06b_90ns", float(y_test.sum()), 13068.0, 0.5)
    ok &= chk("recovery_mean_0.1_pooled_vs_G81_90ns", rec(COND_MEAN[0.1], "pooled"), 0.747, 5e-3)
    ok &= chk("recovery_mean_0.1_macro_vs_G81_90ns", rec(COND_MEAN[0.1], "macro"), 0.695, 5e-3)
    ok &= chk("recovery_mean_0.5_pooled_vs_G81_90ns", rec(COND_MEAN[0.5], "pooled"), 1.025, 5e-3)
    ok &= chk("recovery_mean_0.5_macro_vs_G81_90ns", rec(COND_MEAN[0.5], "macro"), 1.008, 5e-3)
    ok &= chk("recovery_mean_1.0_pooled_vs_G81_mean_only_full", rec(COND_MEAN[1.0], "pooled"), 1.084, 5e-3)
    ok &= chk("recovery_mean_1.0_macro_vs_G81_mean_only_full", rec(COND_MEAN[1.0], "macro"), 1.060, 5e-3)
    ok &= chk("stats6_definition_maxdiff_vs_k06b_history", worst, 0.0, 1e-4)
    (OUT / "premise_check.json").write_text(json.dumps(
        dict(checks=checks, all_ok=bool(ok)), indent=2, ensure_ascii=False))
    if not ok:
        print("PREMISE FAILED: 土台の再現が崩れているため、ここで止める。", flush=True)
        sys.exit(2)
    print("premise OK (%.0fs)" % (time.time() - t0), flush=True)

    # ---- 回復率と増え分 (平均だけ・統計量6種の両方) ----
    FAMS = {"mean": COND_MEAN, "stats6": COND_STATS6}
    recovery = {c: dict(pooled=float(rec(c, "pooled")), macro=float(rec(c, "macro")))
                for c in CONDITIONS if c != "static_only"}
    increments = {}
    for fam, cmap in FAMS.items():
        increments[fam] = {}
        for kind in ("pooled", "macro"):
            increments[fam][kind] = {}
            prev = None
            for L in LENGTHS:
                r = recovery[cmap[L]][kind]
                if prev is not None:
                    increments[fam][kind][f"{L:.1f}"] = float(r - prev)
                prev = r

    def min_length(cmap, kind, thr):
        for L in LENGTHS:
            if recovery[cmap[L]][kind] >= thr:
                return L
        return None

    def largest_step(cmap, kind):
        inc = increments["stats6" if cmap is COND_STATS6 else "mean"][kind]
        key = max(inc, key=lambda k: inc[k])
        Lto = float(key)
        return dict(from_ns=Lto - 0.1, to_ns=Lto, increment=inc[key])

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
        for fam, cmap, apk, rk in (("mean", COND_MEAN, "auprc_prefix", "recovery_prefix"),
                                   ("stats6", COND_STATS6, "auprc_stats6", "recovery_stats6")):
            entry[apk] = {}
            entry[rk] = {}
            for L in LENGTHS:
                ap = per_protein_ap[cmap[L]][p]
                entry[apk][f"{L:.1f}"] = ap
                if ap is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                    entry[rk][f"{L:.1f}"] = float((ap - ap_s) / entry["full_gain"])
                else:
                    entry[rk][f"{L:.1f}"] = None
        per_protein[p] = entry

    # タンパク質ごとの回復率の分布と増え分の分布 (利得 > 0 のタンパク質だけ)
    def per_protein_summary(rk):
        sub = [e for e in per_protein.values() if e[rk].get("0.0") is not None]
        dist = {}
        for L in LENGTHS:
            vals = np.array([e[rk][f"{L:.1f}"] for e in sub])
            dist[f"{L:.1f}"] = dict(n=int(len(vals)), median=float(np.median(vals)),
                                    q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)),
                                    frac_ge_0p8=float(np.mean(vals >= 0.8)))
        incs = {}
        for i in range(1, len(LENGTHS)):
            L0, L1 = LENGTHS[i - 1], LENGTHS[i]
            vals = np.array([e[rk][f"{L1:.1f}"] - e[rk][f"{L0:.1f}"] for e in sub])
            incs[f"{L1:.1f}"] = dict(median=float(np.median(vals)),
                                     q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)),
                                     min=float(vals.min()), max=float(vals.max()),
                                     frac_positive=float(np.mean(vals > 0)),
                                     frac_ge_0p1=float(np.mean(vals >= 0.1)))
        return dict(n_proteins=len(sub), recovery_distribution=dist, increment_distribution=incs)

    pp_summary = {fam: per_protein_summary(rk) for fam, rk in
                  (("mean", "recovery_prefix"), ("stats6", "recovery_stats6"))}

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
        precision = np.divide(cum_tp := np.cumsum(tp_g), cum_tot := np.cumsum(tot_g),
                              out=np.zeros_like(cum_tp), where=cum_tot > 0)
        return float(np.sum((tp_g / cum_tp[-1]) * precision))

    defined = [p for p, _, _ in spans if per_protein_ap["static_only"][p] is not None]
    ap_mat = np.array([[per_protein_ap[c][p] for c in CONDITIONS] for p in defined])
    cond_idx = {c: j for j, c in enumerate(CONDITIONS)}

    rng = np.random.default_rng(BOOT_SEED)
    nP = len(TEST)
    boot_rec = {c: {kind: [] for kind in ("pooled", "macro")}
                for c in CONDITIONS if c != "static_only"}
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
            for c in boot_rec:
                ap = pooled_ap_boot(boot_prep[c], mult)
                boot_rec[c]["pooled"].append((ap - ap_s) / g)
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
                for c in boot_rec:
                    boot_rec[c]["macro"].append(
                        (m[cond_idx[c]] - m[cond_idx["static_only"]]) / gm)
        if (b + 1) % 2000 == 0:
            print("bootstrap %d/%d (%.0fs)" % (b + 1, N_BOOT, time.time() - t0), flush=True)

    bootstrap = {}
    for kind in ("pooled", "macro"):
        bootstrap[kind] = {}
        for c in boot_rec:
            arr = np.array(boot_rec[c][kind])
            if len(arr) == 0 or np.isnan(arr).any():
                raise RuntimeError(f"bootstrap {kind} {c}: invalid values (n={len(arr)}, nan={int(np.isnan(arr).sum())})")
            bootstrap[kind][c] = dict(
                n_boot=int(len(arr)), seed=BOOT_SEED,
                point_estimate=recovery[c][kind],
                ci_lower=float(np.percentile(arr, 2.5)),
                ci_upper=float(np.percentile(arr, 97.5)),
                boot_mean=float(arr.mean()), boot_std=float(arr.std()))
    print("bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)

    # ---- 読み方への答え (主は統計量6種) ----
    def step_per_protein(cmap, rk):
        step = largest_step(cmap, "pooled")
        Lfrom, Lto = step["from_ns"], step["to_ns"]
        vals = []
        for p, entry in per_protein.items():
            r_from = entry[rk].get(f"{Lfrom:.1f}")
            r_to = entry[rk].get(f"{Lto:.1f}")
            if r_from is not None and r_to is not None:
                vals.append(r_to - r_from)
        vals = np.array(vals)
        return dict(from_ns=Lfrom, to_ns=Lto, n_proteins=int(len(vals)),
                    median=float(np.median(vals)),
                    q25=float(np.percentile(vals, 25)),
                    q75=float(np.percentile(vals, 75)),
                    min=float(vals.min()), max=float(vals.max()),
                    frac_positive=float(np.mean(vals > 0)),
                    frac_ge_0p1=float(np.mean(vals >= 0.1)))

    readings = dict(
        stats6=dict(
            min_length_recovery_0_8=dict(pooled=min_length(COND_STATS6, "pooled", 0.8),
                                         macro=min_length(COND_STATS6, "macro", 0.8)),
            min_length_recovery_0_9=dict(pooled=min_length(COND_STATS6, "pooled", 0.9),
                                         macro=min_length(COND_STATS6, "macro", 0.9)),
            largest_step=dict(pooled=largest_step(COND_STATS6, "pooled"),
                              macro=largest_step(COND_STATS6, "macro")),
            step_per_protein=step_per_protein(COND_STATS6, "recovery_stats6")),
        mean_only=dict(
            min_length_recovery_0_8=dict(pooled=min_length(COND_MEAN, "pooled", 0.8),
                                         macro=min_length(COND_MEAN, "macro", 0.8)),
            min_length_recovery_0_9=dict(pooled=min_length(COND_MEAN, "pooled", 0.9),
                                         macro=min_length(COND_MEAN, "macro", 0.9)),
            largest_step=dict(pooled=largest_step(COND_MEAN, "pooled"),
                              macro=largest_step(COND_MEAN, "macro")),
            step_per_protein=step_per_protein(COND_MEAN, "recovery_prefix")),
        l08_mean_only_reference=dict(
            min_length_recovery_0_8=dict(pooled=0.2, macro=0.2),
            min_length_recovery_0_9=dict(pooled=0.4, macro=0.4),
            largest_step="0.1->0.2 ns (+0.133 pooled / +0.110 macro)"))

    # 同じ長さでの統計量6種と平均だけの回復率の差
    diff_vs_mean = {f"{L:.1f}": dict(
        pooled=float(recovery[COND_STATS6[L]]["pooled"] - recovery[COND_MEAN[L]]["pooled"]),
        macro=float(recovery[COND_STATS6[L]]["macro"] - recovery[COND_MEAN[L]]["macro"]))
        for L in LENGTHS}

    out = {
        "label": "HB formation 90ns, guidance recovery vs short-MD covered length with 6 history statistics (experiment-D definition), strong base (K06b)",
        "horizon_ns": 90,
        "stats6_definition": {
            "source": "MolCasta-InternalResearch branch research/fixed-1ns-3phen-6h-20261121, GPT playground/ATLAS-real-trajectory-2026-11-19/fixed_1ns_history_3phen_6h.py, features_from_history (local copy: ~/workspace/kinescope/work/fixed_1ns_history_3phen_6h.py)",
            "statistics": ["first value", "mean", "std (ddof=0)", "min", "last - first",
                           "fraction of frames <= bound"],
            "bound_angstrom": HB_BOUND,
            "bound_source": "THRESH['hb'] = 3.5 in the experiment-D script; also the HB formation label threshold",
            "dtype": "float32 (as in experiment D)",
            "relation_to_k06b_history10": "K06(b) history features (~/workspace/k06b/run/score_b.py) are [first, mean, std, min, max, last-first, last-prev, frac<=3.5, frac<=4.5, frac increasing frames]; the experiment-D 6 statistics are exactly columns [0,1,2,3,5,7] of those 10. score_b.py also defines gen6 with the identical 6 statistics for generated series.",
            "definition_check": "stats6 computed from g05_series over all 11 frames vs k06b_out history columns [0,1,2,3,5,7], max abs diff over all 108 proteins x 3 replicas recorded in premise_checks (stats6_definition_maxdiff_vs_k06b_history).",
        },
        "length_definition": {
            "unit": "covered length (ns), same definition as L01; prefix = frames 0..(L/0.1) of the 0-1 ns history, 100 ps spacing",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=len(FRAMES[L]), frames=FRAMES[L]) for L in LENGTHS},
            "note": "Differs from the G-series naming at 1.0 ns: G145's '1.0 ns' prefix is frames [0..9], while L01/L02's covered 1.0 ns is frames [0..10].",
        },
        "base": {
            "name": "K06(b) strong base: frozen T1I2006 + HistGBM sidecar (160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained on 8 calibration proteins, evaluated on 100 unseen proteins x 3 replicas",
            "static_features": 5, "history_features": 10,
            "n_calibration": int(len(y_cal)), "n_test": int(len(y_test)),
            "test_positives_90ns": int(y_test.sum()),
        },
        "premise_checks": checks,
        "references": {
            "k06b_90ns_static_auprc_pooled": 0.4418, "k06b_90ns_static_auprc_macro": 0.4357,
            "k06b_90ns_full_auprc_pooled": 0.4683, "k06b_90ns_full_auprc_macro": 0.4630,
            "k06b_90ns_gain_pooled": 0.0265, "k06b_90ns_gain_macro": 0.0273,
            "k06b_90ns_test_positives": 13068,
            "g81_90ns_recovery_0.1ns": dict(pooled=0.747, macro=0.695),
            "g81_90ns_recovery_0.5ns": dict(pooled=1.025, macro=1.008),
            "g81_90ns_recovery_mean_only_full": dict(pooled=1.084, macro=1.060),
            "l08_90ns_mean_only_recovery_exact": {
                "prefix_mean_0.0ns": dict(pooled=0.693477, macro=0.632363),
                "prefix_mean_0.1ns": dict(pooled=0.746623, macro=0.694986),
                "prefix_mean_0.2ns": dict(pooled=0.879530, macro=0.804829),
                "prefix_mean_0.3ns": dict(pooled=0.895750, macro=0.859136),
                "prefix_mean_0.4ns": dict(pooled=0.960041, macro=0.925104),
                "prefix_mean_0.5ns": dict(pooled=1.025324, macro=1.007727),
                "prefix_mean_0.6ns": dict(pooled=1.059393, macro=1.050874),
                "prefix_mean_0.7ns": dict(pooled=1.154762, macro=1.132766),
                "prefix_mean_0.8ns": dict(pooled=1.110948, macro=1.097370),
                "prefix_mean_0.9ns": dict(pooled=1.149462, macro=1.137657),
                "prefix_mean_1.0ns": dict(pooled=1.084366, macro=1.059703)},
            "l08_90ns_readings": {
                "min_length_recovery_0_8": dict(pooled=0.2, macro=0.2),
                "min_length_recovery_0_9": dict(pooled=0.4, macro=0.4),
                "largest_step": "0.1->0.2 ns (+0.133 pooled / +0.110 macro)"},
        },
        "metrics_90ns": results,
        "base_gain_90ns": dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        "recovery_vs_real_history_gain": recovery,
        "recovery_increment_per_0.1ns": increments,
        "recovery_stats6_minus_mean": diff_vs_mean,
        "readings": readings,
        "per_protein_summary": pp_summary,
        "bootstrap": dict(
            n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
            resampling_unit="protein (100 test proteins, resampled with replacement)",
            note="Within each resample, static-only, full-history, prefix-mean and prefix-stats6 AUPRCs are recomputed and the gain and recovery are recomputed inside the sample. Models are not refit; per-candidate predicted probabilities are resampled. Pooled AUPRC uses a numpy implementation of sklearn's average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every condition. Macro AUPRC within a resample is the multiplicity-weighted mean of per-protein AUPRCs over proteins whose AUPRC is defined.",
            n_draws_with_nonpositive_gain=n_bad_gain,
            results=bootstrap),
        "per_protein": per_protein,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # CSV (主は統計量6種。平均だけの回復率も併記)
    lines = ["covered_ns,n_frames,auprc_stats6_pooled,auprc_stats6_macro,recovery_stats6_pooled,recovery_stats6_macro,"
             "recovery_stats6_pooled_ci_lower,recovery_stats6_pooled_ci_upper,recovery_stats6_macro_ci_lower,recovery_stats6_macro_ci_upper,"
             "increment_stats6_pooled,increment_stats6_macro,recovery_mean_pooled,recovery_mean_macro,"
             "recovery_stats6_minus_mean_pooled,auprc_mean_pooled,auc_stats6,brier_stats6,logloss_stats6"]
    for L in LENGTHS:
        cs, cm = COND_STATS6[L], COND_MEAN[L]
        m = results[cs]
        lines.append(",".join([
            f"{L:.1f}", str(len(FRAMES[L])),
            f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}",
            f"{recovery[cs]['pooled']:.6f}", f"{recovery[cs]['macro']:.6f}",
            f"{bootstrap['pooled'][cs]['ci_lower']:.6f}",
            f"{bootstrap['pooled'][cs]['ci_upper']:.6f}",
            f"{bootstrap['macro'][cs]['ci_lower']:.6f}",
            f"{bootstrap['macro'][cs]['ci_upper']:.6f}",
            f"{increments['stats6']['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{increments['stats6']['macro'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{recovery[cm]['pooled']:.6f}", f"{recovery[cm]['macro']:.6f}",
            f"{diff_vs_mean[f'{L:.1f}']['pooled']:.6f}",
            f"{results[cm]['auprc']:.6f}",
            f"{m['auc']:.6f}", f"{m['brier']:.6f}", f"{m['logloss']:.6f}"]))
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    # タンパク質ごとの CSV
    plines = ["protein,n,positives,auprc_static,auprc_full,full_gain,"
              + ",".join(f"auprc_stats6_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"recovery_stats6_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"auprc_mean_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"recovery_mean_{L:.1f}ns" for L in LENGTHS)]
    for p, entry in per_protein.items():
        row = [p, str(entry["n"]), str(entry["positives"]),
               "" if entry["auprc_static"] is None else f"{entry['auprc_static']:.6f}",
               "" if entry["auprc_full"] is None else f"{entry['auprc_full']:.6f}",
               "" if entry["full_gain"] is None else f"{entry['full_gain']:.6f}"]
        for apk, rk in (("auprc_stats6", "recovery_stats6"), ("auprc_prefix", "recovery_prefix")):
            row += ["" if entry[apk][f"{L:.1f}"] is None else f"{entry[apk][f'{L:.1f}']:.6f}" for L in LENGTHS]
            row += ["" if entry[rk][f"{L:.1f}"] is None else f"{entry[rk][f'{L:.1f}']:.6f}" for L in LENGTHS]
        plines.append(",".join(row))
    (OUT / "per_protein.csv").write_text("\n".join(plines) + "\n")

    # 予測確率 (ブートストラップの再現用)
    np.savez_compressed(
        OUT / "test_predictions.npz",
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]),
        label=y_test.astype(np.int8),
        **{c: preds[c].astype(np.float32) for c in CONDITIONS})
    print("saved results.json / results.csv / per_protein.csv / test_predictions.npz (%.0fs total)" % (time.time() - t0), flush=True)
    print(json.dumps(readings, indent=2, ensure_ascii=False), flush=True)


# draw (TEST の添字) -> defined リスト内の位置 (-1 は未定義)
_orig_main = main


def main_with_map():
    global draw_to_defined
    orig2 = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
    test_proteins = orig2["test_proteins"]
    defined_set = []
    for p in test_proteins:
        z = np.load(K06B / f"{p}.npz")
        y = np.concatenate([z[f"r{r}_label"][:, HORIZON_IDX] for r in (1, 2, 3)])
        defined_set.append(0 < y.sum() < len(y))
    draw_to_defined = np.full(len(test_proteins), -1, dtype=int)
    pos = 0
    for i, d in enumerate(defined_set):
        if d:
            draw_to_defined[i] = pos
            pos += 1
    main()


if __name__ == "__main__":
    main_with_map()
