"""L39: 統計量6種で L01 の曲がりを強い土台で測る (K06(b) の条件、水素結合の形成、1 ns先)。

L02 (強い土台、履歴は切り取りの平均だけ) で測った「覆う長さに対する回復率の曲がり」を、
履歴の平均だけではなく統計量6種を使った場合に測り直し、必要な長さが縮むかを見る。

統計量6種の定義は実験D (固定した 1 ns 履歴の実験) の features_from_history そのまま:
  最初の値、平均、標準偏差 (ddof=0)、最小値、最後-最初、閾値以下だった割合。
  閾値は水素結合の形成の境界 3.5 Å (実験Dの THRESH["hb"]=3.5)。
  出典: MolCasta-InternalResearch ブランチ research/fixed-1ns-3phen-6h-20261121 の
  GPT遊び場/ATLAS-実軌道-2026-11-19/fixed_1ns_history_3phen_6h.py の features_from_history。
  切り取り (フレーム 0..k) に対しては、同じ式を部分系列に適用する
  (「最後」は切り取りの最後のフレーム、「割合」は切り取り内での割合)。

K06(b) の履歴特徴10個との関係:
  10個は g05_series の11フレームから
    [最初, 平均, 標準偏差, 最小, 最大, 最後-最初, 最後-直前,
     3.5 Å以下割合, 4.5 Å以下割合, 増加割合]
  で再現できる (このスクリプトの premise で全108タンパク質×3レプリカを数値照合する)。
  実験Dの6種はそのうち [最初, 平均, 標準偏差, 最小, 最後-最初, 3.5 Å以下割合] で、
  10個は6種に「最大、最後-直前、4.5 Å以下割合、増加割合」の4つを足した拡張になっている。
  したがって全部入り (要約値10個) は6種の上位集合で、覆う長さ 1.0 ns の6種条件は
  全部入りから4特徴を落とした条件になる。

長さの定義は L01・L02 にそのまま合わせる:
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 覆う長さ L ns の切り取りはフレーム 0〜(L/0.1) を使う。
  0 ns -> [0] (1フレーム)、0.1 ns -> [0,1]、…、1.0 ns -> [0..10] (11フレーム全部)。
- 起点フレーム10 (1 ns) は変えず、ラベルは 1 ns先。

条件 (すべて条件ごとに学習し直す):
- 静的のみ: 静的特徴5個。
- 全部入り (full_history): 静的特徴＋実履歴の要約値10個。回復率の分母 (実履歴の利得) の参照。
- 切り取りの平均だけ: 静的特徴＋切り取りの平均だけ。覆う長さ 0.0〜1.0 ns の11通り。
  L02 とまったく同じ作り方で、premise の照合と L02 との比較に使う。
- 切り取りの統計量6種: 静的特徴＋切り取りの統計量6種。覆う長さ 0.0〜1.0 ns の11通り (主役)。

土台 (K06(b) と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、seed 20261121)。
  較正タンパク質 8つで学習し、未見の100タンパク質×3レプリカ (163,074行) で評価する。
- 系列は G05 で作り直したもの (~/workspace/kinescope/work/g05_series/) を使う。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。
全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る (L02 と同じまとめ方)。

読み方 (結果を見る前に固定、L02 と同じ):
1. 回復率が 0.8 と 0.9 に届く最小の覆う長さを、統合値 (pooled) で読む (macro も併記)。
   L02 (平均だけ) の答え (0.8 は 0.7 ns、0.9 は全体で 0.8 ns) と比べて縮んだかを明記する。
2. 段の判定は、隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、その大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、パーセンタイル法)。
   各標本の中でも静的のみ・全部入り・各長さの AUPRC を測り直し、
   利得と回復率をその標本の中で計算し直す (G147・G148・G162 と同じ段取り)。
4. 段の前後ではタンパク質ごとの値も記録する。

土台の再現の確認 (premise。これが崩れたら測定は無効として止める):
- 統計量の照合: 全108タンパク質×3レプリカで、g05_series の11フレームから計算した
  要約値10個が K06(b) の保存値 (k06b_out の r{r}_history) と 1e-4 未満で一致すること
  (float32 の丸めの範囲。6種はその部分集合なので、6種の定義が保存値と結びつく)。
- 1 ns先で静的のみ・全部入りの AUPRC (pooled / macro) が k06b の記録
  (0.4591 / 0.4539、0.5293 / 0.5251) と 5e-4 以内で一致すること (L02 と同じ)。
- 平均だけの回復率が L02 の記録 (results.json の recovery_vs_real_history_gain) と
  全11長さで 5e-3 以内で一致すること (6種のうち平均が L02 と同じ入力である確認)。
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
L02_RESULTS = Path("/home/hatch/workspace/kinescope/repo/experiments/"
                   "L02_hb_formation_leverage_strong_base_1ns/results.json")
OUT = Path(__file__).resolve().parent
SEED = 20261121
BOOT_SEED = 42
N_BOOT = 10000
HORIZON_IDX = 0  # 1 ns先
THRESH_HB = 3.5  # 実験Dの THRESH["hb"] (水素結合の形成の境界、Å)

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]
ALLP = CAL + TEST

# L01 の長さの定義: 覆う長さ L ns -> フレーム 0..(L/0.1)
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(int(round(L / 0.1)) + 1)) for L in LENGTHS}
COND_MEAN = {L: f"prefix_mean_{L:.1f}ns" for L in LENGTHS}
COND_6 = {L: f"prefix_6stats_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = (["static_only", "full_history"]
              + [COND_MEAN[L] for L in LENGTHS]
              + [COND_6[L] for L in LENGTHS])


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


def six_stats(sub):
    """実験Dの features_from_history を部分系列 sub (k フレーム × 候補数) に適用する。

    並びは [最初の値, 平均, 標準偏差 (ddof=0), 最小値, 最後-最初, 3.5 Å以下だった割合]。
    """
    return np.stack([sub[0], sub.mean(0), sub.std(0), sub.min(0),
                     sub[-1] - sub[0], (sub <= THRESH_HB).mean(0)],
                    axis=1).astype(np.float32)


def ten_stats(series):
    """K06(b) の要約値10個を11フレームの系列から再計算する (premise の照合用)。"""
    return np.stack([series[0], series.mean(0), series.std(0), series.min(0), series.max(0),
                     series[-1] - series[0], series[-1] - series[-2],
                     (series <= 3.5).mean(0), (series <= 4.5).mean(0),
                     (np.diff(series, axis=0) > 0).mean(0)], axis=1)


def check_history_stats():
    """g05_series から計算した要約値10個が K06(b) の保存値と一致するかを全数照合する。"""
    worst = 0.0
    worst6 = 0.0
    n_checked = 0
    for p in ALLP:
        z = np.load(K06B / f"{p}.npz")
        s = np.load(SERIES / f"{p}.npz")
        for r in (1, 2, 3):
            hist = z[f"r{r}_history"].astype(np.float64)
            calc = ten_stats(s[f"r{r}_series"]).astype(np.float64)
            assert hist.shape == calc.shape, (p, r, hist.shape, calc.shape)
            d = np.abs(hist - calc).max()
            d6 = np.abs(hist[:, [0, 1, 2, 3, 5, 7]] - calc[:, [0, 1, 2, 3, 5, 7]]).max()
            worst = max(worst, float(d))
            worst6 = max(worst6, float(d6))
            n_checked += 1
    return dict(n_protein_replicas=n_checked,
                max_abs_diff_10stats=worst, max_abs_diff_6stats_subset=worst6,
                tol=1e-4, ok=bool(worst < 1e-4))


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

    # ---- premise 1: 統計量の定義の照合 (系列 -> K06(b) 保存値) ----
    stat_check = check_history_stats()
    print("history stats check:", stat_check, "(%.0fs)" % (time.time() - t0), flush=True)
    if not stat_check["ok"]:
        (OUT / "premise_check.json").write_text(json.dumps(
            dict(history_stats_check=stat_check, all_ok=False), indent=2, ensure_ascii=False))
        print("PREMISE FAILED: 統計量の定義が K06(b) の保存値と結びつかないため、ここで止める。",
              flush=True)
        sys.exit(2)

    cal, test = load(CAL), load(TEST)
    y_cal = cal["label"][:, HORIZON_IDX]
    y_test = test["label"][:, HORIZON_IDX]
    print("cal", len(y_cal), "test", len(y_test), "test pos 1ns", int(y_test.sum()), flush=True)

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
        # 平均だけ: L02 のコードとまったく同じ作り方
        feats[COND_MEAN[L]] = (
            np.column_stack([cal["static"], cal["series"][idx].mean(axis=0)[:, None].astype(np.float32)]),
            np.column_stack([test["static"], test["series"][idx].mean(axis=0)[:, None].astype(np.float32)]))
        # 統計量6種: 実験Dの式を切り取りに適用
        feats[COND_6[L]] = (
            np.column_stack([cal["static"], six_stats(cal["series"][idx])]),
            np.column_stack([test["static"], six_stats(test["series"][idx])]))

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

    # ---- premise 2: 土台の再現 (L02 と同じ照合 + 平均だけの回復率の全長さ照合) ----
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
    ok &= chk("static_pooled_vs_k06b", results["static_only"]["auprc"], 0.4591, 5e-4)
    ok &= chk("static_macro_vs_k06b", results["static_only"]["macro_auprc"], 0.4539, 5e-4)
    ok &= chk("full_pooled_vs_k06b", results["full_history"]["auprc"], 0.5293, 5e-4)
    ok &= chk("full_macro_vs_k06b", results["full_history"]["macro_auprc"], 0.5251, 5e-4)
    l02 = json.loads(L02_RESULTS.read_text())
    l02_rec = l02["recovery_vs_real_history_gain"]
    for L in LENGTHS:
        c = COND_MEAN[L]
        ok &= chk(f"mean_recovery_{L:.1f}_pooled_vs_L02", rec(c, "pooled"),
                  l02_rec[c]["pooled"], 5e-3)
        ok &= chk(f"mean_recovery_{L:.1f}_macro_vs_L02", rec(c, "macro"),
                  l02_rec[c]["macro"], 5e-3)
    (OUT / "premise_check.json").write_text(json.dumps(
        dict(history_stats_check=stat_check, checks=checks, all_ok=bool(ok)),
        indent=2, ensure_ascii=False))
    if not ok:
        print("PREMISE FAILED: 土台の再現が崩れているため、ここで止める。", flush=True)
        sys.exit(2)
    print("premise OK (%.0fs)" % (time.time() - t0), flush=True)

    # ---- 回復率と増え分 (6種が主役、平均だけは比較用) ----
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
                  for fam, cm in (("mean_only", COND_MEAN), ("stats6", COND_6))}

    def min_length(condmap, kind, thr):
        for L in LENGTHS:
            if recovery[condmap[L]][kind] >= thr:
                return L
        return None

    def largest_step(condmap, kind):
        inc = increments["stats6" if condmap is COND_6 else "mean_only"][kind]
        key = max(inc, key=lambda k: inc[k])
        Lto = float(key)
        return dict(from_ns=Lto - 0.1, to_ns=Lto, increment=inc[key])

    # ---- タンパク質ごとの回復率 (6種と平均だけの両方) ----
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
            entry["auprc_prefix"] = {}
            entry["recovery_prefix"] = {}
            for L in LENGTHS:
                ap = per_protein_ap[condmap[L]][p]
                entry["auprc_prefix"][f"{L:.1f}"] = ap
                if ap is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                    entry["recovery_prefix"][f"{L:.1f}"] = float((ap - ap_s) / entry["full_gain"])
                else:
                    entry["recovery_prefix"][f"{L:.1f}"] = None
            pp[p] = entry
        return pp

    per_protein = {"stats6": per_protein_for(COND_6), "mean_only": per_protein_for(COND_MEAN)}

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
        # 重み 0 のグループでは cum_tot が 0 になる。recall の増え分は 0 なので
        # precision は 0 として扱う (0/0 の NaN が AP に伝わるのを防ぐ。L02 と同じ対策)。
        precision = np.divide(cum_tp, cum_tot, out=np.zeros_like(cum_tp), where=cum_tot > 0)
        return float(np.sum((tp_g / cum_tp[-1]) * precision))

    defined = [p for p, _, _ in spans if per_protein_ap["static_only"][p] is not None]
    ap_mat = np.array([[per_protein_ap[c][p] for c in CONDITIONS] for p in defined])
    cond_idx = {c: j for j, c in enumerate(CONDITIONS)}

    rng = np.random.default_rng(BOOT_SEED)
    nP = len(TEST)
    fams = (("mean_only", COND_MEAN), ("stats6", COND_6))
    boot_rec = {fam: {kind: {f"{L:.1f}": [] for L in LENGTHS} for kind in ("pooled", "macro")}
                for fam, _ in fams}
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
            for fam, cm in fams:
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
                for fam, cm in fams:
                    for L in LENGTHS:
                        boot_rec[fam]["macro"][f"{L:.1f}"].append(
                            (m[cond_idx[cm[L]]] - m[cond_idx["static_only"]]) / gm)
        if (b + 1) % 2000 == 0:
            print("bootstrap %d/%d (%.0fs)" % (b + 1, N_BOOT, time.time() - t0), flush=True)

    bootstrap = {}
    for fam, cm in fams:
        bootstrap[fam] = {}
        for kind in ("pooled", "macro"):
            bootstrap[fam][kind] = {}
            for L in LENGTHS:
                arr = np.array(boot_rec[fam][kind][f"{L:.1f}"])
                if len(arr) == 0 or np.isnan(arr).any():
                    raise RuntimeError(f"bootstrap {fam} {kind} {L}: invalid values (n={len(arr)}, nan={int(np.isnan(arr).sum())})")
                bootstrap[fam][kind][f"{L:.1f}"] = dict(
                    n_boot=int(len(arr)), seed=BOOT_SEED,
                    point_estimate=recovery[cm[L]][kind],
                    ci_lower=float(np.percentile(arr, 2.5)),
                    ci_upper=float(np.percentile(arr, 97.5)),
                    boot_mean=float(arr.mean()), boot_std=float(arr.std()))
    print("bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)

    # ---- 段の前後のタンパク質ごとの増え分 (6種の pooled の段の位置で) ----
    step = largest_step(COND_6, "pooled")
    Lfrom, Lto = step["from_ns"], step["to_ns"]
    step_inc_protein = []
    for p, entry in per_protein["stats6"].items():
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

    readings = dict(
        stats6=dict(
            min_length_recovery_0_8=dict(pooled=min_length(COND_6, "pooled", 0.8),
                                         macro=min_length(COND_6, "macro", 0.8)),
            min_length_recovery_0_9=dict(pooled=min_length(COND_6, "pooled", 0.9),
                                         macro=min_length(COND_6, "macro", 0.9)),
            largest_step=dict(pooled=step, macro=largest_step(COND_6, "macro")),
            step_per_protein=step_per_protein),
        mean_only_recomputed=dict(
            min_length_recovery_0_8=dict(pooled=min_length(COND_MEAN, "pooled", 0.8),
                                         macro=min_length(COND_MEAN, "macro", 0.8)),
            min_length_recovery_0_9=dict(pooled=min_length(COND_MEAN, "pooled", 0.9),
                                         macro=min_length(COND_MEAN, "macro", 0.9)),
            largest_step=dict(pooled=largest_step(COND_MEAN, "pooled"),
                              macro=largest_step(COND_MEAN, "macro"))),
        l02_record_mean_only=dict(
            min_length_recovery_0_8=dict(pooled=l02["readings"]["min_length_recovery_0_8"]["pooled"],
                                         macro=l02["readings"]["min_length_recovery_0_8"]["macro"]),
            min_length_recovery_0_9=dict(pooled=l02["readings"]["min_length_recovery_0_9"]["pooled"],
                                         macro=l02["readings"]["min_length_recovery_0_9"]["macro"])))

    # 同じ長さでの回復率の差 (6種 - 平均だけ)
    diff_vs_mean = {f"{L:.1f}": dict(
        pooled=float(recovery[COND_6[L]]["pooled"] - recovery[COND_MEAN[L]]["pooled"]),
        macro=float(recovery[COND_6[L]]["macro"] - recovery[COND_MEAN[L]]["macro"]))
        for L in LENGTHS}

    out = {
        "label": "HB formation 1ns, guidance recovery vs short-MD covered length with 6 history statistics (experiment-D definition), strong base (K06b)",
        "horizon_ns": 1,
        "length_definition": {
            "unit": "covered length (ns), same definition as L01/L02; prefix = frames 0..(L/0.1) of the 0-1 ns history, 100 ps spacing",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=len(FRAMES[L]), frames=FRAMES[L]) for L in LENGTHS},
            "note": "Differs from the G-series naming at 1.0 ns: G145's '1.0 ns' prefix is frames [0..9], while L01/L02/L39's covered 1.0 ns is frames [0..10].",
        },
        "statistics6_definition": {
            "source": "MolCasta-InternalResearch branch research/fixed-1ns-3phen-6h-20261121, GPT playground/ATLAS-observed-2026-11-19/fixed_1ns_history_3phen_6h.py, features_from_history(series, bound)",
            "order": ["first value", "mean", "std (ddof=0)", "min", "last - first",
                      "fraction of frames <= threshold"],
            "threshold_A": THRESH_HB,
            "threshold_source": "THRESH['hb'] = 3.5 in the same script (HB formation boundary)",
            "application_to_prefix": "the same formula applied to the prefix sub-series frames 0..k; 'last' is the last frame of the prefix and the fraction is taken within the prefix",
            "dtype": "float32 (as in the source script)",
            "relation_to_k06b_history10": "K06(b)'s 10 history features are [first, mean, std, min, max, last-first, last-prev, frac<=3.5A, frac<=4.5A, frac increasing], recomputed from the g05 series and verified against the stored k06b_out history for all 108 proteins x 3 replicas (see premise history_stats_check). The experiment-D 6 statistics are the subset at positions [0,1,2,3,5,7]; the 10 are the 6 plus [max, last-prev, frac<=4.5A, frac increasing]. Full history (10 stats on all 11 frames) is therefore a superset of the covered-1.0 ns 6-stats condition.",
        },
        "base": {
            "name": "K06(b) strong base: frozen T1I2006 + HistGBM sidecar (160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained on 8 calibration proteins, evaluated on 100 unseen proteins x 3 replicas",
            "static_features": 5, "history_features_full": 10,
            "n_calibration": int(len(y_cal)), "n_test": int(len(y_test)),
            "test_positives_1ns": int(y_test.sum()),
        },
        "premise_checks": checks,
        "history_stats_check": stat_check,
        "references": {
            "k06b_static_auprc_pooled": 0.4591, "k06b_static_auprc_macro": 0.4539,
            "k06b_full_auprc_pooled": 0.5293, "k06b_full_auprc_macro": 0.5251,
            "k06b_gain_pooled": 0.0702, "k06b_gain_macro": 0.0712,
            "l02_recovery_mean_only": {c: l02_rec[c] for c in l02_rec if c.startswith("prefix_mean")},
        },
        "metrics_1ns": results,
        "base_gain_1ns": dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        "recovery_vs_real_history_gain": recovery,
        "recovery_increment_per_0.1ns": increments,
        "recovery_diff_stats6_minus_mean": diff_vs_mean,
        "readings": readings,
        "bootstrap": dict(
            n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
            resampling_unit="protein (100 test proteins, resampled with replacement)",
            note="Same procedure as L02. Within each resample, static-only, full-history and each prefix condition's AUPRCs are recomputed and the gain and recovery are recomputed inside the sample. Models are not refit. Both families (mean-only and 6-stats) are bootstrapped from the same draws. Pooled AUPRC uses a numpy implementation of sklearn's average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every condition. Macro AUPRC within a resample is the multiplicity-weighted mean of per-protein AUPRCs over proteins whose AUPRC is defined.",
            n_draws_with_nonpositive_gain=n_bad_gain,
            results=bootstrap),
        "per_protein": per_protein,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # CSV (主表は6種、平均だけの回復率と差も併記)
    lines = ["covered_ns,n_frames,auprc_pooled,auprc_macro,recovery_pooled,recovery_macro,"
             "recovery_pooled_ci_lower,recovery_pooled_ci_upper,recovery_macro_ci_lower,recovery_macro_ci_upper,"
             "increment_pooled,increment_macro,"
             "mean_only_recovery_pooled,mean_only_recovery_macro,"
             "mean_only_recovery_pooled_ci_lower,mean_only_recovery_pooled_ci_upper,"
             "recovery_diff_pooled,recovery_diff_macro,auc,brier,logloss"]
    for L in LENGTHS:
        c6, cm = COND_6[L], COND_MEAN[L]
        m = results[c6]
        lines.append(",".join([
            f"{L:.1f}", str(len(FRAMES[L])),
            f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}",
            f"{recovery[c6]['pooled']:.6f}", f"{recovery[c6]['macro']:.6f}",
            f"{bootstrap['stats6']['pooled'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['stats6']['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{bootstrap['stats6']['macro'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['stats6']['macro'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{increments['stats6']['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{increments['stats6']['macro'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{recovery[cm]['pooled']:.6f}", f"{recovery[cm]['macro']:.6f}",
            f"{bootstrap['mean_only']['pooled'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['mean_only']['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{diff_vs_mean[f'{L:.1f}']['pooled']:.6f}",
            f"{diff_vs_mean[f'{L:.1f}']['macro']:.6f}",
            f"{m['auc']:.6f}", f"{m['brier']:.6f}", f"{m['logloss']:.6f}"]))
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    # タンパク質ごとの CSV (6種)
    plines = ["protein,n,positives,auprc_static,auprc_full,full_gain,"
              + ",".join(f"auprc_{L:.1f}ns" for L in LENGTHS) + ","
              + ",".join(f"recovery_{L:.1f}ns" for L in LENGTHS)]
    for p, entry in per_protein["stats6"].items():
        row = [p, str(entry["n"]), str(entry["positives"]),
               "" if entry["auprc_static"] is None else f"{entry['auprc_static']:.6f}",
               "" if entry["auprc_full"] is None else f"{entry['auprc_full']:.6f}",
               "" if entry["full_gain"] is None else f"{entry['full_gain']:.6f}"]
        row += ["" if entry["auprc_prefix"][f"{L:.1f}"] is None else f"{entry['auprc_prefix'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        row += ["" if entry["recovery_prefix"][f"{L:.1f}"] is None else f"{entry['recovery_prefix'][f'{L:.1f}']:.6f}" for L in LENGTHS]
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
    print(json.dumps(diff_vs_mean, indent=2, ensure_ascii=False), flush=True)


# draw (TEST の添字) -> defined リスト内の位置 (-1 は未定義)。L02 と同じ段取り。
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
