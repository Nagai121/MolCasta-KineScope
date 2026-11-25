"""L07: L02 の曲がりを予測幅 50 ns で測る (K06(b) の条件、水素結合の形成、50 ns先)。

L02 (強い土台、1 ns先) で測った「覆う長さに対する回復率の曲がり」を、
同じ方法・同じ土台で予測幅だけを 50 ns に変えて測る。
L02・L04 (5 ns先)・L05 (10 ns先) では段はなく、増え幅最大は最初の 0→0.1 ns だった。
50 ns先でも段がないか、線 (0.8 / 0.9) に届く長さがどう変わるかを見る。

長さの定義は L01 にそのまま合わせる (L01 と比べられることが大事):
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 覆う長さ L ns の切り取りはフレーム 0〜(L/0.1) を使う。
  0 ns -> [0] (1フレーム)、0.1 ns -> [0,1]、…、1.0 ns -> [0..10] (11フレーム全部)。
- 注意: G 系列の「1.0 ns」の切り取り (フレーム [0..9]、G145) とは違う。
  L01 の 1.0 ns は 11フレーム全部の平均で、G05 の mean_only_full と同じ条件になる。
- 起点フレーム10 (1 ns) は変えず、ラベルは 50 ns先 (ラベルの列の添字 4、
  予測幅の並びは [1, 5, 10, 30, 50, 90] ns)。

条件 (すべて条件ごとに学習し直す):
- 静的のみ: 静的特徴5個。
- 全部入り (full_history): 静的特徴＋実履歴の要約値10個。回復率の分母 (実履歴の利得) の参照。
- 切り取りの平均だけ: 静的特徴＋切り取りの平均だけ。覆う長さ 0.0〜1.0 ns の11通り。

土台 (K06(b) と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、seed 20261121)。
  較正タンパク質 8つで学習し、未見の100タンパク質×3レプリカ (163,074行) で評価する。
- 系列は G05 で作り直したもの (~/workspace/kinescope/work/g05_series/) を使う。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。
全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る (G05 と同じまとめ方)。

読み方 (結果を見る前に固定):
1. 回復率が 0.8 と 0.9 に届く最小の覆う長さを、統合値 (pooled) で読む (macro も併記)。
2. 段の判定は、隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、その大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、パーセンタイル法)。
   各標本の中でも静的のみ・全部入り・各長さの AUPRC を測り直し、
   利得と回復率をその標本の中で計算し直す (G147・G148・G162 と同じ段取り)。
4. 段の前後ではタンパク質ごとの値も記録する。

土台の再現の確認 (premise。これが崩れたら測定は無効として止める):
- 50 ns先で静的のみ・全部入りの AUPRC (pooled / macro) が k06b の記録
  (~/workspace/k06b/results_100.json の by_horizon の "50":
  0.4470 / 0.4407、0.4825 / 0.4752) と 5e-4 以内で一致すること (G05 の基準)。
- テスト陽性が k06b の記録の 12,819件と一致すること。
- 重なる長さの回復率が G81 (強い土台の切り取り、50 ns先) の記録と
  5e-3 以内で一致すること:
  覆う長さ 0.1 ns (G81 の 0.1 ns、フレーム [0,1]) で 0.524 / 0.540、
  0.5 ns (フレーム 0〜5) で 0.781 / 0.791、
  1.0 ns (L01 の定義 = 全部の平均だけ、G81 の mean_only_full) で 0.906 / 0.917。
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
HORIZON_IDX = 4  # 50 ns先 (予測幅の並びは [1, 5, 10, 30, 50, 90])

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# L01 の長さの定義: 覆う長さ L ns -> フレーム 0..(L/0.1)
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(int(round(L / 0.1)) + 1)) for L in LENGTHS}
COND_PREFIX = {L: f"prefix_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = ["static_only", "full_history"] + [COND_PREFIX[L] for L in LENGTHS]


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
    # 同じスコアのグループの終わり位置
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
    print("cal", len(y_cal), "test", len(y_test), "test pos 50ns", int(y_test.sum()), flush=True)

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
        feats[COND_PREFIX[L]] = (
            np.column_stack([cal["static"], cal["series"][idx].mean(axis=0)[:, None].astype(np.float32)]),
            np.column_stack([test["static"], test["series"][idx].mean(axis=0)[:, None].astype(np.float32)]))

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
    ok &= chk("static_pooled_vs_k06b_50ns", results["static_only"]["auprc"], 0.4470, 5e-4)
    ok &= chk("static_macro_vs_k06b_50ns", results["static_only"]["macro_auprc"], 0.4407, 5e-4)
    ok &= chk("full_pooled_vs_k06b_50ns", results["full_history"]["auprc"], 0.4825, 5e-4)
    ok &= chk("full_macro_vs_k06b_50ns", results["full_history"]["macro_auprc"], 0.4752, 5e-4)
    ok &= chk("test_positives_vs_k06b_50ns", float(y_test.sum()), 12819.0, 0.5)
    ok &= chk("recovery_0.1_pooled_vs_G81_50ns", rec(COND_PREFIX[0.1], "pooled"), 0.524, 5e-3)
    ok &= chk("recovery_0.1_macro_vs_G81_50ns", rec(COND_PREFIX[0.1], "macro"), 0.540, 5e-3)
    ok &= chk("recovery_0.5_pooled_vs_G81_50ns", rec(COND_PREFIX[0.5], "pooled"), 0.781, 5e-3)
    ok &= chk("recovery_0.5_macro_vs_G81_50ns", rec(COND_PREFIX[0.5], "macro"), 0.791, 5e-3)
    ok &= chk("recovery_1.0_pooled_vs_G81_mean_only_full", rec(COND_PREFIX[1.0], "pooled"), 0.906, 5e-3)
    ok &= chk("recovery_1.0_macro_vs_G81_mean_only_full", rec(COND_PREFIX[1.0], "macro"), 0.917, 5e-3)
    (OUT / "premise_check.json").write_text(json.dumps(
        dict(checks=checks, all_ok=bool(ok)), indent=2, ensure_ascii=False))
    if not ok:
        print("PREMISE FAILED: 土台の再現が崩れているため、ここで止める。", flush=True)
        sys.exit(2)
    print("premise OK (%.0fs)" % (time.time() - t0), flush=True)

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
    # numpy 版 AP の検証 (全データで sklearn と一致すること)
    for cname in CONDITIONS:
        ap_np = ap_numpy(y_test, preds[cname].astype(np.float64))
        assert abs(ap_np - results[cname]["auprc"]) < 1e-9, (cname, ap_np, results[cname]["auprc"])
    print("numpy AP verified vs sklearn for all conditions", flush=True)

    # pooled 用: 条件ごとにスコア降順の並びを固定し、同点グループでまとめて重み付きで数える
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
        # 重み 0 のグループ (選ばれなかったタンパク質だけの同点グループ) では
        # cum_tot が 0 になる。そのグループの recall の増え分は 0 なので、
        # precision は 0 として扱う (0/0 の NaN が AP に伝わるのを防ぐ)。
        precision = np.divide(cum_tp, cum_tot, out=np.zeros_like(cum_tp), where=cum_tot > 0)
        return float(np.sum((tp_g / cum_tp[-1]) * precision))

    # macro 用: タンパク質×条件の AUPRC 行列 (定義できるタンパク質だけ)
    defined = [p for p, _, _ in spans if per_protein_ap["static_only"][p] is not None]
    didx = [i for i, (p, _, _) in enumerate(spans) if p in defined]
    ap_mat = np.array([[per_protein_ap[c][p] for c in CONDITIONS] for p in defined])  # (n_defined, n_cond)
    cond_idx = {c: j for j, c in enumerate(CONDITIONS)}

    rng = np.random.default_rng(BOOT_SEED)
    nP = len(TEST)
    boot_rec = {kind: {f"{L:.1f}": [] for L in LENGTHS} for kind in ("pooled", "macro")}
    n_bad_gain = dict(pooled=0, macro=0)
    for b in range(N_BOOT):
        draw = rng.integers(0, nP, nP)
        mult = np.bincount(draw, minlength=nP).astype(np.float64)
        # pooled
        ap_s = pooled_ap_boot(boot_prep["static_only"], mult)
        ap_f = pooled_ap_boot(boot_prep["full_history"], mult)
        g = ap_f - ap_s
        if g <= 0:
            n_bad_gain["pooled"] += 1
        else:
            for L in LENGTHS:
                ap = pooled_ap_boot(boot_prep[COND_PREFIX[L]], mult)
                boot_rec["pooled"][f"{L:.1f}"].append((ap - ap_s) / g)
        # macro (定義できるタンパク質の重み付き平均。重みは再標本化の重なり具合)
        dpos = draw_to_defined[draw]  # -1 = 未定義のタンパク質
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
                        (m[cond_idx[COND_PREFIX[L]]] - m[cond_idx["static_only"]]) / gm)
        if (b + 1) % 2000 == 0:
            print("bootstrap %d/%d (%.0fs)" % (b + 1, N_BOOT, time.time() - t0), flush=True)

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

    readings = dict(
        min_length_recovery_0_8=dict(pooled=min_length("pooled", 0.8), macro=min_length("macro", 0.8)),
        min_length_recovery_0_9=dict(pooled=min_length("pooled", 0.9), macro=min_length("macro", 0.9)),
        largest_step=dict(pooled=step, macro=largest_step("macro")),
        step_per_protein=step_per_protein)

    out = {
        "label": "HB formation 50ns, guidance recovery vs short-MD covered length, strong base (K06b)",
        "horizon_ns": 50,
        "length_definition": {
            "unit": "covered length (ns), same definition as L01; prefix = frames 0..(L/0.1) of the 0-1 ns history, 100 ps spacing",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=len(FRAMES[L]), frames=FRAMES[L]) for L in LENGTHS},
            "note": "Differs from the G-series naming at 1.0 ns: G145's '1.0 ns' prefix is frames [0..9], while L01/L02/L05's covered 1.0 ns is frames [0..10] (the full-history mean, G05's/G81's mean_only_full).",
            "g05_mapping": {
                "G05 0.1 ns (frames 0,1)": "covered 0.1 ns",
                "G05 0.5 ns (frames 0..5)": "covered 0.5 ns",
                "G05 mean_only_full (0-1 ns mean)": "covered 1.0 ns",
            },
        },
        "base": {
            "name": "K06(b) strong base: frozen T1I2006 + HistGBM sidecar (160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained on 8 calibration proteins, evaluated on 100 unseen proteins x 3 replicas",
            "static_features": 5, "history_features": 10,
            "n_calibration": int(len(y_cal)), "n_test": int(len(y_test)),
            "test_positives_50ns": int(y_test.sum()),
        },
        "premise_checks": checks,
        "references": {
            "k06b_50ns_static_auprc_pooled": 0.4470, "k06b_50ns_static_auprc_macro": 0.4407,
            "k06b_50ns_full_auprc_pooled": 0.4825, "k06b_50ns_full_auprc_macro": 0.4752,
            "k06b_50ns_gain_pooled": 0.0354, "k06b_50ns_gain_macro": 0.0344,
            "k06b_50ns_test_positives": 12819,
            "g81_50ns_recovery_0.1ns": dict(pooled=0.524, macro=0.540),
            "g81_50ns_recovery_0.5ns": dict(pooled=0.781, macro=0.791),
            "g81_50ns_recovery_mean_only_full": dict(pooled=0.906, macro=0.917),
            "g81_50ns_recovery_exact": {
                "prefix_mean_0.1ns": dict(pooled=0.5238064980419734, macro=0.5403630913030245),
                "prefix_mean_0.5ns": dict(pooled=0.7806433900785225, macro=0.7910237403595737),
                "mean_only_full": dict(pooled=0.9059701849645032, macro=0.9166705258680196)},
            "l05_10ns_recovery": {
                "prefix_mean_0.0ns": dict(pooled=0.452, macro=0.445),
                "prefix_mean_0.1ns": dict(pooled=0.686, macro=0.683),
                "prefix_mean_0.2ns": dict(pooled=0.668, macro=0.674),
                "prefix_mean_0.3ns": dict(pooled=0.783, macro=0.779),
                "prefix_mean_0.4ns": dict(pooled=0.889, macro=0.885),
                "prefix_mean_0.5ns": dict(pooled=0.794, macro=0.827),
                "prefix_mean_0.6ns": dict(pooled=0.906, macro=0.907),
                "prefix_mean_0.7ns": dict(pooled=0.915, macro=0.894),
                "prefix_mean_0.8ns": dict(pooled=1.014, macro=0.967),
                "prefix_mean_0.9ns": dict(pooled=1.097, macro=1.060),
                "prefix_mean_1.0ns": dict(pooled=1.054, macro=1.033)},
            "l05_10ns_readings": {
                "min_length_recovery_0_8": dict(pooled=0.4, macro=0.4),
                "min_length_recovery_0_9": dict(pooled=0.6, macro=0.6),
                "largest_step": "0->0.1 ns (+0.234 pooled / +0.238 macro)"},
            "l02_1ns_recovery": {
                "prefix_mean_0.0ns": dict(pooled=0.353, macro=0.377),
                "prefix_mean_0.1ns": dict(pooled=0.545, macro=0.553),
                "prefix_mean_0.2ns": dict(pooled=0.573, macro=0.588),
                "prefix_mean_0.3ns": dict(pooled=0.670, macro=0.682),
                "prefix_mean_0.4ns": dict(pooled=0.680, macro=0.694),
                "prefix_mean_0.5ns": dict(pooled=0.695, macro=0.686),
                "prefix_mean_0.6ns": dict(pooled=0.774, macro=0.785),
                "prefix_mean_0.7ns": dict(pooled=0.836, macro=0.824),
                "prefix_mean_0.8ns": dict(pooled=0.930, macro=0.896),
                "prefix_mean_0.9ns": dict(pooled=1.028, macro=0.994),
                "prefix_mean_1.0ns": dict(pooled=0.975, macro=0.957)},
            "l02_1ns_readings": {
                "min_length_recovery_0_8": dict(pooled=0.7, macro=0.7),
                "min_length_recovery_0_9": dict(pooled=0.8, macro=0.9),
                "largest_step": "0->0.1 ns (+0.192 pooled / +0.176 macro)"},
        },
        "metrics_50ns": results,
        "base_gain_50ns": dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        "recovery_vs_real_history_gain": recovery,
        "recovery_increment_per_0.1ns": increments,
        "readings": readings,
        "bootstrap": dict(
            n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
            resampling_unit="protein (100 test proteins, resampled with replacement)",
            note="Within each resample, static-only, full-history and prefix-mean AUPRCs are recomputed and the gain and recovery are recomputed inside the sample. Models are not refit; per-candidate predicted probabilities are resampled. Pooled AUPRC uses a numpy implementation of sklearn's average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every condition. Macro AUPRC within a resample is the multiplicity-weighted mean of per-protein AUPRCs over proteins whose AUPRC is defined.",
            n_draws_with_nonpositive_gain=n_bad_gain,
            results=bootstrap),
        "per_protein": per_protein,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # CSV
    lines = ["covered_ns,n_frames,auprc_pooled,auprc_macro,recovery_pooled,recovery_macro,"
             "recovery_pooled_ci_lower,recovery_pooled_ci_upper,recovery_macro_ci_lower,recovery_macro_ci_upper,"
             "increment_pooled,increment_macro,auc,brier,logloss"]
    for L in LENGTHS:
        c = COND_PREFIX[L]
        m = results[c]
        lines.append(",".join([
            f"{L:.1f}", str(len(FRAMES[L])),
            f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}",
            f"{recovery[c]['pooled']:.6f}", f"{recovery[c]['macro']:.6f}",
            f"{bootstrap['pooled'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{bootstrap['macro'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['macro'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{increments['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{increments['macro'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{m['auc']:.6f}", f"{m['brier']:.6f}", f"{m['logloss']:.6f}"]))
    (OUT / "results.csv").write_text("\n".join(lines) + "\n")

    # タンパク質ごとの CSV
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
    # defined かどうかは per_protein_ap が要るので、ここでは静的のみの定義可能性を
    # ラベルから直接判定する (陽性が 0 でも全部でもないタンパク質)
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
