"""L45: 統計量6種で L09 の曲がりを強い土台で測る (K75 の条件、水素結合の解離、1 ns先)。

L09 (水素結合の解離、強い土台、1 ns先、履歴は切り取りの平均だけ) で測った
「覆う長さに対する回復率の曲がり」を、履歴の平均だけではなく統計量6種を
使った場合に測り直し、必要な長さが縮むかを見る。平均だけの条件も同じ実行で
学習し直し、L09 の記録との照合 (前提確認) と、6種との同じ長さでの比較に使う。

統計量6種の定義は実験D (固定した 1 ns 履歴の実験) の features_from_history そのまま
(L39・L40 と同じ):
  最初の値、平均、標準偏差 (ddof=0)、最小値、最後-最初、閾値以下だった割合。
  閾値は水素結合の境界 3.5 Å (実験Dの THRESH["hb"]=3.5)。
  出典: MolCasta-InternalResearch ブランチ research/fixed-1ns-3phen-6h-20261121 の
  GPT遊び場/ATLAS-実軌道-2026-11-19/fixed_1ns_history_3phen_6h.py の features_from_history。
  値は float32 に落とす (同スクリプトの作り方どおり)。
  切り取り (フレーム 0..k) に対しては、同じ式を部分系列に適用する
  (「最後」は切り取りの最後のフレーム、「割合」は切り取り内での割合)。

K75 の履歴特徴10個との関係:
  10個は11フレームの系列から
    [最初, 平均, 標準偏差, 最小, 最大, 最後-最初, 最後-直前,
     3.5 Å以下割合, 4.5 Å以下割合, 増加割合]
  で再現できるはずで、このスクリプトの premise で較正+テストの全候補について
  数値照合する。実験Dの6種はそのうち [最初, 平均, 標準偏差, 最小, 最後-最初,
  3.5 Å以下割合] で、10個は6種に「最大、最後-直前、4.5 Å以下割合、増加割合」の
  4つを足した拡張になっている。したがって全部入り (要約値10個) は6種の上位集合で、
  覆う長さ 1.0 ns の6種条件は、全部入りから4特徴を落とした条件にあたる。

長さの定義は L01・L02・L09 にそのまま合わせる:
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 覆う長さ L ns の切り取りはフレーム 0〜(L/0.1) を使う。
  0 ns -> [0] (1フレーム)、0.1 ns -> [0,1]、…、1.0 ns -> [0..10] (11フレーム全部)。
- 起点フレーム10 (1 ns) は変えず、ラベルは 1 ns先。

条件 (すべて条件ごとに学習し直す):
- 静的のみ: 静的特徴5個。
- 全部入り (full_history): 静的特徴＋実履歴の要約値10個。回復率の分母 (実履歴の利得) の参照。
- 切り取りの平均だけ: 静的特徴＋切り取りの平均だけ。覆う長さ 0.0〜1.0 ns の11通り。
  L09 とまったく同じ作り方 (K2-18 の系統: 系列の平均を float64 のまま使う)。
- 切り取りの統計量6種: 静的特徴＋切り取りの統計量6種。覆う長さ 0.0〜1.0 ns の11通り (主役)。

土台 (K75・K2-18 と同じ、水素結合の解離の強い土台):
- HistGradientBoosting の補正モデル (160回、学習率 0.065、葉 15、
  葉あたり最小 20、L2 0.15、seed 20261121)。較正タンパク質で学習し、
  未見の100タンパク質×3レプリカ (76,342行) で評価する。
- 入力は K2-18 の input_bundle.npz を使う。静的特徴と系列は K2-18 の系統
  (K75・K2-77・K2-84) と同じく float64 にそろえて使う (6種のブロックだけは
  実験Dの作り方どおり float32 に落としてから並べる)。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。
全体 (pooled) とタンパク質ごとの平均 (macro、100タンパク質) の両方で測る
(L09 と同じまとめ方)。K75 の照合用に、タンパク質×レプリカ (300群) の
平均も併せて出す。

読み方 (結果を見る前に固定、L09・L39 と同じ):
1. 回復率が 0.8 と 0.9 に届く最小の覆う長さを、統合値 (pooled) で読む (macro も併記)。
   L09 (平均だけ) の答えと比べて縮んだかを明記する。
2. 段の判定は、隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、その大きさ。
3. 信頼区間はタンパク質単位のブートストラップ 10,000回 (seed 42、パーセンタイル法)。
   各標本の中でも静的のみ・全部入り・各長さの AUPRC を測り直し、
   利得と回復率をその標本の中で計算し直す。6種と平均だけの両方の系列を同じ抽出で測る。
4. 段の前後ではタンパク質ごとの値も記録する。

土台の再現の確認 (premise):
- 統計量の定義の照合: 較正+テストの全候補で、11フレームの系列から計算した
  統計量6種 (部分集合の位置 [0,1,2,3,5,7]) が、保存されている履歴特徴10個の
  該当列と 1e-4 未満で一致すること (float32 の丸めの範囲)。
- 土台の再現: 1 ns先で静的のみ・全部入りの AUPRC (pooled / macro / macro300) と
  利得が、L09 の results.json の記録と 5e-4 以内で一致すること。
- 平均だけの回復率が L09 の記録と全11長さ×全体・平均で 5e-3 以内で一致すること。
- 覆う長さ 1.0 ns の平均だけの回復率 (pooled) が既存の記録 (K2-18 の
  全部の平均だけの回復率 0.9343) と 5e-3 以内で一致すること (外部記録との照合)。
- all_ok は上の全項目が ok のときに限って true にする (L09 で、1項目が
  ok:false なのに all_ok が true になっていた問題の修正)。
  止める条件 (hard) は、統計量の定義の照合・土台の再現・平均だけの回復率の照合。
  外部記録との照合だけが落ちた場合は、all_ok=false を記録した上で続け、
  判断理由を README に明記する。
"""
import json
import os
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
HORIZON_NS = 1
HORIZON_IDX = 0  # HORIZONS_ALL = (1, 5, 10, 30, 50, 90)
THRESH_HB = 3.5  # 実験Dの THRESH["hb"] (水素結合の境界、Å)

REF = {"static_pooled": 0.5060719789337638, "static_macro": 0.5221953617635298, "static_macro300": 0.5329610220169362, "full_pooled": 0.6141881772536423, "full_macro": 0.6246928969663523, "full_macro300": 0.63447293727364, "gain_pooled": 0.10811619831987851, "gain_macro": 0.10249753520282257, "mean_recovery": {"0.0": {"pooled": 0.4434478789161302, "macro": 0.4300614503254031}, "0.1": {"pooled": 0.5423093771885374, "macro": 0.5427488826974132}, "0.2": {"pooled": 0.6214015716161188, "macro": 0.6186213303664637}, "0.3": {"pooled": 0.6990209992717867, "macro": 0.6889700368818172}, "0.4": {"pooled": 0.7158829975207125, "macro": 0.6967131827171602}, "0.5": {"pooled": 0.7678806676841321, "macro": 0.7504200377363607}, "0.6": {"pooled": 0.7986978764010756, "macro": 0.7822788202434656}, "0.7": {"pooled": 0.8560138129617115, "macro": 0.8427130203089553}, "0.8": {"pooled": 0.9065625433095846, "macro": 0.8967016830103263}, "0.9": {"pooled": 0.9377406144984305, "macro": 0.9285845529294904}, "1.0": {"pooled": 0.926155649915475, "macro": 0.9224354722472313}}, "record_readings_mean_only": {"min_length_recovery_0_8": {"pooled": 0.7, "macro": 0.7}, "min_length_recovery_0_9": {"pooled": 0.8, "macro": 0.9}}, "cov10_recovery_pooled": 0.9343, "cov10_source": "K2-18", "test_n": 76342, "test_positives": 12696}

LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(int(round(L / 0.1)) + 1)) for L in LENGTHS}
COND_MEAN = {L: f"prefix_mean_{L:.1f}ns" for L in LENGTHS}
COND_6 = {L: f"prefix_6stats_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = (["static_only", "full_history"]
              + [COND_MEAN[L] for L in LENGTHS]
              + [COND_6[L] for L in LENGTHS])
FAMS = (("mean_only", COND_MEAN), ("stats6", COND_6))


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


def six_stats(S):
    """実験Dの features_from_history を部分系列 S (候補数 × k フレーム) に適用する。

    並びは [最初の値, 平均, 標準偏差 (ddof=0), 最小値, 最後-最初, 3.5 Å以下だった割合]。
    値は float32 に落とす (出典のスクリプトの作り方どおり)。
    """
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
    """系列から計算した統計量が保存されている履歴特徴と一致するかを全数照合する。"""
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


def main():
    t0 = time.time()
    b = np.load(BUNDLE)
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
        stat_check = fs["history_stats_check"]
        hard_ok = fs["hard_ok"]
        print("loaded saved fits (checkpoint)", flush=True)
    else:
        # ---- premise 1: 統計量の定義の照合 (系列 -> 保存されている履歴特徴) ----
        stat_check = check_history_stats(cal, test)
        print("history stats check:", stat_check, "(%.0fs)" % (time.time() - t0), flush=True)
        if not stat_check["ok"]:
            (OUT / "premise_check.json").write_text(json.dumps(
                dict(history_stats_check=stat_check, checks=[], all_ok=False, hard_ok=False),
                indent=2, ensure_ascii=False))
            print("PREMISE FAILED: 統計量の定義が保存されている履歴特徴と結びつかないため、ここで止める。",
                  flush=True)
            sys.exit(2)

        # 条件ごとの特徴量
        feats = {}
        feats["static_only"] = (cal["static"], test["static"])
        feats["full_history"] = (np.column_stack([cal["static"], cal["hist"]]),
                                 np.column_stack([test["static"], test["hist"]]))
        for L in LENGTHS:
            idx = FRAMES[L]
            # 平均だけ: L09 とまったく同じ作り方 (系列の平均を float64 のまま使う)
            feats[COND_MEAN[L]] = (
                np.column_stack([cal["static"], cal["series"][:, idx].mean(axis=1)[:, None]]),
                np.column_stack([test["static"], test["series"][:, idx].mean(axis=1)[:, None]]))
            # 統計量6種: 実験Dの式を切り取りに適用 (float32 に落としてから並べる)
            feats[COND_6[L]] = (
                np.column_stack([cal["static"], six_stats(cal["series"][:, idx])]),
                np.column_stack([test["static"], six_stats(test["series"][:, idx])]))

        preds = {}
        results = {}
        per_protein_ap = {c: {} for c in CONDITIONS}
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
            m["macro300_auprc"] = float(np.mean(gaps))
            m["n_groups_macro300"] = len(gaps)
            results[cname] = m
            print(cname, "pooled %.4f macro %.4f macro300 %.4f (%.0fs)"
                  % (m["auprc"], m["macro_auprc"], m["macro300_auprc"], time.time() - t0), flush=True)

        # ---- premise 2: 土台の再現と平均だけの回復率の照合 (L09 の記録と突き合わせ) ----
        checks = []

        def chk(name, got, ref, tol):
            ok = abs(got - ref) < tol
            checks.append(dict(check=name, got=float(got), reference=float(ref), tol=tol, ok=bool(ok)))
            print("check", name, "got %.4f ref %.4f tol %g -> %s" % (got, ref, tol, ok), flush=True)
            return ok

        gain_pooled0 = results["full_history"]["auprc"] - results["static_only"]["auprc"]
        gain_macro0 = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

        def rec0(cname, kind):
            if kind == "pooled":
                return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled0
            return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro0

        hard = True
        hard &= chk("static_pooled_vs_L09", results["static_only"]["auprc"], REF["static_pooled"], 5e-4)
        hard &= chk("static_macro_vs_L09", results["static_only"]["macro_auprc"], REF["static_macro"], 5e-4)
        hard &= chk("static_macro300_vs_L09", results["static_only"]["macro300_auprc"], REF["static_macro300"], 5e-4)
        hard &= chk("full_pooled_vs_L09", results["full_history"]["auprc"], REF["full_pooled"], 5e-4)
        hard &= chk("full_macro_vs_L09", results["full_history"]["macro_auprc"], REF["full_macro"], 5e-4)
        hard &= chk("full_macro300_vs_L09", results["full_history"]["macro300_auprc"], REF["full_macro300"], 5e-4)
        hard &= chk("gain_pooled_vs_L09", gain_pooled0, REF["gain_pooled"], 5e-4)
        hard &= chk("gain_macro_vs_L09", gain_macro0, REF["gain_macro"], 5e-4)
        for L in LENGTHS:
            c = COND_MEAN[L]
            hard &= chk(f"mean_recovery_{L:.1f}_pooled_vs_L09", rec0(c, "pooled"),
                        REF["mean_recovery"][f"{L:.1f}"]["pooled"], 5e-3)
            hard &= chk(f"mean_recovery_{L:.1f}_macro_vs_L09", rec0(c, "macro"),
                        REF["mean_recovery"][f"{L:.1f}"]["macro"], 5e-3)
        # 外部記録 (K2 系列) との照合。all_ok には含めるが、止める条件にはしない。
        chk("mean_recovery_1.0_pooled_vs_K2record", rec0(COND_MEAN[1.0], "pooled"),
            REF["cov10_recovery_pooled"], 5e-3)
        all_ok = bool(stat_check["ok"]) and all(c["ok"] for c in checks)
        hard_ok = bool(stat_check["ok"]) and bool(hard)
        (OUT / "premise_check.json").write_text(json.dumps(
            dict(history_stats_check=stat_check, checks=checks,
                 all_ok=all_ok, hard_ok=hard_ok),
            indent=2, ensure_ascii=False))
        if not hard_ok:
            print("PREMISE FAILED: 土台の再現が崩れているため、ここで止める。", flush=True)
            sys.exit(2)
        if not all_ok:
            print("premise: hard checks OK, but all_ok=false (外部記録との照合のみ不一致)。"
                  "判断理由は README に明記する。", flush=True)
        print("premise done (%.0fs), all_ok=%s hard_ok=%s" % (time.time() - t0, all_ok, hard_ok), flush=True)
        np.savez_compressed(
            preds_path,
            proteins=np.concatenate([np.full(e_ - s_, p) for p, s_, e_ in spans]),
            label=y_test.astype(np.int8),
            **{f"p_{c}": preds[c].astype(np.float32) for c in CONDITIONS})
        fits_path.write_text(json.dumps(
            dict(results=results, per_protein_ap=per_protein_ap, premise_checks=checks,
                 history_stats_check=stat_check, hard_ok=hard_ok),
            indent=2, ensure_ascii=False))
        print("saved fits checkpoint (%.0fs)" % (time.time() - t0), flush=True)

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

    def rec(cname, kind):
        if kind == "pooled":
            return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

    # ---- 回復率と増え分 (6種が主役、平均だけは比較用) ----
    recovery = {c: dict(pooled=float(rec(c, "pooled")), macro=float(rec(c, "macro")))
                for c in CONDITIONS if c != "static_only"}
    increments = {}
    for fam, cm in FAMS:
        increments[fam] = {}
        for kind in ("pooled", "macro"):
            inc, prev = {}, None
            for L in LENGTHS:
                r = recovery[cm[L]][kind]
                if prev is not None:
                    inc[f"{L:.1f}"] = float(r - prev)
                prev = r
            increments[fam][kind] = inc

    def min_length(cm, kind, thr):
        for L in LENGTHS:
            if recovery[cm[L]][kind] >= thr:
                return L
        return None

    def largest_step(fam, cm, kind):
        inc = increments[fam][kind]
        key = max(inc, key=lambda k: inc[k])
        Lto = float(key)
        return dict(from_ns=round(Lto - 0.1, 1), to_ns=Lto, increment=inc[key])

    # ---- タンパク質ごとの回復率 (6種と平均だけの両方) ----
    def per_protein_for(cm):
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
                ap = per_protein_ap[cm[L]][p]
                entry["auprc_prefix"][f"{L:.1f}"] = ap
                if ap is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                    entry["recovery_prefix"][f"{L:.1f}"] = float((ap - ap_s) / entry["full_gain"])
                else:
                    entry["recovery_prefix"][f"{L:.1f}"] = None
            pp[p] = entry
        return pp

    per_protein = {fam: per_protein_for(cm) for fam, cm in FAMS}

    def dist_for(pp):
        dist = {}
        for L in LENGTHS:
            vals = np.array([e["recovery_prefix"][f"{L:.1f}"] for e in pp.values()
                             if e["recovery_prefix"][f"{L:.1f}"] is not None])
            dist[f"{L:.1f}"] = dict(
                n_proteins=int(len(vals)), median=float(np.median(vals)),
                q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)),
                frac_ge_0p8=float(np.mean(vals >= 0.8)))
        incd = {}
        for k in range(1, len(LENGTHS)):
            Lf, Lt = LENGTHS[k - 1], LENGTHS[k]
            vals = []
            for e in pp.values():
                rf = e["recovery_prefix"].get(f"{Lf:.1f}")
                rt = e["recovery_prefix"].get(f"{Lt:.1f}")
                if rf is not None and rt is not None:
                    vals.append(rt - rf)
            vals = np.array(vals)
            incd[f"{Lt:.1f}"] = dict(
                from_ns=Lf, to_ns=Lt, n_proteins=int(len(vals)),
                median=float(np.median(vals)), q25=float(np.percentile(vals, 25)),
                q75=float(np.percentile(vals, 75)),
                frac_positive=float(np.mean(vals > 0)))
        return dist, incd

    pp_dist, pp_inc = {}, {}
    for fam, _ in FAMS:
        pp_dist[fam], pp_inc[fam] = dist_for(per_protein[fam])

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

    nP = len(proteins)
    boot_path = OUT / "boot_checkpoint.npz"
    rng = np.random.default_rng(BOOT_SEED)
    all_draws = rng.integers(0, nP, size=(N_BOOT, nP)).astype(np.int16)
    ARR_KEYS = ["rec_pooled_mean_only", "rec_pooled_stats6", "rec_macro_mean_only", "rec_macro_stats6"]
    if boot_path.exists():
        zb = np.load(boot_path)
        assert (zb["draws"] == all_draws).all()
        arrs = {k: zb[k].astype(np.float64) for k in ARR_KEYS}
        n_done = int(zb["n_done"])
        n_bad_gain = dict(pooled=int(zb["n_bad_pooled"]), macro=int(zb["n_bad_macro"]))
        print("resuming bootstrap from draw", n_done, flush=True)
    else:
        arrs = {k: np.full((N_BOOT, len(LENGTHS)), np.nan) for k in ARR_KEYS}
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
                for fam, cm in FAMS:
                    arr = arrs[f"rec_pooled_{fam}"]
                    for j, L in enumerate(LENGTHS):
                        ap = pooled_ap_boot(boot_prep[cm[L]], mult)
                        arr[bidx, j] = (ap - ap_s) / g
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
                        arr = arrs[f"rec_macro_{fam}"]
                        for j, L in enumerate(LENGTHS):
                            arr[bidx, j] = (m[cond_idx[cm[L]]] - m[cond_idx["static_only"]]) / gm
        n_done = hi
        tmp = OUT / "boot_checkpoint.tmp.npz"
        np.savez_compressed(tmp, draws=all_draws,
                            **{k: arrs[k].astype(np.float32) for k in ARR_KEYS},
                            n_done=n_done,
                            n_bad_pooled=n_bad_gain["pooled"], n_bad_macro=n_bad_gain["macro"])
        os.replace(tmp, boot_path)
        print("bootstrap %d/%d (%.0fs)" % (n_done, N_BOOT, time.time() - t0), flush=True)
    print("bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)

    bootstrap = {}
    for fam, cm in FAMS:
        bootstrap[fam] = {}
        for kind in ("pooled", "macro"):
            arr_all = arrs[f"rec_{kind}_{fam}"]
            bootstrap[fam][kind] = {}
            for j, L in enumerate(LENGTHS):
                col = arr_all[:, j]
                arr = col[~np.isnan(col)]
                if len(arr) == 0 or np.isnan(arr).any():
                    raise RuntimeError(f"bootstrap {fam} {kind} {L}: invalid values (n={len(arr)})")
                bootstrap[fam][kind][f"{L:.1f}"] = dict(
                    n_boot=int(len(arr)), seed=BOOT_SEED,
                    point_estimate=recovery[cm[L]][kind],
                    ci_lower=float(np.percentile(arr, 2.5)),
                    ci_upper=float(np.percentile(arr, 97.5)),
                    boot_mean=float(arr.mean()), boot_std=float(arr.std()))

    # ---- 段の前後のタンパク質ごとの増え分 (6種の pooled の段の位置で) ----
    step6 = largest_step("stats6", COND_6, "pooled")
    Lfrom, Lto = step6["from_ns"], step6["to_ns"]
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
            largest_step=dict(pooled=step6, macro=largest_step("stats6", COND_6, "macro")),
            step_per_protein=step_per_protein),
        mean_only_recomputed=dict(
            min_length_recovery_0_8=dict(pooled=min_length(COND_MEAN, "pooled", 0.8),
                                         macro=min_length(COND_MEAN, "macro", 0.8)),
            min_length_recovery_0_9=dict(pooled=min_length(COND_MEAN, "pooled", 0.9),
                                         macro=min_length(COND_MEAN, "macro", 0.9)),
            largest_step=dict(pooled=largest_step("mean_only", COND_MEAN, "pooled"),
                              macro=largest_step("mean_only", COND_MEAN, "macro"))),
        l09_record_mean_only=REF["record_readings_mean_only"])

    diff_vs_mean = {f"{L:.1f}": dict(
        pooled=float(recovery[COND_6[L]]["pooled"] - recovery[COND_MEAN[L]]["pooled"]),
        macro=float(recovery[COND_6[L]]["macro"] - recovery[COND_MEAN[L]]["macro"]))
        for L in LENGTHS}

    out = {
        "label": f"HB dissociation {HORIZON_NS}ns, guidance recovery vs short-MD covered length with 6 history statistics (experiment-D definition), strong base (K75)",
        "experiment": "L45",
        "horizon_ns": HORIZON_NS,
        "length_definition": {
            "unit": "covered length (ns), same definition as L01/L02; prefix = frames 0..(L/0.1) of the 0-1 ns history, 100 ps spacing",
            "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=len(FRAMES[L]), frames=FRAMES[L]) for L in LENGTHS},
            "note": "Differs from the G-series naming at 1.0 ns: the G-series '1.0 ns' prefix is frames [0..9], while L01/L02's covered 1.0 ns is frames [0..10] (the full-history mean).",
        },
        "statistics6_definition": {
            "source": "MolCasta-InternalResearch branch research/fixed-1ns-3phen-6h-20261121, GPT playground/ATLAS-observed-2026-11-19/fixed_1ns_history_3phen_6h.py, features_from_history(series, bound); same definition as L39/L40",
            "order": ["first value", "mean", "std (ddof=0)", "min", "last - first",
                      "fraction of frames <= threshold"],
            "threshold_A": THRESH_HB,
            "threshold_source": "THRESH['hb'] = 3.5 in the same script (HB boundary)",
            "application_to_prefix": "the same formula applied to the prefix sub-series frames 0..k; 'last' is the last frame of the prefix and the fraction is taken within the prefix",
            "dtype": "float32 (as in the source script)",
            "relation_to_k75_history10": "K75's 10 history features are [first, mean, std, min, max, last-first, last-prev, frac<=3.5A, frac<=4.5A, frac increasing], stored in the K2-18 bundle and recomputed from the stored series for all calibration+test candidates (see history_stats_check). The experiment-D 6 statistics are the subset at positions [0,1,2,3,5,7]; the 10 are the 6 plus [max, last-prev, frac<=4.5A, frac increasing]. Full history (10 stats on all 11 frames) is therefore a superset of the covered-1.0 ns 6-stats condition.",
        },
        "base": {
            "name": "K75 strong base for HB dissociation: HistGBM correction model (160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained on calibration proteins, evaluated on 100 unseen proteins x 3 replicas (input bundle from K2-18)",
            "static_features": 5, "history_features_full": 10,
            "n_calibration": int(len(y_cal)), "n_test": int(len(y_test)),
            f"test_positives_{HORIZON_NS}ns": int(y_test.sum()),
        },
        "premise_checks": checks,
        "history_stats_check": stat_check,
        "premise_all_ok": bool(stat_check["ok"]) and all(c["ok"] for c in checks),
        "premise_hard_ok": bool(hard_ok),
        "references": {
            "l09_record": REF,
        },
        f"metrics_{HORIZON_NS}ns": results,
        f"base_gain_{HORIZON_NS}ns": dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        "recovery_vs_real_history_gain": recovery,
        "recovery_increment_per_0.1ns": increments,
        "recovery_diff_stats6_minus_mean": diff_vs_mean,
        "readings": readings,
        "per_protein_recovery_distribution": pp_dist,
        "per_protein_increments": pp_inc,
        "bootstrap": dict(
            n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
            resampling_unit="protein (100 test proteins, resampled with replacement)",
            note="Within each resample, static-only, full-history and each prefix condition's AUPRCs are recomputed and the gain and recovery are recomputed inside the sample. Models are not refit; per-candidate predicted probabilities are resampled. Both families (mean-only and 6-stats) are bootstrapped from the same draws. Pooled AUPRC uses a numpy implementation of sklearn's average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every condition. Macro AUPRC within a resample is the multiplicity-weighted mean of per-protein AUPRCs over proteins whose AUPRC is defined.",
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

    np.savez_compressed(
        OUT / "test_predictions.npz",
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]),
        label=y_test.astype(np.int8),
        **{c: preds[c].astype(np.float32) for c in CONDITIONS})
    print("saved results.json / results.csv / per_protein.csv / test_predictions.npz (%.0fs total)" % (time.time() - t0), flush=True)
    print(json.dumps(readings, indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(diff_vs_mean, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
