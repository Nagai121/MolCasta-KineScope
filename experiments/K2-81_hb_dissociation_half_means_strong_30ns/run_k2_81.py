"""K2-81: K2-18 の問い（強い土台での水素結合の解離の前半と後半、平均だけ）を 30 ns先で測る.

K2-18 では、強い土台の水素結合の解離の 1 ns先で、前半の平均だけの利得が
全部入りの 0.738 / 0.720、後半だけが 0.757 / 0.757（合算 / マクロ）で、
前半と後半はほぼ同じ大きさだった（どちらにも当てはまらなかった）。
ここでは同じ問いを 30 ns先で測る。5 ns先（K2-73）では、合算で
前半 0.782・後半 0.808 とどちらにも当てはまらず、マクロでは
前半 0.808・後半 0.804 と「続いている状態が効いている」になった。
タンパク質ごとの形の数え上げも K2-18 と同じく記録する。

条件は強い土台（K06(b) の条件、凍結した本体 T1I2006 + HistGradientBoosting
の補正モデル、160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
seed 20261121）で、水素結合の解離、30 ns先。較正 8 タンパク質（6,144件）
で学習し、未見の 100 タンパク質×3 レプリカ（76,342件）で評価します。

入力は、静的な特徴量に、つぎのどれか 1 つを足した形で比べます:
- 前半（フレーム [0..4]）の平均
- 後半（フレーム [6..10]）の平均
- 前半と後半の両方（2 つの平均を並べたもの）
- 全部（フレーム [0..10]）の平均

入力は K2-18 の input_bundle.npz（K75 のキャッシュの静的な特徴・履歴・
ラベルと、軌道から取り直した 11 フレームの系列をまとめたもの。K2-18 の
ディレクトリに置いてある、コミット済みのファイル）をそのまま使います。
ラベルは 6 つの予測幅すべて入っているため、30 ns先は HIDX=3 で読みます。

土台の再現: 静的のみと全部の 10 統計量の AUPRC が、K75 の記録
（30 ns先、静的のみ 0.6042 / 0.6261、全部入り 0.6663 / 0.6828、
利得 +0.0621 / +0.0567、合算 / マクロ）と一致することを確かめます。
なお、K75 の記録は予測幅ごとの行列（訓練した予測幅 × 評価した予測幅）で
あるため、30 ns先で測り直す実験の照合には、訓練も評価も 30 ns先の
対角の値（static_train[30] と history_train[30] の「30」の列）を使います。
訓練が 1 ns先の行（static_train[1] / history_train[1] の「30」の列）は
読み出しの転移の値であって、測り直しの照合には使いません。
マクロは K75 と同じく、タンパク質×レプリカごとの AUPRC の平均です。
あわせて、K2-07 と同じタンパク質ごと（3 レプリカをまとめた 100 個）
の平均も記録します。

読み方（K2-01 と同じく先に固定）。割合の分母は全部入りの利得です。
- 後半の利得が全部入りの 8 割以上で、前半の利得が全部入りの 5 割以下
  なら「今の状態が効いている」。
- 前半も後半も全部入りの 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する。
合算とマクロ（K75 の定義と K2-07 の定義の両方）で判定します。

実行はローカルの仮想マシンで行います（6 条件の学習で数分。K2 系列では
重い計算は Colab に逃がす決め事がありますが、K2-07 の強い土台の
6 条件の学習が約 1 分で済んだのと同じ規模のため、重い計算には
あたらないと判断しました）。sklearn の学習はスレッド数を 1 に絞ります。
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from pathlib import Path

import numpy as np
import sklearn

# 入力は K2-18 の input_bundle.npz をそのまま使います（コミット済み）。
BUNDLE = (Path(__file__).resolve().parent.parent
          / "K2-18_hb_dissociation_half_means_strong_base"
          / "input_bundle.npz")
OUTDIR = Path(__file__).resolve().parent

FRAMES_FIRST = [0, 1, 2, 3, 4]       # 前半（0〜0.4 ns）
FRAMES_SECOND = [6, 7, 8, 9, 10]     # 後半（0.6〜1.0 ns）
FRAMES_FULL = list(range(11))        # 全部（0〜1.0 ns）
HIDX = 3  # 30 ns先

# K75 の記録との照合用の期待値（30 ns先）
REF = {"static_pooled": 0.604201, "static_macro": 0.626123,
       "full_pooled": 0.666274, "full_macro": 0.682811,
       "test_n": 76342, "test_positives": 15905, "cal_n": 6144}
TOL = 5e-4  # ローカル実行のため、仮想マシン側の実験と同じ許容差


def main():
    print("sklearn", sklearn.__version__, "numpy", np.__version__, flush=True)
    bundle_path = BUNDLE
    print("using bundle:", bundle_path, flush=True)
    print("bundle bytes:", bundle_path.stat().st_size, flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 log_loss, roc_auc_score)

    with np.load(bundle_path) as z:
        cal_static = z["cal_static"].astype(np.float64)
        cal_hist = z["cal_history"].astype(np.float64)
        cal_series = z["cal_series"].astype(np.float64)  # (N, 11)
        cal_label = z["cal_label"]
        te_static = z["test_static"].astype(np.float64)
        te_hist = z["test_history"].astype(np.float64)
        te_series = z["test_series"].astype(np.float64)  # (N, 11)
        te_label = z["test_label"]
        te_protein = z["test_protein"]
        te_replica = z["test_replica"]
    print("cal", cal_static.shape, "test", te_static.shape, flush=True)

    # 前提の確認：系列から計算した平均がキャッシュの history の 1 列目と一致する
    for name, s, h in (("cal", cal_series, cal_hist), ("test", te_series, te_hist)):
        d = np.max(np.abs(s.mean(axis=1) - h[:, 1]))
        print(f"{name} series mean vs history col 1 max diff: {d:.2e}", flush=True)
        assert d < 1e-4, (name, d)

    def mean_of(S, fr):
        return S[:, fr].mean(axis=1, keepdims=True)

    def predict(xtr, ytr, xte):
        if np.unique(ytr).size < 2:
            return np.full(len(xte), float(ytr.mean()))
        m = HistGradientBoostingClassifier(max_iter=160, learning_rate=.065,
                                           max_leaf_nodes=15, min_samples_leaf=20,
                                           l2_regularization=.15, random_state=20261121)
        m.fit(xtr, ytr)
        return m.predict_proba(xte)[:, 1]

    def metrics(y, p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
        return dict(n=int(len(y)), positives=int(y.sum()),
                    auc=float(roc_auc_score(y, p)),
                    auprc=float(average_precision_score(y, p)),
                    brier=float(brier_score_loss(y, p)),
                    logloss=float(log_loss(y, p, labels=[0, 1])))

    def macro_auprc(y, p, groups):
        """groups ごとの AUPRC の平均（両方のクラスがあるグループだけ）。"""
        vals = []
        for g in np.unique(groups):
            m = groups == g
            if 0 < y[m].sum() < m.sum():
                vals.append(average_precision_score(y[m], p[m]))
        return float(np.mean(vals)), len(vals)

    # マクロのグループ分け：K75 はタンパク質×レプリカ、K2-07 はタンパク質
    grp_rep = np.array([f"{p}_R{r}" for p, r in zip(te_protein, te_replica)])
    grp_prot = te_protein

    ytr, yte = cal_label[:, HIDX], te_label[:, HIDX]
    CONDS = ["static_only", "full_history", "first_half_mean",
             "second_half_mean", "both_halves", "full_mean"]

    def feats_for(cname):
        if cname == "static_only":
            return cal_static, te_static
        if cname == "full_history":
            return (np.column_stack([cal_static, cal_hist]),
                    np.column_stack([te_static, te_hist]))
        if cname == "first_half_mean":
            return (np.column_stack([cal_static, mean_of(cal_series, FRAMES_FIRST)]),
                    np.column_stack([te_static, mean_of(te_series, FRAMES_FIRST)]))
        if cname == "second_half_mean":
            return (np.column_stack([cal_static, mean_of(cal_series, FRAMES_SECOND)]),
                    np.column_stack([te_static, mean_of(te_series, FRAMES_SECOND)]))
        if cname == "both_halves":
            return (np.column_stack([cal_static, mean_of(cal_series, FRAMES_FIRST),
                                     mean_of(cal_series, FRAMES_SECOND)]),
                    np.column_stack([te_static, mean_of(te_series, FRAMES_FIRST),
                                     mean_of(te_series, FRAMES_SECOND)]))
        if cname == "full_mean":
            return (np.column_stack([cal_static, mean_of(cal_series, FRAMES_FULL)]),
                    np.column_stack([te_static, mean_of(te_series, FRAMES_FULL)]))
        raise ValueError(cname)

    res, preds = {}, {}
    per_protein = {}
    for cname in CONDS:
        xtr, xte = feats_for(cname)
        p = predict(xtr, ytr, xte)
        preds[cname] = p
        m = metrics(yte, p)
        m["macro_auprc"], m["macro_n_groups"] = macro_auprc(yte, p, grp_rep)
        m["macro_protein_auprc"], m["macro_protein_n"] = macro_auprc(yte, p, grp_prot)
        res[cname] = m
        per_protein[cname] = {}
        for prot in np.unique(te_protein):
            mm = te_protein == prot
            per_protein[cname][str(prot)] = metrics(yte[mm], p[mm])
        print(cname, {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in m.items()}, flush=True)

    # K75 の記録との照合
    assert res["static_only"]["n"] == REF["test_n"]
    assert res["static_only"]["positives"] == REF["test_positives"]
    assert len(cal_label) == REF["cal_n"]
    checks = {}
    for c, key_p, key_m in (("static_only", "static_pooled", "static_macro"),
                            ("full_history", "full_pooled", "full_macro")):
        dp = res[c]["auprc"] - REF[key_p]
        dm = res[c]["macro_auprc"] - REF[key_m]
        checks[c] = {"pooled_diff": float(dp), "macro_diff": float(dm)}
        assert abs(dp) < TOL, (c, "pooled", res[c]["auprc"])
        assert abs(dm) < TOL, (c, "macro", res[c]["macro_auprc"])
    print("base reproduction vs K75: OK", checks, flush=True)

    gain_full = res["full_history"]["auprc"] - res["static_only"]["auprc"]
    gain_full_macro = res["full_history"]["macro_auprc"] - res["static_only"]["macro_auprc"]
    gain_full_mprot = (res["full_history"]["macro_protein_auprc"]
                       - res["static_only"]["macro_protein_auprc"])
    gain_mean_full = res["full_mean"]["auprc"] - res["static_only"]["auprc"]
    gains = {c: res[c]["auprc"] - res["static_only"]["auprc"] for c in CONDS}
    gains_macro = {c: res[c]["macro_auprc"] - res["static_only"]["macro_auprc"]
                   for c in CONDS}
    gains_mprot = {c: res[c]["macro_protein_auprc"]
                   - res["static_only"]["macro_protein_auprc"] for c in CONDS}
    ratios = {c: gains[c] / gain_full for c in CONDS}
    ratios_macro = {c: gains_macro[c] / gain_full_macro for c in CONDS}
    ratios_mprot = {c: gains_mprot[c] / gain_full_mprot for c in CONDS}
    ratios_vs_mean = {c: gains[c] / gain_mean_full for c in CONDS}

    def reading(first, second):
        if second >= 0.8 and first <= 0.5:
            return "今の状態が効いている"
        if first >= 0.8 and second >= 0.8:
            return "続いている状態が効いている"
        return "どちらでもない（数値をそのまま記録）"

    judgment = {
        "pooled": reading(ratios["first_half_mean"], ratios["second_half_mean"]),
        "macro": reading(ratios_macro["first_half_mean"], ratios_macro["second_half_mean"]),
        "macro_protein": reading(ratios_mprot["first_half_mean"],
                                 ratios_mprot["second_half_mean"]),
        "pooled_vs_full_mean": reading(ratios_vs_mean["first_half_mean"],
                                       ratios_vs_mean["second_half_mean"]),
    }
    print("gains", {c: round(v, 4) for c, v in gains.items()}, flush=True)
    print("ratios pooled", {c: round(v, 3) for c, v in ratios.items()}, flush=True)
    print("ratios macro", {c: round(v, 3) for c, v in ratios_macro.items()}, flush=True)
    print("judgment", judgment, flush=True)

    # タンパク質ごとの形の数え上げ（K2-07 と同じ。分母はそのタンパク質の全部入りの利得）
    shapes = {"now_state": 0, "continuing": 0, "other": 0, "excluded_gain_le_0.01": 0}
    prot_frac = {}
    for prot in np.unique(te_protein):
        gf = (per_protein["full_history"][str(prot)]["auprc"]
              - per_protein["static_only"][str(prot)]["auprc"])
        if gf <= 0.01:
            shapes["excluded_gain_le_0.01"] += 1
            continue
        f = (per_protein["first_half_mean"][str(prot)]["auprc"]
             - per_protein["static_only"][str(prot)]["auprc"]) / gf
        s = (per_protein["second_half_mean"][str(prot)]["auprc"]
             - per_protein["static_only"][str(prot)]["auprc"]) / gf
        prot_frac[str(prot)] = {"first": float(f), "second": float(s)}
        if s >= 0.8 and f <= 0.5:
            shapes["now_state"] += 1
        elif f >= 0.8 and s >= 0.8:
            shapes["continuing"] += 1
        else:
            shapes["other"] += 1
    print("per-protein shapes", shapes, flush=True)

    out = {
        "label": "HB dissociation, first-half vs second-half history means on the strong base at 30 ns (K2-18 question at 30 ns)",
        "phenomenon": "HB dissociation (strong base, K75 conditions)",
        "horizon_ns": 30,
        "frames": {"first_half": FRAMES_FIRST, "second_half": FRAMES_SECOND,
                   "full": FRAMES_FULL,
                   "note": "frame 5 belongs to neither half"},
        "sklearn_version": sklearn.__version__,
        "metrics": res,
        "per_protein_metrics": per_protein,
        "gains": gains,
        "gains_macro": gains_macro,
        "gains_macro_protein": gains_mprot,
        "gain_ratios_vs_full_history": ratios,
        "gain_ratios_vs_full_history_macro": ratios_macro,
        "gain_ratios_vs_full_history_macro_protein": ratios_mprot,
        "gain_ratios_vs_full_mean": ratios_vs_mean,
        "reading_rule": ("ratios are vs the full-history gain (K2-01 convention); "
                         "second >= 0.8 and first <= 0.5 -> current state; "
                         "both >= 0.8 -> persistent state; else record numbers"),
        "judgment": judgment,
        "per_protein_shapes": shapes,
        "per_protein_fractions": prot_frac,
        "checks_vs_K75": checks,
        "reference": REF,
    }
    (OUTDIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUTDIR / "results.json", flush=True)
    # 予測確率は float32 に落として圧縮して保存する（2026-11-24 の
    # ディスククォータ超過のため。保存に失敗しても results.json は
    # 残る順番にしておく。予測確率はスクリプトを再実行すれば作り直せる）
    try:
        np.savez_compressed(
            OUTDIR / "test_predictions.npz",
            **{f"pred_{c}": preds[c].astype(np.float32) for c in CONDS},
            y_test=yte, protein=te_protein, replica=te_replica)
        print("saved", OUTDIR / "test_predictions.npz", flush=True)
    except OSError as e:
        print(f"WARNING: test_predictions.npz not saved ({e}); "
              "predictions are regenerable by re-running this script", flush=True)
    print("RESULT_JSON " + json.dumps(out, ensure_ascii=False), flush=True)
    print("RESULT " + json.dumps({
        "first_ratio": ratios["first_half_mean"],
        "second_ratio": ratios["second_half_mean"],
        "both_ratio": ratios["both_halves"],
        "full_mean_ratio": ratios["full_mean"],
        "judgment": judgment}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
