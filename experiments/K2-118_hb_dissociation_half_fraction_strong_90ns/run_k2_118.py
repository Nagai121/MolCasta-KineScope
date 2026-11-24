"""K2-118: K2-14 の問い（強い土台での水素結合の解離の割合の前半と
後半）を 90 ns先で測る（水素結合の解離、90 ns先、割合だけ）.

平均だけの系列は 90 ns先（K2-84：前半の平均だけの利得は全部入りの
合算で 0.893・マクロで 0.897、後半だけが合算で 0.578・マクロで 0.565、
読み方は 4 通りともどちらにも当てはまらず、タンパク質ごとの形は
「今の状態」3・「続いている状態」14・どちらでもない 62・利得が小さく
判定から外したもの 21）で測り終えている。ここでは、同じ問いを割合だけ
で測る。強い土台での解離の割合の前半と後半は、まだどの予測幅でも
測っていない。

条件は強い土台（K06(b) の条件、凍結した本体 T1I2006 + HistGradientBoosting
の補正モデル、160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
seed 20261121）で、水素結合の解離、90 ns先、割合だけ。割合は、ペアが
境界（3.5 Å、K71 と同じ）より近い時間の割合で、与えられたフレームの
範囲について、境界以下だったフレームの数 ÷ その範囲のフレームの数で
計算する。較正 8 タンパク質（6,144件）で学習し、未見の 100 タンパク質
×3 レプリカ（76,342件）で評価する。入力は K2-18 の input_bundle.npz
（K75 のキャッシュの静的な特徴・履歴・ラベルと、軌道から取り直した
11 フレームの系列をまとめたもの）をそのまま使う。ラベルの列は
[1, 5, 10, 30, 50, 90] ns の順で、90 ns先は列 5。

入力は、静的な特徴量に、つぎのどれか 1 つを足した形で比べる:
- 前半（フレーム [0..4]）の割合
- 後半（フレーム [6..10]）の割合
- 前半と後半の両方（2 つの割合を並べたもの）
- 全部（フレーム [0..10]）の割合

土台の再現: 静的のみと全部の 10 統計量の AUPRC が、K75 の記録
（90 ns先、静的のみ 0.6272 / 0.6494、全部入り 0.6758 / 0.6931、
利得 +0.0486 / +0.0436、合算 / マクロ）と一致することを確かめる。
K75 の記録は予測幅ごとの行列なので、照合には訓練も評価も 90 ns先の
対角の値を使い、訓練が 1 ns先の行（転移の値）は使わない（K2-73 で
分かった注意点）。
マクロは K75 と同じく、タンパク質×レプリカごとの AUPRC の平均。
あわせて、K2-07 と同じタンパク質ごと（3 レプリカをまとめた 100 個）
の平均も記録する。

読み方（K2-01 と同じく先に固定）。割合の分母は全部入りの利得。
- 後半の利得が全部入りの 8 割以上で、前半の利得が全部入りの 5 割以下
  なら「今の状態が効いている」。
- 前半も後半も全部入りの 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する。
合算とマクロ（K75 の定義と K2-07 の定義の両方）で判定する。分母を
全部の割合の利得にしても判定が変わる場合は、両方を記録する。

注意点:
- K2-18 は Colab（sklearn 1.6.1）で動かし、土台の再現の照合の許容差
  を 1e-3 とした。ここではこの仮想マシン（sklearn 1.9.1）で動かす。
  K75 の記録がどちらの版で作られたかは記録にないため、許容差は
  K2-18 と同じ 1e-3 とし、実際の差を results.json の checks に記録する。
- sklearn の学習はスレッド数を 1 に絞る。
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import sklearn
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

BUNDLE = Path("/home/hatch/workspace/kinescope/repo/experiments/"
              "K2-18_hb_dissociation_half_means_strong_base/input_bundle.npz")
OUT = Path(__file__).resolve().parent

FRAMES_FIRST = [0, 1, 2, 3, 4]       # 前半（0〜0.4 ns）
FRAMES_SECOND = [6, 7, 8, 9, 10]     # 後半（0.6〜1.0 ns）
FRAMES_FULL = list(range(11))        # 全部（0〜1.0 ns）
HIDX = 5  # 90 ns先（ラベルの列は [1, 5, 10, 30, 50, 90] ns の順）
BOUND = 3.5  # 割合の境界（K71 と同じ）

# K75 の記録との照合用の期待値（90 ns先、90 ns先で学習したモデル＝対角）
REF = {"static_pooled": 0.627218, "static_macro": 0.649445,
       "full_pooled": 0.675771, "full_macro": 0.693082,
       "test_n": 76342, "test_positives": 16758, "cal_n": 6144}
TOL = 1e-3  # K2-18 と同じ許容差（sklearn の版違いを吸収するため）


def main():
    print("sklearn", sklearn.__version__, "numpy", np.__version__, flush=True)
    with np.load(BUNDLE) as z:
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

    def frac_of(S, fr, bound=BOUND):
        """S: (N, 11)。指定フレームの境界以下割合を返す (N, 1)。"""
        return (S[:, fr] <= bound).mean(axis=1, keepdims=True)

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
    print(f"--- horizon 90 ns: cal pos {int(ytr.sum())} test pos {int(yte.sum())} ---",
          flush=True)
    CONDS = ["static_only", "full_history", "first_half_frac",
             "second_half_frac", "both_halves_frac", "full_frac"]

    def feats_for(cname):
        if cname == "static_only":
            return cal_static, te_static
        if cname == "full_history":
            return (np.column_stack([cal_static, cal_hist]),
                    np.column_stack([te_static, te_hist]))
        if cname == "first_half_frac":
            return (np.column_stack([cal_static, frac_of(cal_series, FRAMES_FIRST)]),
                    np.column_stack([te_static, frac_of(te_series, FRAMES_FIRST)]))
        if cname == "second_half_frac":
            return (np.column_stack([cal_static, frac_of(cal_series, FRAMES_SECOND)]),
                    np.column_stack([te_static, frac_of(te_series, FRAMES_SECOND)]))
        if cname == "both_halves_frac":
            return (np.column_stack([cal_static, frac_of(cal_series, FRAMES_FIRST),
                                     frac_of(cal_series, FRAMES_SECOND)]),
                    np.column_stack([te_static, frac_of(te_series, FRAMES_FIRST),
                                     frac_of(te_series, FRAMES_SECOND)]))
        if cname == "full_frac":
            return (np.column_stack([cal_static, frac_of(cal_series, FRAMES_FULL)]),
                    np.column_stack([te_static, frac_of(te_series, FRAMES_FULL)]))
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
    gain_frac_full = res["full_frac"]["auprc"] - res["static_only"]["auprc"]
    gain_frac_full_macro = (res["full_frac"]["macro_auprc"]
                            - res["static_only"]["macro_auprc"])
    gain_frac_full_mprot = (res["full_frac"]["macro_protein_auprc"]
                            - res["static_only"]["macro_protein_auprc"])
    gains = {c: res[c]["auprc"] - res["static_only"]["auprc"] for c in CONDS}
    gains_macro = {c: res[c]["macro_auprc"] - res["static_only"]["macro_auprc"]
                   for c in CONDS}
    gains_mprot = {c: res[c]["macro_protein_auprc"]
                   - res["static_only"]["macro_protein_auprc"] for c in CONDS}
    ratios = {c: gains[c] / gain_full for c in CONDS}
    ratios_macro = {c: gains_macro[c] / gain_full_macro for c in CONDS}
    ratios_mprot = {c: gains_mprot[c] / gain_full_mprot for c in CONDS}
    ratios_vs_frac = {c: gains[c] / gain_frac_full for c in CONDS}
    ratios_vs_frac_macro = {c: gains_macro[c] / gain_frac_full_macro for c in CONDS}
    ratios_vs_frac_mprot = {c: gains_mprot[c] / gain_frac_full_mprot for c in CONDS}

    def reading(first, second):
        if second >= 0.8 and first <= 0.5:
            return "今の状態が効いている"
        if first >= 0.8 and second >= 0.8:
            return "続いている状態が効いている"
        return "どちらでもない（数値をそのまま記録）"

    judgment = {
        "pooled": reading(ratios["first_half_frac"], ratios["second_half_frac"]),
        "macro": reading(ratios_macro["first_half_frac"], ratios_macro["second_half_frac"]),
        "macro_protein": reading(ratios_mprot["first_half_frac"],
                                 ratios_mprot["second_half_frac"]),
        "pooled_vs_full_frac": reading(ratios_vs_frac["first_half_frac"],
                                       ratios_vs_frac["second_half_frac"]),
        "macro_vs_full_frac": reading(ratios_vs_frac_macro["first_half_frac"],
                                      ratios_vs_frac_macro["second_half_frac"]),
        "macro_protein_vs_full_frac": reading(ratios_vs_frac_mprot["first_half_frac"],
                                              ratios_vs_frac_mprot["second_half_frac"]),
    }
    print("gains", {c: round(v, 4) for c, v in gains.items()}, flush=True)
    print("ratios pooled", {c: round(v, 3) for c, v in ratios.items()}, flush=True)
    print("ratios macro", {c: round(v, 3) for c, v in ratios_macro.items()}, flush=True)
    print("ratios vs full-frac pooled", {c: round(v, 3) for c, v in ratios_vs_frac.items()}, flush=True)
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
        f = (per_protein["first_half_frac"][str(prot)]["auprc"]
             - per_protein["static_only"][str(prot)]["auprc"]) / gf
        s = (per_protein["second_half_frac"][str(prot)]["auprc"]
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
        "label": "HB dissociation, first-half vs second-half history fraction on the strong base at 90 ns (K2-14 question at 90 ns, fraction only)",
        "phenomenon": "HB dissociation (strong base, K75 conditions)",
        "horizon_ns": 90,
        "bound_A": BOUND,
        "fraction": "fraction of frames with pair distance <= 3.5 A",
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
        "gain_ratios_vs_full_frac": ratios_vs_frac,
        "gain_ratios_vs_full_frac_macro": ratios_vs_frac_macro,
        "gain_ratios_vs_full_frac_macro_protein": ratios_vs_frac_mprot,
        "reading_rule": ("ratios are vs the full-history gain (K2-01 convention); "
                         "second >= 0.8 and first <= 0.5 -> current state; "
                         "both >= 0.8 -> persistent state; else record numbers"),
        "judgment": judgment,
        "per_protein_shapes": shapes,
        "per_protein_fractions": prot_frac,
        "checks_vs_K75": checks,
        "reference": REF,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    np.savez_compressed(
        OUT / "test_predictions.npz",
        **{f"pred_{c}": preds[c].astype(np.float32) for c in CONDS},
        y_test=yte, protein=te_protein, replica=te_replica)
    print("saved", OUT / "results.json", OUT / "test_predictions.npz", flush=True)


if __name__ == "__main__":
    main()
