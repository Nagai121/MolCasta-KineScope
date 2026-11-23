"""G72: 解離で、6 個の統計量の切り取りでも長い予測幅での
回復率を測る（資料Dの条件、水素結合の解離、5・10・30・50・90 ns先）。

G70 は平均だけの切り取りで長い予測幅を測った。0.5 を超える最小の
切り取りは全予測幅で 0.1 ns だった。ここでは同じ予測幅・切り取りで、
6 個の統計量を入れたときの回復率を測る。

タンパク質ごとの回復率も出す。解離のテストは 1mix_A と 2qt4_A の
2タンパク質だけなので、タンパク質ごとの表は 2行になる。定義は
G48・G66 と同じで、タンパク質ごとの利得を分母にする（利得の絶対値が
0.01 以下のタンパク質と、陽性が 0件・全件のタンパク質は回復率を
出さない）。平均と中央値の両方を出すが、2タンパク質しかないため
平均は 2つの値の平均にすぎない点に注意する。

土台（資料Dの条件、水素結合の解離、G70 と同じ）:
- K46 のキャッシュ（~/workspace/kinescope/work/data_cache_k46/ の
  {protein}_R{rep}.npz、静的特徴7個・11フレームの履歴・ラベル）。
  学習 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算、
  候補は学習 4,051件・テスト 2,080件、起点フレーム10（1 ns）、
  履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、seed 42）、条件ごとに学習し直す。
- 6 統計量の定義: [最初, 平均, 標準偏差, 最小, 最後−最初,
  3.5 Å以下割合] を切り取りのフレームで計算する（G48 と同じ形で、
  境界は解離の 3.5 Å）。
- 切り取り: 0.1 ns = フレーム [0,1]、0.25 ns = フレーム [0,1,2]、
  0.5 ns = フレーム [0,1,2,3,4,5]。
- 土台の利得は K46 の各予測幅の記録値。静的のみ・全部入りの AUPRC
  が記録値と小数第4位まで一致することを各予測幅で確かめる。
- 注意: K46 のキャッシュの hist は (11, n) の並びで保存されている
  ため、読み込み時に転置して (n, 11) に直す（G70 と同じ）。

判定（先に固定）: 各予測幅で、回復率 0.5 を超える最小の切り取りを
出す。タンパク質ごとの回復率も併記する。
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
OUT = Path("/home/hatch/workspace/g72")
BOUND = 3.5
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = [(1, 5), (2, 10), (3, 30), (4, 50), (5, 90)]
# K46 の記録値（静的のみ・全部入り）。照合用。
REF = {
    5: (0.4346, 0.6103), 10: (0.4496, 0.5974), 30: (0.4409, 0.5731),
    50: (0.4088, 0.5231), 90: (0.5118, 0.5797),
}
PREFIX_FRAMES = {"0.1ns": [0, 1], "0.25ns": [0, 1, 2], "0.5ns": [0, 1, 2, 3, 4, 5]}
PREFIX_TAGS = ["0.1ns", "0.25ns", "0.5ns"]


def load_with_protein(prots):
    xs, hs, ys, ps = [], [], [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            xs.append(z["X"])
            hs.append(z["hist"].T)  # キャッシュは (11, n) なので (n, 11) に直す
            ys.append(z["y"])
            ps.extend([p] * len(z["y"]))
    return (np.concatenate(xs), np.concatenate(hs), np.concatenate(ys),
            np.array(ps))


def six_stats(series):
    return np.column_stack([series[:, 0], series.mean(axis=1), series.std(axis=1),
                            series.min(axis=1), series[:, -1] - series[:, 0],
                            (series <= BOUND).mean(axis=1)]).astype(np.float32)


def fit_predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def main():
    Xtr, Htr, Ytr, _ = load_with_protein(TRAIN_P)
    Xte, Hte, Yte, Pte = load_with_protein(TEST_P)
    print("train", Xtr.shape, "test", Xte.shape, flush=True)

    feats = {"static_only": (Xtr, Xte),
             "full_hist6": (np.column_stack([Xtr, six_stats(Htr)]),
                            np.column_stack([Xte, six_stats(Hte)]))}
    for tag in PREFIX_TAGS:
        fr = PREFIX_FRAMES[tag]
        feats[f"prefix_hist6_{tag}"] = (
            np.column_stack([Xtr, six_stats(Htr[:, fr])]),
            np.column_stack([Xte, six_stats(Hte[:, fr])]))

    out = {"reference_k46": {str(h): {"static": s, "full": f}
                             for h, (s, f) in REF.items()},
           "horizons": {}, "smallest_prefix_over_0.5_pooled": {},
           "smallest_prefix_over_0.5_macro": {}}
    per_protein_all = []
    for h_idx, h_ns in HORIZONS:
        ytr, yte = Ytr[:, h_idx], Yte[:, h_idx]
        print(f"=== horizon {h_ns} ns, test positives {int(yte.sum())} ===", flush=True)
        preds, results = {}, {}
        for name, (xtr, xte) in feats.items():
            p = fit_predict(xtr, ytr, xte)
            preds[name] = p
            results[name] = metrics(yte, p)
            print(f"  {name} auprc {results[name]['auprc']:.4f}", flush=True)
        ref_s, ref_f = REF[h_ns]
        assert abs(results["static_only"]["auprc"] - ref_s) < 5e-4, (h_ns, results["static_only"])
        assert abs(results["full_hist6"]["auprc"] - ref_f) < 5e-4, (h_ns, results["full_hist6"])
        gain = results["full_hist6"]["auprc"] - results["static_only"]["auprc"]
        recovery = {n: float((m["auprc"] - results["static_only"]["auprc"]) / gain)
                    for n, m in results.items() if n != "static_only"}
        print(f"  base gain {gain:.4f}, pooled recovery:",
              {n: round(v, 3) for n, v in recovery.items()}, flush=True)

        conds = [f"prefix_hist6_{tag}" for tag in PREFIX_TAGS]
        per_protein = []
        for p in sorted(set(Pte)):
            idx = np.where(Pte == p)[0]
            row = {"protein": p, "horizon_ns": h_ns, "n": int(len(idx)),
                   "positives": int(yte[idx].sum())}
            ok = (0 < yte[idx].sum() < len(idx))
            for name in ("static_only", "full_hist6") + tuple(conds):
                row[f"auprc_{name}"] = (float(average_precision_score(yte[idx], preds[name][idx]))
                                        if ok else None)
            if ok:
                g = row["auprc_full_hist6"] - row["auprc_static_only"]
                if abs(g) > 0.01:
                    for name in conds:
                        row[f"recovery_{name}"] = float(
                            (row[f"auprc_{name}"] - row["auprc_static_only"]) / g)
            per_protein.append(row)
        per_protein_all.extend(per_protein)
        summary = {}
        for name in conds:
            defined = [r for r in per_protein if f"recovery_{name}" in r]
            vals = [r[f"recovery_{name}"] for r in defined]
            summary[name] = {
                "n_proteins_defined": len(defined),
                "macro_recovery": float(np.mean(vals)) if vals else None,
                "median_recovery": float(np.median(vals)) if vals else None,
                "min_recovery": float(np.min(vals)) if vals else None,
                "max_recovery": float(np.max(vals)) if vals else None,
            }
        print(f"  per-protein summary {h_ns}ns:",
              json.dumps({k: (round(v["macro_recovery"], 3) if v["macro_recovery"] is not None else None)
                          for k, v in summary.items()}), flush=True)

        out["horizons"][str(h_ns)] = {
            "metrics": results, "base_gain": gain,
            "pooled_recovery_vs_real_history_gain": recovery,
            "per_protein": per_protein, "per_protein_summary": summary,
        }
        out["smallest_prefix_over_0.5_pooled"][str(h_ns)] = next(
            (tag for tag in PREFIX_TAGS if recovery[f"prefix_hist6_{tag}"] > 0.5), None)
        out["smallest_prefix_over_0.5_macro"][str(h_ns)] = next(
            (tag for tag in PREFIX_TAGS
             if (summary[f"prefix_hist6_{tag}"]["macro_recovery"] or -9) > 0.5), None)

    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_ns", "condition", "n", "positives", "auc", "auprc", "brier",
                    "logloss", "recovery_vs_real_history_gain"])
        for h_ns in (5, 10, 30, 50, 90):
            h = out["horizons"][str(h_ns)]
            for cname, m in h["metrics"].items():
                w.writerow([h_ns, cname, m["n"], m["positives"], m["auc"], m["auprc"],
                            m["brier"], m["logloss"],
                            "" if cname == "static_only"
                            else h["pooled_recovery_vs_real_history_gain"][cname]])
    with open(OUT / "per_protein.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_ns", "protein", "n", "positives", "auprc_static_only",
                    "auprc_full_hist6"] +
                   [f"auprc_prefix_hist6_{tag}" for tag in PREFIX_TAGS] +
                   [f"recovery_prefix_hist6_{tag}" for tag in PREFIX_TAGS])
        for row in per_protein_all:
            w.writerow([row["horizon_ns"], row["protein"], row["n"], row["positives"],
                        row["auprc_static_only"], row["auprc_full_hist6"]] +
                       [row.get(f"auprc_prefix_hist6_{tag}") for tag in PREFIX_TAGS] +
                       [row.get(f"recovery_prefix_hist6_{tag}") for tag in PREFIX_TAGS])
    print("saved", OUT / "results.json")
    print("smallest prefix over 0.5 (pooled):", out["smallest_prefix_over_0.5_pooled"])
    print("smallest prefix over 0.5 (macro):", out["smallest_prefix_over_0.5_macro"])


if __name__ == "__main__":
    main()
