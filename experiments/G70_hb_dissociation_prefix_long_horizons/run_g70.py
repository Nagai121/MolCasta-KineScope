"""G70: 解離で、長い予測幅でも短い切り取りで指導力が戻るか
（資料Dの条件、水素結合の解離、5・10・30・50・90 ns先）。

G06 は 1 ns 先だけだったので、長い予測幅での形が空白だった。
ここでは予測幅 5・10・30・50・90 ns、切り取り 0.1・0.25・0.5 ns
で、平均だけの回復率を測る。土台の利得は K46 の各予測幅の記録値。

土台（資料Dの条件、水素結合の解離、G61 と同じ）:
- K46 のキャッシュ（~/workspace/kinescope/work/data_cache_k46/ の
  {protein}_R{rep}.npz、静的特徴7個・11フレームの履歴・ラベル）。
  学習 1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算、
  候補は学習 4,051件・テスト 2,080件、起点フレーム10（1 ns）、
  履歴 0〜1 ns の11フレーム（100 ps 間隔）。
- HistGradientBoosting（120回、学習率0.075、葉31、葉あたり最小15件、
  L2 0.02、種42）、条件ごとに学習し直す。
- 切り取り: 0.1 ns = フレーム [0,1]、0.25 ns = フレーム [0,1,2]、
  0.5 ns = フレーム [0,1,2,3,4,5]。平均だけを入れる。
- 土台の静的のみ・全部入りの AUPRC は、K46 の記録値と小数第4位まで
  一致することを各予測幅で確かめる。

判定（先に固定）: G60 と同じく、各予測幅で回復率 0.5 を超える最小の
切り取りを出す。
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
OUT = Path("/home/hatch/workspace/g70")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 3.5
HORIZONS = [(1, 5), (2, 10), (3, 30), (4, 50), (5, 90)]
# K46 の記録値（静的のみ・全部入り）。照合用。
REF = {
    5: (0.4346, 0.6103), 10: (0.4496, 0.5974), 30: (0.4409, 0.5731),
    50: (0.4088, 0.5231), 90: (0.5118, 0.5797),
}
PREFIX_FRAMES = {"0.1ns": [0, 1], "0.25ns": [0, 1, 2], "0.5ns": [0, 1, 2, 3, 4, 5]}
PREFIX_TAGS = ["0.1ns", "0.25ns", "0.5ns"]


def load(prots):
    xs, hs, ys = [], [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            xs.append(z["X"])
            hs.append(z["hist"].T)  # キャッシュは (11, n) なので (n, 11) に直す
            ys.append(z["y"])
    return np.concatenate(xs), np.concatenate(hs), np.concatenate(ys)


def hist6(series):
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
    Xtr, Htr, Ytr = load(TRAIN_P)
    Xte, Hte, Yte = load(TEST_P)
    print("train", Xtr.shape, "test", Xte.shape, flush=True)

    feats = {"static_only": (Xtr, Xte),
             "base_hist_D6": (np.column_stack([Xtr, hist6(Htr)]),
                              np.column_stack([Xte, hist6(Hte)]))}
    for tag in PREFIX_TAGS:
        fr = PREFIX_FRAMES[tag]
        feats[f"prefix_mean_{tag}"] = (
            np.column_stack([Xtr, Htr[:, fr].mean(axis=1)[:, None].astype(np.float32)]),
            np.column_stack([Xte, Hte[:, fr].mean(axis=1)[:, None].astype(np.float32)]))

    out = {"reference_k46": {str(h): {"static": s, "full": f}
                             for h, (s, f) in REF.items()},
           "horizons": {}, "smallest_prefix_over_0.5": {}}
    csv_rows = []
    for h_idx, h_ns in HORIZONS:
        ytr, yte = Ytr[:, h_idx], Yte[:, h_idx]
        print(f"=== horizon {h_ns} ns, test positives {int(yte.sum())} ===", flush=True)
        results = {}
        for name, (xtr, xte) in feats.items():
            p = fit_predict(xtr, ytr, xte)
            results[name] = metrics(yte, p)
            print(f"  {name} auprc {results[name]['auprc']:.4f}", flush=True)
        ref_s, ref_f = REF[h_ns]
        assert abs(results["static_only"]["auprc"] - ref_s) < 5e-4, (h_ns, results["static_only"])
        assert abs(results["base_hist_D6"]["auprc"] - ref_f) < 5e-4, (h_ns, results["base_hist_D6"])
        gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
        recovery = {n: float((m["auprc"] - results["static_only"]["auprc"]) / gain)
                    for n, m in results.items() if n != "static_only"}
        print(f"  base gain {gain:.4f}, recovery:",
              {n: round(v, 3) for n, v in recovery.items()}, flush=True)
        out["horizons"][str(h_ns)] = {
            "metrics": results, "base_gain": gain,
            "recovery_vs_real_history_gain": recovery,
        }
        out["smallest_prefix_over_0.5"][str(h_ns)] = next(
            (tag for tag in PREFIX_TAGS if recovery[f"prefix_mean_{tag}"] > 0.5), None)
        for cname, m in results.items():
            csv_rows.append([h_ns, cname, m["n"], m["positives"], m["auc"], m["auprc"],
                             m["brier"], m["logloss"],
                             "" if cname == "static_only" else recovery[cname]])
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_ns", "condition", "n", "positives", "auc", "auprc", "brier",
                    "logloss", "recovery_vs_real_history_gain"])
        w.writerows(csv_rows)
    print("saved", OUT / "results.json")
    print("smallest prefix over 0.5:", out["smallest_prefix_over_0.5"])


if __name__ == "__main__":
    main()
