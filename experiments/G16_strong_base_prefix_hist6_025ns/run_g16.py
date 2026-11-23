"""G16: 強い土台で、0.25 ns の切り取りの 6 統計量はどうなるか
(K06(b) の条件、水素結合の形成、1 ns先)。

G12 の続き。G12 では 0.1 ns の 6 統計量が平均だけより悪く
(回復率全体 0.446 に対して平均だけ 0.545)、0.5 ns では良くなった
(0.768 に対して 0.695)。そのあいだの 0.25 ns ではどうか。
G05 で作り直した系列 (~/workspace/kinescope/work/g05_series/)
を再利用する。比較のため、0.25 ns の平均だけも同じ土台で測る。

6 統計量の定義 (G02/G12 と同じ、資料Dの D6 と同じ):
[最初, 平均, 標準偏差, 最小, 最後−最初, 3.5 Å以下割合] を
切り取りのフレームで計算する。0.25 ns の切り取りはフレーム
0, 1, 2 の3フレームで、100 ps 刻みに合わないため実際に覆う長さは
0.2 ns になる (G02 と同じ)。

土台 (K06(b) と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20件、L2 0.15、
  seed 20261121)。較正タンパク質 8つで学習し、未見の100タンパク質
  ×3レプリカ (163,074行) で評価する。
- 土台の再現の確認: 1 ns先で静的のみ AUPRC (macro) 0.4539、
  実履歴 0.5251、全体 (pooled) では静的のみ 0.4591、実履歴 0.5293
  (k06b_results.json)。これと合わなければ測定は無効。
- 0.1 ns / 0.5 ns の条件の再現の確認 (G05/G12):
  0.1 ns の平均だけは全体 0.4974 / タンパク質ごとの平均 0.4933、
  0.1 ns の 6 統計量は全体 0.4904 / タンパク質ごとの平均 0.4872、
  0.5 ns の平均だけは全体 0.5079 / タンパク質ごとの平均 0.5027、
  0.5 ns の 6 統計量は全体 0.5130 / タンパク質ごとの平均 0.5074。

判定 (先に固定): 0.25 ns の 6 統計量で回復率が平均だけ
(G05 の 0.1 ns は全体 0.545、タンパク質ごとの平均 0.553)を
上回れば、0.1 ns での悪化は切り取りの短さの問題。平均だけを
下回れば、強い土台では短い切り取りの 6 統計量は一貫して足しに
ならないことになる。同じ長さの 0.25 ns の平均だけとの比べも
あわせて見る。
"""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/g05_series")
OUT = Path(__file__).resolve().parent
SEED = 20261121
HORIZON_IDX = 0  # 1 ns先

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
}


def load(proteins):
    static, hist, frozen, label, series = [], [], [], [], []
    for p in proteins:
        z = np.load(K06B / f"{p}.npz")
        s = np.load(SERIES / f"{p}.npz")
        for r in (1, 2, 3):
            static.append(z[f"r{r}_static"])
            hist.append(z[f"r{r}_history"])
            frozen.append(z[f"r{r}_frozen"])
            label.append(z[f"r{r}_label"])
            series.append(s[f"r{r}_series"])
    return dict(static=np.concatenate(static), hist=np.concatenate(hist),
                frozen=np.concatenate(frozen), label=np.concatenate(label),
                series=np.concatenate(series, axis=1))


def hist6(series, bound=3.5):
    """6 statistics of a (frames, N) series: [first, mean, std, min,
    last-first, frac<=bound]. Same definition as G02/G12 / Doc D D6."""
    return np.stack(
        [series[0], series.mean(axis=0), series.std(axis=0), series.min(axis=0),
         series[-1] - series[0], (series <= bound).mean(axis=0)],
        axis=1).astype(np.float32)


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


def main():
    cal, test = load(CAL), load(TEST)
    y_cal, y_test = cal["label"][:, HORIZON_IDX], test["label"][:, HORIZON_IDX]
    print("cal", len(y_cal), "test", len(y_test), "test pos 1ns", int(y_test.sum()), flush=True)

    full_mean_cal = cal["series"].mean(axis=0)
    full_mean_test = test["series"].mean(axis=0)

    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n

    def with_feats(cal_f, test_f):
        return (np.column_stack([cal["static"], cal_f]),
                np.column_stack([test["static"], test_f]))

    conditions = {}
    conditions["static_only"] = (cal["static"], test["static"])
    conditions["full_history"] = (
        np.column_stack([cal["static"], cal["hist"]]),
        np.column_stack([test["static"], test["hist"]]))
    conditions["mean_only_full"] = with_feats(
        full_mean_cal[:, None].astype(np.float32),
        full_mean_test[:, None].astype(np.float32))
    for name, (label, idx, covered) in CUTS.items():
        conditions[f"prefix_mean_{name}"] = with_feats(
            cal["series"][idx].mean(axis=0)[:, None].astype(np.float32),
            test["series"][idx].mean(axis=0)[:, None].astype(np.float32))
        conditions[f"prefix_hist6_{name}"] = with_feats(
            hist6(cal["series"][idx]), hist6(test["series"][idx]))

    results = {}
    for cname, (xtr, xte) in conditions.items():
        pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
        m = metrics(y_test, pred)
        macro = [average_precision_score(y_test[s:e], pred[s:e])
                 for _, s, e in spans if 0 < y_test[s:e].sum() < e - s]
        m["macro_auprc"] = float(np.mean(macro))
        results[cname] = m
        print(cname, "pooled auprc %.4f macro auprc %.4f" % (m["auprc"], m["macro_auprc"]), flush=True)

    # 土台の再現の確認 (k06b_results.json の 1 ns先)
    assert abs(results["static_only"]["macro_auprc"] - 0.4539) < 5e-4, results["static_only"]
    assert abs(results["full_history"]["macro_auprc"] - 0.5251) < 5e-4, results["full_history"]
    assert abs(results["static_only"]["auprc"] - 0.4591) < 5e-4, results["static_only"]
    assert abs(results["full_history"]["auprc"] - 0.5293) < 5e-4, results["full_history"]
    # 0.1 ns / 0.5 ns の条件の再現の確認 (G05/G12)
    assert abs(results["prefix_mean_0.1ns"]["auprc"] - 0.4974) < 5e-4, results["prefix_mean_0.1ns"]
    assert abs(results["prefix_hist6_0.1ns"]["auprc"] - 0.4904) < 5e-4, results["prefix_hist6_0.1ns"]
    assert abs(results["prefix_mean_0.5ns"]["auprc"] - 0.5079) < 5e-4, results["prefix_mean_0.5ns"]
    assert abs(results["prefix_hist6_0.5ns"]["auprc"] - 0.5130) < 5e-4, results["prefix_hist6_0.5ns"]

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]
    print("base gain pooled %.4f macro %.4f" % (gain_pooled, gain_macro), flush=True)
    recovery = {}
    for cname, m in results.items():
        if cname == "static_only":
            continue
        recovery[cname] = dict(
            pooled=float((m["auprc"] - results["static_only"]["auprc"]) / gain_pooled),
            macro=float((m["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro))
        print(cname, "recovery pooled %.3f macro %.3f" % (
            recovery[cname]["pooled"], recovery[cname]["macro"]), flush=True)

    out = {
        "cuts": {k: {"label": v[0], "frames": v[1], "covered_length": v[2]} for k, v in CUTS.items()},
        "hist6_definition": "[first, mean, std, min, last-first, frac<=3.5] computed on the prefix frames (same as G02/G12 prefix_hist6 / Doc D D6)",
        "references": {
            "real_history_gain_1ns_pooled": 0.0702,
            "real_history_gain_1ns_macro": 0.0712,
            "static_only_auprc_pooled": 0.4591,
            "static_only_auprc_macro": 0.4539,
            "full_history_auprc_pooled": 0.5293,
            "full_history_auprc_macro": 0.5251,
            "prefix_mean_0.1ns_recovery_pooled": 0.545,
            "prefix_mean_0.1ns_recovery_macro": 0.553,
        },
        "metrics_1ns": results,
        "base_gain_1ns": dict(pooled=gain_pooled, macro=gain_macro),
        "recovery_vs_real_history_gain": recovery,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
