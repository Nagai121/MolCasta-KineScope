"""G05: 短い実MDの切り取りは、強い土台でも指導力を戻すか
(K06(b) の条件、水素結合の形成)。

G02 (弱い土台、資料D) の強い土台版。実の 0〜1 ns の履歴を、始め
の短い部分だけに切って使う。切る長さは 0.1 ns と 0.5 ns の2通り。

切り取りの定義 (G02 と同じ、結果を見る前に固定):
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 切る長さ L に対して、時刻が L 以下のフレームを使う。
  - 0.1 ns -> フレーム 0, 1 (2フレーム、覆う長さ 0.1 ns)
  - 0.5 ns -> フレーム 0〜5 (6フレーム、覆う長さ 0.5 ns)
- 起点フレーム10 (1 ns) は変えない。ラベルは 1 ns先。

(a) 切った履歴の平均と 0〜1 ns の平均との残りのばらつき
    = (切った平均 - 0〜1 ns の平均) の候補ごとの差の標準偏差。
    G02 の (a) と同じ。テスト (100タンパク質) と較正 (8タンパク質)
    で別々に測る。
(b) 切った平均だけを履歴の特徴として強い土台の補正モデルに入れた
    ときの 1 ns先の利得の回復率。条件は「静的特徴＋切った平均だけ」
    (K21 の「平均だけ」と同じ形。他の統計量は入れない)。
    回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得)。
    実履歴の利得は、全体 (pooled) +0.0702、タンパク質ごとの平均
    (macro) +0.0712。回復率は両方で測る。
    参照として、実履歴の平均だけ (K21 の「平均だけ」) も同じ形で測る。

土台 (K06(b) と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
  seed 20261121)。較正タンパク質 8つで学習し、未見の100タンパク質
  ×3レプリカ (163,074行) で評価する。
- 土台の再現の確認: 1 ns先で静的のみ AUPRC (macro) 0.4539、
  実履歴 0.5251、全体 (pooled) では静的のみ 0.4591、実履歴 0.5293
  (k06b_results.json)。これと合わなければ測定は無効。

読み方 (先に固定): 0.1 ns の切り取りで回復率が5割以上なら、強い
土台でも短い実MDの経路が使える。5割に届かなければ、強い土台では
短い切り取りでは足りない。
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
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
}


def load(proteins):
    """Returns dicts of concatenated arrays per replica-split."""
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

    # (a) 残りのばらつき
    residual = {}
    for name, (label, idx, covered) in CUTS.items():
        pm_cal = cal["series"][idx].mean(axis=0)
        pm_test = test["series"][idx].mean(axis=0)
        residual[name] = dict(
            cut_label=label, frames=idx, covered_length=covered,
            cal_n=int(len(pm_cal)), test_n=int(len(pm_test)),
            cal_residual_std=float(np.std(pm_cal - full_mean_cal)),
            test_residual_std=float(np.std(pm_test - full_mean_test)),
            cal_residual_mean=float(np.mean(pm_cal - full_mean_cal)),
            test_residual_mean=float(np.mean(pm_test - full_mean_test)),
        )
        print(name, "residual std test", round(residual[name]["test_residual_std"], 4),
              "cal", round(residual[name]["cal_residual_std"], 4), flush=True)

    # per-protein spans for macro AUPRC (eval_b.py と同じ)
    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n

    # (b) 1 ns先の利得の回復率
    conditions = {}
    conditions["static_only"] = (cal["static"], test["static"])
    conditions["full_history"] = (
        np.column_stack([cal["static"], cal["hist"]]),
        np.column_stack([test["static"], test["hist"]]))
    conditions["mean_only_full"] = (
        np.column_stack([cal["static"], full_mean_cal[:, None].astype(np.float32)]),
        np.column_stack([test["static"], full_mean_test[:, None].astype(np.float32)]))
    for name, (label, idx, covered) in CUTS.items():
        conditions[f"prefix_mean_{name}"] = (
            np.column_stack([cal["static"], cal["series"][idx].mean(axis=0)[:, None].astype(np.float32)]),
            np.column_stack([test["static"], test["series"][idx].mean(axis=0)[:, None].astype(np.float32)]))

    results = {}
    for cname, (xtr, xte) in conditions.items():
        pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
        m = metrics(y_test, pred)
        macro = [average_precision_score(y_test[s:e], pred[s:e])
                 for _, s, e in spans if 0 < y_test[s:e].sum() < e - s]
        m["macro_auprc"] = float(np.mean(macro))
        results[cname] = m
        print(cname, "pooled auprc %.4f macro auprc %.4f" % (m["auprc"], m["macro_auprc"]), flush=True)

    # frozen alone (no sidecar), for reference
    fz = metrics(y_test, test["frozen"])
    fz["macro_auprc"] = float(np.mean([
        average_precision_score(y_test[s:e], test["frozen"][s:e])
        for _, s, e in spans if 0 < y_test[s:e].sum() < e - s]))
    results["frozen_only"] = fz
    print("frozen_only", "pooled auprc %.4f macro auprc %.4f" % (fz["auprc"], fz["macro_auprc"]), flush=True)

    # 土台の再現の確認 (k06b_results.json の 1 ns先)
    assert abs(results["static_only"]["macro_auprc"] - 0.4539) < 5e-4, results["static_only"]
    assert abs(results["full_history"]["macro_auprc"] - 0.5251) < 5e-4, results["full_history"]
    assert abs(results["static_only"]["auprc"] - 0.4591) < 5e-4, results["static_only"]
    assert abs(results["full_history"]["auprc"] - 0.5293) < 5e-4, results["full_history"]

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]
    print("base gain pooled %.4f macro %.4f" % (gain_pooled, gain_macro), flush=True)
    recovery = {}
    for cname, m in results.items():
        if cname in ("static_only", "frozen_only"):
            continue
        recovery[cname] = dict(
            pooled=float((m["auprc"] - results["static_only"]["auprc"]) / gain_pooled),
            macro=float((m["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro))
        print(cname, "recovery pooled %.3f macro %.3f" % (
            recovery[cname]["pooled"], recovery[cname]["macro"]), flush=True)

    out = {
        "cuts": {k: {"label": v[0], "frames": v[1], "covered_length": v[2]} for k, v in CUTS.items()},
        "residual_spread_vs_full_mean": residual,
        "references": {
            "real_history_gain_1ns_pooled": 0.0702,
            "real_history_gain_1ns_macro": 0.0712,
            "static_only_auprc_pooled": 0.4591,
            "static_only_auprc_macro": 0.4539,
            "full_history_auprc_pooled": 0.5293,
            "full_history_auprc_macro": 0.5251,
            "replica_lower_bound_A": 0.249,
        },
        "metrics_1ns": results,
        "base_gain_1ns": dict(pooled=gain_pooled, macro=gain_macro),
        "recovery_vs_real_history_gain": recovery,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
