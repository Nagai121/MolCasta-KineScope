"""G52: 塩橋の解離で、0.25 ns の切り取りはどうなるか
(K53 の条件、塩橋の解離、1 ns先)。

G09 の続き。G09 では 0.1 ns の切り取りの平均だけが 0.693、0.5 ns の
切り取りの平均だけが 0.951 だった。そのあいだの 0.25 ns の切り取りは
まだ測っていない。0.25 ns の切り取りの平均だけと 6 統計量の両方を
入れる。0.1 ns と 0.5 ns の条件もあわせて測り、G09 と合うかを確かめる
ことで、手順の正しさを確かめる。

切り取りの定義 (G02/G09 と同じ、結果を見る前に固定):
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 0.1 ns -> フレーム 0, 1 (2フレーム、覆う長さ 0.1 ns)
- 0.25 ns -> フレーム 0, 1, 2 (3フレーム、実際に覆う長さは 0.2 ns)
- 0.5 ns -> フレーム 0〜5 (6フレーム、覆う長さ 0.5 ns)
- 起点フレーム10 (1 ns) は変えない。ラベルは 1 ns先。

6 統計量の定義: [最初, 平均, 標準偏差, 最小, 最後−最初,
4.0 Å以下割合] を切り取りのフレームで計算する (K53 と同じ
閾値 4.0 Å)。

土台 (K53 の条件、G09 と同じ):
- K53 キャッシュ (~/workspace/kinescope/work/k53_cache/、suffix salt_diss)、
  K15 の split.json の分割。学習58タンパク質 (735件)、テスト30タンパク質
  (360件、1 ns先の陽性 101件)。静的特徴5個。
- HistGradientBoosting (120回、学習率 0.075、葉31、葉あたり最小15件、
  L2 0.02、seed 42)、条件ごとに学習し直す。1 ns先だけを測る。
- 土台の再現の確認: 1 ns先で静的のみ AUPRC 0.3734、全部入り 0.5522、
  利得 +0.1788 (K53)。0.1 ns の平均だけは AUPRC 0.4973、回復率
  0.693、0.1 ns の 6 統計量は AUPRC 0.4745、回復率 0.566、
  0.5 ns の平均だけは AUPRC 0.5434、回復率 0.951、
  0.5 ns の 6 統計量は AUPRC 0.5495、回復率 0.985 (G09)。
  これらと合わなければ測定は無効。

(a) 切った平均と 0〜1 ns の平均との残りのばらつき (テスト・学習)。
(b) 切った条件の 1 ns先の利得の回復率。参照は実履歴の利得 +0.1788。

判定 (先に固定): 0.25 ns の 6 統計量が 0.8 以上になれば、塩橋の解離
でも 0.25 ns の切り取りで短い実MDの道が立つことになる。0.5 未満なら、
0.25 ns の切り取りでは足りないことになる。そのあいだなら、部分的に
しか戻らないことになる。
"""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K53 = Path("/home/hatch/workspace/kinescope/work/k53_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
BOUND = 4.0
HORIZON_IDX = 0  # 1 ns先

ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")

CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
}


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K53 / f"{p}_R{r}_salt_diss.npz")
            parts.append({k: z[k] for k in z.files})
    return parts


def cat(parts, key):
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def main():
    split = json.loads((K15_DIR / "split.json").read_text())
    train_p = ORIG_TRAIN + tuple(split["train_additions"])
    test_p = ORIG_TEST + tuple(split["test_additions"])
    train, test = load(train_p), load(test_p)
    Xtr, Xte = cat(train, "X"), cat(test, "X")
    Htr, Hte = cat(train, "hist"), cat(test, "hist")  # (n, 11 frames)
    Ytr, Yte = cat(train, "y"), cat(test, "y")
    ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
    print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)

    full_mean_tr, full_mean_te = Htr.mean(1), Hte.mean(1)

    # (a) 残りのばらつき
    residual = {}
    for name, (label, idx, covered) in CUTS.items():
        pm_tr = Htr[:, idx].mean(1)
        pm_te = Hte[:, idx].mean(1)
        residual[name] = dict(
            cut_label=label, frames=idx, covered_length=covered,
            train_n=int(len(pm_tr)), test_n=int(len(pm_te)),
            train_residual_std=float(np.std(pm_tr - full_mean_tr)),
            test_residual_std=float(np.std(pm_te - full_mean_te)),
            train_residual_mean=float(np.mean(pm_tr - full_mean_tr)),
            test_residual_mean=float(np.mean(pm_te - full_mean_te)),
        )
        print(name, "residual std test", round(residual[name]["test_residual_std"], 4),
              "train", round(residual[name]["train_residual_std"], 4), flush=True)

    # (b) 1 ns先の利得の回復率
    conditions = {}
    conditions["static_only"] = (Xtr, Xte)
    conditions["full_hist6"] = (np.column_stack([Xtr, hist6(Htr)]),
                                np.column_stack([Xte, hist6(Hte)]))
    conditions["mean_only_full"] = (
        np.column_stack([Xtr, full_mean_tr[:, None].astype(np.float32)]),
        np.column_stack([Xte, full_mean_te[:, None].astype(np.float32)]))
    for name, (label, idx, covered) in CUTS.items():
        conditions[f"prefix_mean_{name}"] = (
            np.column_stack([Xtr, Htr[:, idx].mean(1)[:, None].astype(np.float32)]),
            np.column_stack([Xte, Hte[:, idx].mean(1)[:, None].astype(np.float32)]))
        conditions[f"prefix_hist6_{name}"] = (
            np.column_stack([Xtr, hist6(Htr[:, idx])]),
            np.column_stack([Xte, hist6(Hte[:, idx])]))

    results = {}
    for cname, (xtr, xte) in conditions.items():
        p = predict(xtr, ytr, xte)
        results[cname] = metrics(yte, p)
        print(cname, "auprc %.4f" % results[cname]["auprc"], flush=True)

    # 土台の再現の確認 (K53 / G09)
    assert abs(results["static_only"]["auprc"] - 0.3734) < 5e-4, results["static_only"]
    assert abs(results["full_hist6"]["auprc"] - 0.5522) < 5e-4, results["full_hist6"]
    assert abs(results["prefix_mean_0.1ns"]["auprc"] - 0.4973) < 5e-4, results["prefix_mean_0.1ns"]
    assert abs(results["prefix_hist6_0.1ns"]["auprc"] - 0.4745) < 5e-4, results["prefix_hist6_0.1ns"]
    assert abs(results["prefix_mean_0.5ns"]["auprc"] - 0.5434) < 5e-4, results["prefix_mean_0.5ns"]
    assert abs(results["prefix_hist6_0.5ns"]["auprc"] - 0.5495) < 5e-4, results["prefix_hist6_0.5ns"]

    gain = results["full_hist6"]["auprc"] - results["static_only"]["auprc"]
    print("base gain %.4f" % gain, flush=True)
    recovery = {}
    for cname, m in results.items():
        if cname == "static_only":
            continue
        recovery[cname] = float((m["auprc"] - results["static_only"]["auprc"]) / gain)
        print(cname, "recovery %.3f" % recovery[cname], flush=True)

    out = {
        "cuts": {k: {"label": v[0], "frames": v[1], "covered_length": v[2]} for k, v in CUTS.items()},
        "hist6_definition": "[first, mean, std, min, last-first, frac<=4.0] computed on the prefix frames",
        "residual_spread_vs_full_mean": residual,
        "references": {
            "real_history_gain_1ns": 0.1788,
            "static_only_auprc_1ns": 0.3734,
            "full_hist6_auprc_1ns": 0.5522,
            "prefix_mean_0.1ns_recovery": 0.693,
            "prefix_hist6_0.1ns_recovery": 0.566,
            "prefix_mean_0.5ns_recovery": 0.951,
            "prefix_hist6_0.5ns_recovery": 0.985,
        },
        "metrics_1ns": results,
        "base_gain_1ns": gain,
        "recovery_vs_real_history_gain": recovery,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
