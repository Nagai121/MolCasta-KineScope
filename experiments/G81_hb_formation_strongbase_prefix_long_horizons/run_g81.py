"""G81: 強い土台で、水素結合の形成の長い予測幅でも短い切り取りで
指導力が戻るか (K06(b) の条件)。

G05 (強い土台、1 ns先だけ) の長い予測幅版。実の 0〜1 ns の履歴を、
始めの短い部分だけに切って使う。切る長さは 0.1 / 0.25 / 0.5 ns の
3通り。予測幅は 1 / 5 / 10 / 30 / 50 / 90 ns。

切り取りの定義 (G02/G05 と同じ、結果を見る前に固定):
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 切る長さ L に対して、時刻が L 以下のフレームを使う。
  - 0.1 ns -> フレーム 0, 1 (2フレーム、覆う長さ 0.1 ns)
  - 0.25 ns -> フレーム 0, 1, 2 (3フレーム、覆う長さは 0.2 ns。
    0.25 ns は 100 ps 刻みに合わない)
  - 0.5 ns -> フレーム 0〜5 (6フレーム、覆う長さ 0.5 ns)
- 起点フレーム10 (1 ns) は変えない。ラベルは各予測幅の先のもの。

測るもの:
- (a) 切った履歴の平均と 0〜1 ns の平均との残りのばらつき
  (G02/G05 と同じ。テスト 100タンパク質と較正 8タンパク質で別々に測る)
- (b) 切った平均だけを履歴の特徴として強い土台の補正モデルに入れた
  ときの各予測幅の利得の回復率。条件は「静的特徴＋切った平均だけ」
  (K21 の「平均だけ」と同じ形。他の統計量は入れない)。
  回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得)。
  利得は全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る。

土台 (K06(b) と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
  seed 20261121)。較正タンパク質 8つで学習し、未見の100タンパク質
  ×3レプリカ (163,074行) で評価する。
- 系列は G05 で作り直したもの (~/workspace/kinescope/work/g05_series/)
  をそのまま使う。作り直した系列から導いた要約値は保存値と 1e-05 Å
  未満で一致することが G05 で 108タンパク質すべてで確かめてある。
- 土台の再現の確認: 各予測幅で静的のみ・実履歴の AUPRC (pooled /
  macro) が k06b_results.json の記録と 5e-4 以内で一致すること。
  利得の記録値 (k06b_results.json / K12 の記録):
  1 ns: pooled +0.0702 / macro +0.0712
  5 ns: pooled +0.0514 / macro +0.0518
  10 ns: pooled +0.0376 / macro +0.0388
  30 ns: pooled +0.0363 / macro +0.0370
  50 ns: pooled +0.0354 / macro +0.0344
  90 ns: pooled +0.0265 / macro +0.0273

読み方 (先に固定): 各予測幅で、回復率 0.5 を超える最小の切り取りを
出す。3つの切り取りのどれでも 0.5 を超えなければ、その予測幅では
「0.5 ns まででも届かない」と記録する。
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
HORIZONS = [1, 5, 10, 30, 50, 90]

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# k06b_results.json の記録値 (再現の確認用)
REF = {
    1: dict(static_pooled=0.4591, static_macro=0.4539, full_pooled=0.5293, full_macro=0.5251),
    5: dict(static_pooled=0.4575, static_macro=0.4505, full_pooled=0.5089, full_macro=0.5023),
    10: dict(static_pooled=0.4505, static_macro=0.4438, full_pooled=0.4881, full_macro=0.4826),
    30: dict(static_pooled=0.4492, static_macro=0.4418, full_pooled=0.4855, full_macro=0.4788),
    50: dict(static_pooled=0.4470, static_macro=0.4407, full_pooled=0.4825, full_macro=0.4752),
    90: dict(static_pooled=0.4418, static_macro=0.4357, full_pooled=0.4683, full_macro=0.4630),
}
# G05 の 1 ns先の記録値 (照合用: pooled / macro)
G05_REF = {
    "prefix_mean_0.1ns": (0.545, 0.553),
    "prefix_mean_0.5ns": (0.695, 0.686),
}

CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
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
    print("cal", len(cal["label"]), "test", len(test["label"]), flush=True)

    full_mean_cal = cal["series"].mean(axis=0)
    full_mean_test = test["series"].mean(axis=0)

    # (a) 残りのばらつき (G02/G05 と同じ)
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

    partial_path = OUT / "results_g81_partial.json"
    done = {}
    if partial_path.exists():
        done = json.loads(partial_path.read_text())
        print("resuming from partial:", sorted(done.keys()), flush=True)

    results, recovery, base_gain = {}, {}, {}
    for hi, h in enumerate(HORIZONS):
        hkey = str(h)
        y_cal, y_test = cal["label"][:, hi], test["label"][:, hi]
        print(f"--- horizon {h} ns: cal pos {int(y_cal.sum())} test pos {int(y_test.sum())} ---", flush=True)
        if hkey in done:
            results[hkey] = done[hkey]["metrics"]
            recovery[hkey] = done[hkey]["recovery"]
            base_gain[hkey] = done[hkey]["base_gain"]
            print("  (loaded from partial)", flush=True)
            continue

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

        m_h = {}
        for cname, (xtr, xte) in conditions.items():
            pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
            m = metrics(y_test, pred)
            macro = [average_precision_score(y_test[s:e], pred[s:e])
                     for _, s, e in spans if 0 < y_test[s:e].sum() < e - s]
            m["macro_auprc"] = float(np.mean(macro))
            m_h[cname] = m
            print(f"  {cname}: pooled auprc {m['auprc']:.4f} macro auprc {m['macro_auprc']:.4f}", flush=True)

        # 土台の再現の確認 (k06b_results.json の記録値)
        ref = REF[h]
        assert abs(m_h["static_only"]["auprc"] - ref["static_pooled"]) < 5e-4, (h, m_h["static_only"])
        assert abs(m_h["static_only"]["macro_auprc"] - ref["static_macro"]) < 5e-4, (h, m_h["static_only"])
        assert abs(m_h["full_history"]["auprc"] - ref["full_pooled"]) < 5e-4, (h, m_h["full_history"])
        assert abs(m_h["full_history"]["macro_auprc"] - ref["full_macro"]) < 5e-4, (h, m_h["full_history"])

        gp = m_h["full_history"]["auprc"] - m_h["static_only"]["auprc"]
        gm = m_h["full_history"]["macro_auprc"] - m_h["static_only"]["macro_auprc"]
        base_gain[hkey] = dict(pooled=float(gp), macro=float(gm))
        print(f"  base gain pooled {gp:.4f} macro {gm:.4f}", flush=True)

        rec_h = {}
        for cname, m in m_h.items():
            if cname == "static_only":
                continue
            rec_h[cname] = dict(
                pooled=float((m["auprc"] - m_h["static_only"]["auprc"]) / gp),
                macro=float((m["macro_auprc"] - m_h["static_only"]["macro_auprc"]) / gm))
            print(f"  {cname}: recovery pooled {rec_h[cname]['pooled']:.3f} macro {rec_h[cname]['macro']:.3f}", flush=True)

        # 1 ns先は G05 の記録値とも照合
        if h == 1:
            for cname, (rp, rm) in G05_REF.items():
                assert abs(rec_h[cname]["pooled"] - rp) < 5e-3, (cname, rec_h[cname])
                assert abs(rec_h[cname]["macro"] - rm) < 5e-3, (cname, rec_h[cname])
            print("  G05 1ns check passed", flush=True)

        results[hkey] = m_h
        recovery[hkey] = rec_h
        done[hkey] = dict(metrics=m_h, recovery=rec_h, base_gain=base_gain[hkey])
        partial_path.write_text(json.dumps(done, indent=2, ensure_ascii=False))

    # 最小の切り取り (先に固定した読み方)
    min_cut = {}
    for h in HORIZONS:
        hkey = str(h)
        rec = recovery[hkey]
        mc = {"pooled": None, "macro": None}
        for name in ("0.1ns", "0.25ns", "0.5ns"):
            cname = f"prefix_mean_{name}"
            if mc["pooled"] is None and rec[cname]["pooled"] > 0.5:
                mc["pooled"] = CUTS[name][0]
            if mc["macro"] is None and rec[cname]["macro"] > 0.5:
                mc["macro"] = CUTS[name][0]
        min_cut[hkey] = mc
        print(f"horizon {h} ns: min cut pooled {mc['pooled']} macro {mc['macro']}", flush=True)

    out = {
        "cuts": {k: {"label": v[0], "frames": v[1], "covered_length": v[2]} for k, v in CUTS.items()},
        "horizons": HORIZONS,
        "residual_spread_vs_full_mean": residual,
        "references": {
            "base_gain_ref": {str(h): dict(
                pooled=REF[h]["full_pooled"] - REF[h]["static_pooled"],
                macro=REF[h]["full_macro"] - REF[h]["static_macro"]) for h in HORIZONS},
            "base_gain_recomputed": base_gain,
            "g05_1ns_prefix_mean": {k: {"pooled": v[0], "macro": v[1]} for k, v in G05_REF.items()},
            "replica_lower_bound_A": 0.249,
        },
        "metrics": results,
        "recovery_vs_real_history_gain": recovery,
        "min_cut_over_0.5": min_cut,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
