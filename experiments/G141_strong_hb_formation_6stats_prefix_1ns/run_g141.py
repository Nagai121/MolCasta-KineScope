"""G141: 強い土台で、水素結合の形成の 6 個の統計量の切り取り
1.0 ns でも回復率を測る (K06(b) の条件)。

G86 (強い土台、6統計量) の 1.0 ns 版。実の 0〜1 ns の履歴を、
始めの短い部分だけに切って、6 個の統計量 (資料D式: 最初の値、平均、
標準偏差、最小値、最後−最初、3.5 A 以下割合) に要約して使う。
切る長さは 0.1 / 0.25 / 0.5 / 1.0 ns の 4通り。予測幅は 1 / 5 / 10 /
30 / 50 / 90 ns。G86 は切り取り 0.1 / 0.25 / 0.5 ns の 3通りだけで、
1.0 ns はまだ測っていない。ここでは 0.1 / 0.25 / 0.5 / 1.0 ns の
はしごを一度に測り直し、0.1 / 0.25 / 0.5 ns の値は G86 の記録と
照合する（小数第4位まで一致することを確かめる）。

切り取りの定義 (G02/G05/G81/G86 と同じ、結果を見る前に固定):
- 軌道の保存間隔は 100 ps (0.1 ns)。フレーム k は時刻 k*0.1 ns。
- 切る長さ L に対して、時刻が L 以下のフレームを使う。
  - 0.1 ns -> フレーム 0, 1 (2フレーム、覆う長さ 0.1 ns)
  - 0.25 ns -> フレーム 0, 1, 2 (3フレーム、覆う長さは 0.2 ns。
    0.25 ns は 100 ps 刻みに合わない)
  - 0.5 ns -> フレーム 0〜5 (6フレーム、覆う長さ 0.5 ns)
  - 1.0 ns -> フレーム 0〜9 (10フレーム、覆う長さは 0.9 ns。
    フレーム 9 は 0.9 ns で、1.0 ns のフレームは起点フレーム10
    そのものなので、切り取りには入れない)
- 起点フレーム10 (1 ns) は変えない。ラベルは各予測幅の先のもの。
- 6統計量は切ったフレームだけで計算する。強い土台の実履歴10特徴の
  並び (score_b.py、K12 の記録で確認) は [最初の値、平均、標準偏差、
  最小値、最大値、最後−最初、最後−1つ前、3.5 A 以下割合、4.5 A 以下
  割合、増えたステップの割合] なので、6統計量は history の列
  [0, 1, 2, 3, 5, 7] にあたる。11フレーム全部で計算した 6 統計量が
  保存値と 1.9e-06 以内で一致することは、実行前に確かめてある。

測るもの:
- 切った 6 統計量を履歴の特徴として強い土台の補正モデルに入れた
  ときの各予測幅の利得の回復率。条件は「静的特徴＋切った 6 統計量」。
  回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得)。
  利得は全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る。
  参考として、0〜1 ns の 11 フレーム全部の 6 統計量 (full_6stats)
  も同じ形で測る。
- タンパク質ごとの回復率: テストの 100 タンパク質ごとに AUPRC を
  出し、タンパク質ごとの回復率 (そのタンパク質の利得を分母にする)
  の平均と中央値も記録する。

土台 (K06(b) と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
  seed 20261121)。較正タンパク質 8つで学習し、未見の100タンパク質
  ×3レプリカ (163,074行) で評価する。
- 系列は G05 で作り直したもの (~/workspace/kinescope/work/g05_series/)
  をそのまま使う。作り直した系列から導いた要約値は保存値と 1e-05 A
  未満で一致することが G05 で 108タンパク質すべてで確かめてある。
- 土台の再現の確認: 各予測幅で静的のみ・実履歴の AUPRC (pooled /
  macro) が k06b_results.json の記録と 5e-4 以内で一致すること。
  利得の記録値 (k06b_results.json / K12 / G81 の記録):
  1 ns: pooled +0.0702 / macro +0.0712
  5 ns: pooled +0.0514 / macro +0.0518
  10 ns: pooled +0.0376 / macro +0.0388
  30 ns: pooled +0.0363 / macro +0.0370
  50 ns: pooled +0.0354 / macro +0.0344
  90 ns: pooled +0.0265 / macro +0.0273

読み方 (先に固定): 各予測幅で、回復率 0.5 を超える最小の切り取りを
出す。4つの切り取りのどれでも 0.5 を超えなければ、その予測幅では
「1.0 ns まででも届かない」と記録する。強い土台では 6 個の統計量が
平均だけより悪くなることがあるので、平均だけの切り取り (G81) の値
との比較も記録する。
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

# G86 の記録値（切り取り 6 統計量の AUPRC、pooled / macro）。照合用。
G86_REF = {
    1: {"0.1ns": (0.4904, 0.4872), "0.25ns": (0.5046, 0.5012), "0.5ns": (0.5130, 0.5074)},
    5: {"0.1ns": (0.4832, 0.4783), "0.25ns": (0.4927, 0.4867), "0.5ns": (0.5007, 0.4948)},
    10: {"0.1ns": (0.4711, 0.4652), "0.25ns": (0.4737, 0.4690), "0.5ns": (0.4776, 0.4725)},
    30: {"0.1ns": (0.4721, 0.4661), "0.25ns": (0.4735, 0.4686), "0.5ns": (0.4770, 0.4715)},
    50: {"0.1ns": (0.4645, 0.4589), "0.25ns": (0.4649, 0.4595), "0.5ns": (0.4714, 0.4653)},
    90: {"0.1ns": (0.4573, 0.4495), "0.25ns": (0.4620, 0.4556), "0.5ns": (0.4665, 0.4608)},
}

CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
    "1.0ns": ("1.0 ns", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "0.9 ns"),
}
ALL_FRAMES = list(range(11))


def six_stats(series, idx):
    """series: (n_frames, n_candidates). D式6統計量を返す (n, 6)。"""
    sub = series[idx]
    return np.column_stack([
        sub[0],
        sub.mean(axis=0),
        sub.std(axis=0),
        sub.min(axis=0),
        sub[-1] - sub[0],
        (sub <= 3.5).mean(axis=0),
    ]).astype(np.float32)


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

    # 実行前の確認: 11フレーム全部の 6 統計量が保存値と一致すること
    full6_cal = six_stats(cal["series"], ALL_FRAMES)
    full6_test = six_stats(test["series"], ALL_FRAMES)
    d_cal = float(np.abs(full6_cal - cal["hist"][:, [0, 1, 2, 3, 5, 7]]).max())
    d_test = float(np.abs(full6_test - test["hist"][:, [0, 1, 2, 3, 5, 7]]).max())
    print(f"full 6stats vs stored history cols: cal max diff {d_cal:.2e}, test {d_test:.2e}", flush=True)
    assert d_cal < 1e-4 and d_test < 1e-4

    # per-protein spans for macro AUPRC (eval_b.py と同じ)
    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n

    partial_path = OUT / "results_g141_partial.json"
    done = {}
    if partial_path.exists():
        done = json.loads(partial_path.read_text())
        print("resuming from partial:", sorted(done.keys()), flush=True)

    results, recovery, base_gain = {}, {}, {}
    per_protein = {}
    for hi, h in enumerate(HORIZONS):
        hkey = str(h)
        y_cal, y_test = cal["label"][:, hi], test["label"][:, hi]
        print(f"--- horizon {h} ns: cal pos {int(y_cal.sum())} test pos {int(y_test.sum())} ---", flush=True)
        if hkey in done:
            results[hkey] = done[hkey]["metrics"]
            recovery[hkey] = done[hkey]["recovery"]
            base_gain[hkey] = done[hkey]["base_gain"]
            per_protein[hkey] = done[hkey]["per_protein_auprc"]
            print("  (loaded from partial)", flush=True)
            continue

        conditions = {}
        conditions["static_only"] = (cal["static"], test["static"])
        conditions["full_history"] = (
            np.column_stack([cal["static"], cal["hist"]]),
            np.column_stack([test["static"], test["hist"]]))
        conditions["full_6stats"] = (
            np.column_stack([cal["static"], full6_cal]),
            np.column_stack([test["static"], full6_test]))
        for name, (label, idx, covered) in CUTS.items():
            conditions[f"prefix_6stats_{name}"] = (
                np.column_stack([cal["static"], six_stats(cal["series"], idx)]),
                np.column_stack([test["static"], six_stats(test["series"], idx)]))

        m_h, pp_h = {}, {}
        for cname, (xtr, xte) in conditions.items():
            pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
            m = metrics(y_test, pred)
            per = []
            for pname, s, e in spans:
                ys, ps = y_test[s:e], pred[s:e]
                if 0 < ys.sum() < e - s:
                    per.append(float(average_precision_score(ys, ps)))
                else:
                    per.append(None)
            valid = [v for v in per if v is not None]
            m["macro_auprc"] = float(np.mean(valid))
            m["n_proteins_valid"] = len(valid)
            m_h[cname] = m
            pp_h[cname] = {pname: v for (pname, _, _), v in zip(spans, per)}
            print(f"  {cname}: pooled auprc {m['auprc']:.4f} macro auprc {m['macro_auprc']:.4f}", flush=True)

        # 土台の再現の確認 (k06b_results.json の記録値)
        ref = REF[h]
        assert abs(m_h["static_only"]["auprc"] - ref["static_pooled"]) < 5e-4, (h, m_h["static_only"])
        assert abs(m_h["static_only"]["macro_auprc"] - ref["static_macro"]) < 5e-4, (h, m_h["static_only"])
        assert abs(m_h["full_history"]["auprc"] - ref["full_pooled"]) < 5e-4, (h, m_h["full_history"])
        assert abs(m_h["full_history"]["macro_auprc"] - ref["full_macro"]) < 5e-4, (h, m_h["full_history"])
        # G86 の記録との照合 (0.1 / 0.25 / 0.5 ns の切り取り 6 統計量)
        for name, (ref_p, ref_m) in G86_REF[h].items():
            cname = f"prefix_6stats_{name}"
            assert abs(m_h[cname]["auprc"] - ref_p) < 5e-4, (h, cname, m_h[cname])
            assert abs(m_h[cname]["macro_auprc"] - ref_m) < 5e-4, (h, cname, m_h[cname])

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
            # タンパク質ごとの回復率 (そのタンパク質の利得を分母にする)
            vals = []
            for pname, _, _ in spans:
                a_s = pp_h["static_only"][pname]
                a_f = pp_h["full_history"][pname]
                a_c = pp_h[cname][pname]
                if a_s is None or a_f is None or a_c is None:
                    continue
                denom = a_f - a_s
                if abs(denom) < 1e-6:
                    continue
                vals.append((a_c - a_s) / denom)
            rec_h[cname]["per_protein_mean"] = float(np.mean(vals)) if vals else None
            rec_h[cname]["per_protein_median"] = float(np.median(vals)) if vals else None
            rec_h[cname]["per_protein_n"] = len(vals)
            print(f"  {cname}: recovery pooled {rec_h[cname]['pooled']:.3f} "
                  f"macro {rec_h[cname]['macro']:.3f} "
                  f"per-protein mean {rec_h[cname]['per_protein_mean']:.3f} "
                  f"median {rec_h[cname]['per_protein_median']:.3f}", flush=True)

        results[hkey] = m_h
        recovery[hkey] = rec_h
        per_protein[hkey] = pp_h
        done[hkey] = dict(metrics=m_h, recovery=rec_h, base_gain=base_gain[hkey],
                          per_protein_auprc=pp_h)
        partial_path.write_text(json.dumps(done, indent=2, ensure_ascii=False))

    # 最小の切り取り (先に固定した読み方)
    min_cut = {}
    for h in HORIZONS:
        hkey = str(h)
        rec = recovery[hkey]
        mc = {"pooled": None, "macro": None}
        for name in ("0.1ns", "0.25ns", "0.5ns", "1.0ns"):
            cname = f"prefix_6stats_{name}"
            if mc["pooled"] is None and rec[cname]["pooled"] > 0.5:
                mc["pooled"] = CUTS[name][0]
            if mc["macro"] is None and rec[cname]["macro"] > 0.5:
                mc["macro"] = CUTS[name][0]
        min_cut[hkey] = mc
        print(f"horizon {h} ns: min cut pooled {mc['pooled']} macro {mc['macro']}", flush=True)

    out = {
        "cuts": {k: {"label": v[0], "frames": v[1], "covered_length": v[2]} for k, v in CUTS.items()},
        "horizons": HORIZONS,
        "six_stats_definition": "first, mean, std, min, last-first, frac<=3.5A; history cols [0,1,2,3,5,7]",
        "full_6stats_vs_stored_max_abs_diff": {"cal": d_cal, "test": d_test},
        "references": {
            "base_gain_ref": {str(h): dict(
                pooled=REF[h]["full_pooled"] - REF[h]["static_pooled"],
                macro=REF[h]["full_macro"] - REF[h]["static_macro"]) for h in HORIZONS},
            "base_gain_recomputed": base_gain,
            "replica_lower_bound_A": 0.249,
        },
        "metrics": results,
        "recovery_vs_real_history_gain": recovery,
        "per_protein_auprc": per_protein,
        "min_cut_over_0.5": min_cut,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
