"""G161: 強い土台で、水素結合の形成の 6 個の統計量の切り取りに
起点のフレームを入れるかどうかで回復率がどう変わるかを見る
(K06(b) の条件、強い土台)。

G155 は資料Dの条件の水素結合の形成で、切り取り 1.0 ns
（フレーム [0,1,2,3,4,5,6,7,8,9]、10フレーム）の 6 個の統計量と、
0〜1 ns の 6 個の統計量（フレーム [0,1,2,3,4,5,6,7,8,9,10]、
11フレーム全部）とで、回復率が大きく違うことを見つけました。
30 ns先では -0.403（1.403 -> 1.000）、10 ns先では +0.286
（0.714 -> 1.000）と、予測幅によって分かれ方が違いました。
ここでは強い土台でも同じように分かれるかどうかを見ます。

2 つの条件（結果を見る前に固定）:
- hist6_10f: フレーム [0..9] の 10 フレームで計算した 6 個の統計量。
  実際に覆う長さは 0.9 ns です。起点フレーム 10 は入らない
  （G141 の「1.0 ns の切り取り」と同じ）。
- hist6_11f: フレーム [0..10] の 11 フレーム全部で計算した 6 個の統計量。
  0〜1 ns の履歴全部です。起点フレーム 10 が入ります。
  資料Dの条件と違い、強い土台では 11 フレーム全部の 6 個の統計量は
  全部入りの条件そのものではありません。全部入りの条件は保存された
  10 個の履歴特徴を使うので、11 フレーム全部の 6 個の統計量の回復率は
  1.0 にはなりません（G141 の記録では 0.972〜1.162）。

6 個の統計量は、最初の値、平均、標準偏差、最小値、最後の値 − 最初の値、
3.5 A 以下の割合です。強い土台の実履歴10特徴の並びは
[最初の値、平均、標準偏差、最小値、最大値、最後−最初、最後−1つ前、
3.5 A 以下割合、4.5 A 以下割合、増えたステップの割合] なので、
6統計量は history の列 [0, 1, 2, 3, 5, 7] にあたります。
回復率 = (その条件の AUPRC - 静的のみ) / (その予測幅での実履歴の利得)。
利得は全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測ります。
土台の利得の記録値 (k06b_results.json / K12 / G81 の記録):
  1 ns: pooled +0.0702 / macro +0.0712
  5 ns: pooled +0.0514 / macro +0.0518
  10 ns: pooled +0.0376 / macro +0.0388
  30 ns: pooled +0.0363 / macro +0.0370
  50 ns: pooled +0.0354 / macro +0.0344
  90 ns: pooled +0.0265 / macro +0.0273

読み方（事前に固定）: 2 つの 6 個の統計量の回復率を予測幅ごとに並べ、
差が大きい予測幅を挙げる。差の大きさは回復率の差と AUPRC の差の両方で見る。
pooled と macro の両方で見ます。

起点のフレーム（フレーム 10）は予測時点そのものの構造なので、静的な
入力の一部として入れるのは正当です。フレーム 10 の距離は静的特徴の
0 列目にも入っています。
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

# G141 の記録値（1.0 ns の切り取り＝フレーム 0〜9 の 6 個の統計量の回復率、照合用）
G141_REF_10F = {
    1: (1.055, 1.028), 5: (0.949, 0.961), 10: (0.954, 0.955),
    30: (1.152, 1.157), 50: (0.918, 0.936), 90: (1.072, 1.112),
}
# G141 の記録値（11 フレーム全部の 6 個の統計量の回復率、照合用）
G141_REF_11F = {
    1: (1.006, 0.992), 5: (0.978, 0.967), 10: (1.041, 1.011),
    30: (1.057, 1.063), 50: (0.972, 0.983), 90: (1.162, 1.165),
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

    partial_path = OUT / "results_g161_partial.json"
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
        conditions["hist6_11f"] = (
            np.column_stack([cal["static"], full6_cal]),
            np.column_stack([test["static"], full6_test]))
        for name, (label, idx, covered) in CUTS.items():
            if name == "1.0ns":
                cname = "hist6_10f"
            else:
                cname = f"prefix_6stats_{name}"
            conditions[cname] = (
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
        # G141 の記録との照合 (フレーム 0〜9 と 0〜10 の 6 個の統計量の AUPRC は
        # 回復率の照合で代わりに確かめる。ここでは後で回復率で照合する)

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

        # G141 の記録との照合（フレーム 0〜9 と 0〜10 の 6 個の統計量の回復率）
        ref10_p, ref10_m = G141_REF_10F[h]
        ref11_p, ref11_m = G141_REF_11F[h]
        assert abs(rec_h["hist6_10f"]["pooled"] - ref10_p) < 5e-3, (h, rec_h["hist6_10f"])
        assert abs(rec_h["hist6_10f"]["macro"] - ref10_m) < 5e-3, (h, rec_h["hist6_10f"])
        assert abs(rec_h["hist6_11f"]["pooled"] - ref11_p) < 5e-3, (h, rec_h["hist6_11f"])
        assert abs(rec_h["hist6_11f"]["macro"] - ref11_m) < 5e-3, (h, rec_h["hist6_11f"])

        results[hkey] = m_h
        recovery[hkey] = rec_h
        per_protein[hkey] = pp_h
        done[hkey] = dict(metrics=m_h, recovery=rec_h, base_gain=base_gain[hkey],
                          per_protein_auprc=pp_h)
        partial_path.write_text(json.dumps(done, indent=2, ensure_ascii=False))

    # 差（11フレームの 6 個の統計量 − 10フレームの 6 個の統計量、先に固定した読み方）
    diff = {}
    for h in HORIZONS:
        hkey = str(h)
        m = results[hkey]
        rec = recovery[hkey]
        diff[hkey] = dict(
            recovery_diff_pooled=float(rec["hist6_11f"]["pooled"] - rec["hist6_10f"]["pooled"]),
            recovery_diff_macro=float(rec["hist6_11f"]["macro"] - rec["hist6_10f"]["macro"]),
            auprc_diff_pooled=float(m["hist6_11f"]["auprc"] - m["hist6_10f"]["auprc"]),
            auprc_diff_macro=float(m["hist6_11f"]["macro_auprc"] - m["hist6_10f"]["macro_auprc"]),
        )
        print(f"{h} ns: recovery diff pooled {diff[hkey]['recovery_diff_pooled']:+.3f} "
              f"macro {diff[hkey]['recovery_diff_macro']:+.3f} "
              f"auprc diff pooled {diff[hkey]['auprc_diff_pooled']:+.4f} "
              f"macro {diff[hkey]['auprc_diff_macro']:+.4f}", flush=True)

    out = {
        "conditions": {
            "hist6_10f": {"label": "フレーム [0..9] の 10 フレームで計算した 6 個の統計量",
                          "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          "covered_length": "0.9 ns"},
            "hist6_11f": {"label": "フレーム [0..10] の 11 フレーム全部で計算した 6 個の統計量",
                          "frames": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          "covered_length": "1.0 ns"},
        },
        "horizons": HORIZONS,
        "six_stats_definition": "first, mean, std, min, last-first, frac<=3.5A; history cols [0,1,2,3,5,7]",
        "full_6stats_vs_stored_max_abs_diff": {"cal": d_cal, "test": d_test},
        "references": {
            "base_gain_ref": {str(h): dict(
                pooled=REF[h]["full_pooled"] - REF[h]["static_pooled"],
                macro=REF[h]["full_macro"] - REF[h]["static_macro"]) for h in HORIZONS},
            "base_gain_recomputed": base_gain,
            "g141_hist6_10f_recovery": {str(h): {"pooled": v[0], "macro": v[1]} for h, v in G141_REF_10F.items()},
            "g141_hist6_11f_recovery": {str(h): {"pooled": v[0], "macro": v[1]} for h, v in G141_REF_11F.items()},
            "replica_lower_bound_A": 0.249,
        },
        "metrics": results,
        "recovery_vs_real_history_gain": recovery,
        "per_protein_auprc": per_protein,
        "diff_hist6_11f_minus_hist6_10f": diff,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    import csv as _csv
    with open(OUT / "results.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["horizon_ns", "base_gain_pooled", "base_gain_macro",
                    "static_auprc_pooled", "static_auprc_macro",
                    "full_auprc_pooled", "full_auprc_macro",
                    "hist6_10f_auprc_pooled", "hist6_10f_auprc_macro",
                    "hist6_10f_recovery_pooled", "hist6_10f_recovery_macro",
                    "hist6_11f_auprc_pooled", "hist6_11f_auprc_macro",
                    "hist6_11f_recovery_pooled", "hist6_11f_recovery_macro",
                    "recovery_diff_pooled", "recovery_diff_macro",
                    "auprc_diff_pooled", "auprc_diff_macro"])
        for h in HORIZONS:
            hkey = str(h)
            m = results[hkey]
            rec = recovery[hkey]
            d = diff[hkey]
            w.writerow([h, round(base_gain[hkey]["pooled"], 4), round(base_gain[hkey]["macro"], 4),
                        round(m["static_only"]["auprc"], 4), round(m["static_only"]["macro_auprc"], 4),
                        round(m["full_history"]["auprc"], 4), round(m["full_history"]["macro_auprc"], 4),
                        round(m["hist6_10f"]["auprc"], 4), round(m["hist6_10f"]["macro_auprc"], 4),
                        round(rec["hist6_10f"]["pooled"], 3), round(rec["hist6_10f"]["macro"], 3),
                        round(m["hist6_11f"]["auprc"], 4), round(m["hist6_11f"]["macro_auprc"], 4),
                        round(rec["hist6_11f"]["pooled"], 3), round(rec["hist6_11f"]["macro"], 3),
                        round(d["recovery_diff_pooled"], 3), round(d["recovery_diff_macro"], 3),
                        round(d["auprc_diff_pooled"], 4), round(d["auprc_diff_macro"], 4)])
    print("saved", OUT / "results.json", OUT / "results.csv")


if __name__ == "__main__":
    main()
