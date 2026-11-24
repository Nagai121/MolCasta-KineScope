"""G175: 強い土台での水素結合の形成の混合を、第22案の生成平均だけで
測り直す (K06(b) の条件、水素結合の形成、1 ns先と 30 ns先)。

背景: G94 は、強い土台の候補に第11〜14案の生成平均が存在しないため
前提確認で止まった。Echizen の委任を受けた統括の判断で、(b) 第22案
だけで測り直すことになり、別の問いとしてこの G175 で実施する。
案 11〜14 の 108 タンパク質分の作り直しはしない。

設計 (止まった G94 と同じ):
- 切り取りは 6 個の統計量 (資料D式: 最初の値、平均、標準偏差、最小値、
  最後−最初、3.5 A 以下割合)。0.1 ns の切り取り = フレーム [0, 1]。
  系列は G05 で作り直したもの (~/workspace/kinescope/work/g05_series/)。
  6統計量の計算は G86 の run_g86.py と同じ。
- 生成平均は K06(b) の保存データの生成6特徴量 (r{r}_gen6) のうち
  平均の列 (gen6[:, 1])。第22案の生成5点系列 (起点＋0.1/0.3/0.5/1.0 ns
  先) の平均で、G21・G33 が使ったのと同じ形。gen8 は使わない。
- 混合 = 静的特徴5つ + 切り取りの 6 統計量 + 生成平均 (1列)。
- 比べる相手は、切り取りだけ (静的特徴5つ + 切り取りの 6 統計量、
  G86 の記録) と静的のみ (静的特徴5つだけ)。
- 予測幅は 1 ns先と 30 ns先。

土台 (K06(b) と同じ):
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  (160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
  seed 20261121)。較正タンパク質 8つで学習し、未見の100タンパク質
  ×3レプリカ (163,074行) で評価する。静的特徴は5個。
- 土台の利得は K06(b) の記録値 (k06b_results.json) で、再計算でも
  一致することを確かめる。1 ns先: 全体 +0.0702 / macro +0.0712。
  30 ns先: 全体 +0.0363 / macro +0.0370。
- 回復率 = (その条件の AUPRC - 静的のみ) / (その予測幅での土台の利得)。
  全体 (pooled) とタンパク質ごとの平均 (macro) の両方で測る。
- 混合の差 = 混合の回復率 - 切り取りだけの回復率。

読み方 (先に固定): 混合の差が +0.05 の線を超えるかどうかを、予測幅
ごとに見る。強い土台ではこれまで混合が負に出ている (G21: 1 ns先、
切り取り平均だけとの差で全体 -0.083 / macro -0.062。G33: 0.5 ns の
切り取り平均で全体 -0.060 / macro -0.027)。
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/g05_series")
OUT = Path(__file__).resolve().parent
SEED = 20261121
HORIZONS = [(0, 1), (3, 30)]  # (label 列の index, 予測幅 ns)

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# k06b_results.json の記録値 (土台の再現の確認用、G86 の REF と同じ)
REF = {
    1: dict(static_pooled=0.4591, static_macro=0.4539,
            full_pooled=0.5293, full_macro=0.5251),
    30: dict(static_pooled=0.4492, static_macro=0.4418,
             full_pooled=0.4855, full_macro=0.4788),
}
# G86 の記録: 0.1 ns の 6 統計量の切り取りだけの回復率
G86_PREFIX_RECOVERY = {1: dict(pooled=0.446, macro=0.468),
                       30: dict(pooled=0.629, macro=0.656)}
# G21 の記録: 1 ns先の生成平均だけ (AUPRC と回復率)
G21_GEN_ONLY = dict(auprc_pooled=0.4568, auprc_macro=0.4520,
                    recovery_pooled=-0.032, recovery_macro=-0.027)

PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り
ALL_FRAMES = list(range(11))


def six_stats(series, idx):
    """series: (n_frames, n_candidates). 資料D式6統計量 (n, 6)。G86 と同じ。"""
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
    static, hist, label, series, gen6 = [], [], [], [], []
    for p in proteins:
        z = np.load(K06B / f"{p}.npz")
        s = np.load(SERIES / f"{p}.npz")
        for r in (1, 2, 3):
            static.append(z[f"r{r}_static"])
            hist.append(z[f"r{r}_history"])
            label.append(z[f"r{r}_label"])
            series.append(s[f"r{r}_series"])
            gen6.append(z[f"r{r}_gen6"])
    return dict(static=np.concatenate(static), hist=np.concatenate(hist),
                label=np.concatenate(label),
                series=np.concatenate(series, axis=1),
                gen6=np.concatenate(gen6))


def learner(X, y):
    return HistGradientBoostingClassifier(
        max_iter=160, learning_rate=.065, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=.15,
        random_state=SEED).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def col(a):
    return a[:, None].astype(np.float32)


def main():
    cal, test = load(CAL), load(TEST)
    print("cal", len(cal["label"]), "test", len(test["label"]), flush=True)

    # 実行前の確認: 11フレーム全部の 6 統計量が保存値と一致すること (G86 と同じ)
    full6_cal = six_stats(cal["series"], ALL_FRAMES)
    full6_test = six_stats(test["series"], ALL_FRAMES)
    d_cal = float(np.abs(full6_cal - cal["hist"][:, [0, 1, 2, 3, 5, 7]]).max())
    d_test = float(np.abs(full6_test - test["hist"][:, [0, 1, 2, 3, 5, 7]]).max())
    print(f"full 6stats vs stored history cols: cal {d_cal:.2e}, test {d_test:.2e}",
          flush=True)
    assert d_cal < 1e-4 and d_test < 1e-4

    # 切り取りの 6 統計量と生成平均
    pre6_cal = six_stats(cal["series"], PREFIX_FRAMES)
    pre6_test = six_stats(test["series"], PREFIX_FRAMES)
    gen_mean_cal = cal["gen6"][:, 1].astype(np.float64)
    gen_mean_test = test["gen6"][:, 1].astype(np.float64)

    # 検証: 生成平均と実の 0〜1 ns の平均の相関 (K43・G21 の記録: テストで 0.786)
    full_mean_test = test["series"].mean(axis=0)
    corr_gm = float(np.corrcoef(gen_mean_test, full_mean_test)[0, 1])
    print(f"corr(gen_mean, real full mean) test = {corr_gm:.4f} (K43/G21: 0.786)",
          flush=True)
    assert abs(corr_gm - 0.786) < 0.01
    relations = dict(
        corr_gen_fullmean_test=corr_gm,
        corr_prefix6mean_gen_test=float(np.corrcoef(pre6_test[:, 1], gen_mean_test)[0, 1]),
        gen_minus_prefix6mean_test_mean=float((gen_mean_test - pre6_test[:, 1]).mean()),
        gen_minus_prefix6mean_test_std=float((gen_mean_test - pre6_test[:, 1]).std()),
    )

    # タンパク質ごとの区間 (macro AUPRC 用、G86 と同じ)
    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n))
        cur += n

    conditions = {
        "static_only": (cal["static"], test["static"]),
        "full_history": (np.column_stack([cal["static"], cal["hist"]]),
                         np.column_stack([test["static"], test["hist"]])),
        "prefix_6stats_0.1ns": (np.column_stack([cal["static"], pre6_cal]),
                                np.column_stack([test["static"], pre6_test])),
        "gen_mean_only": (np.column_stack([cal["static"], col(gen_mean_cal)]),
                          np.column_stack([test["static"], col(gen_mean_test)])),
        "mixed_6stats_gen22": (
            np.column_stack([cal["static"], pre6_cal, col(gen_mean_cal)]),
            np.column_stack([test["static"], pre6_test, col(gen_mean_test)])),
    }

    results, recovery, base_gain, mixed_delta = {}, {}, {}, {}
    per_protein = {}
    predictions = {}
    for hi, h in HORIZONS:
        hkey = str(h)
        y_cal, y_test = cal["label"][:, hi], test["label"][:, hi]
        print(f"--- horizon {h} ns: cal pos {int(y_cal.sum())} "
              f"test pos {int(y_test.sum())} ---", flush=True)

        m_h, pp_h = {}, {}
        for cname, (xtr, xte) in conditions.items():
            pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
            predictions[f"h{h}_{cname}"] = pred.astype(np.float64)
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
            print(f"  {cname}: pooled auprc {m['auprc']:.4f} "
                  f"macro auprc {m['macro_auprc']:.4f}", flush=True)
        predictions[f"h{h}_label"] = y_test.astype(np.int64)

        # 土台の再現の確認 (k06b_results.json の記録値)
        ref = REF[h]
        assert abs(m_h["static_only"]["auprc"] - ref["static_pooled"]) < 5e-4, (h, "static pooled")
        assert abs(m_h["static_only"]["macro_auprc"] - ref["static_macro"]) < 5e-4, (h, "static macro")
        assert abs(m_h["full_history"]["auprc"] - ref["full_pooled"]) < 5e-4, (h, "full pooled")
        assert abs(m_h["full_history"]["macro_auprc"] - ref["full_macro"]) < 5e-4, (h, "full macro")
        print(f"  base reproduction OK (h={h})", flush=True)

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
                  f"per-protein median {rec_h[cname]['per_protein_median']:.3f}",
                  flush=True)

        # G86 の記録との照合: 切り取りだけの回復率
        g86 = G86_PREFIX_RECOVERY[h]
        got = rec_h["prefix_6stats_0.1ns"]
        print(f"  G86 check: prefix recovery pooled {got['pooled']:.4f} "
              f"(G86: {g86['pooled']}) macro {got['macro']:.4f} (G86: {g86['macro']})",
              flush=True)
        assert abs(got["pooled"] - g86["pooled"]) < 5e-3, (h, "G86 pooled")
        assert abs(got["macro"] - g86["macro"]) < 5e-3, (h, "G86 macro")

        # G21 の記録との照合 (1 ns先の生成平均だけ)
        if h == 1:
            g = m_h["gen_mean_only"]
            rg = rec_h["gen_mean_only"]
            print(f"  G21 check: gen_mean_only auprc pooled {g['auprc']:.4f} "
                  f"(G21: {G21_GEN_ONLY['auprc_pooled']}) macro {g['macro_auprc']:.4f} "
                  f"(G21: {G21_GEN_ONLY['auprc_macro']}), recovery pooled "
                  f"{rg['pooled']:.3f} (G21: {G21_GEN_ONLY['recovery_pooled']}) "
                  f"macro {rg['macro']:.3f} (G21: {G21_GEN_ONLY['recovery_macro']})",
                  flush=True)
            assert abs(g["auprc"] - G21_GEN_ONLY["auprc_pooled"]) < 5e-4
            assert abs(g["macro_auprc"] - G21_GEN_ONLY["auprc_macro"]) < 5e-4

        mixed_delta[hkey] = dict(
            pooled=float(rec_h["mixed_6stats_gen22"]["pooled"]
                         - rec_h["prefix_6stats_0.1ns"]["pooled"]),
            macro=float(rec_h["mixed_6stats_gen22"]["macro"]
                        - rec_h["prefix_6stats_0.1ns"]["macro"]),
            auprc_pooled=float(m_h["mixed_6stats_gen22"]["auprc"]
                               - m_h["prefix_6stats_0.1ns"]["auprc"]),
            auprc_macro=float(m_h["mixed_6stats_gen22"]["macro_auprc"]
                              - m_h["prefix_6stats_0.1ns"]["macro_auprc"]))
        print(f"  mixed - prefix: pooled {mixed_delta[hkey]['pooled']:+.3f} "
              f"macro {mixed_delta[hkey]['macro']:+.3f} "
              f"(line: +0.05)", flush=True)

        results[hkey] = m_h
        recovery[hkey] = rec_h
        per_protein[hkey] = pp_h

    out = {
        "design": {
            "base": "K06(b): frozen T1I2006 + HistGB correction (160 iter, lr 0.065, "
                    "leaves 15, min_samples_leaf 20, L2 0.15, seed 20261121); "
                    "calibration 8 proteins, test 100 proteins x 3 replicas",
            "prefix": "6 stats (first, mean, std, min, last-first, frac<=3.5A) "
                      "of frames [0, 1] (0.1 ns cut) from G05 series",
            "generated_mean": "gen6[:, 1] (mean column of proposal-22 generated "
                              "6 features; 5-point forward series). gen8 not used. "
                              "Same form as G21/G33.",
            "mixed": "static 5 + prefix 6 stats + generated mean (1 column)",
            "horizons_ns": [1, 30],
            "recovery": "(AUPRC - static_only) / base gain at that horizon; "
                        "base gain = full_history - static_only (K06(b) records)",
            "mixed_delta": "mixed recovery - prefix-only recovery; line +0.05",
        },
        "full_6stats_vs_stored_max_abs_diff": {"cal": d_cal, "test": d_test},
        "relations": relations,
        "base_gain": base_gain,
        "metrics": results,
        "recovery_vs_real_history_gain": recovery,
        "mixed_minus_prefix": mixed_delta,
        "per_protein_auprc": per_protein,
        "references": {
            "k06b_base_gain": {"1": {"pooled": 0.0702, "macro": 0.0712},
                               "30": {"pooled": 0.0363, "macro": 0.0370}},
            "g86_prefix_6stats_0.1ns_recovery": {
                "1": G86_PREFIX_RECOVERY[1], "30": G86_PREFIX_RECOVERY[30]},
            "g21_mixed_minus_prefix_mean_only_1ns": {"pooled": -0.083, "macro": -0.062},
            "g33_mixed_minus_prefix_mean_only_0.5ns_1ns": {"pooled": -0.060, "macro": -0.027},
        },
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    np.savez_compressed(OUT / "test_predictions.npz", **predictions)

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_ns", "condition", "n", "positives", "auprc_pooled",
                    "auprc_macro", "auc", "brier", "logloss",
                    "recovery_pooled", "recovery_macro",
                    "recovery_per_protein_median"])
        for hi, h in HORIZONS:
            hkey = str(h)
            for cname, m in results[hkey].items():
                rec = recovery[hkey].get(cname, {})
                w.writerow([h, cname, m["n"], m["positives"],
                            f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}",
                            f"{m['auc']:.6f}", f"{m['brier']:.6f}",
                            f"{m['logloss']:.6f}",
                            f"{rec['pooled']:.6f}" if rec else "",
                            f"{rec['macro']:.6f}" if rec else "",
                            f"{rec['per_protein_median']:.6f}"
                            if rec and rec.get("per_protein_median") is not None else ""])
        for hi, h in HORIZONS:
            d = mixed_delta[str(h)]
            w.writerow([h, "mixed_minus_prefix", "", "", "", "", "", "", "",
                        f"{d['pooled']:.6f}", f"{d['macro']:.6f}", ""])
    print("saved", OUT / "results.json", OUT / "results.csv",
          OUT / "test_predictions.npz", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
