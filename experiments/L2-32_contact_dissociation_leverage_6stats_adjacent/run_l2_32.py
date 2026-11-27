"""L2-32: 短いMDの長さに対して、指導力がどれだけ戻るかの曲がり具合
（統計量6種の形、資料Dの条件、接触の解離、予測幅 1/5/10/30/50/90 ns、
区間を起点フレーム10の直前に置き直した測り直し）.

L2 系列の実験。L57〜L62（L21〜L26 の曲がりを統計量6種で測ったもの）と
同じ土台・同じ候補・同じ予測器で、履歴の区間だけを置き直す。
L では区間は先頭から（フレーム 0〜k）で、短い長さでは起点フレーム10との
間に時間の空きがあった。E01 で、この空きが回復率を水増ししていたことが
分かっている。ここでは区間を起点の直前に置く（E01 の Protocol A と
同じ置き方）。

長さの定義（結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 覆う長さ L ns の区間は、フレーム (10 − L/0.1) 〜 10 の L/0.1 + 1
  フレーム（起点フレーム10を含む、起点に接した区間）。
  長さ 0 はフレーム10だけ、0.1 ns はフレーム 9〜10、…、
  1.0 ns はフレーム 0〜10 で、L の全部入りと同じ条件になる。
- 起点フレーム10（1 ns）は変えない。ラベルは予測幅ごと。

条件:
- 静的のみ: 静的特徴5個だけ。
- 全部入り（full_hist6）: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム）。
  回復率の分母（実履歴の利得）の参照。
- 区間の6統計量: 静的特徴＋区間（起点の直前）の6統計量。
  覆う長さ 0.0〜1.0 ns の 11通り。
- 区間の平均だけ: 同じ区間の平均だけを足した条件（平均だけの形との比較用）。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。

特徴量の型は L57〜L62（G71）と同じ作り方にそろえる（静的特徴を float64、
区間の平均を float32、6統計量は float32）。型の違いが回復率をずらす
ことが L23〜L26 で分かっているため。

読み方（L と同じ、結果を見る前に固定）:
1. 回復率 0.8 に届く最小の長さと、0.9 に届く最小の長さを、全体で読む。
2. 隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、その大きさ。
3. 信頼区間は資料Dの条件の流儀で、全体・1mix_A・2qt4_A の3つの回復率の
   最小値〜最大値とする。
4. 増え幅が最大の場所では、その前後でタンパク質ごとの値も記録する
   （タンパク質ごとの回復率は全長さで記録する）。
あわせて、平均だけと比べて必要な長さが縮んだかも記録する。

前提確認（始める前に固定）:
- 静的のみ・全部入りの AUPRC と利得が、対応する L（L57〜L62）の
  results.json と 5e-4 以内で一致すること。一致しなければ止める。
- 長さ 1.0 ns の区間の6統計量は全部入りと同じ条件で、AUPRC が
  一致すること（構成上同じ特徴量になる）。

実行はこの仮想マシンで行う（L57〜L62 と同じ規模）。sklearn の学習は
スレッド数を 1 に絞る。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
REPO_EXP = Path("/home/hatch/workspace/kinescope/repo/experiments")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.5
REF_DIRS = {1: "L57_contact_dissociation_leverage_6stats_1ns",
            5: "L58_contact_dissociation_leverage_6stats_5ns",
            10: "L59_contact_dissociation_leverage_6stats_10ns",
            30: "L60_contact_dissociation_leverage_6stats_30ns",
            50: "L61_contact_dissociation_leverage_6stats_50ns",
            90: "L62_contact_dissociation_leverage_6stats_90ns"}

# 覆う長さ（ns）-> フレーム数。区間はフレーム (11-nf)〜10（起点に接する）。
LENGTHS = {round(0.1 * k, 1): k + 1 for k in range(11)}


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
            d = {k: z[k] for k in z.files}
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    # K50 のキャッシュの hist は (N, 11) で保存されているので転置しない
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
    return np.stack(
        [series[:, 0], series.mean(1), series.std(1), series.min(1),
         series[:, -1] - series[:, 0], (series <= bound).mean(1)],
        axis=1).astype(np.float32)


def window(hist, nf):
    # 起点フレーム10の直前の区間: 最後の nf フレーム
    return hist[:, 11 - nf:]


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.075, max_leaf_nodes=31,
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


def auprc(y, p):
    return float(average_precision_score(y, p))


def clean_nan(o):
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def min_length_for(rec, threshold, lengths_sorted):
    for L in lengths_sorted:
        if rec[L] >= threshold:
            return L
    return None


def compare_len(new, ref):
    if ref is None and new is None:
        return "same (both not reached)"
    if ref is None:
        return "newly_reached_with_adjacent"
    if new is None:
        return "no_longer_reached_with_adjacent"
    if new < ref:
        return "shorter_with_adjacent"
    if new > ref:
        return "longer_with_adjacent"
    return "same"


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = (cat(train, "X").astype(np.float64),
            cat(test, "X").astype(np.float64))
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
assert Xtr.shape[0] == 6383 and Xte.shape[0] == 2915, (Xtr.shape, Xte.shape)
prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print("train", Xtr.shape, "test", Xte.shape, {p: len(v) for p, v in prot_idx.items()}, flush=True)

lengths_sorted = sorted(LENGTHS)
out = {
    "experiment": "L2-32",
    "reference_experiments": "L57-L62",
    "phenomenon": "contact dissociation (Doc D, k50_cache)",
    "window_definition": {
        "unit": "covered length (ns); window = frames (10 - L/0.1) .. 10 of the 0-1 ns history, adjacent to the origin frame 10 (E01 Protocol A placement)",
        "lengths": {f"{L:.1f}": {"covered_ns": L, "n_frames": nf,
                                 "frames": list(range(11 - nf, 11))}
                    for L, nf in LENGTHS.items()},
    },
    "feature_dtypes": "static features float64; window mean float32; hist6 float32 (G71/L57 construction)",
    "by_horizon": {},
}
npz_store = {"test_protein": test_protein}
csv_rows = []

for hi, h in enumerate(HORIZONS):
    ref = json.loads((REPO_EXP / REF_DIRS[h] / "results.json").read_text())
    ref_static = ref["base_reproduction"]["static_only_auprc"]
    ref_full = ref["base_reproduction"]["full_hist6_auprc"]
    ref_gain = ref["base_reproduction"]["full_gain"]
    ref_rec6 = {k: v["recovery_pooled"] for k, v in ref["lengths_hist6"].items()}
    ref_min08 = ref["readings"]["smallest_length_recovery_0.8_pooled"]
    ref_min09 = ref["readings"]["smallest_length_recovery_0.9_pooled"]
    ytr, yte = Ytr[:, hi], Yte[:, hi]
    print(f"== horizon {h} ns: test pos {int(yte.sum())} ==", flush=True)

    preds = {}
    preds["static_only"] = predict(Xtr, ytr, Xte)
    f6tr, f6te = hist6(Htr), hist6(Hte)
    preds["full_hist6"] = predict(
        np.concatenate([Xtr, f6tr], axis=1), ytr,
        np.concatenate([Xte, f6te], axis=1))
    results = {}
    for L, nf in LENGTHS.items():
        s6tr, s6te = hist6(window(Htr, nf)), hist6(window(Hte, nf))
        preds[f"adj_hist6_{L:.1f}ns"] = predict(
            np.concatenate([Xtr, s6tr], axis=1), ytr,
            np.concatenate([Xte, s6te], axis=1))
        results[f"adj_hist6_{L:.1f}ns"] = metrics(yte, preds[f"adj_hist6_{L:.1f}ns"])
        mtr = window(Htr, nf).mean(1)[:, None].astype(np.float32)
        mte = window(Hte, nf).mean(1)[:, None].astype(np.float32)
        preds[f"adj_mean_{L:.1f}ns"] = predict(
            np.concatenate([Xtr, mtr], axis=1), ytr,
            np.concatenate([Xte, mte], axis=1))
        results[f"adj_mean_{L:.1f}ns"] = metrics(yte, preds[f"adj_mean_{L:.1f}ns"])
    results["static_only"] = metrics(yte, preds["static_only"])
    results["full_hist6"] = metrics(yte, preds["full_hist6"])

    # 前提確認（合わなければ止める）
    base_gain = results["full_hist6"]["auprc"] - results["static_only"]["auprc"]
    print(f"  static {results['static_only']['auprc']:.6f} (ref {ref_static:.6f}) "
          f"full {results['full_hist6']['auprc']:.6f} (ref {ref_full:.6f}) "
          f"gain {base_gain:.6f} (ref {ref_gain:.6f})", flush=True)
    assert abs(results["static_only"]["auprc"] - ref_static) < 5e-4
    assert abs(results["full_hist6"]["auprc"] - ref_full) < 5e-4
    assert abs(base_gain - ref_gain) < 5e-4
    assert abs(results["adj_hist6_1.0ns"]["auprc"] - results["full_hist6"]["auprc"]) < 1e-9
    print("  premise check vs L reference: OK", flush=True)

    recovery6, recoverym = {}, {}
    for L in lengths_sorted:
        recovery6[L] = float((results[f"adj_hist6_{L:.1f}ns"]["auprc"]
                              - results["static_only"]["auprc"]) / base_gain)
        recoverym[L] = float((results[f"adj_mean_{L:.1f}ns"]["auprc"]
                              - results["static_only"]["auprc"]) / base_gain)
    increments = {}
    prev = None
    for L in lengths_sorted:
        if prev is not None:
            increments[L] = recovery6[L] - recovery6[prev]
        prev = L
    increments_m = {}
    prev = None
    for L in lengths_sorted:
        if prev is not None:
            increments_m[L] = recoverym[L] - recoverym[prev]
        prev = L

    min08 = min_length_for(recovery6, 0.8, lengths_sorted)
    min09 = min_length_for(recovery6, 0.9, lengths_sorted)
    min08m = min_length_for(recoverym, 0.8, lengths_sorted)
    min09m = min_length_for(recoverym, 0.9, lengths_sorted)
    step_L = max(increments, key=lambda L: increments[L])
    step_Lm = max(increments_m, key=lambda L: increments_m[L])
    print(f"  hist6 min08 {min08} min09 {min09} largest step at {step_L} "
          f"+{increments[step_L]:.4f} | mean min08 {min08m} min09 {min09m}", flush=True)

    per_protein = {}
    for L in lengths_sorted:
        key = f"adj_hist6_{L:.1f}ns"
        row = {}
        for prot in TEST_P:
            idx = prot_idx[prot]
            a_s = auprc(yte[idx], preds["static_only"][idx])
            a_f = auprc(yte[idx], preds["full_hist6"][idx])
            a_p = auprc(yte[idx], preds[key][idx])
            g = a_f - a_s
            row[prot] = {
                "n": int(len(idx)), "positives": int(yte[idx].sum()),
                "auprc_static": a_s, "auprc_full": a_f, "auprc_prefix": a_p,
                "full_gain": g,
                "recovery": (a_p - a_s) / g if g else float("nan")}
        per_protein[L] = row
    intervals = {}
    for L in lengths_sorted:
        three = [recovery6[L]] + [per_protein[L][p]["recovery"] for p in TEST_P]
        intervals[L] = {"pooled": recovery6[L],
                        "per_protein": {p: per_protein[L][p]["recovery"] for p in TEST_P},
                        "ci_lower": float(min(three)), "ci_upper": float(max(three))}
    step_i = lengths_sorted.index(step_L)
    step_context = {}
    for L in lengths_sorted[max(0, step_i - 1): step_i + 1]:
        step_context[f"{L:.1f}"] = {
            "pooled": recovery6[L],
            "per_protein": {p: per_protein[L][p]["recovery"] for p in TEST_P}}

    out["by_horizon"][str(h)] = {
        "horizon_ns": h,
        "reference_experiment": REF_DIRS[h].split("_")[0],
        "metrics": results,
        "base_gain": base_gain,
        "recovery_hist6_adjacent": {f"{L:.1f}": recovery6[L] for L in lengths_sorted},
        "recovery_mean_adjacent": {f"{L:.1f}": recoverym[L] for L in lengths_sorted},
        "recovery_hist6_reference_L": ref_rec6,
        "recovery_increment_per_0.1ns_hist6": {f"{L:.1f}": increments[L] for L in sorted(increments)},
        "readings_fixed_in_advance": {
            "min_length_for_recovery_0.8_pooled": min08,
            "min_length_for_recovery_0.9_pooled": min09,
            "min_length_for_recovery_0.8_pooled_mean_only": min08m,
            "min_length_for_recovery_0.9_pooled_mean_only": min09m,
            "min_length_for_recovery_0.8_pooled_reference_L": ref_min08,
            "min_length_for_recovery_0.9_pooled_reference_L": ref_min09,
            "largest_increment": {"covered_ns": step_L, "increment": increments[step_L]},
            "largest_increment_mean_only": {"covered_ns": step_Lm, "increment": increments_m[step_Lm]},
            "step_context_per_protein": step_context,
            "intervals_3value_min_max": {f"{L:.1f}": intervals[L] for L in lengths_sorted},
        },
        "comparison_vs_mean_only": {
            "for_0.8": compare_len(min08, min08m), "for_0.9": compare_len(min09, min09m)},
        "comparison_vs_L_window": {
            "for_0.8": compare_len(min08, ref_min08),
            "for_0.9": compare_len(min09, ref_min09)},
        "per_protein_hist6": {f"{L:.1f}": per_protein[L] for L in lengths_sorted},
        "checks_vs_L_reference": {
            "static_only_auprc_reference": ref_static,
            "full_hist6_auprc_reference": ref_full,
            "base_gain_reference": ref_gain},
    }
    for L in lengths_sorted:
        key = f"adj_hist6_{L:.1f}ns"
        m, iv = results[key], intervals[L]
        csv_rows.append([h, L, LENGTHS[L], 11 - LENGTHS[L], 10,
                         m["positives"], results["static_only"]["auprc"],
                         results["full_hist6"]["auprc"], m["auprc"], m["auc"],
                         m["brier"], m["logloss"], results[f"adj_mean_{L:.1f}ns"]["auprc"],
                         base_gain, recovery6[L], per_protein[L]["1mix_A"]["recovery"],
                         per_protein[L]["2qt4_A"]["recovery"], iv["ci_lower"], iv["ci_upper"],
                         increments.get(L, ""), recoverym[L],
                         float(ref_rec6[f"{L:.1f}"])])
    npz_store[f"y_{h}ns"] = yte.astype(np.int8)
    npz_store[f"p_static_only_{h}ns"] = preds["static_only"].astype(np.float32)
    npz_store[f"p_full_hist6_{h}ns"] = preds["full_hist6"].astype(np.float32)
    for L in lengths_sorted:
        npz_store[f"p_adj_hist6_{L:.1f}ns_{h}ns"] = preds[f"adj_hist6_{L:.1f}ns"].astype(np.float32)
        npz_store[f"p_adj_mean_{L:.1f}ns_{h}ns"] = preds[f"adj_mean_{L:.1f}ns"].astype(np.float32)

(OUT / "results.json").write_text(json.dumps(clean_nan(out), indent=2, ensure_ascii=False) + "\n")
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "covered_ns", "n_frames", "frame_start", "frame_end",
                "test_positives", "auprc_static", "auprc_full", "auprc_adj_hist6",
                "auc_adj_hist6", "brier_adj_hist6", "logloss_adj_hist6",
                "auprc_adj_mean", "full_gain", "recovery_hist6_pooled",
                "recovery_hist6_1mix_A", "recovery_hist6_2qt4_A",
                "interval_lower", "interval_upper",
                "increment_hist6_from_previous_0.1ns", "recovery_mean_pooled",
                "recovery_hist6_reference_L"])
    w.writerows(csv_rows)
np.savez_compressed(OUT / "test_predictions.npz", **npz_store)
print("saved results.json / results.csv / test_predictions.npz", flush=True)
