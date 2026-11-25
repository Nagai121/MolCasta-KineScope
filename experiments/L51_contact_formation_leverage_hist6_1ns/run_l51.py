"""L51: 短いMDの長さに対して、指導力がどれだけ戻るかの曲がり具合
（統計量6種の形、資料Dの条件、接触の形成、1 ns先）.

L 系列の実験。L15（平均だけの形）と同じ土台・同じ覆う長さで、
切り取りの履歴の特徴量を平均だけから資料Dの6統計量（最初・平均・
標準偏差・最小・最後−最初・4.5 Å以下の割合）に変えて曲がりを測る。
同じ実行の中で平均だけの曲がりも測り直し、L15 の記録と
一致するか確かめる。

長さの呼び方（L01・L15 と同じ、結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 長さは「覆う長さ」で統一する。覆う長さ L ns の切り取りは
  フレーム 0〜(L/0.1) を使う（フレーム数は L/0.1 + 1）。
- 起点フレーム10（1 ns）は変えない。ラベルは 1 ns先。

条件:
- 静的のみ: 静的特徴5個だけ。
- 全部入り（base_hist_D6）: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム）。
  回復率の分母（実履歴の利得）の参照。
- 切り取りの6統計量: 静的特徴＋切り取り（フレーム 0〜k）の6統計量。
  覆う長さ 0.0〜1.0 ns の 11通り。覆う長さ 1.0 ns では全部入りと
  同じ条件になり、回復率は 1 になるはず。
- 切り取りの平均だけ（照合用）: L15 と同じ条件を測り直す。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。

読み方（結果を見る前に固定）:
1. 回復率 0.8 に届く最小の長さと、0.9 に届く最小の長さを、全体で読む。
2. 段の判定は、隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、
   その大きさで読む。
3. 信頼区間は資料Dの条件の流儀で、全体・1mix_A・2qt4_A の3つの回復率の
   最小値〜最大値とする。
4. 段らしい場所があれば、その前後でタンパク質ごとの値も記録する
   （タンパク質ごとの回復率は全長さで記録する）。
あわせて、平均だけと比べて必要な長さが縮んだか、届かなかった予測幅で
届くようになったかを記録する。

土台の再現の確認（始める前に固定）:
- 静的のみ・全部入りの AUPRC と利得が L15 の results.json と
  5e-4 以内で一致すること。一致しなければ止める。
- 平均だけの曲がり（測り直し）が L15 の回復率と 2e-3 以内で
  一致すること。
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
REPO_EXP = Path("/home/hatch/workspace/kinescope/repo/experiments")
REF_RESULTS = REPO_EXP / "L15_contact_formation_leverage_1ns" / "results.json"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_NS = 1
HORIZON_IDX = 0
BOUND = 4.5

# 覆う長さ（ns）-> フレーム数。フレーム 0〜(n-1) を使う（L01・L15 と同じ定義）。
LENGTHS = {round(0.1 * k, 1): k + 1 for k in range(11)}


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact.npz")
            d = {k: z[k] for k in z.files}
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    # k13_cache の hist は (n, 11) で、フレームが軸 1（転置しない）
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
    return np.stack(
        [series[:, 0], series.mean(1), series.std(1), series.min(1),
         series[:, -1] - series[:, 0], (series <= bound).mean(1)],
        axis=1).astype(np.float32)


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


ref = json.loads(REF_RESULTS.read_text())
ref_static = ref["metrics"]["static_only"]["auprc"]
ref_full = ref["metrics"]["base_hist_D6"]["auprc"]
ref_gain = ref["base_gain"]
ref_mean_recovery = ref["recovery_vs_real_history_gain"]

train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape,
      f"test pos {HORIZON_NS}ns", int(yte.sum()), flush=True)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

# ---- 条件ごとの学習と評価 ----
preds = {}
preds["static_only"] = predict(Xtr, ytr, Xte)
f6tr, f6te = hist6(Htr), hist6(Hte)
preds["base_hist_D6"] = predict(
    np.concatenate([Xtr, f6tr], axis=1), ytr,
    np.concatenate([Xte, f6te], axis=1))

results = {}
for L, nf in LENGTHS.items():
    key6 = f"prefix_hist6_{L:.1f}ns"
    s6tr, s6te = hist6(Htr[:, :nf]), hist6(Hte[:, :nf])
    preds[key6] = predict(
        np.concatenate([Xtr, s6tr], axis=1), ytr,
        np.concatenate([Xte, s6te], axis=1))
    results[key6] = metrics(yte, preds[key6])
    keym = f"prefix_mean_{L:.1f}ns"
    mtr = Htr[:, :nf].mean(1)[:, None].astype(np.float32)
    mte = Hte[:, :nf].mean(1)[:, None].astype(np.float32)
    preds[keym] = predict(
        np.concatenate([Xtr, mtr], axis=1), ytr,
        np.concatenate([Xte, mte], axis=1))
    results[keym] = metrics(yte, preds[keym])
    print(key6, "frames", nf, "auprc", round(results[key6]["auprc"], 4),
          "|", keym, "auprc", round(results[keym]["auprc"], 4), flush=True)

results["static_only"] = metrics(yte, preds["static_only"])
results["base_hist_D6"] = metrics(yte, preds["base_hist_D6"])
print("static_only auprc", round(results["static_only"]["auprc"], 4),
      "base_hist_D6 auprc", round(results["base_hist_D6"]["auprc"], 4), flush=True)

# 土台の再現の確認（合わなければ止める）
base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
print("base gain:", round(base_gain, 4), " reference:", round(ref_gain, 4), flush=True)
assert abs(results["static_only"]["auprc"] - ref_static) < 5e-4, results["static_only"]
assert abs(results["base_hist_D6"]["auprc"] - ref_full) < 5e-4, results["base_hist_D6"]
assert abs(base_gain - ref_gain) < 5e-4, base_gain
# 覆う長さ 1.0 ns の6統計量は全部入りと同じ条件（回復率 1 のはず）
assert abs(results["prefix_hist6_1.0ns"]["auprc"] - results["base_hist_D6"]["auprc"]) < 1e-9
print("base reproduction vs L15: OK", flush=True)

# ---- 回復率（6統計量と平均だけ） ----
lengths_sorted = sorted(LENGTHS)
recovery6, recoverym = {}, {}
for L in lengths_sorted:
    recovery6[L] = float((results[f"prefix_hist6_{L:.1f}ns"]["auprc"]
                          - results["static_only"]["auprc"]) / base_gain)
    recoverym[L] = float((results[f"prefix_mean_{L:.1f}ns"]["auprc"]
                          - results["static_only"]["auprc"]) / base_gain)
    print(f"covered {L:.1f} ns recovery hist6 {recovery6[L]:.4f} "
          f"mean {recoverym[L]:.4f} (ref {ref_mean_recovery[f'{L:.1f}']:.4f})", flush=True)

# 平均だけの曲がりの照合（L15 の記録と一致するか）
max_dev = max(abs(recoverym[L] - ref_mean_recovery[f"{L:.1f}"]) for L in lengths_sorted)
print("mean-only curve max deviation vs L15:", round(max_dev, 6), flush=True)
assert max_dev < 2e-3, max_dev
print("mean-only curve cross-check vs L15: OK", flush=True)

increments = {}
prev = None
for L in lengths_sorted:
    if prev is not None:
        increments[L] = recovery6[L] - recovery6[prev]
    prev = L


def min_length_for(rec, threshold):
    for L in lengths_sorted:
        if rec[L] >= threshold:
            return L
    return None


min_len_08 = min_length_for(recovery6, 0.8)
min_len_09 = min_length_for(recovery6, 0.9)
min_len_08_m = min_length_for(recoverym, 0.8)
min_len_09_m = min_length_for(recoverym, 0.9)
largest_step_L = max(increments, key=lambda L: increments[L])
print("hist6 min length for 0.8:", min_len_08, " for 0.9:", min_len_09,
      " (mean-only:", min_len_08_m, "/", min_len_09_m, ")",
      " largest step at", largest_step_L, "+", round(increments[largest_step_L], 4),
      flush=True)

# ---- タンパク質ごとの回復率（6統計量） ----
per_protein = {}
for L in lengths_sorted:
    key = f"prefix_hist6_{L:.1f}ns"
    row = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        a_s = auprc(yte[idx], preds["static_only"][idx])
        a_f = auprc(yte[idx], preds["base_hist_D6"][idx])
        a_p = auprc(yte[idx], preds[key][idx])
        g = a_f - a_s
        row[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc_static": a_s, "auprc_full": a_f, "auprc_prefix": a_p,
            "full_gain": g,
            "recovery": (a_p - a_s) / g if g else float("nan"),
        }
    per_protein[L] = row

# ---- 区間（資料Dの流儀: 全体・1mix_A・2qt4_A の3値の最小〜最大） ----
intervals = {}
for L in lengths_sorted:
    three = [recovery6[L]] + [per_protein[L][p]["recovery"] for p in TEST_P]
    intervals[L] = {"pooled": recovery6[L],
                    "per_protein": {p: per_protein[L][p]["recovery"] for p in TEST_P},
                    "ci_lower": float(min(three)), "ci_upper": float(max(three))}

out = {
    "label": f"Contact formation {HORIZON_NS}ns, guidance recovery vs short-MD covered length, hist6 prefix (leverage curve)",
    "phenomenon": "contact formation (Doc D, k13_cache)",
    "experiment": "L51",
    "reference_experiment": "L15",
    "horizon_ns": HORIZON_NS,
    "length_definition": {
        "unit": "covered length (ns); prefix = frames 0..(n_frames-1) of the 0-1 ns history, 100 ps spacing (L01 definition)",
        "lengths": {f"{L:.1f}": {"covered_ns": L, "n_frames": nf,
                                 "frames": list(range(nf))}
                    for L, nf in LENGTHS.items()},
    },
    "metrics": results,
    "base_gain": base_gain,
    "recovery_hist6": {f"{L:.1f}": recovery6[L] for L in lengths_sorted},
    "recovery_mean_remeasured": {f"{L:.1f}": recoverym[L] for L in lengths_sorted},
    "recovery_mean_reference_L15": ref_mean_recovery,
    "mean_curve_max_deviation_vs_reference": max_dev,
    "recovery_increment_per_0.1ns_hist6": {f"{L:.1f}": increments[L] for L in sorted(increments)},
    "readings_fixed_in_advance": {
        "min_length_for_recovery_0.8_pooled": min_len_08,
        "min_length_for_recovery_0.9_pooled": min_len_09,
        "min_length_for_recovery_0.8_pooled_mean_only": min_len_08_m,
        "min_length_for_recovery_0.9_pooled_mean_only": min_len_09_m,
        "largest_increment": {"covered_ns": largest_step_L,
                              "increment": increments[largest_step_L]},
        "intervals_3value_min_max": {f"{L:.1f}": intervals[L] for L in lengths_sorted},
    },
    "per_protein_hist6": {f"{L:.1f}": per_protein[L] for L in lengths_sorted},
    "checks_vs_L15": {
        "static_only_auprc_reference": ref_static,
        "full_hist6_auprc_reference": ref_full,
        "base_gain_reference": ref_gain},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["covered_ns", "n_frames", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_hist6", "auc_prefix_hist6",
                "brier_prefix_hist6", "logloss_prefix_hist6",
                "auprc_prefix_mean", "full_gain",
                "recovery_hist6_pooled", "recovery_hist6_1mix_A",
                "recovery_hist6_2qt4_A", "interval_lower", "interval_upper",
                "increment_hist6_from_previous_0.1ns", "recovery_mean_remeasured"])
    for L in lengths_sorted:
        key = f"prefix_hist6_{L:.1f}ns"
        m, iv = results[key], intervals[L]
        w.writerow([L, LENGTHS[L], m["positives"],
                    results["static_only"]["auprc"], results["base_hist_D6"]["auprc"],
                    m["auprc"], m["auc"], m["brier"], m["logloss"],
                    results[f"prefix_mean_{L:.1f}ns"]["auprc"], base_gain,
                    recovery6[L], per_protein[L]["1mix_A"]["recovery"],
                    per_protein[L]["2qt4_A"]["recovery"],
                    iv["ci_lower"], iv["ci_upper"], increments.get(L, ""), recoverym[L]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    y=yte.astype(np.int8),
    p_static_only=preds["static_only"].astype(np.float32),
    p_base_hist_D6=preds["base_hist_D6"].astype(np.float32),
    **{f"p_prefix_hist6_{L:.1f}ns": preds[f"prefix_hist6_{L:.1f}ns"].astype(np.float32)
       for L in lengths_sorted},
    **{f"p_prefix_mean_{L:.1f}ns": preds[f"prefix_mean_{L:.1f}ns"].astype(np.float32)
       for L in lengths_sorted},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
