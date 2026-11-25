"""L16: 短いMDの長さに対して、指導力がどれだけ戻るかの曲がり具合
（資料Dの条件、接触の形成、5 ns先）.

L 系列の実験。方法は L01・L02 と同じで、長さの定義も L01 に合わせる。
実の 0〜1 ns の履歴を始めの短い部分だけに切って使い、切り取りの長さを
0.1 ns 刻みで変えたときの、平均だけの回復率の曲がり具合を測る。

長さの呼び方（L01 と同じ、結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 長さは「覆う長さ」で統一する。覆う長さ L ns の切り取りは
  フレーム 0〜(L/0.1) を使う（フレーム数は L/0.1 + 1）。
  - 覆う長さ 0 ns   -> フレーム 0 のみ（1フレーム。起点の距離そのもの）
  - 覆う長さ 0.1 ns -> フレーム 0, 1（2フレーム）
  - ...
  - 覆う長さ 1.0 ns -> フレーム 0〜10（11フレーム。0〜1 ns の全部）
- 起点フレーム10（1 ns）は変えない。ラベルは 5 ns先。

条件:
- 静的のみ: 静的特徴5個だけ。
- 全部入り（base_hist_D6）: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム、
  境界 4.5 Å）。
- 切り取りの平均だけ: 静的特徴＋切り取り（フレーム 0〜k）の距離の平均だけ。
  覆う長さ 0.0〜1.0 ns の 11通り。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。

読み方（結果を見る前に固定）:
1. 回復率 0.8 に届く最小の長さと、0.9 に届く最小の長さを、統合値で読む。
2. 段の判定は、隣り合う長さの間の回復率の増え幅がいちばん大きい場所と、
   その大きさで読む。
3. 信頼区間は資料Dの条件の流儀で、統合・1mix_A・2qt4_A の3つの回復率の
   最小値〜最大値とする（テスト 2タンパク質の土台では区間は粗い）。
4. 段らしい場所があれば、その前後でタンパク質ごとの値も記録する
   （タンパク質ごとの回復率は全長さで記録する）。

土台の再現の期待値（K27 の記録、接触の形成・資料D、5 ns先）:
静的のみ 0.287、全部入り 0.5192、テスト 7,384件・
陽性 214件。覆う長さ 1.0 ns の平均だけは K27 の only_mean
（0.4479）とも照らす。
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_NS = 5
HORIZON_IDX = 1
BOUND = 4.5

# 覆う長さ（ns）-> フレーム数。フレーム 0〜(n-1) を使う（L01 と同じ定義）。
LENGTHS = {round(0.1 * k, 1): k + 1 for k in range(11)}

# K27 の記録との照合用の期待値（接触の形成、資料D、5 ns先、合算）
EXPECTED = {"static_only": 0.287, "base_hist_D6": 0.5192,
            "mean_full": 0.4479,
            "test_n": 7384, "test_positives": 214}


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
    key = f"prefix_mean_{L:.1f}ns"
    mtr = Htr[:, :nf].mean(1)[:, None].astype(np.float32)
    mte = Hte[:, :nf].mean(1)[:, None].astype(np.float32)
    preds[key] = predict(
        np.concatenate([Xtr, mtr], axis=1), ytr,
        np.concatenate([Xte, mte], axis=1))
    results[key] = metrics(yte, preds[key])
    print(key, "frames", nf,
          {k: round(v, 4) if isinstance(v, float) else v
           for k, v in results[key].items()}, flush=True)

results["static_only"] = metrics(yte, preds["static_only"])
results["base_hist_D6"] = metrics(yte, preds["base_hist_D6"])
print("static_only", {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in results["static_only"].items()}, flush=True)
print("base_hist_D6", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in results["base_hist_D6"].items()}, flush=True)

# 土台の再現の確認（合わなければ止める）
base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
print("base gain:", round(base_gain, 4), flush=True)
assert results["static_only"]["n"] == EXPECTED["test_n"]
assert results["static_only"]["positives"] == EXPECTED["test_positives"]
assert abs(results["static_only"]["auprc"] - EXPECTED["static_only"]) < 5e-4, results["static_only"]
assert abs(results["base_hist_D6"]["auprc"] - EXPECTED["base_hist_D6"]) < 5e-4, results["base_hist_D6"]
# 覆う長さ 1.0 ns の平均だけは K27 の only_mean と一致するはず
assert abs(results["prefix_mean_1.0ns"]["auprc"] - EXPECTED["mean_full"]) < 5e-4, results["prefix_mean_1.0ns"]
print("base reproduction vs K27: OK", flush=True)

# ---- 回復率と増え分（統合） ----
recovery = {}
for L in LENGTHS:
    key = f"prefix_mean_{L:.1f}ns"
    recovery[L] = float((results[key]["auprc"] - results["static_only"]["auprc"]) / base_gain)
    print(f"covered {L:.1f} ns recovery {recovery[L]:.4f}", flush=True)

lengths_sorted = sorted(LENGTHS)
increments = {}
prev = None
for L in lengths_sorted:
    if prev is not None:
        increments[L] = recovery[L] - recovery[prev]
    prev = L


def min_length_for(threshold):
    for L in lengths_sorted:
        if recovery[L] >= threshold:
            return L
    return None


min_len_08 = min_length_for(0.8)
min_len_09 = min_length_for(0.9)
largest_step_L = max(increments, key=lambda L: increments[L])
print("min length for 0.8:", min_len_08, " for 0.9:", min_len_09,
      " largest step at", largest_step_L, "+", round(increments[largest_step_L], 4),
      flush=True)

# ---- タンパク質ごとの回復率 ----
per_protein = {}
for L in lengths_sorted:
    key = f"prefix_mean_{L:.1f}ns"
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

# ---- 信頼区間（資料Dの流儀: 統合・1mix_A・2qt4_A の3値の最小〜最大） ----
intervals = {}
for L in lengths_sorted:
    three = [recovery[L]] + [per_protein[L][p]["recovery"] for p in TEST_P]
    intervals[L] = {"pooled": recovery[L],
                    "per_protein": {p: per_protein[L][p]["recovery"] for p in TEST_P},
                    "ci_lower": float(min(three)), "ci_upper": float(max(three))}
    print(f"covered {L:.1f} ns interval [{intervals[L]['ci_lower']:+.4f}, "
          f"{intervals[L]['ci_upper']:+.4f}]", flush=True)

out = {
    "label": f"Contact formation {HORIZON_NS}ns, guidance recovery vs short-MD covered length (leverage curve)",
    "phenomenon": "contact formation (Doc D, k13_cache)",
    "experiment": "L16",
    "horizon_ns": HORIZON_NS,
    "length_definition": {
        "unit": "covered length (ns); prefix = frames 0..(n_frames-1) of the 0-1 ns history, 100 ps spacing (L01 definition)",
        "lengths": {f"{L:.1f}": {"covered_ns": L, "n_frames": nf,
                                 "frames": list(range(nf))}
                    for L, nf in LENGTHS.items()},
    },
    "metrics": results,
    "base_gain": base_gain,
    "recovery_vs_real_history_gain": {f"{L:.1f}": recovery[L] for L in lengths_sorted},
    "recovery_increment_per_0.1ns": {f"{L:.1f}": increments[L] for L in sorted(increments)},
    "readings_fixed_in_advance": {
        "min_length_for_recovery_0.8_pooled": min_len_08,
        "min_length_for_recovery_0.9_pooled": min_len_09,
        "largest_increment": {"covered_ns": largest_step_L,
                              "increment": increments[largest_step_L]},
        "intervals_3value_min_max": {f"{L:.1f}": intervals[L] for L in lengths_sorted},
    },
    "per_protein": {f"{L:.1f}": per_protein[L] for L in lengths_sorted},
    "checks_vs_K27": {
        "static_only_auprc_expected": EXPECTED["static_only"],
        "full_hist6_auprc_expected": EXPECTED["base_hist_D6"],
        "mean_full_auprc_expected": EXPECTED["mean_full"]},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["covered_ns", "n_frames", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_mean", "auc_prefix_mean",
                "brier_prefix_mean", "logloss_prefix_mean", "full_gain",
                "recovery_pooled", "recovery_1mix_A", "recovery_2qt4_A",
                "interval_lower", "interval_upper",
                "increment_from_previous_0.1ns"])
    for L in lengths_sorted:
        key = f"prefix_mean_{L:.1f}ns"
        m, iv = results[key], intervals[L]
        w.writerow([L, LENGTHS[L], m["positives"],
                    results["static_only"]["auprc"], results["base_hist_D6"]["auprc"],
                    m["auprc"], m["auc"], m["brier"], m["logloss"], base_gain,
                    recovery[L], per_protein[L]["1mix_A"]["recovery"],
                    per_protein[L]["2qt4_A"]["recovery"],
                    iv["ci_lower"], iv["ci_upper"], increments.get(L, "")])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    y=yte.astype(np.int8),
    p_static_only=preds["static_only"].astype(np.float32),
    p_base_hist_D6=preds["base_hist_D6"].astype(np.float32),
    **{f"p_prefix_mean_{L:.1f}ns": preds[f"prefix_mean_{L:.1f}ns"].astype(np.float32)
       for L in lengths_sorted},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
