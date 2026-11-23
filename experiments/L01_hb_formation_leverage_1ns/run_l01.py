"""L01: 短いMDの長さに対して、指導力がどれだけ戻るかの曲がり具合
（資料Dの条件、水素結合の形成、1 ns先）.

L 系列の最初の実験。実の 0〜1 ns の履歴を始めの短い部分だけに切って使い、
切り取りの長さを 0.1 ns 刻みで細かく変えたときの、平均だけの回復率の
曲がり具合を測る。G02 では 0.1 / 0.25 / 0.5 ns の3通りだけだったのを、
11通りに増やす。

長さの呼び方（結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 長さは「覆う長さ」で統一する。覆う長さ L ns の切り取りは
  フレーム 0〜(L/0.1) を使う（フレーム数は L/0.1 + 1）。
  - 覆う長さ 0 ns   -> フレーム 0 のみ（1フレーム。起点の距離そのもの）
  - 覆う長さ 0.1 ns -> フレーム 0, 1（2フレーム）
  - ...
  - 覆う長さ 1.0 ns -> フレーム 0〜10（11フレーム。0〜1 ns の全部）
- G02 の呼び方との対応:
  - G02 の 0.1 ns（フレーム 0, 1）   = 覆う長さ 0.1 ns
  - G02 の 0.25 ns（フレーム 0〜2）  = 覆う長さ 0.2 ns
  - G02 の 0.5 ns（フレーム 0〜5）   = 覆う長さ 0.5 ns
  - G02 の「平均だけ（0〜1 ns）」    = 覆う長さ 1.0 ns
- 起点フレーム10（1 ns）は変えない。ラベルは 1 ns先（フレーム20）。

条件:
- 静的のみ: 静的特徴7個だけ。
- 全部入り（base_hist_D6）: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム）。
- 切り取りの平均だけ: 静的特徴＋切り取り（フレーム 0〜k）の距離の平均だけ。
  覆う長さ 0.0〜1.0 ns の 11通り。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。
実履歴の利得は土台の再現から +0.1178 になるはず（資料Dと同じ）。

読み方（結果を見る前に固定）:
- 回復率 0.8 に届く最小の長さと、0.9 に届く最小の長さを出す。
- 0.1 ns 刻みごとの回復率の増え分を出す。分母はシミュレーションの長さ
  （0.1 ns 長くしたときに回復率がどれだけ増えるかで読む）。
- 実測の費用への換算はしない（G04 の後にする）。

ブートストラップ（結果を見る前に固定、G49・G59 と同じ）:
- 回数: 10,000回。乱数 seed: 42。区間はパーセンタイル法。
- 再標本化の単位: タンパク質。テストは 1mix_A と 2qt4_A の2つだけなので、
  取りうる組み合わせは3通りしかなく、分布も3つの値しか取らない。
  95% 区間はその3値の最小値〜最大値になる。区間は粗い（留保として記録）。
- 各標本の中でも静的のみ・全部入り・その長さの平均だけの AUPRC を測り直し、
  利得と回復率をその標本の中で計算し直す。モデルは学習し直さず、
  候補ごとの予測確率を再標本化する（テスト集合の取り方の揺れだけを見る）。
- 計算の工夫は G49 と同じ: 3通りの組み合わせの値を正確に計算し、
  10,000回の抽出は組み合わせへの写像として行う。

土台の再現の期待値: 1 ns先で静的のみ AUPRC 0.3637、全部入り 0.4815、
利得 +0.1178。G02 との照合: 覆う長さ 0.1 ns で回復率 0.570（AUPRC 0.4308）、
0.2 ns で 0.647（0.4399）、0.5 ns で 0.770（0.4544）、
1.0 ns で 1.042（0.4865、G02 の「平均だけ（0〜1 ns）」）。
"""
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 0  # 1 ns先
N_BOOT = 10000  # 結果を見る前に固定（G49・G59 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G49・G59 と同じ）

# 覆う長さ（ns）-> フレーム数。フレーム 0〜(n-1) を使う。
LENGTHS = {round(0.1 * k, 1): k + 1 for k in range(11)}

# G02 の記録との照合用の期待値（覆う長さ -> (AUPRC, 回復率)）
EXPECTED_G02 = {
    0.1: (0.4308, 0.570),
    0.2: (0.4399, 0.647),   # G02 の「0.25 ns」（実際に覆う長さ 0.2 ns）
    0.5: (0.4544, 0.770),
    1.0: (0.4865, 1.042),   # G02 の「平均だけ（0〜1 ns）」
}


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)


def hist6(series, bound=3.5):
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
print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)

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

# 土台の再現の確認
base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
print("base gain (should be ~0.1178):", round(base_gain, 4), flush=True)
assert abs(results["static_only"]["auprc"] - 0.3637) < 5e-4, results["static_only"]
assert abs(results["base_hist_D6"]["auprc"] - 0.4815) < 5e-4, results["base_hist_D6"]
assert abs(base_gain - 0.1178) < 5e-4, base_gain

# ---- 回復率と増え分 ----
recovery = {}
for L in LENGTHS:
    key = f"prefix_mean_{L:.1f}ns"
    recovery[L] = float((results[key]["auprc"] - results["static_only"]["auprc"]) / base_gain)
    print(f"covered {L:.1f} ns recovery {recovery[L]:.4f}", flush=True)

# G02 の記録との照合
for L, (exp_auprc, exp_rec) in EXPECTED_G02.items():
    key = f"prefix_mean_{L:.1f}ns"
    assert abs(results[key]["auprc"] - exp_auprc) < 5e-4, (L, results[key]["auprc"], exp_auprc)
    assert abs(recovery[L] - exp_rec) < 2e-3, (L, recovery[L], exp_rec)
print("G02 cross-check OK (covered 0.1/0.2/0.5/1.0 ns)", flush=True)

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
print("min length for 0.8:", min_len_08, " for 0.9:", min_len_09, flush=True)

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

# ---- タンパク質単位のブートストラップ（G49・G59 と同じ） ----
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))
rng = np.random.default_rng(BOOT_SEED)
combo_idx = rng.integers(0, len(COMBOS), size=N_BOOT)


def recovery_in_sample(idx, p_prefix):
    a_s = auprc(yte[idx], preds["static_only"][idx])
    a_f = auprc(yte[idx], preds["base_hist_D6"][idx])
    a_p = auprc(yte[idx], p_prefix[idx])
    g = a_f - a_s
    if g == 0:
        return float("nan")
    return (a_p - a_s) / g


boot = {}
for L in lengths_sorted:
    key = f"prefix_mean_{L:.1f}ns"
    three = {}
    for combo in COMBOS:
        idx = np.concatenate([prot_idx[prot] for prot in combo])
        three["+".join(combo)] = recovery_in_sample(idx, preds[key])
    vals = np.array([three["+".join(COMBOS[i])] for i in combo_idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    boot[L] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": recovery[L],
        "ci_lower": float(lo), "ci_upper": float(hi),
        "three_resample_values": three,
        "n_distinct_boot_values": int(len(np.unique(np.round(vals, 12)))),
    }
    print(f"covered {L:.1f} ns bootstrap: point {recovery[L]:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] distinct={boot[L]['n_distinct_boot_values']}",
          flush=True)

out = {
    "label": "HB formation 1ns, guidance recovery vs short-MD covered length (leverage curve)",
    "horizon_ns": 1,
    "length_definition": {
        "unit": "covered length (ns); prefix = frames 0..(n_frames-1) of the 0-1 ns history, 100 ps spacing",
        "lengths": {f"{L:.1f}": {"covered_ns": L, "n_frames": nf,
                                 "frames": list(range(nf))}
                    for L, nf in LENGTHS.items()},
        "g02_mapping": {
            "G02 0.1 ns (frames 0,1)": "covered 0.1 ns",
            "G02 0.25 ns (frames 0,1,2; actually covers 0.2 ns)": "covered 0.2 ns",
            "G02 0.5 ns (frames 0..5)": "covered 0.5 ns",
            "G02 mean_only_full (0-1 ns mean)": "covered 1.0 ns",
        },
    },
    "references": {
        "real_history_gain_1ns": 0.1178,
        "note_on_K21": "K21's mean-only values (1 ns: 0.957 macro / 0.975 micro) are under the "
                       "strong base K06(b) with a different reference gain (+0.0712), so they are "
                       "not directly comparable to this D-condition curve. Under the D condition "
                       "the full-length mean-only recovery is 1.042 (G02, K64).",
        "cost_conversion": "Not done in L01. Conversion to measured compute cost comes after G04.",
    },
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (1mix_A, 2qt4_A; 2 proteins only, so only 3 "
                  "resample combinations exist and the bootstrap distribution takes only "
                  "3 distinct values; intervals are coarse)",
                  "note": "Within each resample, static-only, full-history and prefix-mean "
                  "AUPRCs are recomputed and the recovery is recomputed inside the sample. "
                  "Models are not refit; per-candidate predicted probabilities are resampled. "
                  "The 10,000 draws are mapped onto the 3 exact combination values, which is "
                  "mathematically identical to recomputing AUPRC on the resampled rows "
                  "(the recovery depends only on the chosen combination)."},
    "metrics_1ns": results,
    "base_gain_1ns": base_gain,
    "recovery_vs_real_history_gain": {f"{L:.1f}": recovery[L] for L in lengths_sorted},
    "recovery_increment_per_0.1ns": {f"{L:.1f}": increments[L] for L in sorted(increments)},
    "min_length_for_recovery_0.8": min_len_08,
    "min_length_for_recovery_0.9": min_len_09,
    "per_protein": {f"{L:.1f}": per_protein[L] for L in lengths_sorted},
    "bootstrap_results": {f"{L:.1f}": boot[L] for L in lengths_sorted},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["covered_ns", "n_frames", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_mean", "auc_prefix_mean",
                "brier_prefix_mean", "logloss_prefix_mean", "full_gain",
                "recovery", "increment_from_previous_0.1ns",
                "boot_ci_lower", "boot_ci_upper"])
    for L in lengths_sorted:
        key = f"prefix_mean_{L:.1f}ns"
        m, b = results[key], boot[L]
        w.writerow([L, LENGTHS[L], m["positives"],
                    results["static_only"]["auprc"], results["base_hist_D6"]["auprc"],
                    m["auprc"], m["auc"], m["brier"], m["logloss"], base_gain,
                    recovery[L], increments.get(L, ""), b["ci_lower"], b["ci_upper"]])

# 候補ごとの予測確率は再利用のために保存する（ブートストラップの再現用）
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    y_1ns=yte,
    p_static_only=preds["static_only"],
    p_base_hist_D6=preds["base_hist_D6"],
    **{f"p_prefix_mean_{L:.1f}ns": preds[f"prefix_mean_{L:.1f}ns"] for L in lengths_sorted},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
