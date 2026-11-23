"""G15: 混合の効き目は2つの推定の誤差の打ち消し合いか
（資料Dの条件、水素結合の形成、1 ns先）.

G11 では切り取り平均と生成平均の 2つを別々の特徴量として入れ、
回復率 0.756 でした（切り取り平均だけ 0.570、生成平均だけ 0.093）。
こんどは 2つの平均の単純な平均を 1つの特徴量として入れ、同じ補正モデルで測る。

条件（1 ns先、すべて条件ごとに学習し直す）:
  - static_only: 静的特徴7個だけ
  - base_hist_D6: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム、参照）
  - prefix_mean_0.1ns: 静的特徴＋切り取り平均だけ（フレーム [0,1]、G02/G11 の再現）
  - gen_mean_only: 静的特徴＋生成平均だけ（K05 第22案の生成5点系列の平均）
  - mixed: 静的特徴＋切り取り平均＋生成平均（2列、G11 の再現）
  - avg_mean: 静的特徴＋(切り取り平均＋生成平均)/2（1列、主条件）
  - diff_mean（参考）: 静的特徴＋(切り取り平均 − 生成平均)（1列）

回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得 +0.1178)。
判定（先に固定）: 単純な平均だけで回復率 0.70 以上なら、効き目の大半は
2つの推定の誤差の打ち消し合い。0.70 未満なら、モデルは 2つの推定の
食い違いそのものを手がかりにしていることになる。
G11 の混合（0.756）と切り取り平均だけ（0.570）はそのまま使ってかまわないが、
ここでは同じコードで再計算して照合する（値が G11 と一致することを確認）。

生成平均の出所: K05 の generated_series.npz（第22案の5点系列）。
  生成5点 = [起点距離 d0（フレーム10）, +0.1, +0.3, +0.5, +1.0 ns] の前方系列。
  対応付けのキーは (軌道ID, ドナー原子, アクセプター原子) で、K09 と同じ。
  生成器のチェックポイント .pt ファイルは読み込まない。
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 0  # 1 ns先
PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り（G02 と同じ）


def load_gen5():
    """生成5点系列: [d0, gen(+0.1,+0.3,+0.5,+1.0 ns)]。キーは (tid, donor, acceptor)。"""
    z = np.load(GEN)
    tids, pair, gen = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
    assert gen5.shape == (14582, 5), gen5.shape
    return {(str(t), int(pair[i, 0]), int(pair[i, 1])): gen5[i]
            for i, t in enumerate(tids)}


GEN5 = load_gen5()


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            g = np.stack([GEN5[(tid, int(da), int(aa))]
                          for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["gen5"] = g
            parts.append(d)
    return parts


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


train = load(TRAIN_P)
test = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Gtr, Gte = cat(train, "gen5"), cat(test, "gen5")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)

# 対応付けの検証: 生成5点の最初の点 = 起点距離 = 履歴の最終フレーム
print("gen5 first vs hist last: max abs diff train %.2e test %.2e" % (
    np.abs(Gtr[:, 0] - Htr[:, -1]).max(), np.abs(Gte[:, 0] - Hte[:, -1]).max()), flush=True)
assert np.abs(Gtr[:, 0] - Htr[:, -1]).max() < 1e-5
assert np.abs(Gte[:, 0] - Hte[:, -1]).max() < 1e-5

full_mean_tr = Htr.mean(1)
full_mean_te = Hte.mean(1)
prefix_mean_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_mean_te = Hte[:, PREFIX_FRAMES].mean(1)
gen_mean_tr = Gtr.mean(1)
gen_mean_te = Gte.mean(1)
avg_mean_tr = 0.5 * (prefix_mean_tr + gen_mean_tr)
avg_mean_te = 0.5 * (prefix_mean_te + gen_mean_te)
diff_mean_tr = prefix_mean_tr - gen_mean_tr
diff_mean_te = prefix_mean_te - gen_mean_te

# 特徴量の誤差（0〜1 ns の実平均との差の標準偏差）
errors = {}
for split, pm, gm, am, fm in (("train", prefix_mean_tr, gen_mean_tr, avg_mean_tr, full_mean_tr),
                              ("test", prefix_mean_te, gen_mean_te, avg_mean_te, full_mean_te)):
    errors[split] = dict(
        n=int(len(pm)),
        prefix_minus_full_mean=float(np.mean(pm - fm)),
        prefix_minus_full_std=float(np.std(pm - fm)),
        gen_minus_full_mean=float(np.mean(gm - fm)),
        gen_minus_full_std=float(np.std(gm - fm)),
        avg_minus_full_mean=float(np.mean(am - fm)),
        avg_minus_full_std=float(np.std(am - fm)),
        corr_prefix_gen=float(np.corrcoef(pm, gm)[0, 1]),
        corr_avg_full=float(np.corrcoef(am, fm)[0, 1]),
        corr_prefix_full=float(np.corrcoef(pm, fm)[0, 1]),
        corr_gen_full=float(np.corrcoef(gm, fm)[0, 1]),
    )
    print(split, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in errors[split].items()}, flush=True)

conditions = {}
conditions["static_only"] = None
conditions["base_hist_D6"] = (hist6(Htr), hist6(Hte))
conditions["prefix_mean_0.1ns"] = (
    prefix_mean_tr[:, None].astype(np.float32),
    prefix_mean_te[:, None].astype(np.float32))
conditions["gen_mean_only"] = (
    gen_mean_tr[:, None].astype(np.float32),
    gen_mean_te[:, None].astype(np.float32))
conditions["mixed"] = (
    np.stack([prefix_mean_tr, gen_mean_tr], axis=1).astype(np.float32),
    np.stack([prefix_mean_te, gen_mean_te], axis=1).astype(np.float32))
conditions["avg_mean"] = (
    avg_mean_tr[:, None].astype(np.float32),
    avg_mean_te[:, None].astype(np.float32))
conditions["diff_mean"] = (
    diff_mean_tr[:, None].astype(np.float32),
    diff_mean_te[:, None].astype(np.float32))

results = {}
for cname, feats in conditions.items():
    if feats is None:
        p = predict(Xtr, ytr, Xte)
    else:
        ftr, fte = feats
        p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                    np.concatenate([Xte, fte], axis=1))
    results[cname] = metrics(yte, p)
    print(cname, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in results[cname].items()}, flush=True)

# 土台の再現の確認
base_gain = results["base_hist_D6"]["auprc"] - results["static_only"]["auprc"]
print("base gain (should be ~0.1178):", round(base_gain, 4), flush=True)
assert abs(results["static_only"]["auprc"] - 0.3637) < 5e-4, results["static_only"]
assert abs(results["base_hist_D6"]["auprc"] - 0.4815) < 5e-4, results["base_hist_D6"]
# G11 の値との照合（混合 0.756、切り取り平均だけ 0.570、生成平均だけ 0.093）
assert abs(results["prefix_mean_0.1ns"]["auprc"] - 0.4308) < 5e-4, results["prefix_mean_0.1ns"]
assert abs(results["gen_mean_only"]["auprc"] - 0.3746) < 5e-4, results["gen_mean_only"]
assert abs(results["mixed"]["auprc"] - 0.4528) < 5e-4, results["mixed"]

recovery = {}
for cname, m in results.items():
    if cname == "static_only":
        continue
    recovery[cname] = float((m["auprc"] - results["static_only"]["auprc"]) / base_gain)

print("recovery:", {k: round(v, 4) for k, v in recovery.items()}, flush=True)
print("avg_mean recovery: %.4f (threshold 0.70)" % recovery["avg_mean"], flush=True)
print("mixed - avg_mean recovery diff: %.4f" % (recovery["mixed"] - recovery["avg_mean"]), flush=True)
print("avg_mean - prefix_mean recovery diff: %.4f" % (recovery["avg_mean"] - recovery["prefix_mean_0.1ns"]), flush=True)

out = {
    "prefix_frames": PREFIX_FRAMES,
    "prefix_label": "0.1 ns",
    "generated_mean_source": "K05 generated_series.npz (method 22, 5-point forward series [d0, +0.1, +0.3, +0.5, +1.0 ns])",
    "avg_mean_definition": "(prefix_mean + gen_mean) / 2, single feature",
    "diff_mean_definition": "prefix_mean - gen_mean, single feature (reference)",
    "feature_errors_vs_full_mean": errors,
    "references": {
        "real_history_gain_1ns": 0.1178,
        "g11_mixed_recovery": 0.756,
        "g11_prefix_mean_recovery_0.1ns": 0.570,
        "g11_gen_mean_recovery": 0.093,
        "g02_prefix_minus_full_std_test_A": 0.396,
        "k08_gen_minus_full_std_A": 0.48,
    },
    "metrics_1ns": results,
    "base_gain_1ns": base_gain,
    "recovery_vs_real_history_gain": recovery,
    "mixed_minus_avg_recovery": recovery["mixed"] - recovery["avg_mean"],
    "avg_minus_prefix_recovery": recovery["avg_mean"] - recovery["prefix_mean_0.1ns"],
    "decision_threshold": 0.70,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

import csv
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss", "recovery_vs_real_history_gain"])
    for cname, m in results.items():
        w.writerow([cname, 1, m["n"], m["positives"], m["auc"], m["auprc"], m["brier"], m["logloss"],
                    "" if cname == "static_only" else recovery[cname]])
print("saved", OUT / "results.json")
