"""G50: 解離で、50 ns先の混合は足しになるか
（資料Dの条件、水素結合の解離、50 ns先）.

形成では G11 で、0.1 ns の切り取り平均に第22案の生成平均を足すと回復率が
0.570 から 0.756 へ +0.186 上がった。解離では G18 で 1 ns先が −0.052、
G44 で 30 ns先が +0.086 だった。50 ns先はまだ測っていない。

作り直しの経緯: 第22案の生成系列（K05 の generated_series.npz）は起点距離が
3.5〜6.0 A の形成候補だけで、解離の候補（起点 3.5 A 以下）には1件も対応
しなかった。そのため、Echizen の判断（選択肢 (a)）で第22案の生成系列を
解離の候補向けに作り直した（gen_g18.py）。生成器のチェックポイントは
weights_only で読み込み、読み込む前に generator.json に記録されたハッシュ
と照合した（K05・K74 と同じ決め事）。候補選びの絞りは、起点の距離が
3.5 A より大きいペアだけを残す作りから、すべての残基をまたぐドナー・
アクセプターのペアを残す作りに変えた（`expand_all`、K54 の gen_k54.py と
同じ変更）。変えたのはその1行だけ。
注意: 生成器は形成の候補だけで学習しているため、解離の候補への当てはめは
学習した範囲の外への当てはめになる。そのことは README に明記する。

条件（50 ns先、すべて条件ごとに学習し直す）:
  - static_only: 静的特徴7個だけ
  - base_hist_D6: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム、参照）
  - prefix_mean_0.1ns: 静的特徴＋切り取り平均だけ（フレーム [0,1]、G02/G06 と同じ）
  - gen_mean_only: 静的特徴＋生成平均だけ（作り直した第22案の生成5点系列の平均）
  - mixed: 静的特徴＋切り取り平均＋生成平均（2列）

生成5点 = [起点距離 d0（フレーム10）, gen(+0.1,+0.3,+0.5,+1.0 ns)] の前方系列。
対応付けのキーは (軌道ID, ドナー原子, アクセプター原子) で、G11 と同じ。
実履歴は起点までの後方系列で、生成系列は起点から先への前方系列。両者は
起点距離 d0 を共有するが時間の向きが逆（G11 と同じ留保）。平均は向きに
依らない量なので、平均の列だけを使うこの実験では向きの違いは平均の定義に
影響しない。

回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得 +0.1142)。
判定（先に固定）: 混合の回復率が切り取り平均だけより +0.05 以上上がれば、
混合は解離でも 50 ns先で足しになる。+0.05 未満なら、混合の効き目は
解離の 50 ns先では出ない。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
GEN_DIR = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
OUT = Path("/home/hatch/workspace/g50")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 4  # 50 ns先（予測幅 1/5/10/30/50/90 ns）
PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り（G02/G06 と同じ）


def load_gen5():
    """作り直した生成5点系列: [d0, gen(+0.1,+0.3,+0.5,+1.0 ns)]。
    キーは (tid, donor, acceptor)。"""
    gen = {}
    for f in sorted(GEN_DIR.glob("atlas-*.npz")):
        z = np.load(f)
        tid = f.stem
        pair, gd = z["pair"], z["gen_dist"]
        gen5 = np.column_stack([pair[:, 2].astype(np.float64),
                                gd.astype(np.float64)])
        assert gen5.shape[1] == 5
        for i in range(len(pair)):
            gen[(tid, int(pair[i, 0]), int(pair[i, 1]))] = gen5[i]
    print("regenerated pairs:", len(gen), flush=True)
    return gen


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


def quality(gen_parts, real_parts):
    """作り直した生成系列の質の確認（K54 の gen_quality と同じ量、
    境界 3.5 A、解離の候補）。生成5点と実の前方5点 [d0, real_sub] を比べる。"""
    out = {}
    for tag, series in (("generated", gen_parts), ("real_forward", real_parts)):
        s = np.concatenate(series)
        out[tag] = dict(
            n=int(len(s)),
            mean=float(s.mean()),
            std=float(s.std()),
            per_pair_std=float(s.std(1).mean()),
            per_pair_range=float((s.max(1) - s.min(1)).mean()),
            frac_le35=float((s <= 3.5).mean()),
            ever_le35=float((s <= 3.5).any(1).mean()),
            crossings_per_pair=float(np.abs(np.diff((s <= 3.5).astype(int), axis=1)).sum(1).mean()),
            drift=float((s[:, -1] - s[:, 0]).mean()),
        )
    return out


train = load(TRAIN_P)
test = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Gtr, Gte = cat(train, "gen5"), cat(test, "gen5")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos 30ns", int(yte.sum()), flush=True)

# 対応付けの検証: 生成5点の最初の点 = 起点距離 = 履歴の最終フレーム
print("gen5 first vs hist last: max abs diff train %.2e test %.2e" % (
    np.abs(Gtr[:, 0] - Htr[:, -1]).max(), np.abs(Gte[:, 0] - Hte[:, -1]).max()), flush=True)
assert np.abs(Gtr[:, 0] - Htr[:, -1]).max() < 1e-5
assert np.abs(Gte[:, 0] - Hte[:, -1]).max() < 1e-5

# 質の確認: 生成5点と実の前方5点 [d0, real_sub]（解離の候補 6,131件）
gen_q = []
real_q = []
for prots in (TRAIN_P, TEST_P):
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(GEN_DIR / f"atlas-{p}-analysis-R{r}.npz")
            pair, gd = z["pair"], z["gen_dist"]
            c = np.load(CACHE / f"{p}_R{r}.npz")
            tid = f"atlas-{p}-analysis-R{r}"
            # 直接の対応付けで取り直す
            pos = {(int(pair[j, 0]), int(pair[j, 1])): j for j in range(len(pair))}
            rows = [pos[(int(da), int(aa))]
                    for da, aa in zip(c["donor_atom"], c["acceptor_atom"])]
            gen_q.append(np.column_stack(
                [pair[rows, 2].astype(np.float64), gd[rows].astype(np.float64)]))
            real_q.append(np.column_stack(
                [pair[rows, 2].astype(np.float64), z["real_sub"][rows].astype(np.float64)]))
q = quality(gen_q, real_q)
print("quality:", json.dumps(q, indent=1), flush=True)

full_mean_tr = Htr.mean(1)
full_mean_te = Hte.mean(1)
prefix_mean_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_mean_te = Hte[:, PREFIX_FRAMES].mean(1)
gen_mean_tr = Gtr.mean(1)
gen_mean_te = Gte.mean(1)

# 切り取り平均と生成平均の関係（テスト側、学習側も併記）
relations = {}
for split, pm, gm, fm in (("train", prefix_mean_tr, gen_mean_tr, full_mean_tr),
                          ("test", prefix_mean_te, gen_mean_te, full_mean_te)):
    d_pg = gm - pm   # 生成平均 - 切り取り平均
    d_gf = gm - fm   # 生成平均 - 0〜1 ns の平均（K08 と同じ量）
    d_pf = pm - fm   # 切り取り平均 - 0〜1 ns の平均
    relations[split] = dict(
        n=int(len(pm)),
        corr_prefix_gen=float(np.corrcoef(pm, gm)[0, 1]),
        corr_gen_full=float(np.corrcoef(gm, fm)[0, 1]),
        corr_prefix_full=float(np.corrcoef(pm, fm)[0, 1]),
        gen_minus_prefix_mean=float(np.mean(d_pg)),
        gen_minus_prefix_std=float(np.std(d_pg)),
        gen_minus_full_mean=float(np.mean(d_gf)),
        gen_minus_full_std=float(np.std(d_gf)),
        prefix_minus_full_mean=float(np.mean(d_pf)),
        prefix_minus_full_std=float(np.std(d_pf)),
    )
    print(split, {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in relations[split].items()}, flush=True)

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
print("base gain (should be ~0.1142):", round(base_gain, 4), flush=True)
assert abs(results["static_only"]["auprc"] - 0.4088) < 5e-4, results["static_only"]
assert abs(results["base_hist_D6"]["auprc"] - 0.5231) < 5e-4, results["base_hist_D6"]
# K46 の 50 ns先の平均だけ（0〜1 ns の平均、AUPRC 0.5046）は切り取り平均とは
# 別の条件なので、ここでは再現の確認はしない。参考値として README に併記する。

recovery = {}
for cname, m in results.items():
    if cname == "static_only":
        continue
    recovery[cname] = float((m["auprc"] - results["static_only"]["auprc"]) / base_gain)

mixed_minus_prefix = recovery["mixed"] - recovery["prefix_mean_0.1ns"]
print("recovery:", {k: round(v, 4) for k, v in recovery.items()}, flush=True)
print("mixed - prefix_mean recovery diff: %.4f (threshold +0.05)" % mixed_minus_prefix, flush=True)

out = {
    "prefix_frames": PREFIX_FRAMES,
    "prefix_label": "0.1 ns",
    "generated_mean_source": "regenerated method-22 series for dissociation candidates (gen_g18.py, ~/workspace/kinescope/work/g18_gen22/, 18 trajectories, 20,713 pairs; 5-point forward series [d0, +0.1, +0.3, +0.5, +1.0 ns])",
    "coverage": {"dissociation_candidates": 6131, "missing": 0},
    "quality_generated_vs_real_forward": q,
    "relations": relations,
    "references": {
        "real_history_gain_50ns": 0.1142,
        "k46_full_mean_recovery_50ns": 0.838,
        "k46_full_mean_auprc_50ns": 0.5046,
        "g18_prefix_mean_recovery_1ns": 0.532,
        "g18_prefix_mean_auprc_1ns": 0.5109,
        "g18_mixed_minus_prefix_1ns": -0.052,
        "g44_mixed_minus_prefix_30ns": 0.086,
        "g11_formation_mixed_minus_prefix": 0.186,
    },
    "metrics_50ns": results,
    "base_gain_50ns": base_gain,
    "recovery_vs_real_history_gain": recovery,
    "mixed_minus_prefix_recovery": mixed_minus_prefix,
    "decision_threshold": 0.05,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss", "recovery_vs_real_history_gain"])
    for cname, m in results.items():
        w.writerow([cname, 50, m["n"], m["positives"], m["auc"], m["auprc"], m["brier"], m["logloss"],
                    "" if cname == "static_only" else recovery[cname]])
print("saved", OUT / "results.json")
