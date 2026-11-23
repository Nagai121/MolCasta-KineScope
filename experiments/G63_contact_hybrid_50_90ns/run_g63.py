"""G63: 接触の形成で、50 ns先と 90 ns先の混合は足しになるか
(資料Dの条件、接触の形成、50 ns先・90 ns先)。

接触の形成の混合で、残っている予測幅を埋める。これまでの結果は、
混合の回復率が切り取り平均だけより +0.05 を超えた案の数が、1 ns先で
4 案中 3 案、5 ns先で 1 案 (G36、第11案 +0.057)、10 ns先で 1 案
(G42、第13案 +0.053)、30 ns先で 0 案 (G55)。

生成平均は K17 の残基レベル生成系列 (第11〜14案) の branch-mean
9点系列を 9点で平均したもの (K17・G36・G42・G55 と同じ)。実側は
K13 キャッシュ (静的特徴5個、11フレーム履歴、ラベル) で、K17 の
ファイルに保存された実側と配列単位で一致することを確かめてから、
位置で結合する。生成器は読み込まない。新しい軌道も生成しない。

条件 (50 ns先と 90 ns先、HistGradientBoosting 120回、学習率0.075、
葉31、葉あたり最小15件、L2 0.02、seed 42、条件ごとに学習し直す):
  static_only、full_hist (静的特徴＋実履歴6統計量、閾値 4.5 Å)、
  prefix_mean_0.1ns (静的特徴＋フレーム [0,1] の平均)、
  gen_mean_only_p<n> (静的特徴＋第 n 案の生成平均)、
  mixed_p<n> (静的特徴＋切り取り平均＋第 n 案の生成平均)、
n は 11〜14。回復率 = (AUPRC − 静的のみ) / (全部入り − 静的のみ)。
土台の利得は 50 ns先で +0.203 (静的のみ 0.2604、全部入り 0.4638)、
90 ns先で +0.185 (静的のみ 0.2872、全部入り 0.4722)、どちらも K16。

読み方 (先に固定): これまでと同じく、混合の回復率が切り取り平均
だけより +0.05 を超える案の数で見る。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")
K17 = Path("/home/hatch/workspace/k17/out")
HERE = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
HORIZONS = {50: 4, 90: 5}  # ns -> y の列
REF = {50: (0.2604, 0.4638), 90: (0.2872, 0.4722)}  # K16 の (静的のみ, 全部入り)
PROPOSALS = (11, 12, 13, 14)


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                       max_leaf_nodes=31, min_samples_leaf=15,
                                       l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def load(prots):
    """実側は K13 キャッシュ、生成平均は K17 のファイル。実側が配列単位で
    一致することを確かめてから位置で結合する。"""
    real, gen = [], {n: [] for n in PROPOSALS}
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K13 / f"{p}_R{r}_contact.npz")
            k = np.load(K17 / f"{p}_R{r}.npz")
            assert np.array_equal(k["X"], z["X"]) and np.array_equal(k["hist"], z["hist"]) \
                and np.array_equal(k["y"], z["y"]), f"real-side mismatch {p} R{r}"
            real.append({kk: z[kk] for kk in z.files})
            for n in PROPOSALS:
                # branch-mean 9点系列 -> 9点で平均
                gen[n].append(k[f"gen{n}"].mean(axis=0).mean(axis=0).astype(np.float64))
    return real, gen


train, gtr = load(TRAIN_P)
test, gte = load(TEST_P)
Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
Htr = np.concatenate([d["hist"] for d in train])
Hte = np.concatenate([d["hist"] for d in test])
Ytr = np.concatenate([d["y"] for d in train])
Yte = np.concatenate([d["y"] for d in test])
print("train", Xtr.shape, "test", Xte.shape, flush=True)

prefix_tr = Htr[:, [0, 1]].mean(1)
prefix_te = Hte[:, [0, 1]].mean(1)
full_mean_tr, full_mean_te = Htr.mean(1), Hte.mean(1)
gen_tr = {n: np.concatenate(gtr[n]) for n in PROPOSALS}
gen_te = {n: np.concatenate(gte[n]) for n in PROPOSALS}

conds = {
    "static_only": (Xtr, Xte),
    "full_hist": (np.column_stack([Xtr, hist6(Htr)]),
                  np.column_stack([Xte, hist6(Hte)])),
    "prefix_mean_0.1ns": (np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32)]),
                          np.column_stack([Xte, prefix_te[:, None].astype(np.float32)])),
}
for n in PROPOSALS:
    conds[f"gen_mean_only_p{n}"] = (
        np.column_stack([Xtr, gen_tr[n][:, None].astype(np.float32)]),
        np.column_stack([Xte, gen_te[n][:, None].astype(np.float32)]))
    conds[f"mixed_p{n}"] = (
        np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32),
                         gen_tr[n][:, None].astype(np.float32)]),
        np.column_stack([Xte, prefix_te[:, None].astype(np.float32),
                         gen_te[n][:, None].astype(np.float32)]))

# 生成平均と実履歴の関係 (テスト側、G55 と同じ形)
rel = {}
for n in PROPOSALS:
    g, pm, fm = gen_te[n], prefix_te, full_mean_te
    rel[f"p{n}"] = dict(
        corr_prefix_gen=float(np.corrcoef(pm, g)[0, 1]),
        gen_minus_prefix_mean=float((g - pm).mean()),
        gen_minus_prefix_std=float((g - pm).std()),
        gen_minus_full_mean=float((g - fm).mean()),
        gen_minus_full_std=float((g - fm).std()),
        corr_gen_full=float(np.corrcoef(g, fm)[0, 1]),
        corr_prefix_full=float(np.corrcoef(pm, fm)[0, 1]))

results = {"relations_test": rel}
rows = []
for ns, h_idx in HORIZONS.items():
    ytr, yte = Ytr[:, h_idx], Yte[:, h_idx]
    out = {}
    for name, (xtr, xte) in conds.items():
        p = predict(xtr, ytr, xte)
        out[name] = metrics(yte, p)
        print(f"{ns}ns {name} AUPRC %.4f AUC %.4f" % (out[name]["auprc"], out[name]["auc"]),
              flush=True)
        (HERE / f"results_partial_{ns}ns.json").write_text(json.dumps(out, indent=2))

    s = out["static_only"]["auprc"]
    f = out["full_hist"]["auprc"]
    ref_s, ref_f = REF[ns]
    print("base check %dns: static %.4f (recorded %.4f), full %.4f (recorded %.4f), "
          "gain %.4f" % (ns, s, ref_s, f, ref_f, f - s), flush=True)
    assert abs(s - ref_s) < 1e-4 and abs(f - ref_f) < 1e-4, f"base mismatch {ns}ns"

    recovery = {name: float((m["auprc"] - s) / (f - s)) for name, m in out.items()}
    delta = {f"p{n}": recovery[f"mixed_p{n}"] - recovery["prefix_mean_0.1ns"]
             for n in PROPOSALS}
    results[f"{ns}ns"] = {"conditions": out, "recovery_vs_full_hist": recovery,
                          "base_gain": float(f - s),
                          "prefix_recovery": recovery["prefix_mean_0.1ns"],
                          "mixed_delta_vs_prefix": delta,
                          "n_proposals_above_0.05": int(sum(v > 0.05 for v in delta.values()))}
    for name, m in out.items():
        rows.append(dict(horizon_ns=ns, condition=name, recovery=recovery[name], **m))
    print(f"{ns}ns recovery:", {n: round(v, 3) for n, v in recovery.items()}, flush=True)
    print(f"{ns}ns mixed delta vs prefix:", {n: round(v, 3) for n, v in delta.items()},
          "above +0.05:", results[f"{ns}ns"]["n_proposals_above_0.05"], flush=True)

(HERE / "results.json").write_text(json.dumps(results, indent=2))
with open(HERE / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", HERE / "results.json", flush=True)
