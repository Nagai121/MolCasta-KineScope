"""G128: 接触の形成で、1.0 ns の切り取りの 6 統計量に生成した平均を
足すとどうなるか (資料Dの条件、接触の形成、1 ns先・10 ns先・50 ns先・
90 ns先)。

G123 の続き。G123 では 5 ns先と 30 ns先を測った。ここでは残りの
予測幅（1・10・50・90 ns先）を測る。1.0 ns の切り取りの 6 統計量だけの
回復率は、接触の形成では G123（5 ns先 1.015、30 ns先 1.129）で初めて
測ったばかりで、1・10・50・90 ns先はここでの測定が初めての記録になる。

生成平均は K17 の残基レベル生成系列 (第11〜14案) の branch-mean
9点系列を 9点で平均したもの (K17・G31 と同じ)。実側は K13 キャッシュ
(静的特徴5個、11フレーム履歴、ラベル) で、K17 のファイルに保存された
実側と配列単位で一致することを確かめてから位置で結合する。生成器は
読み込まない。新しい軌道も生成しない。

フレームの注記: 軌道の保存間隔は 100 ps (0.1 ns) で、フレーム k は
時刻 k×0.1 ns。1.0 ns の切り取りはフレーム [0,1,2,3,4,5,6,7,8,9]
(10フレーム、0〜0.9 ns を覆う)。「1.0 ns」の切り取りは、呼び名の
1.0 ns に対して実際は 0.9 ns の範囲である。起点フレーム10 (1 ns) は
切り取りに入らない。全部入りと 1.0 ns の切り取りは、起点のフレーム10
が入るかどうかだけが違う。ラベルは 1 ns先 (フレーム20)、10 ns先
(フレーム110)、50 ns先 (フレーム510)、90 ns先 (フレーム910) のまま。

条件 (1 ns先・10 ns先・50 ns先・90 ns先、HistGradientBoosting 120回、
学習率0.075、葉31、葉あたり最小15件、L2 0.02、seed 42、条件ごとに
学習し直す):
  static_only、full_hist (静的特徴＋実履歴6統計量、閾値 4.5 Å)、
  prefix_hist6_1.0ns (静的特徴＋フレーム [0,1,2,3,4,5,6,7,8,9] の6統計量)、
  prefix_mean_1.0ns (静的特徴＋フレーム [0,1,2,3,4,5,6,7,8,9] の平均、参考)、
  gen_mean_only_p<n> (静的特徴＋第 n 案の生成平均)、
  mixed_p<n> (静的特徴＋1.0 ns の切り取りの6統計量＋第 n 案の生成平均)、
n は 11〜14。回復率 = (AUPRC − 静的のみ) / (全部入り − 静的のみ)。
土台の利得は 1 ns先で +0.227 (静的のみ 0.3398、全部入り 0.5669)、
10 ns先で +0.203 (静的のみ 0.3204、全部入り 0.5234)、
50 ns先で +0.203 (静的のみ 0.2604、全部入り 0.4638)、
90 ns先で +0.185 (静的のみ 0.2872、全部入り 0.4722)、いずれも
K16・K13・G75 と同じ。

読み方 (先に固定): 混合の回復率が 6 統計量の切り取りだけの回復率
より +0.05 を超えるかどうかで見る。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
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
HORIZONS = {1: 0, 10: 2, 50: 4, 90: 5}  # ns -> y の列
REF = {1: (0.3398, 0.5669), 10: (0.3204, 0.5234), 50: (0.2604, 0.4638), 90: (0.2872, 0.4722)}  # K16・G75 の (静的のみ, 全部入り)
# 1.0 ns 切り取りの6統計量だけの回復率は、接触の形成ではこれまで測られていない
# (G75 は 0.1・0.25・0.5 ns まで)。ここでの測定が初めての記録になるため、
# 先行する記録との照合は行わない。
PROPOSALS = (11, 12, 13, 14)
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り (0〜0.9 ns を覆う。起点フレーム10 は入らない)


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

prefix_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_te = Hte[:, PREFIX_FRAMES].mean(1)
gen_tr = {n: np.concatenate(gtr[n]) for n in PROPOSALS}
gen_te = {n: np.concatenate(gte[n]) for n in PROPOSALS}

conds = {
    "static_only": (Xtr, Xte),
    "full_hist": (np.column_stack([Xtr, hist6(Htr)]),
                  np.column_stack([Xte, hist6(Hte)])),
    "prefix_hist6_1.0ns": (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
                           np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])])),
    "prefix_mean_1.0ns": (np.column_stack([Xtr, prefix_tr[:, None].astype(np.float32)]),
                          np.column_stack([Xte, prefix_te[:, None].astype(np.float32)])),
}
for n in PROPOSALS:
    conds[f"gen_mean_only_p{n}"] = (
        np.column_stack([Xtr, gen_tr[n][:, None].astype(np.float32)]),
        np.column_stack([Xte, gen_te[n][:, None].astype(np.float32)]))
    conds[f"mixed_p{n}"] = (
        np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES]),
                         gen_tr[n][:, None].astype(np.float32)]),
        np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES]),
                         gen_te[n][:, None].astype(np.float32)]))

results = {}
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
    print("prefix %dns: prefix_hist6_1.0ns recovery %.4f (no prior record; first measurement)"
          % (ns, recovery["prefix_hist6_1.0ns"]), flush=True)
    delta = {f"p{n}": recovery[f"mixed_p{n}"] - recovery["prefix_hist6_1.0ns"]
             for n in PROPOSALS}
    delta_auprc = {f"p{n}": out[f"mixed_p{n}"]["auprc"] - out["prefix_hist6_1.0ns"]["auprc"]
                   for n in PROPOSALS}
    results[f"{ns}ns"] = {"conditions": out, "recovery_vs_full_hist": recovery,
                          "base_gain": float(f - s),
                          "prefix_hist6_recovery": recovery["prefix_hist6_1.0ns"],
                          "mixed_delta_vs_prefix_hist6": delta,
                          "mixed_delta_vs_prefix_hist6_auprc": delta_auprc,
                          "n_proposals_above_0.05": int(sum(v > 0.05 for v in delta.values()))}
    for name, m in out.items():
        rows.append(dict(horizon_ns=ns, condition=name, recovery=recovery[name], **m))
    print(f"{ns}ns recovery:", {n: round(v, 3) for n, v in recovery.items()}, flush=True)
    print(f"{ns}ns mixed delta vs prefix_hist6:", {n: round(v, 3) for n, v in delta.items()},
          "above +0.05:", results[f"{ns}ns"]["n_proposals_above_0.05"], flush=True)

(HERE / "results.json").write_text(json.dumps(results, indent=2))
with open(HERE / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", HERE / "results.json", flush=True)
