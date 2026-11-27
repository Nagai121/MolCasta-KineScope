"""L2-24 h5 前提確認のずれの原因調査 (2026-11-28)。

目的: 解離・予測幅 5 ns で、長さ 1.0 ns の平均だけの回復率が
  自分 0.9056 に対し E01 0.9473 (差 -0.0417、許容幅 0.03 を超過)。
静的のみと全部入りは 0.005 で通っている。
L2-23 の調査と同じ手順で、ずれが評価手順の行の順番への敏感さの
範囲内かどうかを、解離・5 ns先の条件で直接測る。

変種 (いずれも予測幅 5 ns、条件は static / full / window_mean_1.0ns の3つ):
  A: E01 にそろえた手順 (ファイル名の辞書順、float32) — 本測定と同じ。0.9056 の再現を確認する。
  B1/B2: A に加え、各タンパク質×レプリカのまとまりの中で行を並べ替える (seed 123 / 777)。
  C: 旧い手順 (results_100.json の順番、float64)。

判定の材料:
  - 変種をまたいだ回復率の動き幅に E01 の 0.9473 が入るか。
  - 平均だけの AUPRC の差 (絶対値) と、土台の利得 (full - static) の大きさ。
    回復率は AUPRC の差を小さい利得で割るので、AUPRC の小さな動きが回復率では大きくなる。
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

CACHE = Path("/home/hatch/workspace/kinescope/work/e01_cache")
RESULTS_100 = Path("/home/hatch/workspace/k06b/results_100.json")
E01 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/E01_true_short_md_guidance/e01_short_md_results.json").read_text())
REF = E01["protocol_a_and_c"]["hb_diss"]["by_horizon"]["5"]["A_true_1.0ns_mean"]
MODE = "hb_diss"
H_NS = 5
# 予測幅の列番号: 1/5/10/30/50/90 ns は列 0/1/2/3/4/5。5 ns先は列 1。
H_IDX = {1: 0, 5: 1, 10: 2, 30: 3, 50: 4, 90: 5}[H_NS]
CLF_KW = dict(max_iter=160, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=25, l2_regularization=0.15,
              random_state=20261126)


def hist_stats(S, thr):
    S = np.asarray(S, dtype=np.float32)
    return np.column_stack([S[:, 0], S.mean(1), S.std(1), S.min(1),
                            S[:, -1] - S[:, 0], (S <= thr).mean(1)]).astype(np.float32)


def load_all(order="alpha", dtype=np.float32):
    if order == "alpha":
        proteins = sorted(p.stem for p in CACHE.glob("*.npz"))
    else:
        proteins = json.loads(RESULTS_100.read_text())["test_proteins"]
    Xs, Ss, Ys, Gs = [], [], [], []
    for p in proteins:
        z = np.load(CACHE / f"{p}.npz")
        for r in (1, 2, 3):
            Xs.append(z[f"r{r}_{MODE}_X"].astype(dtype))
            Ss.append(z[f"r{r}_{MODE}_series"].astype(dtype))
            Ys.append(z[f"r{r}_{MODE}_y"].astype(np.int64))
            Gs.append(np.full(len(Xs[-1]), p))
    X = np.concatenate(Xs)
    S = np.concatenate(Ss)
    Y = np.concatenate(Ys)[:, H_IDX]
    G = np.concatenate(Gs)
    # 各 (タンパク質, レプリカ) のまとまりの境目
    blocks = []
    start = 0
    for p in proteins:
        for r in (1, 2, 3):
            n = len(np.load(CACHE / f"{p}.npz")[f"r{r}_{MODE}_y"])
            blocks.append((start, start + n))
            start += n
    return X, S, Y, G, blocks


def oof_predict(X, y, G):
    pred = np.zeros(len(y), dtype=np.float64)
    for tr, te in GroupKFold(5).split(X, y, G):
        clf = HistGradientBoostingClassifier(**CLF_KW)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def run_variant(name, order="alpha", dtype=np.float32, shuffle_seed=None):
    X, S, Y, G, blocks = load_all(order, dtype)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        idx = np.arange(len(Y))
        for a, b in blocks:
            idx[a:b] = rng.permutation(idx[a:b])
        X, S, Y, G = X[idx], S[idx], Y[idx], G[idx]
    feats = {
        "static": X,
        "full": np.hstack([X, hist_stats(S, 3.5)]),
        "mean1.0": np.hstack([X, S.mean(axis=1, keepdims=True).astype(np.float32)]),
    }
    aps = {}
    for c, F in feats.items():
        aps[c] = average_precision_score(Y, oof_predict(F, Y, G))
    gain = aps["full"] - aps["static"]
    rec = (aps["mean1.0"] - aps["static"]) / gain
    print(f"{name}: static {aps['static']:.6f} full {aps['full']:.6f} "
          f"mean1.0 {aps['mean1.0']:.6f} gain {gain:.6f} recovery {rec:.6f} "
          f"(E01 recovery {REF['recovery_pooled']:.6f})", flush=True)
    return rec, aps


if __name__ == "__main__":
    print("E01 refs h5 diss: static %.6f full %.6f mean1.0 %.6f recovery %.6f" % (
        REF["ap_static"], REF["ap_full"], REF["ap_cond"], REF["recovery_pooled"]), flush=True)
    run_variant("A (E01-aligned)", "alpha", np.float32, None)
    run_variant("B1 (shuffle seed 123)", "alpha", np.float32, 123)
    run_variant("B2 (shuffle seed 777)", "alpha", np.float32, 777)
    run_variant("C (old order, float64)", "results100", np.float64, None)
