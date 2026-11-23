"""G48: 塩橋の形成で、タンパク質ごとに見た回復率を出す
(K15 の条件、塩橋の形成、1 ns先)。

G40 の続き。G40 では 0.25 ns の 6 統計量について、全体で見ると
0.334 まで落ち込むのに、タンパク質ごとに見ると 0.672 で平均だけを
上回ることが分かった。G43 では 0.75 ns の 6 統計量が全体で 0.511、
平均だけが 0.901 だった。0.5 ns と 0.75 ns の切り取りについても、
タンパク質ごとの回復率を出して、平均だけと 6 統計量を比べる。

G27・G43 の結果をそのまま使う。ただし G27・G43 は行ごとの予測値を
保存していないため、タンパク質ごとの回復率は、同じ手順（同じキャッシュ、
同じ分割、同じモデル、同じ seed）で予測値を出し直して計算する。
出し直した全体の AUPRC が G27・G43 の値と合うことを確かめてから、
タンパク質ごとの回復率を出す。全体の値は新しく測るものではなく、
G27・G43 の再現である。

タンパク質ごとの回復率の定義 (G40 と同じ):
- テストの30タンパク質それぞれについて、静的のみ・全部入り・
  0.5 ns / 0.75 ns の平均だけ・0.5 ns / 0.75 ns の 6 統計量の
  AUPRC を出す。
- 回復率 = (条件 − 静的のみ) / (全部入り − 静的のみ)。
- 陽性が 0件のタンパク質や、全部入りの利得が 0.01 以下のタンパク質は
  回復率は出せないので、件数と陽性数だけを記録する。
- タンパク質ごとの平均 (macro) と中央値も出す。タンパク質ごとの
  回復率は 1タンパク質あたり約39件・陽性約3件と小さく、大きく振れる
  ため、平均だけでなく中央値も見る。

判定 (先に固定): タンパク質ごとの平均で 0.75 ns の 6 統計量も
0.6 未満なら、塩橋の形成では 6 統計量が長い切り取りでも本当に足を
引っ張ることになる。タンパク質ごとの平均が 0.8 以上なら、全体で見た
不足は集計の仕方によるものだということになる。

土台 (K15 の条件、G27・G43 と同じ):
- K15 キャッシュ (~/workspace/kinescope/work/k15_cache/、suffix salt)、
  split.json の分割。学習58タンパク質 (2,557件)、テスト30タンパク質
  (1,170件、1 ns先の陽性 93件)。静的特徴5個。
- HistGradientBoosting (120回、学習率 0.075、葉31、葉あたり最小15件、
  L2 0.02、seed 42)、条件ごとに学習し直す。1 ns先だけを測る。
- 6 統計量の定義: [最初, 平均, 標準偏差, 最小, 最後−最初,
  4.0 Å以下割合] を切り取りのフレームで計算する。
"""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
OUT = Path(__file__).resolve().parent
BOUND = 4.0

ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
# G27・G43 の 1 ns先の値 (results.json から)。出し直しの照合用。
REF = {
    "static_only": 0.240221, "full_hist6": 0.384196, "mean_only_full": 0.368221,
    "prefix_mean_0.5ns": 0.335168, "prefix_hist6_0.5ns": 0.343364,
    "prefix_mean_0.75ns": 0.369993, "prefix_hist6_0.75ns": 0.313785,
}
PREFIX_FRAMES = {"0.5ns": [0, 1, 2, 3, 4, 5], "0.75ns": [0, 1, 2, 3, 4, 5, 6, 7]}


def load_with_protein(prots):
    """K15 キャッシュを読み、行ごとにタンパク質名も付ける。"""
    xs, hs, ys, ps = [], [], [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15 / f"{p}_R{r}_salt.npz")
            xs.append(z["X"])
            hs.append(z["hist"])  # (n, 11) フレーム
            ys.append(z["y"])
            ps.extend([p] * len(z["y"]))
    return (np.concatenate(xs), np.concatenate(hs), np.concatenate(ys),
            np.array(ps))


def six_stats(series):
    """切り取りの系列から 6 統計量を作る。series は (n, k) フレーム。"""
    return np.column_stack([series[:, 0], series.mean(axis=1), series.std(axis=1),
                            series.min(axis=1), series[:, -1] - series[:, 0],
                            (series <= BOUND).mean(axis=1)]).astype(np.float32)


def fit_predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def main():
    split = json.loads((K15_DIR / "split.json").read_text())
    train_p = ORIG_TRAIN + tuple(split["train_additions"])
    test_p = ORIG_TEST + tuple(split["test_additions"])
    Xtr, Htr, Ytr, _ = load_with_protein(train_p)
    Xte, Hte, Yte, Pte = load_with_protein(test_p)
    ytr, yte = Ytr[:, 0], Yte[:, 0]  # 1 ns先
    print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)

    full_mean_tr = Htr.mean(axis=1)
    full_mean_te = Hte.mean(axis=1)

    feats = {}
    feats["static_only"] = (Xtr, Xte)
    feats["full_hist6"] = (np.column_stack([Xtr, six_stats(Htr)]),
                           np.column_stack([Xte, six_stats(Hte)]))
    feats["mean_only_full"] = (
        np.column_stack([Xtr, full_mean_tr[:, None].astype(np.float32)]),
        np.column_stack([Xte, full_mean_te[:, None].astype(np.float32)]))
    for tag, frames in PREFIX_FRAMES.items():
        feats[f"prefix_mean_{tag}"] = (
            np.column_stack([Xtr, Htr[:, frames].mean(axis=1)[:, None].astype(np.float32)]),
            np.column_stack([Xte, Hte[:, frames].mean(axis=1)[:, None].astype(np.float32)]))
        feats[f"prefix_hist6_{tag}"] = (
            np.column_stack([Xtr, six_stats(Htr[:, frames])]),
            np.column_stack([Xte, six_stats(Hte[:, frames])]))

    preds, results = {}, {}
    for name, (xtr, xte) in feats.items():
        p = fit_predict(xtr, ytr, xte)
        preds[name] = p
        results[name] = metrics(yte, p)
        print(name, "auprc %.4f (G27/G43 %.4f, diff %+.6f)"
              % (results[name]["auprc"], REF[name], results[name]["auprc"] - REF[name]),
              flush=True)
        assert abs(results[name]["auprc"] - REF[name]) < 5e-4, name

    gain = results["full_hist6"]["auprc"] - results["static_only"]["auprc"]
    print("base gain %.4f" % gain, flush=True)
    recovery = {n: float((m["auprc"] - results["static_only"]["auprc"]) / gain)
                for n, m in results.items() if n != "static_only"}
    print("pooled recovery:", {n: round(v, 3) for n, v in recovery.items()}, flush=True)

    # タンパク質ごとの AUPRC と回復率
    conds = ("prefix_mean_0.5ns", "prefix_hist6_0.5ns",
             "prefix_mean_0.75ns", "prefix_hist6_0.75ns")
    per_protein = []
    for p in sorted(set(Pte)):
        idx = np.where(Pte == p)[0]
        row = {"protein": p, "n": int(len(idx)), "positives": int(yte[idx].sum())}
        ok = (0 < yte[idx].sum() < len(idx))
        for name in ("static_only", "full_hist6") + conds:
            row[f"auprc_{name}"] = (float(average_precision_score(yte[idx], preds[name][idx]))
                                    if ok else None)
        if ok:
            g = row["auprc_full_hist6"] - row["auprc_static_only"]
            if abs(g) > 0.01:
                for name in conds:
                    row[f"recovery_{name}"] = float(
                        (row[f"auprc_{name}"] - row["auprc_static_only"]) / g)
        per_protein.append(row)

    summary = {}
    for name in conds:
        defined = [r for r in per_protein if f"recovery_{name}" in r]
        vals = [r[f"recovery_{name}"] for r in defined]
        summary[name] = {
            "n_proteins_defined": len(defined),
            "macro_recovery": float(np.mean(vals)) if vals else None,
            "median_recovery": float(np.median(vals)) if vals else None,
            "min_recovery": float(np.min(vals)) if vals else None,
            "max_recovery": float(np.max(vals)) if vals else None,
        }
    for tag in ("0.5ns", "0.75ns"):
        h = [r for r in per_protein if f"recovery_prefix_hist6_{tag}" in r]
        summary[f"n_proteins_hist6_below_mean_{tag}"] = int(sum(
            1 for r in h
            if r[f"auprc_prefix_hist6_{tag}"] < r[f"auprc_prefix_mean_{tag}"]))
        summary[f"n_proteins_compared_{tag}"] = len(h)
    print("per-protein summary:", json.dumps(summary, indent=1), flush=True)

    out = {
        "g27_g43_reference_auprc_1ns": REF,
        "note": ("Pooled values are reproduced from G27/G43 by re-running the same "
                 "procedure (same cache, split, model, seed); per-protein recovery "
                 "is computed from the regenerated predictions. G27/G43 saved no "
                 "per-row predictions."),
        "metrics_1ns": results,
        "base_gain_1ns": gain,
        "pooled_recovery_vs_real_history_gain": recovery,
        "per_protein": per_protein,
        "per_protein_summary": summary,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
