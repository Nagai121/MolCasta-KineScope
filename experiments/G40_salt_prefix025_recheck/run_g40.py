"""G40: 塩橋の形成で、0.25 ns の切り取りの 6 統計量が 0.334 まで落ち込む
のは本物か (K15 の条件、塩橋の形成、1 ns先)。

G27 の続き。G27 では 0.1 ns の平均だけが 0.647、0.5 ns の 6 統計量が
0.716 なのに、0.25 ns の 6 統計量だけが 0.334 と大きく落ち込み、単調
ではなかった。この実験では、同じ手順で 0.25 ns の 6 統計量を測り直す。
あわせて 0.25 ns の平均だけも測って比べ、タンパク質ごとの回復率も出す。

測り直しは G27 のコードのコピーではなく、同じ手順を別に書き直して
行う。0.1 ns と 0.5 ns の条件もあわせて測り直し、G27 と合うかを確かめる
ことで、書き直した側の手順が正しいことを確かめる。G27 との差が
0.001 未満なら、0.334 は手順がそのまま出した値であり、計算の側の
問題ではない。

判定 (先に固定): 測り直しで 0.25 ns の 6 統計量が 0.5 以上になれば、
0.334 は計算の側の問題だったことになる。測り直しても 0.334 前後で、
平均だけも同じくらい低ければ、0.25 ns の切り取りでは 6 統計量が
本当に足を引っ張ることになる。

土台 (K15 の条件、G27 と同じ):
- K15 キャッシュ (~/workspace/kinescope/work/k15_cache/、suffix salt)、
  split.json の分割。学習58タンパク質 (2,557件)、テスト30タンパク質
  (1,170件、1 ns先の陽性 93件)。静的特徴5個。
- HistGradientBoosting (120回、学習率 0.075、葉31、葉あたり最小15件、
  L2 0.02、seed 42)、条件ごとに学習し直す。1 ns先だけを測る。
- 6 統計量の定義: [最初, 平均, 標準偏差, 最小, 最後−最初,
  4.0 Å以下割合] を切り取りのフレームで計算する。

タンパク質ごとの回復率: テストの30タンパク質それぞれについて、
静的のみ・全部入り・0.25 ns の平均だけ・0.25 ns の 6 統計量の AUPRC
を出し、回復率 = (条件 − 静的のみ) / (全部入り − 静的のみ) を出す。
陽性が 0件のタンパク質や、全部入りの利得が 0.01 以下のタンパク質は
回復率は出せないので、件数と陽性数だけを記録する。

支えの確かめ: テストの行を 2,000 回取り直して、0.25 ns の 6 統計量と
平均だけの AUPRC の差の 95% 区間を出す。落ち込みが偶然の揺れかどうか
の目安にする。
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
# G27 の 1 ns先の値 (results.json から)。測り直しの照合用。
G27 = {
    "static_only": 0.240221, "full_hist6": 0.384196, "mean_only_full": 0.368221,
    "prefix_mean_0.1ns": 0.333328, "prefix_hist6_0.1ns": 0.338800,
    "prefix_mean_0.25ns": 0.328853, "prefix_hist6_0.25ns": 0.288342,
    "prefix_mean_0.5ns": 0.335168, "prefix_hist6_0.5ns": 0.343364,
}
PREFIX_FRAMES = {"0.1ns": [0, 1], "0.25ns": [0, 1, 2], "0.5ns": [0, 1, 2, 3, 4, 5]}


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
    first = series[:, 0]
    mean = series.mean(axis=1)
    std = series.std(axis=1)
    minimum = series.min(axis=1)
    delta = series[:, -1] - series[:, 0]
    frac = (series <= BOUND).mean(axis=1)
    return np.column_stack([first, mean, std, minimum, delta, frac]).astype(np.float32)


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
        print(name, "auprc %.4f (G27 %.4f, diff %+.6f)"
              % (results[name]["auprc"], G27[name], results[name]["auprc"] - G27[name]),
              flush=True)

    gain = results["full_hist6"]["auprc"] - results["static_only"]["auprc"]
    print("base gain %.4f" % gain, flush=True)
    recovery = {n: float((m["auprc"] - results["static_only"]["auprc"]) / gain)
                for n, m in results.items() if n != "static_only"}
    print("recovery:", {n: round(v, 3) for n, v in recovery.items()}, flush=True)

    # タンパク質ごとの AUPRC と回復率 (0.25 ns の平均だけと 6 統計量)
    per_protein = []
    for p in sorted(set(Pte)):
        idx = np.where(Pte == p)[0]
        row = {"protein": p, "n": int(len(idx)), "positives": int(yte[idx].sum())}
        ok = (0 < yte[idx].sum() < len(idx))
        for name in ("static_only", "full_hist6", "prefix_mean_0.25ns", "prefix_hist6_0.25ns"):
            row[f"auprc_{name}"] = (float(average_precision_score(yte[idx], preds[name][idx]))
                                    if ok else None)
        if ok:
            g = row["auprc_full_hist6"] - row["auprc_static_only"]
            if abs(g) > 0.01:
                row["recovery_prefix_mean_0.25ns"] = float(
                    (row["auprc_prefix_mean_0.25ns"] - row["auprc_static_only"]) / g)
                row["recovery_prefix_hist6_0.25ns"] = float(
                    (row["auprc_prefix_hist6_0.25ns"] - row["auprc_static_only"]) / g)
        per_protein.append(row)
    defined = [r for r in per_protein if "recovery_prefix_hist6_0.25ns" in r]
    worse = [r for r in defined
             if r["auprc_prefix_hist6_0.25ns"] < r["auprc_prefix_mean_0.25ns"]]
    macro = {
        "n_proteins_defined": len(defined),
        "n_proteins_hist6_below_mean": len(worse),
        "macro_recovery_prefix_mean_0.25ns": float(np.mean(
            [r["recovery_prefix_mean_0.25ns"] for r in defined])) if defined else None,
        "macro_recovery_prefix_hist6_0.25ns": float(np.mean(
            [r["recovery_prefix_hist6_0.25ns"] for r in defined])) if defined else None,
    }
    print("per-protein:", macro, flush=True)

    # 支えの確かめ: ブートストラップで 0.25 ns の 6 統計量と平均だけの差の区間
    rng = np.random.default_rng(0)
    n = len(yte)
    diffs, recs = [], []
    for _ in range(2000):
        b = rng.integers(0, n, size=n)
        if yte[b].sum() in (0, n):
            continue
        a_h = average_precision_score(yte[b], preds["prefix_hist6_0.25ns"][b])
        a_m = average_precision_score(yte[b], preds["prefix_mean_0.25ns"][b])
        a_s = average_precision_score(yte[b], preds["static_only"][b])
        a_f = average_precision_score(yte[b], preds["full_hist6"][b])
        diffs.append(a_h - a_m)
        recs.append((a_h - a_s) / (a_f - a_s) if abs(a_f - a_s) > 1e-9 else np.nan)
    diffs = np.array(diffs)
    recs = np.array(recs, dtype=float)
    boot = {
        "n_boot": len(diffs),
        "auprc_diff_hist6_minus_mean_0.25ns": float(
            results["prefix_hist6_0.25ns"]["auprc"] - results["prefix_mean_0.25ns"]["auprc"]),
        "diff_95ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "diff_fraction_below_0": float((diffs < 0).mean()),
        "recovery_hist6_0.25ns_95ci": [float(np.nanpercentile(recs, 2.5)),
                                       float(np.nanpercentile(recs, 97.5))],
    }
    print("bootstrap:", boot, flush=True)

    # 0.25 ns の切り取りの 6 統計量の中身 (テスト側)
    s25 = six_stats(Hte[:, PREFIX_FRAMES["0.25ns"]])
    s01 = six_stats(Hte[:, PREFIX_FRAMES["0.1ns"]])
    s05 = six_stats(Hte[:, PREFIX_FRAMES["0.5ns"]])
    feat_diag = {}
    for tag, s in (("0.1ns", s01), ("0.25ns", s25), ("0.5ns", s05)):
        feat_diag[tag] = {
            "frac_le4_mean": float(s[:, 5].mean()),
            "frac_le4_zero_share": float((s[:, 5] == 0).mean()),
            "std_mean": float(s[:, 2].mean()),
            "std_median": float(np.median(s[:, 2])),
            "min_mean": float(s[:, 3].mean()),
        }
    print("feature diag:", feat_diag, flush=True)

    out = {
        "g27_reference_auprc_1ns": G27,
        "metrics_1ns": results,
        "base_gain_1ns": gain,
        "recovery_vs_real_history_gain": recovery,
        "diff_vs_g27": {n: float(results[n]["auprc"] - G27[n]) for n in results},
        "per_protein": per_protein,
        "per_protein_summary": macro,
        "bootstrap": boot,
        "feature_diagnostics_test": feat_diag,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
