"""G80: 水素結合の形成の混合で、切り取りを 6 個の統計量にした場合も測る
（資料Dの条件、水素結合の形成、1 ns先と 30 ns先）。

0.1 ns の切り取りの 6 個の統計量に、第11〜14案の生成平均を別々の
特徴量として足す。比べる相手は 6 個の統計量の切り取りだけの回復率。
30 ns先は、平均だけの混合が +0.489 ともっとも大きかった幅（G17）。

生成平均の出所（第11〜14案）:
K11 の系列 experiments/K11_method_generality/series/atlas-*.npz。
第11〜14案は弾性モード・揺らぎの生成で、系列は 4分岐×8点。
生成平均は K11 の k11_compare.py と同じ定義で、4分岐×8点の平均
（nanmean）。対応付けのキーは (軌道ID, ドナー原子, アクセプター原子)
で、K05・G11 と同じ。候補は K05 と完全に同じ 14,582件。

参考として第22案の生成平均（K05 の generated_series.npz、5点系列の
平均、G11 と同じ）も 1条件だけ追加する。割り当て文の「方法22の
生成平均（案 11〜14）」は、案 11〜14 が明示されているため第11〜14案
を主として測り、第22案は参考の追加条件とする。読み方の +0.05 の線は
第11〜14案の 4条件に適用する。

条件（予測幅ごと、すべて条件ごとに学習し直す）:
  - static_only: 静的特徴7個だけ
  - full_hist6: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム、参照）
  - prefix_hist6_0.1ns: 静的特徴＋切り取り 0.1 ns（フレーム [0,1]）の
    6統計量（境界 3.5 Å）
  - gen_mean_only_p{11..14,22}: 静的特徴＋生成平均だけ（各案）
  - mixed_p{11..14,22}: 静的特徴＋切り取り 0.1 ns の 6統計量＋生成平均
    （別々の特徴量として 7列）

回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得)。
土台の静的のみ・全部入りは K06(a) の記録値と小数第4位まで一致する
ことを各予測幅で確かめる（1 ns先 0.3637/0.4815、30 ns先 0.3639/0.4264）。

判定（先に固定）: 混合の回復率から切り取り 6統計量だけの回復率を引いた
差が +0.05 を超えれば、生成平均は 6統計量の切り取りの上でも足しに
なる。+0.05 未満なら、生成平均は 6統計量の切り取りの上では足しに
ならない。
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
K11_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K11_method_generality/series")
K05_GEN = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
OUT = Path("/home/hatch/workspace/g80")
BOUND = 3.5
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = [(0, 1), (3, 30)]
REF = {1: (0.3637, 0.4815), 30: (0.3639, 0.4264)}  # K06(a) の記録値
PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り
PROPOSALS = ["p11", "p12", "p13", "p14", "p22"]


def load_gen_means():
    """第11〜14案（K11）と第22案（K05）の生成平均。キーは (tid, donor, acceptor)。
    起点距離も併せて返し、対応付けの検証に使う。"""
    gen = {}
    origin = {}
    for p in TRAIN_P + TEST_P:
        for r in (1, 2, 3):
            tid = f"atlas-{p}-analysis-R{r}"
            z = np.load(K11_DIR / f"atlas-{p}-analysis-R{r}.npz")
            pair = z["pair"]
            for tag, key in (("p11", "elas11"), ("p12", "elas12"),
                             ("p13", "elas13"), ("p14", "elas14")):
                m = np.nanmean(z[key].astype(np.float64), axis=(1, 2))
                for i in range(len(pair)):
                    gen.setdefault((tid, int(pair[i, 0]), int(pair[i, 1])), {})[tag] = m[i]
            for i in range(len(pair)):
                origin[(tid, int(pair[i, 0]), int(pair[i, 1]))] = float(pair[i, 2])
    z = np.load(K05_GEN)
    tids, pair, g = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), g.astype(np.float64)])
    assert gen5.shape == (14582, 5), gen5.shape
    for i, t in enumerate(tids):
        gen[(str(t), int(pair[i, 0]), int(pair[i, 1]))]["p22"] = float(gen5[i].mean())
    return gen, origin


GEN, ORIGIN = load_gen_means()


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            g = np.stack([[GEN[(tid, int(da), int(aa))][tag] for tag in PROPOSALS]
                          for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["genmean"] = g
            d["origin"] = np.array([ORIGIN[(tid, int(da), int(aa))]
                                    for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            parts.append(d)
    return parts


def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)


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
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def main():
    train = load(TRAIN_P)
    test = load(TEST_P)
    Xtr, Xte = cat(train, "X"), cat(test, "X")
    Htr, Hte = cat(train, "hist"), cat(test, "hist")
    Gtr, Gte = cat(train, "genmean"), cat(test, "genmean")
    Ytr, Yte = cat(train, "y"), cat(test, "y")
    print("train", Xtr.shape, "test", Xte.shape, flush=True)

    # 対応付けの検証: K11 の起点距離 = 履歴の最終フレーム（起点距離）
    Otr, Ote = cat(train, "origin"), cat(test, "origin")
    print("K11 origin vs hist last: max abs diff train %.2e test %.2e" % (
        np.abs(Otr - Htr[:, -1]).max(), np.abs(Ote - Hte[:, -1]).max()), flush=True)
    assert np.abs(Otr - Htr[:, -1]).max() < 1e-5
    assert np.abs(Ote - Hte[:, -1]).max() < 1e-5

    prefix_mean_tr = Htr[:, PREFIX_FRAMES].mean(1)
    prefix_mean_te = Hte[:, PREFIX_FRAMES].mean(1)
    full_mean_tr = Htr.mean(1)
    full_mean_te = Hte.mean(1)
    relations = {}
    for j, tag in enumerate(PROPOSALS):
        gm_tr, gm_te = Gtr[:, j], Gte[:, j]
        relations[tag] = {}
        for split, pm, gm, fm in (("train", prefix_mean_tr, gm_tr, full_mean_tr),
                                  ("test", prefix_mean_te, gm_te, full_mean_te)):
            relations[tag][split] = dict(
                corr_prefix_gen=float(np.corrcoef(pm, gm)[0, 1]),
                corr_gen_full=float(np.corrcoef(gm, fm)[0, 1]),
                corr_prefix_full=float(np.corrcoef(pm, fm)[0, 1]),
                gen_minus_prefix_mean=float(np.mean(gm - pm)),
                gen_minus_prefix_std=float(np.std(gm - pm)),
                gen_minus_full_mean=float(np.mean(gm - fm)),
                gen_minus_full_std=float(np.std(gm - fm)))
        print(tag, "test:", {k: round(v, 4) for k, v in relations[tag]["test"].items()},
              flush=True)

    out = {"horizons": {}}
    rows = []
    for h_idx, h_ns in HORIZONS:
        ytr, yte = Ytr[:, h_idx], Yte[:, h_idx]
        print(f"=== horizon {h_ns} ns, test positives {int(yte.sum())} ===", flush=True)
        conds = {
            "static_only": (Xtr, Xte),
            "full_hist6": (np.column_stack([Xtr, hist6(Htr)]),
                           np.column_stack([Xte, hist6(Hte)])),
            "prefix_hist6_0.1ns": (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
                                   np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])])),
        }
        for j, tag in enumerate(PROPOSALS):
            conds[f"gen_mean_only_{tag}"] = (
                np.column_stack([Xtr, Gtr[:, j:j + 1].astype(np.float32)]),
                np.column_stack([Xte, Gte[:, j:j + 1].astype(np.float32)]))
            conds[f"mixed_{tag}"] = (
                np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES]),
                                 Gtr[:, j:j + 1].astype(np.float32)]),
                np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES]),
                                 Gte[:, j:j + 1].astype(np.float32)]))
        results = {}
        for cname, (xtr, xte) in conds.items():
            p = predict(xtr, ytr, xte)
            results[cname] = metrics(yte, p)
            print(f"  {cname} auprc {results[cname]['auprc']:.4f}", flush=True)
        ref_s, ref_f = REF[h_ns]
        assert abs(results["static_only"]["auprc"] - ref_s) < 5e-4, (h_ns, results["static_only"])
        assert abs(results["full_hist6"]["auprc"] - ref_f) < 5e-4, (h_ns, results["full_hist6"])
        gain = results["full_hist6"]["auprc"] - results["static_only"]["auprc"]
        recovery = {n: float((m["auprc"] - results["static_only"]["auprc"]) / gain)
                    for n, m in results.items() if n != "static_only"}
        prefix_rec = recovery["prefix_hist6_0.1ns"]
        mixed_delta = {tag: float(recovery[f"mixed_{tag}"] - prefix_rec)
                       for tag in PROPOSALS}
        print(f"  base gain {gain:.4f}, prefix_hist6 recovery {prefix_rec:.3f}, "
              f"mixed deltas { {t: round(v, 3) for t, v in mixed_delta.items()} }", flush=True)
        out["horizons"][str(h_ns)] = {
            "metrics": results, "base_gain": gain,
            "recovery_vs_real_history_gain": recovery,
            "prefix_hist6_0.1ns_recovery": prefix_rec,
            "mixed_minus_prefix_hist6": mixed_delta,
            "relations_test": {tag: relations[tag]["test"] for tag in PROPOSALS},
        }
        for cname, m in results.items():
            rows.append([h_ns, cname, m["n"], m["positives"], m["auc"], m["auprc"],
                         m["brier"], m["logloss"],
                         "" if cname == "static_only" else recovery[cname],
                         "" if not cname.startswith("mixed_")
                         else mixed_delta[cname.split("_", 1)[1]]])
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_ns", "condition", "n", "positives", "auc", "auprc", "brier",
                    "logloss", "recovery_vs_real_history_gain", "mixed_minus_prefix_hist6"])
        w.writerows(rows)
    print("saved", OUT / "results.json")
    for h_ns in (1, 30):
        print(f"  {h_ns} ns mixed deltas:",
              {t: round(out["horizons"][str(h_ns)]["mixed_minus_prefix_hist6"][t], 3)
               for t in PROPOSALS})


if __name__ == "__main__":
    main()
