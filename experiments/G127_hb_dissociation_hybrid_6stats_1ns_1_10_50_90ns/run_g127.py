"""G127: 水素結合の解離の混合で、切り取り 1.0 ns の 6 個の統計量に
した場合の残りの予測幅も測る（資料Dの条件、水素結合の解離、1・10・50・90 ns先）。

切り取り 1.0 ns（フレーム [0,1,2,3,4,5,6,7,8,9]）の 6 個の統計量に、
第11〜14案の生成平均を別々の特徴量として足す。比べる相手は 1.0 ns の
6 個の統計量の切り取りだけの回復率。G122 は 5 ns先と 30 ns先だけだった。
残りの 1・10・50・90 ns先を同じ条件で測る。平均だけの混合では、1 ns先は
-0.052、10 ns先は +0.019（G61）、50 ns先は +0.015（G50）、90 ns先は
-0.206（G54）だった。

切り取りのフレームの範囲についての注記: フレームは 0.1 ns 刻みで、
フレーム 0 が起点（0 ns）、フレーム 9 が 0.9 ns です。切り取り 1.0 ns
と呼ぶフレーム [0,1,2,3,4,5,6,7,8,9] は、実際には 0〜0.9 ns を覆い
ます（0.25 ns の切り取りのフレーム [0,1,2] が実際には 0.2 ns を覆う
のと同じ形。G110 の README にも同じ注記があります）。ここでは割り当て
文の通りフレーム [0,1,2,3,4,5,6,7,8,9] を使い、「切り取り 1.0 ns」
と呼ぶことにします。

生成平均の出所（第11〜14案）:
K54 の残基レベル生成系列 `~/workspace/kinescope/work/k54_gen_residue/`
（`{protein}_R{rep}.npz`、gen11〜gen14 は 4分岐×9点、donor_atom・
acceptor_atom 付き）。G18 の README に選択肢 (b) として記録された
もので、解離の候補 6,131件を全件覆う。生成平均は K17・G13・G84 と
同じ定義で、4分岐×9点の平均。対応付けはファイルの並びそのままで、
donor_atom・acceptor_atom が K46 キャッシュと一致することを確かめる。

参考として第22案の生成平均（G18 で作り直したもの、
`~/workspace/kinescope/work/g18_gen22/`、atlas-*.npz、pair と
gen_dist の 4列。gen5 = [起点距離, gen_dist] の 5点系列の平均、G44・
G50 と同じ）も 1条件だけ追加する。割り当て文の「G18 で作り直した
方法22の生成平均（案 11〜14）」は、案 11〜14 が明示されているため
第11〜14案を主として測り、第22案は参考の追加条件とする（G80・G95・
G111・G122 と同じ読み方）。読み方の +0.05 の線は第11〜14案の 4条件に
適用する。第22案は生成器が学習した範囲（形成の候補だけ）の外への
当てはめである点に留保がつく（G18 の README 第7節）。

条件（予測幅ごと、すべて条件ごとに学習し直す）:
  - static_only: 静的特徴7個だけ
  - full_hist6: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム、参照）
  - prefix_hist6_1.0ns: 静的特徴＋切り取り 1.0 ns（フレーム [0,1,2,3,4,5,6,7,8,9]、
    実際には 0〜0.9 ns を覆います）の 6統計量（境界 3.5 Å）
  - prefix_mean_1.0ns: 静的特徴＋切り取り 1.0 ns の平均だけ（参考）
  - gen_mean_only_p{11..14,22}: 静的特徴＋生成平均だけ（各案）
  - mixed_p{11..14,22}: 静的特徴＋切り取り 1.0 ns の 6統計量＋生成平均
    （別々の特徴量として 7列）

回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得)。
土台の静的のみ・全部入りは K46 の記録値と小数第4位まで一致する
ことを各予測幅で確かめる（1 ns先 0.4160/0.5945、10 ns先 0.4496/0.5974、
50 ns先 0.4088/0.5231、90 ns先 0.5118/0.5797）。

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

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
K54_DIR = Path("/home/hatch/workspace/kinescope/work/k54_gen_residue")
GEN22_DIR = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
OUT = Path(__file__).resolve().parent
BOUND = 3.5
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = [(0, 1), (2, 10), (4, 50), (5, 90)]
REF = {1: (0.4160, 0.5945), 10: (0.4496, 0.5974),
       50: (0.4088, 0.5231), 90: (0.5118, 0.5797)}  # K46 の記録値
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り
PROPOSALS = ["p11", "p12", "p13", "p14", "p22"]


def load_gen22():
    """G18 で作り直した第22案の生成系列。キーは (tid, donor, acceptor)。
    gen5 = [起点距離, gen_dist の 4列] の 5点系列。"""
    gen = {}
    for f in sorted(GEN22_DIR.glob("atlas-*.npz")):
        z = np.load(f)
        tid = f.stem
        pair, gd = z["pair"], z["gen_dist"]
        gen5 = np.column_stack([pair[:, 2].astype(np.float64),
                                gd.astype(np.float64)])
        assert gen5.shape[1] == 5, gen5.shape
        for i in range(len(pair)):
            gen[(tid, int(pair[i, 0]), int(pair[i, 1]))] = gen5[i]
    print("regenerated #22 pairs:", len(gen), flush=True)
    return gen


GEN22 = load_gen22()


def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            k = np.load(K54_DIR / f"{p}_R{r}.npz")
            assert np.array_equal(k["donor_atom"], d["donor_atom"]) and \
                np.array_equal(k["acceptor_atom"], d["acceptor_atom"]), \
                f"candidate mismatch {p} R{r}"
            g = np.stack([k[f"gen{11 + j}"].mean(axis=0).mean(axis=0).astype(np.float64)
                          for j in range(4)], axis=1)  # (N, 4): p11..p14
            g22 = np.array([GEN22[(tid, int(da), int(aa))].mean()
                            for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["genmean"] = np.column_stack([g, g22[:, None]])  # (N, 5): p11..p14, p22
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
            "prefix_hist6_1.0ns": (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
                                   np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])])),
            "prefix_mean_1.0ns": (np.column_stack([Xtr, prefix_mean_tr[:, None].astype(np.float32)]),
                                  np.column_stack([Xte, prefix_mean_te[:, None].astype(np.float32)])),
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
        prefix_rec = recovery["prefix_hist6_1.0ns"]
        mixed_delta = {tag: float(recovery[f"mixed_{tag}"] - prefix_rec)
                       for tag in PROPOSALS}
        print(f"  base gain {gain:.4f}, prefix_hist6 recovery {prefix_rec:.3f}, "
              f"mixed deltas { {t: round(v, 3) for t, v in mixed_delta.items()} }", flush=True)
        out["horizons"][str(h_ns)] = {
            "metrics": results, "base_gain": gain,
            "recovery_vs_real_history_gain": recovery,
            "prefix_hist6_1.0ns_recovery": prefix_rec,
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
    for h_ns in (1, 10, 50, 90):
        print(f"  {h_ns} ns mixed deltas:",
              {t: round(out["horizons"][str(h_ns)]["mixed_minus_prefix_hist6"][t], 3)
               for t in PROPOSALS})


if __name__ == "__main__":
    main()
