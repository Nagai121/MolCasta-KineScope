"""G82: 塩橋の形成の混合で、切り取りを 6 個の統計量にした場合も測る
（K15 の広げた条件、塩橋の形成、1 ns先）。

0.1 ns の切り取りの 6 個の統計量に、G19 で作り直した第11〜14案の
生成平均を別々の特徴量として足す。比べる相手は 6 個の統計量の
切り取りだけの回復率。1 ns先は、平均だけの混合が 4 案すべて線を
超えた幅（G19: 混合の差 +0.190/+0.212/+0.108/+0.231）。

生成平均の出所: G19 の gen/<タンパク質>_R<レプリカ>.npz。
第11〜14案の生成系列（4分岐×9点）を K15 の88タンパク質分作り直した
もので、生成平均は G19 と同じ定義で 4分岐×9点の平均。対応付けは
ファイルの並びそのままで、res_i・res_j が K15 キャッシュと一致する
ことを確かめる（G19 と同じ）。

条件（1 ns先、すべて条件ごとに学習し直す）:
  - static_only: 静的特徴5個だけ
  - full_hist6: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム、参照）
  - prefix_hist6_0.1ns: 静的特徴＋切り取り 0.1 ns（フレーム [0,1]）の
    6統計量（境界 4.0 Å）
  - gen_mean_only_p{11..14}: 静的特徴＋生成平均だけ（各案）
  - mixed_p{11..14}: 静的特徴＋切り取り 0.1 ns の 6統計量＋生成平均
    （別々の特徴量として 7列）

回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得)。
土台の静的のみ・全部入りは K15 の記録値と小数第4位まで一致する
ことを確かめる（1 ns先 0.2402/0.3842、利得 +0.1440）。

判定（先に固定）: 混合の回復率から切り取り 6統計量だけの回復率を
引いた差が +0.05 を超えれば、生成平均は 6統計量の切り取りの上でも
足しになる。土台の利得が +0.144 と小さいので、AUPRC の差の絶対値も
記録する。
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K15_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion")
GEN_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/G19_salt_prefix_plus_generated/gen")
OUT = Path("/home/hatch/workspace/g82")
BOUND = 4.0
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
HORIZON_IDX = 0  # 1 ns先
PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り
PROPOSALS = ["p11", "p12", "p13", "p14"]
REF_STATIC, REF_FULL = 0.2402, 0.3842  # K15 の記録値（1 ns先）


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


def load(prots):
    real, gen = [], {tag: [] for tag in PROPOSALS}
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15 / f"{p}_R{r}_salt.npz")
            d = {k: z[k] for k in z.files}
            g = np.load(GEN_DIR / f"{p}_R{r}.npz")
            assert np.array_equal(g["res_i"], d["res_i"]) and np.array_equal(g["res_j"], d["res_j"]), \
                f"residue pair mismatch {p} R{r}"
            real.append(d)
            for j, tag in enumerate(PROPOSALS):
                # 4分岐×9点の平均（G19 と同じ定義）
                gen[tag].append(g[f"gen{11 + j}"].mean(axis=0).mean(axis=0).astype(np.float64))
    return real, gen


def main():
    split = json.loads((K15_DIR / "split.json").read_text())
    train_p = ORIG_TRAIN + tuple(split["train_additions"])
    test_p = ORIG_TEST + tuple(split["test_additions"])
    train, gtr = load(train_p)
    test, gte = load(test_p)
    Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
    Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
    Htr = np.concatenate([d["hist"] for d in train]).astype(np.float64)
    Hte = np.concatenate([d["hist"] for d in test]).astype(np.float64)
    Ytr = np.concatenate([d["y"] for d in train])
    Yte = np.concatenate([d["y"] for d in test])
    GMtr = {tag: np.concatenate(gtr[tag]) for tag in PROPOSALS}
    GMte = {tag: np.concatenate(gte[tag]) for tag in PROPOSALS}
    ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
    print("train", Xtr.shape, "test", Xte.shape, "test pos 1ns", int(yte.sum()), flush=True)

    prefix_mean_tr = Htr[:, PREFIX_FRAMES].mean(1)
    prefix_mean_te = Hte[:, PREFIX_FRAMES].mean(1)
    full_mean_tr = Htr.mean(1)
    full_mean_te = Hte.mean(1)
    relations = {}
    for tag in PROPOSALS:
        gm_tr, gm_te = GMtr[tag], GMte[tag]
        relations[tag] = {}
        for split_name, pm, gm, fm in (("train", prefix_mean_tr, gm_tr, full_mean_tr),
                                       ("test", prefix_mean_te, gm_te, full_mean_te)):
            relations[tag][split_name] = dict(
                corr_prefix_gen=float(np.corrcoef(pm, gm)[0, 1]),
                corr_gen_full=float(np.corrcoef(gm, fm)[0, 1]),
                corr_prefix_full=float(np.corrcoef(pm, fm)[0, 1]),
                gen_minus_prefix_mean=float(np.mean(gm - pm)),
                gen_minus_prefix_std=float(np.std(gm - pm)),
                gen_minus_full_mean=float(np.mean(gm - fm)),
                gen_minus_full_std=float(np.std(gm - fm)))
        print(tag, "test:", {k: round(v, 4) for k, v in relations[tag]["test"].items()},
              flush=True)

    conds = {
        "static_only": (Xtr, Xte),
        "full_hist6": (np.column_stack([Xtr, hist6(Htr)]),
                       np.column_stack([Xte, hist6(Hte)])),
        "prefix_hist6_0.1ns": (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
                               np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])])),
    }
    for tag in PROPOSALS:
        conds[f"gen_mean_only_{tag}"] = (
            np.column_stack([Xtr, GMtr[tag][:, None].astype(np.float32)]),
            np.column_stack([Xte, GMte[tag][:, None].astype(np.float32)]))
        conds[f"mixed_{tag}"] = (
            np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES]),
                             GMtr[tag][:, None].astype(np.float32)]),
            np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES]),
                             GMte[tag][:, None].astype(np.float32)]))
    results = {}
    for cname, (xtr, xte) in conds.items():
        p = predict(xtr, ytr, xte)
        results[cname] = metrics(yte, p)
        print(f"  {cname} auprc {results[cname]['auprc']:.4f}", flush=True)
    assert abs(results["static_only"]["auprc"] - REF_STATIC) < 5e-4, results["static_only"]
    assert abs(results["full_hist6"]["auprc"] - REF_FULL) < 5e-4, results["full_hist6"]
    gain = results["full_hist6"]["auprc"] - results["static_only"]["auprc"]
    recovery = {n: float((m["auprc"] - results["static_only"]["auprc"]) / gain)
                for n, m in results.items() if n != "static_only"}
    prefix_rec = recovery["prefix_hist6_0.1ns"]
    mixed_delta = {tag: float(recovery[f"mixed_{tag}"] - prefix_rec) for tag in PROPOSALS}
    mixed_auprc_delta = {tag: float(results[f"mixed_{tag}"]["auprc"]
                                    - results["prefix_hist6_0.1ns"]["auprc"])
                         for tag in PROPOSALS}
    print(f"  base gain {gain:.4f}, prefix_hist6 recovery {prefix_rec:.3f}, "
          f"mixed deltas { {t: round(v, 3) for t, v in mixed_delta.items()} }, "
          f"mixed AUPRC deltas { {t: round(v, 4) for t, v in mixed_auprc_delta.items()} }",
          flush=True)
    out = {"horizon_ns": 1, "metrics": results, "base_gain": gain,
           "recovery_vs_real_history_gain": recovery,
           "prefix_hist6_0.1ns_recovery": prefix_rec,
           "mixed_minus_prefix_hist6": mixed_delta,
           "mixed_minus_prefix_hist6_auprc": mixed_auprc_delta,
           "relations_test": {tag: relations[tag]["test"] for tag in PROPOSALS}}
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_ns", "condition", "n", "positives", "auc", "auprc", "brier",
                    "logloss", "recovery_vs_real_history_gain", "mixed_minus_prefix_hist6",
                    "mixed_minus_prefix_hist6_auprc"])
        for cname, m in results.items():
            tag = cname.split("_", 1)[1] if cname.startswith("mixed_") else ""
            w.writerow([1, cname, m["n"], m["positives"], m["auc"], m["auprc"],
                        m["brier"], m["logloss"],
                        "" if cname == "static_only" else recovery[cname],
                        mixed_delta.get(tag, ""), mixed_auprc_delta.get(tag, "")])
    print("saved", OUT / "results.json")
    print("  mixed deltas:", {t: round(v, 3) for t, v in mixed_delta.items()})
    print("  mixed AUPRC deltas:", {t: round(v, 4) for t, v in mixed_auprc_delta.items()})


if __name__ == "__main__":
    main()
