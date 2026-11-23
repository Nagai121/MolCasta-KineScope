"""G57: 塩橋の形成で、10 ns先の混合は足しになるか
(K15 拡張条件の塩橋の形成、10 ns先)。

G19・G53 の続き。G19 では 1 ns先で 4 案すべてが線を超え、G53 では
5 ns先で 3 案が線を超えた。10 ns先はまだ測っていない。作り直した
第11〜14案の生成系列はそのまま使う (G19 の gen/、K17 と同じ作り方で
88タンパク質×3レプリカ分作り直したもの)。生成平均は、案ごとの
branch-mean 9点系列を 9点で平均したもの (G13・G19 と同じ)。

条件 (10 ns先、HistGradientBoosting 120回、学習率0.075、葉31、
葉あたり最小15件、L2 0.02、seed 42、条件ごとに学習し直す):
  static_only、full_hist (静的特徴＋実履歴6統計量、閾値 4.0 Å)、
  prefix_mean_only (静的特徴＋0.1 ns の切り取り平均、フレーム [0,1])、
  gen_mean_only_<n> (静的特徴＋第 n 案の生成平均)、
  mixed_<n> (静的特徴＋切り取り平均＋第 n 案の生成平均)、
n は 11〜14。回復率 = (AUPRC − 静的のみ) / (全部入り − 静的のみ)。
土台の利得は +0.059 (K15 の 10 ns先: 静的のみ 0.2034、全部入り
0.2627、テスト 1,170件、10 ns先の陽性 131件)。

判定 (先に固定): 混合の回復率が切り取り平均だけより +0.05 以上
上がる案があれば、塩橋の形成でも 10 ns先で混合が足しになることに
なる。どの案も +0.05 未満なら、塩橋の形成の混合は短い予測幅に
限られることになる。
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K15_CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
GEN_DIR = Path("/home/hatch/workspace/kinescope/repo/experiments/G19_salt_prefix_plus_generated/gen")
OUT = Path(__file__).resolve().parent
SPLIT = json.loads((Path(__file__).resolve().parent.parent
                    / "K15_saltbridge_expansion" / "split.json").read_text())
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A") + tuple(SPLIT["train_additions"])
TEST_P = ("1mix_A", "2qt4_A") + tuple(SPLIT["test_additions"])
BOUND = 4.0
PROPOSALS = (11, 12, 13, 14)
HORIZON_IDX = 2  # 10 ns先 (予測幅は 1/5/10/30/50/90 ns)


def hist6(series, bound):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02,
                                       random_state=42)
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
    real, gen = [], {n: [] for n in PROPOSALS}
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15_CACHE / f"{p}_R{r}_salt.npz")
            d = {k: z[k] for k in z.files}
            g = np.load(GEN_DIR / f"{p}_R{r}.npz")
            assert np.array_equal(g["res_i"], d["res_i"]) and np.array_equal(g["res_j"], d["res_j"]), \
                f"residue pair mismatch {p} R{r}"
            real.append(d)
            for n in PROPOSALS:
                # branch-mean 9点系列 -> 9点で平均 (G13・G19 と同じ)
                gen[n].append(g[f"gen{n}"].mean(axis=0).mean(axis=0).astype(np.float64))
    return real, gen


train, gtr = load(TRAIN_P)
test, gte = load(TEST_P)


def cat(parts, key):
    return np.concatenate([d[key] for d in parts])


Xtr = cat(train, "X").astype(np.float64); Xte = cat(test, "X").astype(np.float64)
Htr = cat(train, "hist").astype(np.float64); Hte = cat(test, "hist").astype(np.float64)
ytr = cat(train, "y")[:, HORIZON_IDX]; yte = cat(test, "y")[:, HORIZON_IDX]
GMtr = {n: np.concatenate(gtr[n]) for n in PROPOSALS}
GMte = {n: np.concatenate(gte[n]) for n in PROPOSALS}
print("train", Xtr.shape, "test", Xte.shape, "test pos 10ns", int(yte.sum()), flush=True)

prefix_tr = Htr[:, [0, 1]].mean(1); prefix_te = Hte[:, [0, 1]].mean(1)
full_mean_tr = Htr.mean(1); full_mean_te = Hte.mean(1)

relations = {}
for n in PROPOSALS:
    gm = GMte[n]
    relations[str(n)] = dict(
        corr_prefix_gen=float(np.corrcoef(prefix_te, gm)[0, 1]),
        corr_gen_full=float(np.corrcoef(gm, full_mean_te)[0, 1]),
        corr_prefix_full=float(np.corrcoef(prefix_te, full_mean_te)[0, 1]),
        gen_minus_prefix_mean=float(np.mean(gm - prefix_te)),
        gen_minus_prefix_std=float(np.std(gm - prefix_te)),
        gen_minus_full_mean=float(np.mean(gm - full_mean_te)),
        gen_minus_full_std=float(np.std(gm - full_mean_te)),
    )
    print(f"relations p{n} (test):", {k: round(v, 4) for k, v in relations[str(n)].items()},
          flush=True)

conds = {
    "static_only": (Xtr, Xte),
    "full_hist": (np.column_stack([Xtr, hist6(Htr, BOUND)]),
                  np.column_stack([Xte, hist6(Hte, BOUND)])),
    "prefix_mean_only": (np.column_stack([Xtr, prefix_tr[:, None]]),
                         np.column_stack([Xte, prefix_te[:, None]])),
}
for n in PROPOSALS:
    conds[f"gen_mean_only_{n}"] = (np.column_stack([Xtr, GMtr[n][:, None]]),
                                   np.column_stack([Xte, GMte[n][:, None]]))
    conds[f"mixed_{n}"] = (np.column_stack([Xtr, prefix_tr[:, None], GMtr[n][:, None]]),
                           np.column_stack([Xte, prefix_te[:, None], GMte[n][:, None]]))

res = {}
for name, (xtr, xte) in conds.items():
    res[name] = metrics(yte, predict(xtr, ytr, xte))
    print(f"{name}: AUPRC {res[name]['auprc']:.4f} AUC {res[name]['auc']:.4f}", flush=True)

s = res["static_only"]["auprc"]; f = res["full_hist"]["auprc"]
for name in conds:
    if name in ("static_only", "full_hist"):
        continue
    res[name]["gain"] = res[name]["auprc"] - s
    res[name]["recovery"] = (res[name]["auprc"] - s) / (f - s)
    print(f"{name}: gain {res[name]['gain']:.4f} recovery {res[name]['recovery']:.3f}", flush=True)
mixed_diff = {str(n): res[f"mixed_{n}"]["recovery"] - res["prefix_mean_only"]["recovery"]
              for n in PROPOSALS}
print("mixed - prefix recovery:", {k: round(v, 3) for k, v in mixed_diff.items()}, flush=True)

# 土台の再現の確認 (K15 の 10 ns先)
print("base check: static %.4f (recorded 0.2034), full %.4f (recorded 0.2627), "
      "gain %.4f (recorded 0.0594)" % (s, f, f - s), flush=True)
assert abs(s - 0.2034) < 1e-4 and abs(f - 0.2627) < 1e-4

results = {"label": "salt formation 10ns, prefix + generated mean (proposals 11-14)",
           "train_n": int(len(Xtr)), "test_n": int(len(Xte)),
           "full_gain": f - s, "conditions": res,
           "relations_test": relations, "mixed_minus_prefix": mixed_diff}
(OUT / "results.json").write_text(json.dumps(results, indent=2))
print("saved to", OUT)
