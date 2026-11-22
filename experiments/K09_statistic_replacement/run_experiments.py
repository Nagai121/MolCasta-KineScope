"""K09: 統計量ごとの差し替え（指導力の担い手の切り分け）

土台（実験D再現、experiments/base_reproduction/、K07/K08と同一）と同一条件。
生成系列（K05、第22案、5点）と実系列（同じ候補を履歴の5点に間引いたもの）
から同じ6統計量を作り、生成の統計量を1つずつ実系列の値に差し替える。

系列の定義（時間の向きの違いに注意）:
  生成5点: [起点距離 d0（フレーム10、1 ns）, +0.1, +0.3, +0.5, +1.0 ns]
      K05 の generated_series.npz の pair[:,2]（起点距離）と gen（前方4点）。
      起点から先への前方系列。
  実5点: 土台の履歴11フレームからフレーム [0, 1, 3, 5, 10] を取ったもの。
      時刻は 0, 0.1, 0.3, 0.5, 1.0 ns。起点までの履歴（後方）で、最後の点が
      起点距離 d0。生成系列とは時間の向きが逆で、共有する点は d0 のみ。
  6統計量は両系列に同じ定義で計算する:
      [最初の値, 平均, 標準偏差, 最小値, 最後-最初, 3.5 A以下だった割合]
  したがって「最初の値」は生成では d0、実ではフレーム0の距離であり、
  「最後-最初」も向きが逆の量である。この違いは解釈の留保として README に
  記録する。K06(a) の流儀と同じく、生成の前方系列の統計量を履歴特徴の
  位置に置く比較である。

条件（学習・テスト両方に同じ処理、条件ごとに学習し直す）:
  static_only, real_history(11点、参照), real5(S+H5、対照), gen5(S+G5、基準),
  swap_<統計量> x 6: 生成の6統計量のうち1つだけを実5点の値に置き換える。
回復率 = (差し替えの AUPRC - 生成5点の AUPRC) / (実5点の AUPRC - 生成5点の AUPRC)
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
REAL5_FRAMES = [0, 1, 3, 5, 10]  # 時刻 0, 0.1, 0.3, 0.5, 1.0 ns
STAT_NAMES = ["first", "mean", "std", "min", "last_minus_first", "frac_le_3.5"]

def load_gen5():
    """生成5点系列: [d0, gen(+0.1,+0.3,+0.5,+1.0 ns)]。キーは (tid, donor, acceptor)。"""
    z = np.load(GEN)
    tids, pair, gen = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
    assert gen5.shape == (14582, 5), gen5.shape
    return {(str(t), int(pair[i, 0]), int(pair[i, 1])): gen5[i]
            for i, t in enumerate(tids)}

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
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)

def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]

def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

train = load(TRAIN_P); test = load(TEST_P)
Xtr = cat(train, "X"); Xte = cat(test, "X")
Htr = cat(train, "hist"); Hte = cat(test, "hist")          # (n, 11) 履歴
Gtr = cat(train, "gen5"); Gte = cat(test, "gen5")          # (n, 5) 生成（前方）
R5tr = Htr[:, REAL5_FRAMES]; R5te = Hte[:, REAL5_FRAMES]   # (n, 5) 実履歴の間引き
print("train", Xtr.shape, Htr.shape, Gtr.shape, "test", Xte.shape, Hte.shape, Gte.shape, flush=True)
# 対応付けの検証: 生成5点の最初の点 = 起点距離 = 履歴の最終フレーム
print("gen5 first vs hist last: max abs diff train %.2e test %.2e" % (
    np.abs(Gtr[:, 0] - Htr[:, -1]).max(), np.abs(Gte[:, 0] - Hte[:, -1]).max()), flush=True)
print("real5 last vs hist last: max abs diff %.2e" % np.abs(R5te[:, -1] - Hte[:, -1]).max(), flush=True)

gen_tr, gen_te = hist6(Gtr), hist6(Gte)
real5_tr, real5_te = hist6(R5tr), hist6(R5te)
real11_tr, real11_te = hist6(Htr), hist6(Hte)

CONDITIONS = {
    "static_only": None,
    "real_history_11pt": (real11_tr, real11_te),
    "real5": (real5_tr, real5_te),
    "gen5": (gen_tr, gen_te),
}
for j, name in enumerate(STAT_NAMES):
    ftr = gen_tr.copy(); fte = gen_te.copy()
    ftr[:, j] = real5_tr[:, j]; fte[:, j] = real5_te[:, j]
    CONDITIONS[f"swap_{name}"] = (ftr, fte)

results = {}
for name, feats in CONDITIONS.items():
    results[name] = {}
    for hi, h in enumerate(HORIZONS):
        ytr = np.concatenate([p["y"] for p in train])[:, hi]
        yte = np.concatenate([p["y"] for p in test])[:, hi]
        if feats is None:
            p = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = feats
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr, np.concatenate([Xte, fte], axis=1))
        results[name][str(h)] = metrics(yte, p)
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[name][str(h)].items()}, flush=True)

recovery = {}
for name in CONDITIONS:
    if name.startswith("swap_"):
        recovery[name] = {}
        for h in HORIZONS:
            denom = results["real5"][str(h)]["auprc"] - results["gen5"][str(h)]["auprc"]
            recovery[name][str(h)] = float((results[name][str(h)]["auprc"] - results["gen5"][str(h)]["auprc"]) / denom) if denom != 0 else float("nan")
        print("recovery", name, {h: round(recovery[name][str(h)], 4) for h in HORIZONS}, flush=True)

payload = {"conditions": results, "auprc_recovery_vs_gen5_to_real5": recovery,
           "horizons": list(HORIZONS), "real5_frames": REAL5_FRAMES,
           "stat_names": STAT_NAMES,
           "notes": "gen5 is forward (origin +0.1/0.3/0.5/1.0 ns); real5 is history frames 0,1,3,5,10 (0/0.1/0.3/0.5/1.0 ns, ending at origin)."}
(OUT / "results.json").write_text(json.dumps(payload, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss", "auprc_recovery_vs_gen5_to_real5"])
    for name in CONDITIONS:
        for h in HORIZONS:
            r = results[name][str(h)]
            rec = recovery.get(name, {}).get(str(h), "") if name.startswith("swap_") else ""
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["brier"], r["logloss"], rec])
print("saved to", OUT)
