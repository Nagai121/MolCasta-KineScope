"""K10b: 尺度合わせで読み出しのずれはどこまで取れるか

K10 の (b)（実履歴で学習した予測器に生成履歴をそのまま当てる）の利得は
静的のみとほぼ同じ水準まで消えた。生成側の6統計量を実側の尺度に合わせる
線形変換（統計量ごとの平均・標準偏差の合わせ）で、このずれがどこまで
取れるかを測る。

変換: 統計量 j ごとに、学習タンパク質だけで求めた
  mu_real_tr[j], sd_real_tr[j] (実5点 hist6, 学習側)
  mu_gen_tr[j],  sd_gen_tr[j]  (生成5点 hist6, 学習側)
を用い、評価側の生成統計量を
  gen_scaled = (gen - mu_gen_tr)/sd_gen_tr * sd_real_tr + mu_real_tr
と変換する。評価側のデータでは変換のパラメータを決めない（漏れなし）。
変換を当てた生成統計量を、K10 (b) と同じ実履歴学習モデルで評価する。
学習側のモデルは K10 (b) と同一（静的7 + 実 hist6 で学習、seed 42）。

条件:
 (b)        K10 の実→生成（尺度合わせなし）。本スクリプトでは再実行せず、
            K10 の results.json から引用する（本スクリプトの初回実行時に
            別ループで再現を確認済み: 全6予測幅で AUPRC が小数第4位まで一致）。
 (b_scaled) 実→生成（尺度合わせあり）  <- K10b の主条件
比較先 (a) 実→実、(c) 生成→生成、静的のみは K10 の results.json から引用。

読み方（先に固定）:
 b_scaled が (c) や (a) に近づけば、ずれは主に尺度の問題。
 ほとんど変わらなければ、ずれは尺度ではなく分布の形の側の問題。

途中結果は results_k10b_partial.json に予測幅ごとに書き出し、仮想マシンの
再起動で中断しても再開できる。
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_k10b_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
REAL5_IDX = [0, 3, 5, 7, 10]
FEAT_NAMES = ["first", "mean", "std", "min", "last_minus_first", "frac_le_3.5"]

def load_gen_series():
    z = np.load(GEN)
    tids, pair, gen = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
    m = {}
    for i, t in enumerate(tids):
        m[(str(t), int(pair[i, 0]), int(pair[i, 1]))] = gen5[i]
    assert len(m) == 14582
    return m

GEN_MAP = load_gen_series()

def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            gen5 = np.stack([GEN_MAP[(tid, int(da), int(aa))]
                             for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["gen5"] = gen5
            parts.append(d)
    return parts

def cat(parts, key):
    return np.concatenate([p[key].T if key == "hist" else p[key] for p in parts])

def hist6(series, bound=3.5):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)

def metrics(y, p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

train = load(TRAIN_P); test = load(TEST_P)
Xtr = cat(train, "X"); Xte = cat(test, "X")
Htr = cat(train, "hist"); Hte = cat(test, "hist")
Gtr = cat(train, "gen5"); Gte = cat(test, "gen5")
Ytr = cat(train, "y"); Yte = cat(test, "y")
Rtr = hist6(Htr[:, REAL5_IDX]); Rte = hist6(Hte[:, REAL5_IDX])
Gtr6 = hist6(Gtr); Gte6 = hist6(Gte)
print("train", Xtr.shape, "test", Xte.shape, flush=True)

# --- affine fit on TRAIN proteins only ---
mu_real = Rtr.mean(0); sd_real = Rtr.std(0)
mu_gen = Gtr6.mean(0); sd_gen = Gtr6.std(0)
assert (sd_gen > 0).all() and (sd_real > 0).all()
scale = sd_real / sd_gen
shift = mu_real - scale * mu_gen
Gte_scaled = (Gte6 * scale + shift).astype(np.float32)
params = {name: {"mu_real_train": float(mu_real[j]), "sd_real_train": float(sd_real[j]),
                 "mu_gen_train": float(mu_gen[j]), "sd_gen_train": float(sd_gen[j]),
                 "scale": float(scale[j]), "shift": float(shift[j])}
          for j, name in enumerate(FEAT_NAMES)}
for name, v in params.items():
    print(name, {k: round(x, 4) for k, x in v.items()}, flush=True)
print("scaled test-gen mean", np.round(Gte_scaled.mean(0), 4),
      "real test mean", np.round(Rte.mean(0), 4), flush=True)
print("scaled test-gen std ", np.round(Gte_scaled.std(0), 4),
      "real test std ", np.round(Rte.std(0), 4), flush=True)

k10 = json.loads((OUT / "results.json").read_text())
results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())["b_scaled"]
    print("resumed from partial:", sorted(results), flush=True)

for hi, h in enumerate(HORIZONS):
    if str(h) in results:
        print("b_scaled", h, "already done, skipped", flush=True)
        continue
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(np.concatenate([Xtr, Rtr], axis=1), Ytr[:, hi])
    p = m.predict_proba(np.concatenate([Xte, Gte_scaled], axis=1))[:, 1]
    results[str(h)] = metrics(Yte[:, hi], p)
    print("b_scaled", h, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[str(h)].items()}, flush=True)
    PARTIAL.write_text(json.dumps({"b_scaled": results, "affine_params_train_only": params}, indent=2))

s1 = k10["conditions"]["static_only"]["1"]["auprc"]
a1 = k10["conditions"]["a_train_real_eval_real"]["1"]["auprc"]
c1 = k10["conditions"]["c_train_gen_eval_gen"]["1"]["auprc"]
b1 = k10["conditions"]["b_train_real_eval_gen"]["1"]["auprc"]
bs1 = results["1"]["auprc"]
summary = {
    "horizon_1ns_auprc": {"static_only": s1, "a_real": a1, "b_unscaled_K10": b1, "b_scaled": bs1, "c_gen": c1},
    "gain_over_static_1ns": {"b_unscaled_K10": float(b1 - s1), "b_scaled": float(bs1 - s1),
                             "c_gen": float(c1 - s1), "a_real": float(a1 - s1)},
    "b_scaled_recovers_fraction_of_c_gain_1ns": float((bs1 - s1) / (c1 - s1)),
    "b_scaled_recovers_fraction_of_a_gain_1ns": float((bs1 - s1) / (a1 - s1)),
    "b_scaled_minus_b_unscaled_auprc_1ns": float(bs1 - b1),
}
print("summary 1ns:", json.dumps(summary, indent=1), flush=True)

payload = {"conditions": {"b_scaled": results,
                          "b_unscaled_K10_reference": k10["conditions"]["b_train_real_eval_gen"]},
           "affine_params_train_only": params,
           "summary_1ns": summary, "horizons": list(HORIZONS),
           "note": "affine gen->real per-feature, fitted on TRAIN proteins only; "
                   "model is K10 (b) real-trained model, unchanged; "
                   "(b) unscaled values are quoted from K10 results.json (reproduction "
                   "confirmed in the first K10b run: all 6 horizons matched K10 (b) to 4 decimals)"}
(OUT / "results_k10b.json").write_text(json.dumps(payload, indent=2))
with open(OUT / "results_k10b.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "brier", "logloss"])
    for name, cond in (("b_scaled", results), ("b_unscaled_K10_reference", k10["conditions"]["b_train_real_eval_gen"])):
        for h in HORIZONS:
            r = cond[str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
