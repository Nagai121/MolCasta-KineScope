"""K16: 接触形成での担い手の切り分け

資料Dの条件（K13 と同じ土台）で、接触形成について2つの切り分けを行う。

パート1（統計量の差し替え、K09 と同じ段取りの変形）:
 接触には生成側の条件が無いので、K09 の「生成の統計量を実の値に戻す」
 差し替えはできない。代わりに、実系列の6統計量を土台に、統計量を
 1つずつ「学習側全体の平均値（定数）」に差し替える。定数にすると
 その統計量は候補ごとの情報を持たなくなるので、指導力がどれだけ
 落ちるかで、その統計量が指導力をどれだけ担っているかを測る。
 差し替えは学習側・テスト側の両方に当て、条件ごとに学習し直す
 （K09 と同じ）。定数は学習側の統計量の平均で決める（テスト側の
 情報は使わない）。

パート2（履歴の入れ替え、K03 と同じ段取り）:
 学習は実履歴のまま、テストの履歴だけを差し替えて評価する。
  - 別ペアへの差し替え: 同じタンパク質・同じ複製の中で、テスト候補の
    履歴（11フレームの距離系列）を別の候補のものにランダムに入れ替える
    （seed 固定、K03 と同じ 123）。
  - 別複製への差し替え: 同じ残基ペア（res_i, res_j）の履歴を別の複製
    （R1←R2、R2←R3、R3←R1）から持ってくる。候補の集合は複製ごとに
    起点距離で決まるため、同じペアが別複製でも候補になっている場合
    だけ差し替えでき、残りは元の履歴のままにする。差し替えできた
    候補だけに絞った評価も併記する（K03 と同じ）。

条件・モデルは K13 と同一: ATLAS 6タンパク質×3レプリカ、学習
1bgf_A・1ifg_A・1j8r_A・2b0a_A、テスト 1mix_A＋2qt4_A 合算。
接触形成（最小重原子間距離、候補は起点で 4.5 A 超、ラベルは予測先で
4.5 A 以下）。静的特徴5個。履歴は 0〜1 ns の11フレームから6統計量
（閾値 4.5 A）。HistGradientBoosting（反復120、学習率0.075、葉31、
min_samples_leaf 15、L2 0.02、seed 42）。予測幅 1/5/10/30/50/90 ns。
指標は AUC・AUPRC・Brier・Log Loss。距離の計算は K13 の抽出済み
キャッシュを使う（抽出時にフレームごとの箱を軸ごとに扱っている）。

途中結果は results_partial.json に条件ごとに書き出す。
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A"); TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
BOUND = 4.5
FEAT_NAMES = ["first", "mean", "std", "min", "last_minus_first", "frac_le_bound"]

def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact.npz")
            parts.append({k: z[k] for k in z.files})
    return parts

def cat(parts, key):
    return np.concatenate([p[key] for p in parts])

def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)], axis=1).astype(np.float32)

def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr); return m.predict_proba(xte)[:, 1]

def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

train = load(TRAIN_P); test = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, "test pos", Yte.sum(0), flush=True)

Ftr = hist6(Htr); Fte = hist6(Hte)
const = Ftr.mean(0)  # train-set mean per statistic
print("train-mean constants:", {n: round(float(c), 4) for n, c in zip(FEAT_NAMES, const)}, flush=True)

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed:", sorted(results), flush=True)

def save():
    PARTIAL.write_text(json.dumps(results, indent=2))

def run_condition(name, ftr, fte, xtr_extra=True):
    """Retrain per condition (part 1 style). ftr/fte are history feature blocks or None."""
    if name in results:
        print(name, "already done, skipped", flush=True); return
    results[name] = {}
    for hi, h in enumerate(HORIZONS):
        if ftr is None:
            p = predict(Xtr, Ytr[:, hi], Xte)
        else:
            p = predict(np.concatenate([Xtr, ftr], axis=1), Ytr[:, hi],
                        np.concatenate([Xte, fte], axis=1))
        results[name][str(h)] = metrics(Yte[:, hi], p)
    print(name, "AUPRC", {h: round(results[name][str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
    save()

# ---- references ----
run_condition("static_only", None, None)
run_condition("base_hist", Ftr, Fte)

# ---- part 1: per-statistic ablation to train-mean constant (both sides, retrain) ----
for j, nm in enumerate(FEAT_NAMES):
    ftr = Ftr.copy(); fte = Fte.copy()
    ftr[:, j] = const[j]; fte[:, j] = const[j]
    run_condition(f"abl_{nm}", ftr, fte)

# ---- part 2: test-only history swaps (K03 style), model trained on real history ----
# (a) different pair within protein+replica
def swap_within(parts):
    outs = []; r = np.random.default_rng(123)
    for p in parts:
        h = p["hist"]; perm = r.permutation(len(h)); outs.append(h[perm])
    return np.concatenate(outs)

Hte_swap_pair = swap_within(test)

# (b) different replica, same residue pair (R1<-R2, R2<-R3, R3<-R1)
def swap_replica(parts):
    outs = []; matched = []
    for pi in range(0, len(parts), 3):
        grp = parts[pi:pi + 3]
        for ri in range(3):
            src = grp[(ri + 1) % 3]; dst = grp[ri]
            key_src = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(src["res_i"], src["res_j"]))}
            h = dst["hist"].copy(); m = np.zeros(len(h), dtype=bool)
            for i, (a, b) in enumerate(zip(dst["res_i"], dst["res_j"])):
                j = key_src.get((int(a), int(b)))
                if j is not None:
                    h[i] = src["hist"][j]; m[i] = True
            print("swap_replica part", pi // 3, "R", ri + 1, "matched", int(m.sum()), "of", len(h), flush=True)
            outs.append(h); matched.append(m)
    return np.concatenate(outs), np.concatenate(matched)

Hte_swap_rep, matched_mask = swap_replica(test)
print("swap_replica matched total:", int(matched_mask.sum()), "of", len(matched_mask), flush=True)

Fte_pair = hist6(Hte_swap_pair)
Fte_rep = hist6(Hte_swap_rep)

for name, fte in (("swap_pair", Fte_pair), ("swap_replica", Fte_rep)):
    if name in results:
        print(name, "already done, skipped", flush=True); continue
    results[name] = {}
    for hi, h in enumerate(HORIZONS):
        p = predict(np.concatenate([Xtr, Ftr], axis=1), Ytr[:, hi],
                    np.concatenate([Xte, fte], axis=1))
        results[name][str(h)] = metrics(Yte[:, hi], p)
    print(name, "AUPRC", {h: round(results[name][str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
    save()

# subset evaluation on replica-swappable candidates only (base vs swap_replica)
if "swap_replica_subset" not in results:
    results["swap_replica_subset"] = {"n": int(matched_mask.sum()), "base_hist": {}, "swap_replica": {}}
    for hi, h in enumerate(HORIZONS):
        pb = predict(np.concatenate([Xtr, Ftr], axis=1), Ytr[:, hi],
                     np.concatenate([Xte, Fte], axis=1))
        pr = predict(np.concatenate([Xtr, Ftr], axis=1), Ytr[:, hi],
                     np.concatenate([Xte, Fte_rep], axis=1))
        results["swap_replica_subset"]["base_hist"][str(h)] = metrics(Yte[matched_mask, hi], pb[matched_mask])
        results["swap_replica_subset"]["swap_replica"][str(h)] = metrics(Yte[matched_mask, hi], pr[matched_mask])
    print("subset base", {h: round(results["swap_replica_subset"]["base_hist"][str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
    print("subset swap", {h: round(results["swap_replica_subset"]["swap_replica"][str(h)]["auprc"], 4) for h in HORIZONS}, flush=True)
    save()

# ---- derived: retained fraction of guidance ----
def retained(cond):
    out = {}
    for h in HORIZONS:
        s = results["static_only"][str(h)]["auprc"]; b = results["base_hist"][str(h)]["auprc"]
        c = results[cond][str(h)]["auprc"]
        out[str(h)] = float((c - s) / (b - s))
    return out

derived = {"constants_train_mean": {n: float(c) for n, c in zip(FEAT_NAMES, const)},
           "swap_replica_matched": int(matched_mask.sum()), "swap_replica_total": int(len(matched_mask)),
           "retained_fraction": {name: retained(name) for name in
                                 [f"abl_{n}" for n in FEAT_NAMES] + ["swap_pair", "swap_replica"]}}
results["_derived"] = derived

# checks vs K13
k13 = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K13_other_phenomena_real/results.json").read_text())
checks = {
    "static_vs_K13_auprc_max_abs_diff": max(abs(results["static_only"][str(h)]["auprc"] - k13["contact"]["static_only"][str(h)]["auprc"]) for h in HORIZONS),
    "base_vs_K13_auprc_max_abs_diff": max(abs(results["base_hist"][str(h)]["auprc"] - k13["contact"]["base_hist_D6"][str(h)]["auprc"]) for h in HORIZONS),
}
results["_checks_vs_K13"] = checks
print("checks vs K13:", checks, flush=True)

(OUT / "results.json").write_text(json.dumps(results, indent=2))
rows = []
for name, by_h in results.items():
    if name.startswith("_") or name == "swap_replica_subset":
        continue
    for h, m in by_h.items():
        rows.append(dict(condition=name, horizon_ns=h, **m))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved to", OUT)
