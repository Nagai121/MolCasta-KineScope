"""G83: 水素結合の形成での混合の差に、残りの予測幅の区間を付ける.

G17・G37・G34 では 10 ns先で混合の差が -0.104/+0.001/-0.073、
G39・G46・G38 では 50 ns先で +0.371/+0.288/+0.030、
G14・G41・G30 では 90 ns先で +0.015/-0.072/+0.098 でした。
10・50・90 ns先 × 切り取り 0.1/0.25/0.5 ns について、混合と
切り取り平均だけの差（mixed_minus_prefix）のブートストラップ 95%
区間を出します（G23・G49・G59 と同じやり方）。1・5・30 ns先は
G49・G59・G23 で区間を付け済みです。

再標本化の単位はタンパク質。テストのタンパク質は 1mix_A と 2qt4_A の
2つだけなので、タンパク質単位の再標本化で取りうる組み合わせは
{1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通りしかありません。
そのためブートストラップ分布も3つの値しか取らず、95% 区間は
その3値の最小値〜最大値になります。区間が粗いことは留保として記録する
（G23・G49・G59・G62・G67 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G23・G49・G59・G62・G67 と同じです。

段取り:
- G17・G37・G34・G39・G46・G38・G14・G41・G30 と同じ段取りでモデルを
  学習し直す（同じ種・同じ条件で決定的なため、合算の AUPRC は記録と
  一致するはずで、一致を確認する）。候補ごとの予測確率を保存し、そこから
  ブートストラップする。生成平均は第22案の生成系列（K05 の
  generated_series.npz）の5点系列の平均を使う。生成器のチェックポイント
  は読み込まない。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り平均だけの回復率。
  回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り平均だけ・
  混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42 です。

読み方（先に固定）:
- +0.05 を超えたセル（10 ns先は無し、50 ns先の 0.1 ns と 0.25 ns、
  90 ns先の 0.5 ns）の区間が 0 を含むかどうかを見る。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
PREFIXES = {"0.1ns": [0, 1], "0.25ns": [0, 1, 2], "0.5ns": [0, 1, 2, 3, 4, 5]}
HORIZONS = {2: 10, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）

# G17・G37・G34・G39・G46・G38・G14・G41・G30 の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    10: {"static_only": 0.3548, "full": 0.4377, "gen_mean_only": 0.3752,
         "full_gain": 0.0829,
         "0.1ns": {"prefix": 0.4187, "mixed": 0.4100, "diff": -0.104},
         "0.25ns": {"prefix": 0.4356, "mixed": 0.4357, "diff": 0.001},
         "0.5ns": {"prefix": 0.4567, "mixed": 0.4507, "diff": -0.073}},
    50: {"static_only": 0.3342, "full": 0.4081, "gen_mean_only": 0.3663,
         "full_gain": 0.0739,
         "0.1ns": {"prefix": 0.3893, "mixed": 0.4167, "diff": 0.371},
         "0.25ns": {"prefix": 0.4084, "mixed": 0.4297, "diff": 0.288},
         "0.5ns": {"prefix": 0.4278, "mixed": 0.4300, "diff": 0.030}},
    90: {"static_only": 0.3035, "full": 0.4246, "gen_mean_only": 0.3356,
         "full_gain": 0.1211,
         "0.1ns": {"prefix": 0.3749, "mixed": 0.3766, "diff": 0.015},
         "0.25ns": {"prefix": 0.4034, "mixed": 0.3948, "diff": -0.072},
         "0.5ns": {"prefix": 0.3832, "mixed": 0.3950, "diff": 0.098}},
}


def load_gen5():
    """第22案の生成5点系列: [d0, gen(+0.1,+0.3,+0.5,+1.0 ns)]。
    キーは (tid, donor, acceptor)。G11 と同じ。"""
    z = np.load(GEN)
    # gen5 = [起点距離 d0（pair[:,2]）, gen(+0.1,+0.3,+0.5,+1.0 ns)]（G11 と同じ）
    gen5 = np.column_stack([z["pair"][:, 2].astype(np.float64),
                            z["gen"].astype(np.float64)])
    gen = {}
    for tid, da, aa, g in zip(z["tid"], z["pair"][:, 0], z["pair"][:, 1], gen5):
        gen[(str(tid), int(da), int(aa))] = g
    print("generated pairs:", len(gen), flush=True)
    return gen


GEN5 = load_gen5()


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            g = np.stack([GEN5[(tid, int(da), int(aa))]
                          for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["gen5"] = g
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)


def hist6(series, bound=3.5):
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


def auprc(y, p):
    return float(average_precision_score(y, p))


def recovery_diff(y, p_static, p_full, p_prefix, p_mixed):
    """その標本の中での mixed_minus_prefix（回復率の差）。分母は標本内の実履歴の利得。"""
    gain = auprc(y, p_full) - auprc(y, p_static)
    if gain == 0:
        return float("nan")
    return (auprc(y, p_mixed) - auprc(y, p_prefix)) / gain


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Gtr, Gte = cat(train, "gen5"), cat(test, "gen5")
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, flush=True)
print("gen5 first vs hist last: max abs diff train %.2e test %.2e" % (
    np.abs(Gtr[:, 0] - Htr[:, -1]).max(), np.abs(Gte[:, 0] - Hte[:, -1]).max()),
    flush=True)
assert np.abs(Gtr[:, 0] - Htr[:, -1]).max() < 1e-5
assert np.abs(Gte[:, 0] - Hte[:, -1]).max() < 1e-5

gen_mean_tr = Gtr.mean(1)
gen_mean_te = Gte.mean(1)
prefix_tr = {k: Htr[:, frames].mean(1) for k, frames in PREFIXES.items()}
prefix_te = {k: Hte[:, frames].mean(1) for k, frames in PREFIXES.items()}

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

results = {}
preds = {}  # horizon_ns -> condition -> test 予測確率
for hidx, hns in HORIZONS.items():
    ytr, yte = Ytr[:, hidx], Yte[:, hidx]
    p = {}
    p["static_only"] = predict(Xtr, ytr, Xte)
    p["full"] = predict(np.concatenate([Xtr, hist6(Htr)], axis=1), ytr,
                        np.concatenate([Xte, hist6(Hte)], axis=1))
    p["gen_mean_only"] = predict(
        np.column_stack([Xtr, gen_mean_tr[:, None].astype(np.float32)]), ytr,
        np.column_stack([Xte, gen_mean_te[:, None].astype(np.float32)]))
    for k in PREFIXES:
        p[f"prefix_{k}"] = predict(
            np.column_stack([Xtr, prefix_tr[k][:, None].astype(np.float32)]), ytr,
            np.column_stack([Xte, prefix_te[k][:, None].astype(np.float32)]))
        p[f"mixed_{k}"] = predict(
            np.column_stack([Xtr, prefix_tr[k][:, None].astype(np.float32),
                             gen_mean_tr[:, None].astype(np.float32)]), ytr,
            np.column_stack([Xte, prefix_te[k][:, None].astype(np.float32),
                             gen_mean_te[:, None].astype(np.float32)]))
    preds[hns] = p
    a = {c: auprc(yte, p[c]) for c in p}
    gain = a["full"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain for c in p if c != "static_only"}
    diff = {k: rec[f"mixed_{k}"] - rec[f"prefix_{k}"] for k in PREFIXES}
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        ap = {c: auprc(yte[idx], p[c][idx]) for c in p}
        g = ap["full"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc": {c: ap[c] for c in p},
            "full_gain": g,
            "recovery_prefix": {k: (ap[f"prefix_{k}"] - ap["static_only"]) / g if g else float("nan")
                                for k in PREFIXES},
            "recovery_mixed": {k: (ap[f"mixed_{k}"] - ap["static_only"]) / g if g else float("nan")
                               for k in PREFIXES},
            "mixed_minus_prefix_auprc": {k: ap[f"mixed_{k}"] - ap[f"prefix_{k}"]
                                         for k in PREFIXES},
            "mixed_minus_prefix_recovery": {k: ((ap[f"mixed_{k}"] - ap[f"prefix_{k}"]) / g
                                                if g else float("nan"))
                                            for k in PREFIXES},
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "per_protein": per_protein,
    }
    # G17・G37・G34・G39・G46・G38・G14・G41・G30 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full", "gen_mean_only"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    for k in PREFIXES:
        assert abs(a[f"prefix_{k}"] - exp[k]["prefix"]) < 5e-4, (hns, k, a[f"prefix_{k}"])
        assert abs(a[f"mixed_{k}"] - exp[k]["mixed"]) < 5e-4, (hns, k, a[f"mixed_{k}"])
        assert abs(diff[k] - exp[k]["diff"]) < 5e-3, (hns, k, diff[k])
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full']:.4f} "
          f"gen {a['gen_mean_only']:.4f} gain {gain:.4f} "
          f"diff " + ", ".join(f"{k} {diff[k]:+.3f}" for k in PREFIXES)
          + f" (record " + ", ".join(f"{k} {exp[k]['diff']:+.3f}" for k in PREFIXES) + ")",
          flush=True)
    print(f"  per-protein mixed-prefix recovery: "
          + "; ".join(f"{prot} " + ", ".join(
              f"{k} {per_protein[prot]['mixed_minus_prefix_recovery'][k]:+.3f}"
              for k in PREFIXES) for prot in TEST_P), flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in preds[h]},
)
print("checkpoint: saved test_predictions.npz", flush=True)

# ---- タンパク質単位のブートストラップ（G23・G49・G59・G62・G67 と同じ回数・seed・パーセンタイル法） ----
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))
rng = np.random.default_rng(BOOT_SEED)
boot = {}
for hidx, hns in HORIZONS.items():
    yte = Yte[:, hidx]
    p = preds[hns]
    boot[hns] = {}
    for k in PREFIXES:
        three = {}
        for combo in COMBOS:
            idx = np.concatenate([prot_idx[prot] for prot in combo])
            three["+".join(combo)] = recovery_diff(
                yte[idx], p["static_only"][idx], p["full"][idx],
                p[f"prefix_{k}"][idx], p[f"mixed_{k}"][idx])
        combo_idx = rng.integers(0, len(COMBOS), size=N_BOOT)
        vals = np.array([three["+".join(COMBOS[i])] for i in combo_idx])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        boot[hns][k] = {
            "n_boot": N_BOOT, "seed": BOOT_SEED,
            "point_estimate": results[hns]["mixed_minus_prefix"][k],
            "ci_lower": float(lo), "ci_upper": float(hi),
            "contains_zero": bool(lo <= 0 <= hi),
            "three_resample_values": three,
            "n_distinct_boot_values": int(len(np.unique(np.round(vals, 12)))),
        }
    print(f"{hns}ns bootstrap: " + ", ".join(
        f"{k} {boot[hns][k]['point_estimate']:+.3f} "
        f"[{boot[hns][k]['ci_lower']:+.3f}, {boot[hns][k]['ci_upper']:+.3f}] "
        f"zero={boot[hns][k]['contains_zero']}" for k in PREFIXES), flush=True)

# +0.05 を超えたセルの区間が 0 を含むかどうか（先に固定した読み方）
ABOVE = [(50, "0.1ns"), (50, "0.25ns"), (90, "0.5ns")]
above_check = {f"{h}ns_{k}": boot[h][k]["contains_zero"] for h, k in ABOVE}
print("above-0.05 cells contain zero:", above_check, flush=True)

out = {
    "label": "HB formation, hybrid uncertainty at 10/50/90ns across prefixes",
    "phenomenon": "hydrogen-bond formation (DocD, K05 generated series)",
    "prefix_frames": PREFIXES,
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (1mix_A, 2qt4_A; 2 proteins only, "
                  "so only 3 resample combinations exist and the bootstrap "
                  "distribution takes only 3 distinct values)",
                  "note": "The 10,000 draws are mapped onto the 3 exact combination "
                  "values; this is mathematically identical to recomputing AUPRC "
                  "on the resampled rows each time, because the difference depends "
                  "only on the chosen combination."},
    "horizons": {str(h): results[h] for h in HORIZONS.values()},
    "bootstrap_results": {str(h): boot[h] for h in HORIZONS.values()},
    "above_0.05_cells_contain_zero": above_check,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "prefix", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix", "auprc_gen_mean_only", "auprc_mixed",
                "full_gain", "recovery_prefix", "recovery_mixed",
                "mixed_minus_prefix", "boot_ci_lower", "boot_ci_upper",
                "ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        for k in PREFIXES:
            b = boot[h][k]
            w.writerow([h, k, r["test_positives"],
                        r["auprc"]["static_only"], r["auprc"]["full"],
                        r["auprc"][f"prefix_{k}"], r["auprc"]["gen_mean_only"],
                        r["auprc"][f"mixed_{k}"], r["full_gain"],
                        r["recovery"][f"prefix_{k}"],
                        r["recovery"][f"mixed_{k}"],
                        r["mixed_minus_prefix"][k],
                        b["ci_lower"], b["ci_upper"], b["contains_zero"]])

print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
