"""G23: 混合の効き目のばらつきが本物か（ブートストラップ 95% 区間）.

G11 では 1 ns先で混合と切り取り平均だけの差が +0.186、
G17 では 10 ns先で -0.104、30 ns先で +0.489、
G14 では 90 ns先で +0.015 と、予測幅でばらついている。
各予測幅で、混合と切り取り平均だけの差（mixed_minus_prefix）の
ブートストラップ 95% 区間を測る。

再標本化の単位はタンパク質。テストのタンパク質は 1mix_A と 2qt4_A の
2つだけなので、タンパク質単位の再標本化で取りうる組み合わせは
{1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通りしかない。
そのためブートストラップ分布も3つの値しか取らず、95% 区間は
その3値の最小値〜最大値になる。区間が粗いことは留保として記録する。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法。

段取り:
- G11・G17・G14 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で
  決定的なため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
  生成器のチェックポイント .pt ファイルは読み込まない（保存済みの
  生成系列を使う）。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り平均だけの回復率。
  回復率の分母はそれぞれの予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り平均だけ・
  混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。

判定（先に固定）:
- 30 ns の +0.489 の区間が 0 を含まなければ本物、含めば揺れ。
- 1 ns の +0.186 についても同じ物差しで見る。
- 10 ns の -0.104 と 90 ns の +0.015 についても区間を併記する。
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り（G02 と同じ）
HORIZONS = {1: 0, 10: 2, 30: 3, 90: 5}  # 予測幅 ns -> y の列番号
N_BOOT = 10000  # 結果を見る前に固定
BOOT_SEED = 42  # 結果を見る前に固定

# G11・G17・G14 の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    1: {"static_only": 0.3637, "full": 0.4815, "prefix": 0.4308, "mixed": 0.4528,
        "full_gain": 0.1178, "mixed_minus_prefix": 0.186},
    10: {"static_only": 0.3548, "full": 0.4377, "prefix": 0.4187, "mixed": 0.4100,
         "full_gain": 0.0829, "mixed_minus_prefix": -0.104},
    30: {"static_only": 0.3639, "full": 0.4264, "prefix": 0.4251, "mixed": 0.4557,
         "full_gain": 0.0626, "mixed_minus_prefix": 0.489},
    90: {"static_only": 0.3035, "full": 0.4246, "prefix": 0.3749, "mixed": 0.3766,
         "full_gain": 0.1211, "mixed_minus_prefix": 0.015},
}


def load_gen5():
    z = np.load(GEN)
    tids, pair, gen = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), gen.astype(np.float64)])
    assert gen5.shape == (14582, 5), gen5.shape
    return {(str(t), int(pair[i, 0]), int(pair[i, 1])): gen5[i]
            for i, t in enumerate(tids)}


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
    np.abs(Gtr[:, 0] - Htr[:, -1]).max(), np.abs(Gte[:, 0] - Hte[:, -1]).max()), flush=True)
assert np.abs(Gtr[:, 0] - Htr[:, -1]).max() < 1e-5
assert np.abs(Gte[:, 0] - Hte[:, -1]).max() < 1e-5

prefix_mean_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_mean_te = Hte[:, PREFIX_FRAMES].mean(1)
gen_mean_tr = Gtr.mean(1)
gen_mean_te = Gte.mean(1)

# タンパク質ごとのテスト候補の行番号
prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

results = {}
preds = {}  # horizon -> condition -> test 予測確率
for h_ns, h_idx in HORIZONS.items():
    ytr, yte = Ytr[:, h_idx], Yte[:, h_idx]
    conds = {
        "static_only": None,
        "full": (hist6(Htr), hist6(Hte)),
        "prefix": (prefix_mean_tr[:, None].astype(np.float32),
                   prefix_mean_te[:, None].astype(np.float32)),
        "mixed": (np.stack([prefix_mean_tr, gen_mean_tr], axis=1).astype(np.float32),
                  np.stack([prefix_mean_te, gen_mean_te], axis=1).astype(np.float32)),
    }
    p = {}
    for cname, feats in conds.items():
        if feats is None:
            p[cname] = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = feats
            p[cname] = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                               np.concatenate([Xte, fte], axis=1))
    preds[h_ns] = p
    a = {c: auprc(yte, p[c]) for c in conds}
    gain = a["full"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain for c in ("full", "prefix", "mixed")}
    diff = rec["mixed"] - rec["prefix"]
    # タンパク質ごとの差（AUPRC の差と回復率の差の両方）
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        ap = {c: auprc(yte[idx], p[c][idx]) for c in conds}
        g = ap["full"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc": {c: ap[c] for c in conds},
            "full_gain": g,
            "recovery_prefix": (ap["prefix"] - ap["static_only"]) / g if g else float("nan"),
            "recovery_mixed": (ap["mixed"] - ap["static_only"]) / g if g else float("nan"),
            "mixed_minus_prefix_auprc": ap["mixed"] - ap["prefix"],
            "mixed_minus_prefix_recovery": ((ap["mixed"] - ap["prefix"]) / g
                                            if g else float("nan")),
        }
    results[h_ns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "per_protein": per_protein,
    }
    # G11・G17・G14 の記録との照合
    exp = EXPECTED[h_ns]
    for c in ("static_only", "full", "prefix", "mixed"):
        assert abs(a[c] - exp[c]) < 5e-4, (h_ns, c, a[c], exp[c])
    assert abs(diff - exp["mixed_minus_prefix"]) < 5e-3, (h_ns, diff)
    print(f"{h_ns}ns: static {a['static_only']:.4f} full {a['full']:.4f} "
          f"prefix {a['prefix']:.4f} mixed {a['mixed']:.4f} gain {gain:.4f} "
          f"diff {diff:.4f} (record {exp['mixed_minus_prefix']:+.3f})", flush=True)
    print(f"  per-protein mixed-prefix recovery: "
          + ", ".join(f"{prot} {per_protein[prot]['mixed_minus_prefix_recovery']:+.4f}"
                      for prot in TEST_P), flush=True)

# ---- タンパク質単位のブートストラップ ----
rng = np.random.default_rng(BOOT_SEED)
boot = {}
for h_ns, h_idx in HORIZONS.items():
    yte = Yte[:, h_idx]
    p = preds[h_ns]
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        chosen = rng.choice(TEST_P, size=len(TEST_P), replace=True)
        idx = np.concatenate([prot_idx[prot] for prot in chosen])
        diffs[b] = recovery_diff(yte[idx], p["static_only"][idx], p["full"][idx],
                                 p["prefix"][idx], p["mixed"][idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # 取りうる3通りの組み合わせの値も直接出す（分布が3値しか取らないため）
    three = {}
    for combo in (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A")):
        idx = np.concatenate([prot_idx[prot] for prot in combo])
        three["+".join(combo)] = recovery_diff(
            yte[idx], p["static_only"][idx], p["full"][idx], p["prefix"][idx], p["mixed"][idx])
    boot[h_ns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[h_ns]["mixed_minus_prefix"],
        "ci_lower": float(lo), "ci_upper": float(hi),
        "contains_zero": bool(lo <= 0 <= hi),
        "three_resample_values": three,
        "n_distinct_boot_values": int(len(np.unique(np.round(diffs, 12)))),
    }
    print(f"{h_ns}ns bootstrap: point {results[h_ns]['mixed_minus_prefix']:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] contains_zero={lo <= 0 <= hi} "
          f"distinct={boot[h_ns]['n_distinct_boot_values']}", flush=True)
    print(f"  three values: " + ", ".join(f"{k} {v:+.4f}" for k, v in three.items()), flush=True)

out = {
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (1mix_A, 2qt4_A; 2 proteins only, "
                  "so only 3 resample combinations exist and the bootstrap "
                  "distribution takes only 3 distinct values)"},
    "horizons": {str(h): results[h] for h in HORIZONS},
    "bootstrap_results": {str(h): boot[h] for h in HORIZONS},
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

import csv
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "test_positives", "auprc_static", "auprc_full",
                "auprc_prefix", "auprc_mixed", "full_gain",
                "recovery_prefix", "recovery_mixed", "mixed_minus_prefix",
                "boot_ci_lower", "boot_ci_upper", "ci_contains_zero"])
    for h in HORIZONS:
        r, b = results[h], boot[h]
        w.writerow([h, r["test_positives"], r["auprc"]["static_only"], r["auprc"]["full"],
                    r["auprc"]["prefix"], r["auprc"]["mixed"], r["full_gain"],
                    r["recovery"]["prefix"], r["recovery"]["mixed"],
                    r["mixed_minus_prefix"], b["ci_lower"], b["ci_upper"],
                    b["contains_zero"]])

# 候補ごとの予測確率は再利用のために保存する（ブートストラップの再現用）
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for h, i in HORIZONS.items()},
    **{f"p_{h}_{c}": preds[h][c] for h in HORIZONS for c in preds[h]},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
