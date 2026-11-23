"""G62: 水素結合の解離での混合の差に、区間を付ける.

G18 では 1 ns先の混合の差が -0.052、G44 では 30 ns先で +0.086、
G50 では 50 ns先で +0.015、G54 では 90 ns先で -0.206 でした。
4 つの予測幅について、混合と切り取り平均だけの差（mixed_minus_prefix）の
ブートストラップ 95% 区間を出します（G49・G59 と同じやり方）。

再標本化の単位はタンパク質。テストのタンパク質は 1mix_A と 2qt4_A の
2つだけなので、タンパク質単位の再標本化で取りうる組み合わせは
{1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通りしかありません。
そのためブートストラップ分布も3つの値しか取らず、95% 区間は
その3値の最小値〜最大値になります。区間が粗いことは留保として記録する
（G49・G59 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（G49・G59・G23 と同じ）。

段取り:
- G18・G44・G50・G54 と同じ段取りでモデルを学習し直す（同じ種・同じ
  条件で決定的なため、合算の AUPRC は記録と一致するはずで、一致を
  確認する）。候補ごとの予測確率を保存し、そこからブートストラップする。
  生成平均は作り直した第22案の生成系列（G18 の gen_g18.py、
  ~/workspace/kinescope/work/g18_gen22/）を使う。生成器の
  チェックポイントは読み込まない。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り平均だけの回復率。
  回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り平均だけ・
  混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42。

読み方（先に固定）:
- 30 ns の +0.086 の区間が 0 を含むかどうかを見る。
- 4 つの区間どうしの重なりを見る。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
GEN_DIR = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り（G02/G06/G18 と同じ）
HORIZONS = {0: 1, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G49・G59・G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G49・G59・G23 と同じ）

# G18・G44・G50・G54 の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    1: {"static_only": 0.4160, "full": 0.5945, "prefix": 0.5109,
        "gen_mean_only": 0.4046, "mixed": 0.5016,
        "full_gain": 0.1785, "mixed_minus_prefix": -0.052},
    30: {"static_only": 0.4409, "full": 0.5731, "prefix": 0.5184,
         "gen_mean_only": 0.4283, "mixed": 0.5297,
         "full_gain": 0.1322, "mixed_minus_prefix": 0.086},
    50: {"static_only": 0.4088, "full": 0.5231, "prefix": 0.4731,
         "gen_mean_only": 0.4231, "mixed": 0.4748,
         "full_gain": 0.1142, "mixed_minus_prefix": 0.015},
    90: {"static_only": 0.5118, "full": 0.5797, "prefix": 0.5599,
         "gen_mean_only": 0.4988, "mixed": 0.5459,
         "full_gain": 0.0679, "mixed_minus_prefix": -0.206},
}


def load_gen5():
    """作り直した生成5点系列: [d0, gen(+0.1,+0.3,+0.5,+1.0 ns)]。
    キーは (tid, donor, acceptor)。G18 の run_g18.py と同じ。"""
    gen = {}
    for f in sorted(GEN_DIR.glob("atlas-*.npz")):
        z = np.load(f)
        tid = f.stem
        pair, gd = z["pair"], z["gen_dist"]
        gen5 = np.column_stack([pair[:, 2].astype(np.float64),
                                gd.astype(np.float64)])
        assert gen5.shape[1] == 5
        for i in range(len(pair)):
            gen[(tid, int(pair[i, 0]), int(pair[i, 1]))] = gen5[i]
    print("regenerated pairs:", len(gen), flush=True)
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
prefix_mean_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_mean_te = Hte[:, PREFIX_FRAMES].mean(1)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

CONDS = ("static_only", "full", "prefix", "gen_mean_only", "mixed")
feats = {
    "static_only": None,
    "full": (hist6(Htr), hist6(Hte)),
    "prefix": (prefix_mean_tr[:, None].astype(np.float32),
               prefix_mean_te[:, None].astype(np.float32)),
    "gen_mean_only": (gen_mean_tr[:, None].astype(np.float32),
                      gen_mean_te[:, None].astype(np.float32)),
    "mixed": (np.stack([prefix_mean_tr, gen_mean_tr], axis=1).astype(np.float32),
              np.stack([prefix_mean_te, gen_mean_te], axis=1).astype(np.float32)),
}

results = {}
preds = {}  # horizon_ns -> condition -> test 予測確率
for hidx, hns in HORIZONS.items():
    ytr, yte = Ytr[:, hidx], Yte[:, hidx]
    p = {}
    for cname in CONDS:
        f = feats[cname]
        if f is None:
            p[cname] = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = f
            p[cname] = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                               np.concatenate([Xte, fte], axis=1))
    preds[hns] = p
    a = {c: auprc(yte, p[c]) for c in CONDS}
    gain = a["full"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain
           for c in ("full", "prefix", "gen_mean_only", "mixed")}
    diff = rec["mixed"] - rec["prefix"]
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "recovery_prefix": (ap["prefix"] - ap["static_only"]) / g if g else float("nan"),
            "recovery_mixed": (ap["mixed"] - ap["static_only"]) / g if g else float("nan"),
            "mixed_minus_prefix_auprc": ap["mixed"] - ap["prefix"],
            "mixed_minus_prefix_recovery": ((ap["mixed"] - ap["prefix"]) / g
                                            if g else float("nan")),
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "per_protein": per_protein,
    }
    # G18・G44・G50・G54 の記録との照合
    exp = EXPECTED[hns]
    for c in CONDS:
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(diff - exp["mixed_minus_prefix"]) < 5e-3, (hns, diff)
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full']:.4f} "
          f"prefix {a['prefix']:.4f} gen {a['gen_mean_only']:.4f} "
          f"mixed {a['mixed']:.4f} gain {gain:.4f} "
          f"diff {diff:+.4f} (record {exp['mixed_minus_prefix']:+.3f})", flush=True)
    print(f"  per-protein mixed-prefix recovery: "
          + ", ".join(f"{prot} {per_protein[prot]['mixed_minus_prefix_recovery']:+.4f}"
                      for prot in TEST_P), flush=True)

# ---- タンパク質単位のブートストラップ（G49・G59・G23 と同じ回数・seed・パーセンタイル法） ----
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))
rng = np.random.default_rng(BOOT_SEED)
boot = {}
for hidx, hns in HORIZONS.items():
    yte = Yte[:, hidx]
    p = preds[hns]
    three = {}
    for combo in COMBOS:
        idx = np.concatenate([prot_idx[prot] for prot in combo])
        three["+".join(combo)] = recovery_diff(
            yte[idx], p["static_only"][idx], p["full"][idx],
            p["prefix"][idx], p["mixed"][idx])
    combo_idx = rng.integers(0, len(COMBOS), size=N_BOOT)
    vals = np.array([three["+".join(COMBOS[i])] for i in combo_idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    boot[hns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[hns]["mixed_minus_prefix"],
        "ci_lower": float(lo), "ci_upper": float(hi),
        "contains_zero": bool(lo <= 0 <= hi),
        "three_resample_values": three,
        "n_distinct_boot_values": int(len(np.unique(np.round(vals, 12)))),
    }
    print(f"{hns}ns bootstrap: point {results[hns]['mixed_minus_prefix']:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] contains_zero={lo <= 0 <= hi} "
          f"distinct={boot[hns]['n_distinct_boot_values']}", flush=True)
    print(f"  three values: " + ", ".join(f"{k} {v:+.4f}" for k, v in three.items()),
          flush=True)

# ---- 区間の重なり（先に固定した読み方） ----
def overlap(a, b):
    return not (a["ci_upper"] < b["ci_lower"] or b["ci_upper"] < a["ci_lower"])


ci_overlap = {}
hlist = list(HORIZONS.values())
for i, h1 in enumerate(hlist):
    for h2 in hlist[i + 1:]:
        ci_overlap[f"{h1}ns_vs_{h2}ns"] = bool(overlap(boot[h1], boot[h2]))
print("ci_overlap:", ci_overlap, flush=True)

out = {
    "label": "HB dissociation, hybrid uncertainty across horizons",
    "phenomenon": "hydrogen-bond dissociation (DocD, K46 cache)",
    "prefix_frames": PREFIX_FRAMES,
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (1mix_A, 2qt4_A; 2 proteins only, "
                  "so only 3 resample combinations exist and the bootstrap "
                  "distribution takes only 3 distinct values)",
                  "note": "The 10,000 draws are mapped onto the 3 exact combination "
                  "values; this is mathematically identical to recomputing AUPRC "
                  "on the resampled rows each time, because the difference depends "
                  "only on the chosen combination."},
    "horizons": {str(h): results[h] for h in hlist},
    "bootstrap_results": {str(h): boot[h] for h in hlist},
    "ci_overlap": ci_overlap,
    "contains_zero_30ns": boot[30]["contains_zero"],
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "test_positives", "auprc_static", "auprc_full",
                "auprc_prefix", "auprc_gen_mean_only", "auprc_mixed", "full_gain",
                "recovery_prefix", "recovery_mixed", "mixed_minus_prefix",
                "boot_ci_lower", "boot_ci_upper", "ci_contains_zero"])
    for h in hlist:
        r, b = results[h], boot[h]
        w.writerow([h, r["test_positives"],
                    r["auprc"]["static_only"], r["auprc"]["full"],
                    r["auprc"]["prefix"], r["auprc"]["gen_mean_only"],
                    r["auprc"]["mixed"], r["full_gain"],
                    r["recovery"]["prefix"], r["recovery"]["mixed"],
                    r["mixed_minus_prefix"], b["ci_lower"], b["ci_upper"],
                    b["contains_zero"]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in hlist for c in CONDS},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
