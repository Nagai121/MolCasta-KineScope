"""G116: 水素結合の解離の混合で、切り取り 0.5 ns の 6 個の統計量にした場合の差に、
区間を付ける.

G106・G111 では、切り取り 0.5 ns の 6 個の統計量に生成平均を足した混合の差が
+0.05 を超えたのは、1 ns先の第11案 +0.052・第14案 +0.109、
30 ns先の第12案 +0.055・第13案 +0.087、10 ns先の第12案 +0.057・第13案 +0.104、
50 ns先の第11案 +0.077・第12案 +0.119・第14案 +0.092・第22案 +0.087、
90 ns先の第11案 +0.138・第12案 +0.109・第14案 +0.063・第22案 +0.159 の
14セルでした。1・5・10・30・50・90 ns先 × 第11〜14案と第22案について、
混合と切り取り 6統計量だけの差（mixed_minus_prefix）のブートストラップ
95% 区間を出します（G59・G62・G67 と同じやり方）。

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
- G106・G111 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
  生成平均は、第11〜14案が K54 の残基レベル生成系列（4分岐×9点の
  平均）、第22案が G18 で作り直した生成系列（5点系列の平均）を使う。
  生成器のチェックポイントは読み込まない。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り 6統計量だけの
  回復率。回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り 6統計量
  だけ・混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
- 90 ns先は土台の利得が小さい（+0.0679）ため、回復率の差だけでは
  読みを誤ります。90 ns先については、混合と切り取り 6統計量だけの
  AUPRC の差の絶対値も記録します（点推定、タンパク質ごとの値、
  ブートストラップ 95% 区間）。AUPRC の差の区間は、全予測幅について
  同じやり方で出します。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42 です。

読み方（先に固定）:
- +0.05 を超えたセル（1 ns先の第11・14案、30 ns先の第12・13案、
  10 ns先の第12・13案、50 ns先の第11・12・14・22案、
  90 ns先の第11・12・14・22案）の区間が 0 を含むかどうかを見る。
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
K54_DIR = Path("/home/hatch/workspace/kinescope/work/k54_gen_residue")
GEN22_DIR = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
OUT = Path(__file__).resolve().parent
BOUND = 3.5
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5]  # 0.5 ns の切り取り（G106・G111 と同じ）
PROPOSALS = ("p11", "p12", "p13", "p14", "p22")
N_BOOT = 10000  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）

# G106・G111 の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    1: {"static_only": 0.4160, "full_hist6": 0.5945, "prefix_hist6_0.5ns": 0.5539,
        "prefix_mean_0.5ns": 0.5211,
        "gen_mean_only_p11": 0.4179, "mixed_p11": 0.5632,
        "gen_mean_only_p12": 0.4021, "mixed_p12": 0.5600,
        "gen_mean_only_p13": 0.4103, "mixed_p13": 0.5491,
        "gen_mean_only_p14": 0.4176, "mixed_p14": 0.5734,
        "gen_mean_only_p22": 0.4046, "mixed_p22": 0.5457,
        "full_gain": 0.1785,
        "mixed_minus_prefix": {"p11": 0.052, "p12": 0.034, "p13": -0.027,
                               "p14": 0.109, "p22": -0.045}},
    5: {"static_only": 0.4346, "full_hist6": 0.6103, "prefix_hist6_0.5ns": 0.6014,
        "prefix_mean_0.5ns": 0.5965,
        "gen_mean_only_p11": 0.4448, "mixed_p11": 0.6072,
        "gen_mean_only_p12": 0.4423, "mixed_p12": 0.6046,
        "gen_mean_only_p13": 0.4453, "mixed_p13": 0.6060,
        "gen_mean_only_p14": 0.4474, "mixed_p14": 0.5992,
        "gen_mean_only_p22": 0.4348, "mixed_p22": 0.6050,
        "full_gain": 0.1757,
        "mixed_minus_prefix": {"p11": 0.033, "p12": 0.018, "p13": 0.026,
                               "p14": -0.013, "p22": 0.020}},
    10: {"static_only": 0.4496, "full_hist6": 0.5974, "prefix_hist6_0.5ns": 0.5943,
         "prefix_mean_0.5ns": 0.5816,
         "gen_mean_only_p11": 0.4362, "mixed_p11": 0.5913,
         "gen_mean_only_p12": 0.4570, "mixed_p12": 0.6026,
         "gen_mean_only_p13": 0.4463, "mixed_p13": 0.6096,
         "gen_mean_only_p14": 0.4677, "mixed_p14": 0.5970,
         "gen_mean_only_p22": 0.4407, "mixed_p22": 0.5766,
         "full_gain": 0.1478,
         "mixed_minus_prefix": {"p11": -0.020, "p12": 0.057, "p13": 0.104,
                                "p14": 0.018, "p22": -0.120}},
    30: {"static_only": 0.4409, "full_hist6": 0.5731, "prefix_hist6_0.5ns": 0.5634,
         "prefix_mean_0.5ns": 0.5486,
         "gen_mean_only_p11": 0.4277, "mixed_p11": 0.5649,
         "gen_mean_only_p12": 0.4457, "mixed_p12": 0.5706,
         "gen_mean_only_p13": 0.4496, "mixed_p13": 0.5748,
         "gen_mean_only_p14": 0.4325, "mixed_p14": 0.5632,
         "gen_mean_only_p22": 0.4283, "mixed_p22": 0.5473,
         "full_gain": 0.1322,
         "mixed_minus_prefix": {"p11": 0.011, "p12": 0.055, "p13": 0.087,
                                "p14": -0.001, "p22": -0.122}},
    50: {"static_only": 0.4088, "full_hist6": 0.5231, "prefix_hist6_0.5ns": 0.4979,
         "prefix_mean_0.5ns": 0.4828,
         "gen_mean_only_p11": 0.4062, "mixed_p11": 0.5067,
         "gen_mean_only_p12": 0.4085, "mixed_p12": 0.5115,
         "gen_mean_only_p13": 0.4046, "mixed_p13": 0.4925,
         "gen_mean_only_p14": 0.4147, "mixed_p14": 0.5084,
         "gen_mean_only_p22": 0.4231, "mixed_p22": 0.5079,
         "full_gain": 0.1142,
         "mixed_minus_prefix": {"p11": 0.077, "p12": 0.119, "p13": -0.048,
                                "p14": 0.092, "p22": 0.087}},
    90: {"static_only": 0.5118, "full_hist6": 0.5797, "prefix_hist6_0.5ns": 0.5740,
         "prefix_mean_0.5ns": 0.6128,
         "gen_mean_only_p11": 0.5055, "mixed_p11": 0.5834,
         "gen_mean_only_p12": 0.5130, "mixed_p12": 0.5814,
         "gen_mean_only_p13": 0.5031, "mixed_p13": 0.5769,
         "gen_mean_only_p14": 0.4983, "mixed_p14": 0.5783,
         "gen_mean_only_p22": 0.4988, "mixed_p22": 0.5848,
         "full_gain": 0.0679,
         "mixed_minus_prefix": {"p11": 0.138, "p12": 0.109, "p13": 0.044,
                                "p14": 0.063, "p22": 0.159}},
}


def load_gen22():
    """G18 で作り直した第22案の生成系列。キーは (tid, donor, acceptor)。
    gen5 = [起点距離, gen_dist の 4列] の 5点系列。G106・G111 と同じ。"""
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
    parts, proteins = [], []
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
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


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


def auprc(y, p):
    return float(average_precision_score(y, p))


def recovery_diff(y, p_static, p_full, p_prefix, p_mixed):
    """その標本の中での mixed_minus_prefix（回復率の差）。分母は標本内の実履歴の利得。"""
    gain = auprc(y, p_full) - auprc(y, p_static)
    if gain == 0:
        return float("nan")
    return (auprc(y, p_mixed) - auprc(y, p_prefix)) / gain


def auprc_diff(y, p_prefix, p_mixed):
    """その標本の中での混合と切り取り 6統計量だけの AUPRC の差の絶対値。"""
    return auprc(y, p_mixed) - auprc(y, p_prefix)


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Gtr, Gte = cat(train, "genmean"), cat(test, "genmean")
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, flush=True)

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

CONDS = ["static_only", "full_hist6", "prefix_hist6_0.5ns", "prefix_mean_0.5ns"]
for tag in PROPOSALS:
    CONDS.append(f"gen_mean_only_{tag}")
    CONDS.append(f"mixed_{tag}")


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname == "prefix_hist6_0.5ns":
        return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
                np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])]))
    if cname == "prefix_mean_0.5ns":
        return (np.column_stack([Xtr, Htr[:, PREFIX_FRAMES].mean(1)[:, None].astype(np.float32)]),
                np.column_stack([Xte, Hte[:, PREFIX_FRAMES].mean(1)[:, None].astype(np.float32)]))
    if cname.startswith("gen_mean_only_"):
        j = PROPOSALS.index(cname[len("gen_mean_only_"):])
        return (np.column_stack([Xtr, Gtr[:, j:j + 1].astype(np.float32)]),
                np.column_stack([Xte, Gte[:, j:j + 1].astype(np.float32)]))
    j = PROPOSALS.index(cname[len("mixed_"):])
    return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES]),
                             Gtr[:, j:j + 1].astype(np.float32)]),
            np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES]),
                             Gte[:, j:j + 1].astype(np.float32)]))


results = {}
preds = {}  # horizon_ns -> condition -> test 予測確率
for hidx, hns in HORIZONS.items():
    ytr, yte = Ytr[:, hidx], Yte[:, hidx]
    p = {}
    for cname in CONDS:
        xtr, xte = feats_for(cname)
        p[cname] = predict(xtr, ytr, xte)
    preds[hns] = p
    a = {c: auprc(yte, p[c]) for c in CONDS}
    gain = a["full_hist6"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain for c in CONDS if c != "static_only"}
    diff = {tag: rec[f"mixed_{tag}"] - rec["prefix_hist6_0.5ns"]
            for tag in PROPOSALS}
    diff_auprc = {tag: a[f"mixed_{tag}"] - a["prefix_hist6_0.5ns"]
                  for tag in PROPOSALS}
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist6"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "recovery_prefix": (ap["prefix_hist6_0.5ns"] - ap["static_only"]) / g if g else float("nan"),
            "recovery_mixed": {tag: (ap[f"mixed_{tag}"] - ap["static_only"]) / g if g else float("nan")
                               for tag in PROPOSALS},
            "mixed_minus_prefix_auprc": {tag: ap[f"mixed_{tag}"] - ap["prefix_hist6_0.5ns"]
                                         for tag in PROPOSALS},
            "mixed_minus_prefix_recovery": {tag: ((ap[f"mixed_{tag}"] - ap["prefix_hist6_0.5ns"]) / g
                                                  if g else float("nan"))
                                            for tag in PROPOSALS},
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff,
        "mixed_minus_prefix_auprc": diff_auprc,
        "per_protein": per_protein,
    }
    # G106・G111 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist6", "prefix_hist6_0.5ns", "prefix_mean_0.5ns"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    for tag in PROPOSALS:
        assert abs(a[f"gen_mean_only_{tag}"] - exp[f"gen_mean_only_{tag}"]) < 5e-4, \
            (hns, tag, a[f"gen_mean_only_{tag}"])
        assert abs(a[f"mixed_{tag}"] - exp[f"mixed_{tag}"]) < 5e-4, (hns, tag, a[f"mixed_{tag}"])
        assert abs(diff[tag] - exp["mixed_minus_prefix"][tag]) < 5e-3, \
            (hns, tag, diff[tag])
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist6']:.4f} "
          f"prefix {a['prefix_hist6_0.5ns']:.4f} gain {gain:.4f} "
          f"diff " + ", ".join(f"{tag} {diff[tag]:+.3f}" for tag in PROPOSALS)
          + f" (record " + ", ".join(f"{tag} {exp['mixed_minus_prefix'][tag]:+.3f}" for tag in PROPOSALS) + ")"
          + f" auprc-diff " + ", ".join(f"{tag} {diff_auprc[tag]:+.4f}" for tag in PROPOSALS),
          flush=True)
    print(f"  per-protein mixed-prefix recovery: "
          + "; ".join(f"{prot} " + ", ".join(
              f"{tag} {per_protein[prot]['mixed_minus_prefix_recovery'][tag]:+.3f}"
              for tag in PROPOSALS) for prot in TEST_P), flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("checkpoint: saved test_predictions.npz", flush=True)

# ---- タンパク質単位のブートストラップ（G23・G49・G59・G62・G67 と同じ回数・seed・パーセンタイル法） ----
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))
rng = np.random.default_rng(BOOT_SEED)
boot = {}
boot_auprc = {}
for hidx, hns in HORIZONS.items():
    yte = Yte[:, hidx]
    p = preds[hns]
    boot[hns] = {}
    boot_auprc[hns] = {}
    for tag in PROPOSALS:
        three = {}
        three_a = {}
        for combo in COMBOS:
            idx = np.concatenate([prot_idx[prot] for prot in combo])
            three["+".join(combo)] = recovery_diff(
                yte[idx], p["static_only"][idx], p["full_hist6"][idx],
                p["prefix_hist6_0.5ns"][idx], p[f"mixed_{tag}"][idx])
            three_a["+".join(combo)] = auprc_diff(
                yte[idx], p["prefix_hist6_0.5ns"][idx], p[f"mixed_{tag}"][idx])
        combo_idx = rng.integers(0, len(COMBOS), size=N_BOOT)
        vals = np.array([three["+".join(COMBOS[i])] for i in combo_idx])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        boot[hns][tag] = {
            "n_boot": N_BOOT, "seed": BOOT_SEED,
            "point_estimate": results[hns]["mixed_minus_prefix"][tag],
            "ci_lower": float(lo), "ci_upper": float(hi),
            "contains_zero": bool(lo <= 0 <= hi),
            "three_resample_values": three,
            "n_distinct_boot_values": int(len(np.unique(np.round(vals, 12)))),
        }
        vals_a = np.array([three_a["+".join(COMBOS[i])] for i in combo_idx])
        lo_a, hi_a = np.percentile(vals_a, [2.5, 97.5])
        boot_auprc[hns][tag] = {
            "n_boot": N_BOOT, "seed": BOOT_SEED,
            "point_estimate": results[hns]["mixed_minus_prefix_auprc"][tag],
            "ci_lower": float(lo_a), "ci_upper": float(hi_a),
            "contains_zero": bool(lo_a <= 0 <= hi_a),
            "three_resample_values": three_a,
            "n_distinct_boot_values": int(len(np.unique(np.round(vals_a, 12)))),
        }
    print(f"{hns}ns bootstrap: " + ", ".join(
        f"{tag} {boot[hns][tag]['point_estimate']:+.3f} "
        f"[{boot[hns][tag]['ci_lower']:+.3f}, {boot[hns][tag]['ci_upper']:+.3f}] "
        f"zero={boot[hns][tag]['contains_zero']}" for tag in PROPOSALS), flush=True)
    print(f"{hns}ns bootstrap (AUPRC diff): " + ", ".join(
        f"{tag} {boot_auprc[hns][tag]['point_estimate']:+.4f} "
        f"[{boot_auprc[hns][tag]['ci_lower']:+.4f}, {boot_auprc[hns][tag]['ci_upper']:+.4f}] "
        f"zero={boot_auprc[hns][tag]['contains_zero']}" for tag in PROPOSALS), flush=True)

# +0.05 を超えたセルの区間が 0 を含むかどうか（先に固定した読み方）
ABOVE = [(1, "p11"), (1, "p14"), (30, "p12"), (30, "p13"),
         (10, "p12"), (10, "p13"),
         (50, "p11"), (50, "p12"), (50, "p14"), (50, "p22"),
         (90, "p11"), (90, "p12"), (90, "p14"), (90, "p22")]
above_check = {f"{h}ns_{tag}": boot[h][tag]["contains_zero"] for h, tag in ABOVE}
print("above-0.05 cells contain zero:", above_check, flush=True)

out = {
    "label": "HB dissociation, 6-stats prefix hybrid uncertainty at 1/5/10/30/50/90ns",
    "phenomenon": "hydrogen-bond dissociation (DocD, K46 cache + K54 residue-level generated means + G18 regenerated proposal-22 series)",
    "prefix_frames": PREFIX_FRAMES,
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
    "bootstrap_results_auprc_diff": {str(h): boot_auprc[h] for h in HORIZONS.values()},
    "above_0.05_cells_contain_zero": above_check,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "proposal", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_hist6", "auprc_gen_mean_only", "auprc_mixed",
                "full_gain", "recovery_prefix", "recovery_mixed",
                "mixed_minus_prefix", "boot_ci_lower", "boot_ci_upper",
                "ci_contains_zero",
                "mixed_minus_prefix_auprc", "auprc_boot_ci_lower",
                "auprc_boot_ci_upper", "auprc_ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        for tag in PROPOSALS:
            b = boot[h][tag]
            ba = boot_auprc[h][tag]
            w.writerow([h, tag, r["test_positives"],
                        r["auprc"]["static_only"], r["auprc"]["full_hist6"],
                        r["auprc"]["prefix_hist6_0.5ns"],
                        r["auprc"][f"gen_mean_only_{tag}"],
                        r["auprc"][f"mixed_{tag}"], r["full_gain"],
                        r["recovery"]["prefix_hist6_0.5ns"],
                        r["recovery"][f"mixed_{tag}"],
                        r["mixed_minus_prefix"][tag],
                        b["ci_lower"], b["ci_upper"], b["contains_zero"],
                        r["mixed_minus_prefix_auprc"][tag],
                        ba["ci_lower"], ba["ci_upper"], ba["contains_zero"]])

print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
