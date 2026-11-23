"""G88: 水素結合の形成の混合で、切り取りを 6 個の統計量にした場合の差に、
区間を付ける.

G80 では、切り取り 0.1 ns の 6 個の統計量に生成平均を足した混合の差が、
1 ns先で第11案 -0.021・第12案 +0.108・第13案 +0.018・第14案 +0.073・
第22案 +0.155、30 ns先で第11案 +0.202・第12案 -0.125・第13案 +0.091・
第14案 +0.163・第22案 +0.547 でした。1・30 ns先 × 第11〜14案と
第22案について、混合と切り取り 6統計量だけの差（mixed_minus_prefix）
のブートストラップ 95% 区間を出します（G59・G62・G67 と同じやり方）。

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
- G80 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
  生成平均は、第11〜14案が K11 の系列（4分岐×8点の平均）、第22案が
  K05 の生成系列（5点系列の平均）を使う。生成器のチェックポイントは
  読み込まない。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り 6統計量だけの
  回復率。回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り 6統計量
  だけ・混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42 です。

読み方（先に固定）:
- +0.05 を超えたセル（1 ns先の第12・14案と第22案、30 ns先の
  第11・13・14案と第22案）の区間が 0 を含むかどうかを見る。
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
K11_DIR = Path(__file__).resolve().parent.parent / "K11_method_generality" / "series"
K05_GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUT = Path(__file__).resolve().parent
BOUND = 3.5
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = {0: 1, 3: 30}  # y の列番号 -> ns
PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り（G80 と同じ）
PROPOSALS = ("p11", "p12", "p13", "p14", "p22")
N_BOOT = 10000  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）

# G80 の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    1: {"static_only": 0.3637, "full_hist6": 0.4815, "prefix_hist6_0.1ns": 0.4324,
        "gen_mean_only_p11": 0.3544, "mixed_p11": 0.4299,
        "gen_mean_only_p12": 0.3519, "mixed_p12": 0.4451,
        "gen_mean_only_p13": 0.3838, "mixed_p13": 0.4344,
        "gen_mean_only_p14": 0.3715, "mixed_p14": 0.4410,
        "gen_mean_only_p22": 0.3746, "mixed_p22": 0.4506,
        "full_gain": 0.1178,
        "mixed_minus_prefix": {"p11": -0.021, "p12": 0.108, "p13": 0.018,
                               "p14": 0.073, "p22": 0.155}},
    30: {"static_only": 0.3639, "full_hist6": 0.4264, "prefix_hist6_0.1ns": 0.4106,
         "gen_mean_only_p11": 0.3745, "mixed_p11": 0.4232,
         "gen_mean_only_p12": 0.3767, "mixed_p12": 0.4028,
         "gen_mean_only_p13": 0.3683, "mixed_p13": 0.4163,
         "gen_mean_only_p14": 0.3734, "mixed_p14": 0.4208,
         "gen_mean_only_p22": 0.3823, "mixed_p22": 0.4448,
         "full_gain": 0.0626,
         "mixed_minus_prefix": {"p11": 0.202, "p12": -0.125, "p13": 0.091,
                                "p14": 0.163, "p22": 0.547}},
}


def load_gen_means():
    """第11〜14案（K11）と第22案（K05）の生成平均。キーは (tid, donor, acceptor)。
    G80 の load_gen_means と同じ。"""
    gen = {}
    origin = {}
    for p in TRAIN_P + TEST_P:
        for r in (1, 2, 3):
            tid = f"atlas-{p}-analysis-R{r}"
            z = np.load(K11_DIR / f"atlas-{p}-analysis-R{r}.npz")
            pair = z["pair"]
            for tag, key in (("p11", "elas11"), ("p12", "elas12"),
                             ("p13", "elas13"), ("p14", "elas14")):
                m = np.nanmean(z[key].astype(np.float64), axis=(1, 2))
                for i in range(len(pair)):
                    gen.setdefault((tid, int(pair[i, 0]), int(pair[i, 1])), {})[tag] = m[i]
            for i in range(len(pair)):
                origin[(tid, int(pair[i, 0]), int(pair[i, 1]))] = float(pair[i, 2])
    z = np.load(K05_GEN)
    tids, pair, g = z["tid"], z["pair"], z["gen"]
    gen5 = np.column_stack([pair[:, 2].astype(np.float64), g.astype(np.float64)])
    assert gen5.shape == (14582, 5), gen5.shape
    for i, t in enumerate(tids):
        gen[(str(t), int(pair[i, 0]), int(pair[i, 1]))]["p22"] = float(gen5[i].mean())
    print("generated pairs:", len(gen), flush=True)
    return gen, origin


GEN, ORIGIN = load_gen_means()


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            d = {k: z[k] for k in z.files}
            tid = f"atlas-{p}-analysis-R{r}"
            g = np.stack([[GEN[(tid, int(da), int(aa))][tag] for tag in PROPOSALS]
                          for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
            d["genmean"] = g
            d["origin"] = np.array([ORIGIN[(tid, int(da), int(aa))]
                                    for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
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


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Gtr, Gte = cat(train, "genmean"), cat(test, "genmean")
Ytr, Yte = cat(train, "y"), cat(test, "y")
print("train", Xtr.shape, "test", Xte.shape, flush=True)
Otr, Ote = cat(train, "origin"), cat(test, "origin")
print("K11 origin vs hist last: max abs diff train %.2e test %.2e" % (
    np.abs(Otr - Htr[:, -1]).max(), np.abs(Ote - Hte[:, -1]).max()), flush=True)
assert np.abs(Otr - Htr[:, -1]).max() < 1e-5
assert np.abs(Ote - Hte[:, -1]).max() < 1e-5

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

CONDS = ["static_only", "full_hist6", "prefix_hist6_0.1ns"]
for tag in PROPOSALS:
    CONDS.append(f"gen_mean_only_{tag}")
    CONDS.append(f"mixed_{tag}")


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname == "prefix_hist6_0.1ns":
        return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
                np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])]))
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
    diff = {tag: rec[f"mixed_{tag}"] - rec["prefix_hist6_0.1ns"]
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
            "recovery_prefix": (ap["prefix_hist6_0.1ns"] - ap["static_only"]) / g if g else float("nan"),
            "recovery_mixed": {tag: (ap[f"mixed_{tag}"] - ap["static_only"]) / g if g else float("nan")
                               for tag in PROPOSALS},
            "mixed_minus_prefix_auprc": {tag: ap[f"mixed_{tag}"] - ap["prefix_hist6_0.1ns"]
                                         for tag in PROPOSALS},
            "mixed_minus_prefix_recovery": {tag: ((ap[f"mixed_{tag}"] - ap["prefix_hist6_0.1ns"]) / g
                                                  if g else float("nan"))
                                            for tag in PROPOSALS},
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "per_protein": per_protein,
    }
    # G80 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist6", "prefix_hist6_0.1ns"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    for tag in PROPOSALS:
        assert abs(a[f"gen_mean_only_{tag}"] - exp[f"gen_mean_only_{tag}"]) < 5e-4, \
            (hns, tag, a[f"gen_mean_only_{tag}"])
        assert abs(a[f"mixed_{tag}"] - exp[f"mixed_{tag}"]) < 5e-4, (hns, tag, a[f"mixed_{tag}"])
        assert abs(diff[tag] - exp["mixed_minus_prefix"][tag]) < 5e-3, \
            (hns, tag, diff[tag])
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist6']:.4f} "
          f"prefix {a['prefix_hist6_0.1ns']:.4f} gain {gain:.4f} "
          f"diff " + ", ".join(f"{tag} {diff[tag]:+.3f}" for tag in PROPOSALS)
          + f" (record " + ", ".join(f"{tag} {exp['mixed_minus_prefix'][tag]:+.3f}" for tag in PROPOSALS) + ")",
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
for hidx, hns in HORIZONS.items():
    yte = Yte[:, hidx]
    p = preds[hns]
    boot[hns] = {}
    for tag in PROPOSALS:
        three = {}
        for combo in COMBOS:
            idx = np.concatenate([prot_idx[prot] for prot in combo])
            three["+".join(combo)] = recovery_diff(
                yte[idx], p["static_only"][idx], p["full_hist6"][idx],
                p["prefix_hist6_0.1ns"][idx], p[f"mixed_{tag}"][idx])
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
    print(f"{hns}ns bootstrap: " + ", ".join(
        f"{tag} {boot[hns][tag]['point_estimate']:+.3f} "
        f"[{boot[hns][tag]['ci_lower']:+.3f}, {boot[hns][tag]['ci_upper']:+.3f}] "
        f"zero={boot[hns][tag]['contains_zero']}" for tag in PROPOSALS), flush=True)

# +0.05 を超えたセルの区間が 0 を含むかどうか（先に固定した読み方）
ABOVE = [(1, "p12"), (1, "p14"), (1, "p22"), (30, "p11"), (30, "p13"), (30, "p14"), (30, "p22")]
above_check = {f"{h}ns_{tag}": boot[h][tag]["contains_zero"] for h, tag in ABOVE}
print("above-0.05 cells contain zero:", above_check, flush=True)

out = {
    "label": "HB formation, 6-stats prefix hybrid uncertainty at 1/30ns",
    "phenomenon": "hydrogen-bond formation (DocD, K11 residue-level + K05 generated means)",
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
    "above_0.05_cells_contain_zero": above_check,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "proposal", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_hist6", "auprc_gen_mean_only", "auprc_mixed",
                "full_gain", "recovery_prefix", "recovery_mixed",
                "mixed_minus_prefix", "boot_ci_lower", "boot_ci_upper",
                "ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        for tag in PROPOSALS:
            b = boot[h][tag]
            w.writerow([h, tag, r["test_positives"],
                        r["auprc"]["static_only"], r["auprc"]["full_hist6"],
                        r["auprc"]["prefix_hist6_0.1ns"],
                        r["auprc"][f"gen_mean_only_{tag}"],
                        r["auprc"][f"mixed_{tag}"], r["full_gain"],
                        r["recovery"]["prefix_hist6_0.1ns"],
                        r["recovery"][f"mixed_{tag}"],
                        r["mixed_minus_prefix"][tag],
                        b["ci_lower"], b["ci_upper"], b["contains_zero"]])

print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
