"""G133: 接触の形成での混合の差に、区間を付ける.

G123 では 1.0 ns の切り取りの 6 統計量に生成平均を足すと、30 ns先で
第12案（+0.087）が +0.05 を超えた。G128 では 10 ns先で第13案
（+0.054）と第14案（+0.079）が +0.05 を超えた。
+0.05 を超えた 3 セルについて、混合と 1.0 ns の切り取りの
6 統計量だけの差（mixed_minus_prefix）のブートストラップ 95%
区間を出します（G49・G59・G62・G67・G118 と同じやり方）。

再標本化の単位はタンパク質。テストのタンパク質は 1mix_A と 2qt4_A の
2つだけなので、タンパク質単位の再標本化で取りうる組み合わせは
{1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通りしかありません。
そのためブートストラップ分布も3つの値しか取らず、95% 区間は
その3値の最小値〜最大値になります。区間が粗いことは留保として記録する
（G49・G59・G62・G67・G118 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（G49・G59・G62・G23・G67 と同じ）。

段取り:
- G123・G128 と同じ段取りでモデルを学習し直す（同じ種・
  同じ条件で決定的なため、合算の AUPRC は記録と一致するはずで、一致を
  確認する）。候補ごとの予測確率を保存し、そこからブートストラップする。
  生成平均は K17 の残基レベル生成系列（第11〜14案）を使う。生成器の
  チェックポイントは読み込まない。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 1.0 ns の切り取りの
  6 統計量だけの回復率。回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取りの
  6 統計量だけ・混合の AUPRC を測り直し、利得と回復率をその標本の中で
  計算し直す。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42。

フレームの注記: 「1.0 ns」の切り取りは、呼び名の 1.0 ns に対して
実際は 0.9 ns の範囲である（フレーム [0,1,2,3,4,5,6,7,8,9] は時刻
0〜0.9 ns で、起点のフレーム10 は入らない）。

読み方（先に固定）:
- +0.05 を超えたセル（30 ns の第12案、10 ns の第13・14案）の区間が
  0 を含むかどうかを見る。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")
K17 = Path("/home/hatch/workspace/k17/out")
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り（G123/G128 と同じ）
PROPOSALS = (11, 12, 13, 14)
HORIZONS = {2: 10, 3: 30}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G49・G59・G62・G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G49・G59・G62・G23 と同じ）

# G123・G128 の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    10: {"static_only": 0.3204, "full": 0.5234, "prefix": 0.5066,
        "gen_mean_only_p11": 0.3211, "mixed_p11": 0.5135,
        "gen_mean_only_p12": 0.3391, "mixed_p12": 0.5124,
        "gen_mean_only_p13": 0.3020, "mixed_p13": 0.5175,
        "gen_mean_only_p14": 0.3121, "mixed_p14": 0.5227,
        "full_gain": 0.2030,
        "mixed_minus_prefix": {"p11": 0.034, "p12": 0.029, "p13": 0.054, "p14": 0.079}},
    30: {"static_only": 0.2868, "full": 0.4386, "prefix": 0.4581,
         "gen_mean_only_p11": 0.2804, "mixed_p11": 0.4572,
         "gen_mean_only_p12": 0.3062, "mixed_p12": 0.4714,
         "gen_mean_only_p13": 0.2829, "mixed_p13": 0.4608,
         "gen_mean_only_p14": 0.3057, "mixed_p14": 0.4548,
         "full_gain": 0.1518,
         "mixed_minus_prefix": {"p11": -0.006, "p12": 0.087, "p13": 0.018, "p14": -0.022}},
}


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                       max_leaf_nodes=31, min_samples_leaf=15,
                                       l2_regularization=.02, random_state=42)
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


def load(prots):
    """実側は K13 キャッシュ、生成平均は K17 のファイルから位置で対応付ける（G13 と同じ）。"""
    real, gen = [], {n: [] for n in PROPOSALS}
    proteins = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K13 / f"{p}_R{r}_contact.npz")
            k = np.load(K17 / f"{p}_R{r}.npz")
            assert np.array_equal(k["X"], z["X"]) and np.array_equal(k["hist"], z["hist"]) \
                and np.array_equal(k["y"], z["y"]), f"real-side mismatch {p} R{r}"
            real.append({kk: z[kk] for kk in z.files})
            proteins.extend([p] * len(z["y"]))
            for n in PROPOSALS:
                # 分岐平均の9点系列 -> 9点の平均
                gen[n].append(k[f"gen{n}"].mean(axis=0).mean(axis=0).astype(np.float64))
    return real, gen, np.array(proteins)


train, gtr, _ = load(TRAIN_P)
test, gte, test_protein = load(TEST_P)
Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
Htr = np.concatenate([d["hist"] for d in train])
Hte = np.concatenate([d["hist"] for d in test])
Ytr = np.concatenate([d["y"] for d in train])
Yte = np.concatenate([d["y"] for d in test])
print("train", Xtr.shape, "test", Xte.shape, flush=True)

gen_tr = {n: np.concatenate(gtr[n]) for n in PROPOSALS}
gen_te = {n: np.concatenate(gte[n]) for n in PROPOSALS}

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

CONDS = ["static_only", "full_hist", "prefix_hist6_1.0ns"]
for n in PROPOSALS:
    CONDS.append(f"gen_mean_only_p{n}")
    CONDS.append(f"mixed_p{n}")


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname == "prefix_hist6_1.0ns":
        return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES])]),
                np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES])]))
    if cname.startswith("gen_mean_only_p"):
        n = int(cname[len("gen_mean_only_p"):])
        return (np.column_stack([Xtr, gen_tr[n][:, None].astype(np.float32)]),
                np.column_stack([Xte, gen_te[n][:, None].astype(np.float32)]))
    n = int(cname[len("mixed_p"):])
    return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES]),
                             gen_tr[n][:, None].astype(np.float32)]),
            np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES]),
                             gen_te[n][:, None].astype(np.float32)]))


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
    gain = a["full_hist"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain for c in CONDS if c != "static_only"}
    diff = {f"p{n}": rec[f"mixed_p{n}"] - rec["prefix_hist6_1.0ns"]
            for n in PROPOSALS}
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "recovery_prefix": (ap["prefix_hist6_1.0ns"] - ap["static_only"]) / g if g else float("nan"),
            "recovery_mixed": {f"p{n}": (ap[f"mixed_p{n}"] - ap["static_only"]) / g if g else float("nan")
                               for n in PROPOSALS},
            "mixed_minus_prefix_auprc": {f"p{n}": ap[f"mixed_p{n}"] - ap["prefix_hist6_1.0ns"]
                                         for n in PROPOSALS},
            "mixed_minus_prefix_recovery": {f"p{n}": ((ap[f"mixed_p{n}"] - ap["prefix_hist6_1.0ns"]) / g
                                                     if g else float("nan"))
                                            for n in PROPOSALS},
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "per_protein": per_protein,
    }
    # G123・G128 の記録との照合
    exp = EXPECTED[hns]
    keymap = {"full": "full_hist", "prefix": "prefix_hist6_1.0ns"}
    for c in ("static_only", "full", "prefix"):
        assert abs(a[keymap.get(c, c)] - exp[c]) < 5e-4, (hns, c, a[keymap.get(c, c)], exp[c])
    for n in PROPOSALS:
        assert abs(a[f"gen_mean_only_p{n}"] - exp[f"gen_mean_only_p{n}"]) < 5e-4, \
            (hns, n, a[f"gen_mean_only_p{n}"])
        assert abs(a[f"mixed_p{n}"] - exp[f"mixed_p{n}"]) < 5e-4, (hns, n, a[f"mixed_p{n}"])
        assert abs(diff[f"p{n}"] - exp["mixed_minus_prefix"][f"p{n}"]) < 5e-3, \
            (hns, n, diff[f"p{n}"])
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"prefix {a['prefix_hist6_1.0ns']:.4f} gain {gain:.4f} "
          f"diff " + ", ".join(f"p{n} {diff[f'p{n}']:+.3f}" for n in PROPOSALS)
          + f" (record " + ", ".join(f"p{n} {exp['mixed_minus_prefix'][f'p{n}']:+.3f}" for n in PROPOSALS) + ")",
          flush=True)
    print(f"  per-protein mixed-prefix recovery: "
          + "; ".join(f"{prot} " + ", ".join(
              f"p{n} {per_protein[prot]['mixed_minus_prefix_recovery'][f'p{n}']:+.3f}"
              for n in PROPOSALS) for prot in TEST_P), flush=True)

# ---- タンパク質単位のブートストラップ（G49・G59・G62・G23 と同じ回数・seed・パーセンタイル法） ----
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))
rng = np.random.default_rng(BOOT_SEED)
boot = {}
for hidx, hns in HORIZONS.items():
    yte = Yte[:, hidx]
    p = preds[hns]
    boot[hns] = {}
    for n in PROPOSALS:
        three = {}
        for combo in COMBOS:
            idx = np.concatenate([prot_idx[prot] for prot in combo])
            three["+".join(combo)] = recovery_diff(
                yte[idx], p["static_only"][idx], p["full_hist"][idx],
                p["prefix_hist6_1.0ns"][idx], p[f"mixed_p{n}"][idx])
        combo_idx = rng.integers(0, len(COMBOS), size=N_BOOT)
        vals = np.array([three["+".join(COMBOS[i])] for i in combo_idx])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        boot[hns][f"p{n}"] = {
            "n_boot": N_BOOT, "seed": BOOT_SEED,
            "point_estimate": results[hns]["mixed_minus_prefix"][f"p{n}"],
            "ci_lower": float(lo), "ci_upper": float(hi),
            "contains_zero": bool(lo <= 0 <= hi),
            "three_resample_values": three,
            "n_distinct_boot_values": int(len(np.unique(np.round(vals, 12)))),
        }
    print(f"{hns}ns bootstrap: " + ", ".join(
        f"p{n} {boot[hns][f'p{n}']['point_estimate']:+.3f} "
        f"[{boot[hns][f'p{n}']['ci_lower']:+.3f}, {boot[hns][f'p{n}']['ci_upper']:+.3f}] "
        f"zero={boot[hns][f'p{n}']['contains_zero']}" for n in PROPOSALS), flush=True)

# +0.05 を超えたセルの区間が 0 を含むかどうか（先に固定した読み方）
ABOVE = [(30, 12), (10, 13), (10, 14)]
above_check = {f"{h}ns_p{n}": boot[h][f"p{n}"]["contains_zero"]
               for h, n in ABOVE}
print("above-0.05 cells contain zero:", above_check, flush=True)

out = {
    "label": "Contact formation, hybrid uncertainty across horizons and proposals",
    "phenomenon": "contact formation (DocD, K13 cache + K17 residue-level generated means)",
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
                "auprc_full", "auprc_prefix", "auprc_gen_mean_only", "auprc_mixed",
                "full_gain", "recovery_prefix", "recovery_mixed",
                "mixed_minus_prefix", "boot_ci_lower", "boot_ci_upper",
                "ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        for n in PROPOSALS:
            b = boot[h][f"p{n}"]
            w.writerow([h, n, r["test_positives"],
                        r["auprc"]["static_only"], r["auprc"]["full_hist"],
                        r["auprc"]["prefix_hist6_1.0ns"],
                        r["auprc"][f"gen_mean_only_p{n}"],
                        r["auprc"][f"mixed_p{n}"], r["full_gain"],
                        r["recovery"]["prefix_hist6_1.0ns"],
                        r["recovery"][f"mixed_p{n}"],
                        r["mixed_minus_prefix"][f"p{n}"],
                        b["ci_lower"], b["ci_upper"], b["contains_zero"]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
