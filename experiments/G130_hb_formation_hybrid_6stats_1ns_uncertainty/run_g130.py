"""G130: 切り取り 1.0 ns の 6 個の統計量の混合の差に、区間を付ける.

対象は G120 と G125 で +0.05 を超えたセルです（全部で 5セル）。
- G120（水素結合の形成、5・30 ns先）: 30 ns先の第12案 +0.114・第14案 +0.115
- G125（水素結合の形成、1・10・50・90 ns先）: 10 ns先の第13案 +0.090、
  50 ns先の第12案 +0.083・第14案 +0.194
各予測幅について、混合と切り取り 1.0 ns の 6統計量だけの差
（mixed_minus_prefix）のブートストラップ 95% 区間を出します
（G59・G62・G67・G109・G114 と同じやり方）。

切り取りは 1.0 ns（フレーム [0,1,2,3,4,5,6,7,8,9]、実際に覆う長さは
0.9 ns）の 6 個の統計量です（G120・G125 と同じ）。フレームは 100 ps
間隔なので、フレーム [0,1,2,3,4,5,6,7,8,9] が実際に覆う長さは 0.9 ns
で、起点フレーム 10 は入りません。0〜1 ns の履歴全部はフレーム 0〜10
の11フレームです。呼び名の「1.0 ns」と実際に覆う長さ（0.9 ns）は違う
（G120・G125 の記録にも同じ注記がある）。
注記: フレーム [0,1,2] は実際には 0.2 ns を覆う（G100・G105・G110・
G120・G125 の記録にも同じ注記がある）。

再標本化の単位はタンパク質です。テストは 1mix_A と 2qt4_A の 2つだけなので、
取りうる組み合わせは {1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通り
しかなく、ブートストラップ分布も3つの値しか取りません。95% 区間は
その3値の最小値〜最大値になります。区間が粗いことは留保として記録する
（G23・G49・G59・G62・G67・G109・G114 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G23・G49・G59・G62・G67・G109・G114 と同じです。

段取り:
- 水素結合の形成について、G120・G125 と同じ段取りでモデルを
  学習し直す（同じ種・同じ条件で決定的なため、合算の AUPRC は記録と
  一致するはずで、一致を確認する）。候補ごとの予測確率を保存し、
  そこからブートストラップする。生成器のチェックポイントは
  読み込まない。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り 6統計量だけの
  回復率。回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り 6統計量
  だけ・混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
- 土台の利得が小さい予測幅（10 ns先 +0.0829、50 ns先 +0.0739、
  30 ns先 +0.0626）では回復率の差だけでは読みを誤るため、混合と切り取り
  6統計量だけの AUPRC の差の絶対値も記録します（点推定、タンパク質ごとの
  値、ブートストラップ 95% 区間）。G109・G114 と同じです。
- 計算の工夫: 2 タンパク質の現象では、再標本化の標本は3通りの組み合わせ
  のどれかでしかなく、差の値は組み合わせだけで決まります。そのため
  3通りの組み合わせの値を正確に計算し、10,000回の抽出は組み合わせへの
  写像として行います。行を並べ替えて AUPRC を 1万回計算し直す G23 の
  やり方と数学的に同じ結果になります（差が組み合わせの関数であるため）。
  抽出の乱数 seed は G23 と同じ 42 です。

読み方（先に固定）:
- +0.05 を超えたセル（上の 5 セル）の区間が 0 を含むかどうかを見る。
  参考として、第22案の 50 ns先（+0.304）も区間を付けて記録するが、
  先に固定した読みの対象は上の 5セルだけである。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
N_BOOT = 10000  # 結果を見る前に固定（G23・G49・G59・G62・G67・G109 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23・G49・G59・G62・G67・G109 と同じ）
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1.0 ns の切り取り（実際に覆う長さ 0.9 ns）

# G120・G125 の記録との照合用の期待値（合算 AUPRC、小数第4位まで）
EXPECTED = {
    1: {"static_only": 0.3637, "full_hist6": 0.4815, "prefix_hist6_1ns": 0.4940,
        "gen_mean_only_p11": 0.3544, "mixed_p11": 0.4850,
        "gen_mean_only_p12": 0.3519, "mixed_p12": 0.4750,
        "gen_mean_only_p13": 0.3838, "mixed_p13": 0.4715,
        "gen_mean_only_p14": 0.3715, "mixed_p14": 0.4826,
        "gen_mean_only_p22": 0.3746, "mixed_p22": 0.4860,
        "full_gain": 0.1178,
        "mixed_minus_prefix": {"p11": -0.076, "p12": -0.161, "p13": -0.190,
                               "p14": -0.096, "p22": -0.068}},
    5: {"static_only": 0.3493, "full_hist6": 0.4648, "prefix_hist6_1ns": 0.4825,
        "gen_mean_only_p11": 0.3482, "mixed_p11": 0.4635,
        "gen_mean_only_p12": 0.3501, "mixed_p12": 0.4815,
        "gen_mean_only_p13": 0.3336, "mixed_p13": 0.4454,
        "gen_mean_only_p14": 0.3351, "mixed_p14": 0.4786,
        "gen_mean_only_p22": 0.3634, "mixed_p22": 0.4941,
        "full_gain": 0.1155,
        "mixed_minus_prefix": {"p11": -0.165, "p12": -0.009, "p13": -0.321,
                               "p14": -0.034, "p22": 0.100}},
    10: {"static_only": 0.3548, "full_hist6": 0.4377, "prefix_hist6_1ns": 0.4140,
         "gen_mean_only_p11": 0.3668, "mixed_p11": 0.4168,
         "gen_mean_only_p12": 0.3474, "mixed_p12": 0.4130,
         "gen_mean_only_p13": 0.3651, "mixed_p13": 0.4214,
         "gen_mean_only_p14": 0.3554, "mixed_p14": 0.4042,
         "gen_mean_only_p22": 0.3752, "mixed_p22": 0.4264,
         "full_gain": 0.0829,
         "mixed_minus_prefix": {"p11": 0.034, "p12": -0.012, "p13": 0.090,
                                "p14": -0.118, "p22": 0.150}},
    30: {"static_only": 0.3639, "full_hist6": 0.4264, "prefix_hist6_1ns": 0.4517,
         "gen_mean_only_p11": 0.3745, "mixed_p11": 0.4505,
         "gen_mean_only_p12": 0.3767, "mixed_p12": 0.4588,
         "gen_mean_only_p13": 0.3683, "mixed_p13": 0.4441,
         "gen_mean_only_p14": 0.3734, "mixed_p14": 0.4589,
         "gen_mean_only_p22": 0.3823, "mixed_p22": 0.4557,
         "full_gain": 0.0626,
         "mixed_minus_prefix": {"p11": -0.019, "p12": 0.114, "p13": -0.121,
                                "p14": 0.115, "p22": 0.065}},
    50: {"static_only": 0.3342, "full_hist6": 0.4081, "prefix_hist6_1ns": 0.4048,
         "gen_mean_only_p11": 0.3319, "mixed_p11": 0.3997,
         "gen_mean_only_p12": 0.3295, "mixed_p12": 0.4110,
         "gen_mean_only_p13": 0.3188, "mixed_p13": 0.4063,
         "gen_mean_only_p14": 0.3246, "mixed_p14": 0.4191,
         "gen_mean_only_p22": 0.3663, "mixed_p22": 0.4272,
         "full_gain": 0.0739,
         "mixed_minus_prefix": {"p11": -0.068, "p12": 0.083, "p13": 0.021,
                                "p14": 0.194, "p22": 0.304}},
    90: {"static_only": 0.3035, "full_hist6": 0.4246, "prefix_hist6_1ns": 0.4300,
         "gen_mean_only_p11": 0.3046, "mixed_p11": 0.4194,
         "gen_mean_only_p12": 0.3123, "mixed_p12": 0.4243,
         "gen_mean_only_p13": 0.3210, "mixed_p13": 0.4179,
         "gen_mean_only_p14": 0.3026, "mixed_p14": 0.4226,
         "gen_mean_only_p22": 0.3356, "mixed_p22": 0.4197,
         "full_gain": 0.1211,
         "mixed_minus_prefix": {"p11": -0.087, "p12": -0.047, "p13": -0.100,
                                "p14": -0.061, "p22": -0.085}},
}

# +0.05 を超えたセル（先に固定した読みの対象）。割り当て文の列挙は 5セル。
ABOVE = [(30, "p12"), (30, "p14"),
         (10, "p13"),
         (50, "p12"), (50, "p14")]


def hist6(series, bound):
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


def cat(parts, key):
    arrs = [p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)


def load_hb_formation():
    """水素結合の形成。DocD キャッシュ＋K11（第11〜14案）＋K05 の第22案。G120・G125 と同じ。"""
    CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
    K11_DIR = REPO / "experiments" / "K11_method_generality" / "series"
    K05_GEN = REPO / "experiments" / "K05_generated_quality" / "generated_series.npz"
    TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
    TEST_P = ("1mix_A", "2qt4_A")
    PROPOSALS = ("p11", "p12", "p13", "p14", "p22")
    gen, origin = {}, {}
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

    def load(prots):
        parts, proteins = [], []
        for p in prots:
            for r in (1, 2, 3):
                z = np.load(CACHE / f"{p}_R{r}.npz")
                d = {k: z[k] for k in z.files}
                tid = f"atlas-{p}-analysis-R{r}"
                g = np.stack([[gen[(tid, int(da), int(aa))][tag] for tag in PROPOSALS]
                              for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
                d["genmean"] = g
                parts.append(d)
                proteins.extend([p] * len(d["y"]))
        return parts, np.array(proteins)

    train, _ = load(TRAIN_P)
    test, test_protein = load(TEST_P)
    return (cat(train, "X"), cat(test, "X"), cat(train, "hist"), cat(test, "hist"),
            cat(train, "y"), cat(test, "y"), cat(train, "genmean"), cat(test, "genmean"),
            test_protein, PROPOSALS, 3.5, {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90})


def main():
    Xtr, Xte, Htr, Hte, Ytr, Yte, Gtr, Gte, test_protein, PROPOSALS, BOUND, HORIZONS = \
        load_hb_formation()
    print("train", Xtr.shape, "test", Xte.shape,
          "test proteins:", {p: int((test_protein == p).sum()) for p in np.unique(test_protein)},
          flush=True)
    prot_idx = {p: np.where(test_protein == p)[0] for p in np.unique(test_protein)}
    test_prots = list(np.unique(test_protein))
    assert test_prots == ["1mix_A", "2qt4_A"], test_prots

    CONDS = ["static_only", "full_hist6", "prefix_hist6_1ns", "prefix_mean_1ns"]
    for tag in PROPOSALS:
        CONDS.append(f"gen_mean_only_{tag}")
        CONDS.append(f"mixed_{tag}")

    def feats_for(cname):
        if cname == "static_only":
            return (Xtr, Xte)
        if cname == "full_hist6":
            return (np.column_stack([Xtr, hist6(Htr, BOUND)]),
                    np.column_stack([Xte, hist6(Hte, BOUND)]))
        if cname == "prefix_hist6_1ns":
            return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES], BOUND)]),
                    np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES], BOUND)]))
        if cname == "prefix_mean_1ns":
            return (np.column_stack([Xtr, Htr[:, PREFIX_FRAMES].mean(1)[:, None].astype(np.float32)]),
                    np.column_stack([Xte, Hte[:, PREFIX_FRAMES].mean(1)[:, None].astype(np.float32)]))
        if cname.startswith("gen_mean_only_"):
            j = PROPOSALS.index(cname[len("gen_mean_only_"):])
            return (np.column_stack([Xtr, Gtr[:, j:j + 1].astype(np.float32)]),
                    np.column_stack([Xte, Gte[:, j:j + 1].astype(np.float32)]))
        j = PROPOSALS.index(cname[len("mixed_"):])
        return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES], BOUND),
                                 Gtr[:, j:j + 1].astype(np.float32)]),
                np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES], BOUND),
                                 Gte[:, j:j + 1].astype(np.float32)]))

    results = {}
    preds = {}
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
        diff = {tag: rec[f"mixed_{tag}"] - rec["prefix_hist6_1ns"]
                for tag in PROPOSALS}
        diff_auprc = {tag: a[f"mixed_{tag}"] - a["prefix_hist6_1ns"]
                      for tag in PROPOSALS}
        per_protein = {}
        for prot in test_prots:
            idx = prot_idx[prot]
            ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
            g = ap["full_hist6"] - ap["static_only"]
            per_protein[prot] = {
                "n": int(len(idx)), "positives": int(yte[idx].sum()),
                "auprc": {c: ap[c] for c in CONDS},
                "full_gain": g,
                "recovery_prefix": (ap["prefix_hist6_1ns"] - ap["static_only"]) / g if g else float("nan"),
                "recovery_mixed": {tag: (ap[f"mixed_{tag}"] - ap["static_only"]) / g if g else float("nan")
                                   for tag in PROPOSALS},
                "mixed_minus_prefix_auprc": {tag: ap[f"mixed_{tag}"] - ap["prefix_hist6_1ns"]
                                             for tag in PROPOSALS},
                "mixed_minus_prefix_recovery": {tag: ((ap[f"mixed_{tag}"] - ap["prefix_hist6_1ns"]) / g
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
        # 記録との照合
        exp = EXPECTED[hns]
        for c in ("static_only", "full_hist6", "prefix_hist6_1ns"):
            assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
        for tag in PROPOSALS:
            assert abs(a[f"gen_mean_only_{tag}"] - exp[f"gen_mean_only_{tag}"]) < 5e-4, \
                (hns, tag, a[f"gen_mean_only_{tag}"])
            assert abs(a[f"mixed_{tag}"] - exp[f"mixed_{tag}"]) < 5e-4, (hns, tag, a[f"mixed_{tag}"])
            assert abs(diff[tag] - exp["mixed_minus_prefix"][tag]) < 5e-3, \
                (hns, tag, diff[tag])
        print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist6']:.4f} "
              f"prefix {a['prefix_hist6_1ns']:.4f} gain {gain:.4f} "
              f"diff " + ", ".join(f"{tag} {diff[tag]:+.3f}" for tag in PROPOSALS)
              + f" (record " + ", ".join(f"{tag} {exp['mixed_minus_prefix'][tag]:+.3f}" for tag in PROPOSALS) + ")"
              + f" auprc-diff " + ", ".join(f"{tag} {diff_auprc[tag]:+.4f}" for tag in PROPOSALS),
              flush=True)
        print(f"  per-protein mixed-prefix recovery: "
              + "; ".join(f"{prot} " + ", ".join(
                  f"{tag} {per_protein[prot]['mixed_minus_prefix_recovery'][tag]:+.3f}"
                  for tag in PROPOSALS) for prot in test_prots), flush=True)

    # ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
    np.savez_compressed(
        OUT / "test_predictions_hb_formation.npz",
        test_protein=test_protein,
        **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
        **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
    )
    print("checkpoint: saved test_predictions_hb_formation.npz", flush=True)

    # ---- ブートストラップ（2 タンパク質: 3通りの組み合わせの値を正確に計算し、
    # 10,000回の抽出は組み合わせへの写像として行う。G23・G109 と同じ） ----
    COMBOS = ((test_prots[0], test_prots[0]),
              (test_prots[0], test_prots[1]),
              (test_prots[1], test_prots[1]))
    rng = np.random.default_rng(BOOT_SEED)
    boot, boot_auprc = {}, {}
    for hidx, hns in HORIZONS.items():
        yte = Yte[:, hidx]
        p = preds[hns]
        boot[hns] = {}
        boot_auprc[hns] = {}
        for tag in PROPOSALS:
            three, three_a = {}, {}
            for combo in COMBOS:
                idx = np.concatenate([prot_idx[prot] for prot in combo])
                three["+".join(combo)] = recovery_diff(
                    yte[idx], p["static_only"][idx], p["full_hist6"][idx],
                    p["prefix_hist6_1ns"][idx], p[f"mixed_{tag}"][idx])
                three_a["+".join(combo)] = auprc_diff(
                    yte[idx], p["prefix_hist6_1ns"][idx], p[f"mixed_{tag}"][idx])
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

    # +0.05 を超えたセル（14セル）の区間が 0 を含むかどうか（先に固定した読み方）
    above_check = {f"{h}ns_{tag}": boot[h][tag]["contains_zero"] for h, tag in ABOVE}
    above_check_auprc = {f"{h}ns_{tag}": boot_auprc[h][tag]["contains_zero"] for h, tag in ABOVE}
    print("above-0.05 cells contain zero:", above_check, flush=True)
    print("above-0.05 cells contain zero (AUPRC diff):", above_check_auprc, flush=True)

    out = {
        "label": "Hybrid 1ns-prefix 6-stats uncertainty (G120/G125)",
        "phenomenon": "hb_formation",
        "prefix_frames": PREFIX_FRAMES,
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                      "resampling_unit": "protein",
                      "note": "Test has only 2 proteins, so only 3 resample combinations "
                      "exist and the bootstrap distribution takes only 3 distinct values; "
                      "the 10,000 draws are mapped onto the 3 exact combination values "
                      "(G23 method)."},
        "above_0.05_cells": [{"horizon_ns": h, "proposal": tag} for h, tag in ABOVE],
        "above_0.05_cells_contain_zero": above_check,
        "above_0.05_cells_contain_zero_auprc_diff": above_check_auprc,
        "horizons": {str(h): results[h] for h in results},
        "bootstrap_results": {str(h): boot[h] for h in boot},
        "bootstrap_results_auprc_diff": {str(h): boot_auprc[h] for h in boot_auprc},
        "test_proteins": test_prots,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phenomenon", "horizon_ns", "proposal", "test_positives", "auprc_static",
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
                w.writerow(["hb_formation", h, tag, r["test_positives"],
                            r["auprc"]["static_only"],
                            r["auprc"]["full_hist6"],
                            r["auprc"]["prefix_hist6_1ns"],
                            r["auprc"][f"gen_mean_only_{tag}"],
                            r["auprc"][f"mixed_{tag}"], r["full_gain"],
                            r["recovery"]["prefix_hist6_1ns"],
                            r["recovery"][f"mixed_{tag}"],
                            r["mixed_minus_prefix"][tag],
                            b["ci_lower"], b["ci_upper"], b["contains_zero"],
                            r["mixed_minus_prefix_auprc"][tag],
                            ba["ci_lower"], ba["ci_upper"], ba["contains_zero"]])

    print("saved", OUT / "results.json", OUT / "results.csv")


if __name__ == "__main__":
    main()
