"""G109: 切り取り 0.25 ns の 6 個の統計量の混合の差に、区間を付ける.

対象は G100・G102・G103・G104 で +0.05 を超えたセルです。
- G100（水素結合の形成、1・30 ns先）: 30 ns先の第13案 +0.106・第22案 +0.624
- G102（塩橋の形成、5・30 ns先）: 5 ns先の第12案 +0.122、30 ns先の
  第11案 +0.117・第12案 +0.167・第14案 +0.051
- G103（水素結合の解離、1・30 ns先）: 1 ns先の第11案 +0.051・第13案 +0.051
- G104（接触の形成、1・30 ns先）: 1 ns先の第13案 +0.062、30 ns先の第11案 +0.064
各現象について、混合と切り取り 0.25 ns の 6統計量だけの差
（mixed_minus_prefix）のブートストラップ 95% 区間を出します
（G59・G62・G67 と同じやり方）。

切り取りは 0.25 ns（フレーム [0,1,2]、実際には 0.2 ns を覆う）の
6 個の統計量です（G100・G102・G103・G104 と同じ）。

再標本化の単位はタンパク質です。ただし塩橋だけはテストが 30 タンパク質
あるため、タンパク質単位の真の再標本化を行います（G73・G96 と同じ）。
それ以外の 3 現象はテストが 1mix_A と 2qt4_A の 2つだけなので、
取りうる組み合わせは {1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通り
しかなく、ブートストラップ分布も3つの値しか取りません。95% 区間は
その3値の最小値〜最大値になります。区間が粗いことは留保として記録する
（G23・G49・G59・G62・G67 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G23・G49・G59・G62・G67 と同じです。

段取り:
- 各現象について、G100・G102・G103・G104 と同じ段取りでモデルを
  学習し直す（同じ種・同じ条件で決定的なため、合算の AUPRC は記録と
  一致するはずで、一致を確認する）。候補ごとの予測確率を保存し、
  そこからブートストラップする。生成器のチェックポイントは
  読み込まない。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り 6統計量だけの
  回復率。回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り 6統計量
  だけ・混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
- 塩橋のセルは土台の利得が小さい（5 ns先 +0.0788、30 ns先 +0.0884）ため、
  回復率の差だけでは読みを誤ります。塩橋については、混合と切り取り
  6統計量だけの AUPRC の差の絶対値も記録します（点推定、タンパク質ごとの
  値、ブートストラップ 95% 区間）。AUPRC の差の区間は、ほかの 3 現象に
  ついても同じやり方で出します。
- 計算の工夫: 2 タンパク質の現象では、再標本化の標本は3通りの組み合わせ
  のどれかでしかなく、差の値は組み合わせだけで決まります。そのため
  3通りの組み合わせの値を正確に計算し、10,000回の抽出は組み合わせへの
  写像として行います。行を並べ替えて AUPRC を 1万回計算し直す G23 の
  やり方と数学的に同じ結果になります（差が組み合わせの関数であるため）。
  塩橋では、1回の抽出につき行の集合を1回だけ作り、静的のみ・全部入り・
  切り取り 6統計量だけの AUPRC は予測幅ごとに1回だけ測って4案で
  使い回します。AUPRC の計算には sklearn と同じ段取りの numpy 版
  （ap_fast、しきい値で区切って精度と再現率を測る）を使い、実際の
  予測確率で sklearn の値と 1e-12 まで一致することを確かめてから使います
  （G73・G96 と同じ）。抽出の乱数 seed は G23 と同じ 42 です。

読み方（先に固定）:
- +0.05 を超えたセル（上の 9 セル）の区間が 0 を含むかどうかを見る。
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
N_BOOT = 10000  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）
PREFIX_FRAMES = [0, 1, 2]  # 0.25 ns の切り取り（実際には 0.2 ns を覆う）

# 各現象の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    "hb_formation": {
        1: {"static_only": 0.3637, "full_hist6": 0.4815, "prefix_hist6_0.25ns": 0.4462,
            "gen_mean_only_p11": 0.3544, "mixed_p11": 0.4505,
            "gen_mean_only_p12": 0.3519, "mixed_p12": 0.4421,
            "gen_mean_only_p13": 0.3838, "mixed_p13": 0.4257,
            "gen_mean_only_p14": 0.3715, "mixed_p14": 0.4281,
            "gen_mean_only_p22": 0.3746, "mixed_p22": 0.4450,
            "full_gain": 0.1178,
            "mixed_minus_prefix": {"p11": 0.037, "p12": -0.035, "p13": -0.174,
                                   "p14": -0.154, "p22": -0.010}},
        30: {"static_only": 0.3639, "full_hist6": 0.4264, "prefix_hist6_0.25ns": 0.4051,
             "gen_mean_only_p11": 0.3745, "mixed_p11": 0.4047,
             "gen_mean_only_p12": 0.3767, "mixed_p12": 0.4027,
             "gen_mean_only_p13": 0.3683, "mixed_p13": 0.4117,
             "gen_mean_only_p14": 0.3734, "mixed_p14": 0.4053,
             "gen_mean_only_p22": 0.3823, "mixed_p22": 0.4441,
             "full_gain": 0.0626,
             "mixed_minus_prefix": {"p11": -0.006, "p12": -0.039, "p13": 0.106,
                                    "p14": 0.004, "p22": 0.624}},
    },
    "salt_formation": {
        5: {"static_only": 0.2513, "full_hist6": 0.3301, "prefix_hist6_0.25ns": 0.3108,
            "gen_mean_only_p11": 0.2799, "mixed_p11": 0.3099,
            "gen_mean_only_p12": 0.2788, "mixed_p12": 0.3204,
            "gen_mean_only_p13": 0.2434, "mixed_p13": 0.2955,
            "gen_mean_only_p14": 0.2601, "mixed_p14": 0.2912,
            "full_gain": 0.0788,
            "mixed_minus_prefix": {"p11": -0.012, "p12": 0.122, "p13": -0.194, "p14": -0.249}},
        30: {"static_only": 0.1582, "full_hist6": 0.2466, "prefix_hist6_0.25ns": 0.2186,
             "gen_mean_only_p11": 0.1582, "mixed_p11": 0.2290,
             "gen_mean_only_p12": 0.1611, "mixed_p12": 0.2333,
             "gen_mean_only_p13": 0.1585, "mixed_p13": 0.2169,
             "gen_mean_only_p14": 0.1548, "mixed_p14": 0.2231,
             "full_gain": 0.0884,
             "mixed_minus_prefix": {"p11": 0.117, "p12": 0.167, "p13": -0.019, "p14": 0.051}},
    },
    "hb_dissociation": {
        1: {"static_only": 0.4160, "full_hist6": 0.5945, "prefix_hist6_0.25ns": 0.5493,
            "prefix_mean_0.25ns": 0.5183,
            "gen_mean_only_p11": 0.4179, "mixed_p11": 0.5585,
            "gen_mean_only_p12": 0.4021, "mixed_p12": 0.5515,
            "gen_mean_only_p13": 0.4103, "mixed_p13": 0.5585,
            "gen_mean_only_p14": 0.4176, "mixed_p14": 0.5530,
            "gen_mean_only_p22": 0.4046, "mixed_p22": 0.5436,
            "full_gain": 0.1785,
            "mixed_minus_prefix": {"p11": 0.051, "p12": 0.012, "p13": 0.051,
                                   "p14": 0.021, "p22": -0.032}},
        30: {"static_only": 0.4409, "full_hist6": 0.5731, "prefix_hist6_0.25ns": 0.5512,
             "prefix_mean_0.25ns": 0.5288,
             "gen_mean_only_p11": 0.4277, "mixed_p11": 0.5493,
             "gen_mean_only_p12": 0.4457, "mixed_p12": 0.5505,
             "gen_mean_only_p13": 0.4496, "mixed_p13": 0.5491,
             "gen_mean_only_p14": 0.4325, "mixed_p14": 0.5439,
             "gen_mean_only_p22": 0.4283, "mixed_p22": 0.5398,
             "full_gain": 0.1322,
             "mixed_minus_prefix": {"p11": -0.014, "p12": -0.005, "p13": -0.016,
                                    "p14": -0.056, "p22": -0.086}},
    },
    "contact_formation": {
        1: {"static_only": 0.3398, "full_hist": 0.5669, "prefix_hist6_0.25ns": 0.4793,
            "prefix_mean_0.25ns": 0.4561,
            "gen_mean_only_p11": 0.3469, "mixed_p11": 0.4728,
            "gen_mean_only_p12": 0.3409, "mixed_p12": 0.4831,
            "gen_mean_only_p13": 0.3527, "mixed_p13": 0.4935,
            "gen_mean_only_p14": 0.3352, "mixed_p14": 0.4798,
            "full_gain": 0.2271,
            "mixed_minus_prefix": {"p11": -0.029, "p12": 0.017, "p13": 0.062, "p14": 0.002}},
        30: {"static_only": 0.2868, "full_hist": 0.4386, "prefix_hist6_0.25ns": 0.4100,
             "prefix_mean_0.25ns": 0.4142,
             "gen_mean_only_p11": 0.2804, "mixed_p11": 0.4196,
             "gen_mean_only_p12": 0.3062, "mixed_p12": 0.3990,
             "gen_mean_only_p13": 0.2829, "mixed_p13": 0.3947,
             "gen_mean_only_p14": 0.3057, "mixed_p14": 0.3945,
             "full_gain": 0.1518,
             "mixed_minus_prefix": {"p11": 0.064, "p12": -0.072, "p13": -0.101, "p14": -0.102}},
    },
}


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


def ap_fast(y, s):
    """sklearn の average_precision_score と同じ段取りの numpy 版。
    予測確率の同じ値（タイブ）をしきい値でひとまとめにして扱う。
    実際の予測確率で sklearn の値と 1e-12 まで一致することを確かめてから使う（G73・G96 と同じ）。"""
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    total_pos = ys.sum()
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(ys)
    change = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[change[1:] - 1, len(ys) - 1]
    tp_t = tp[ends]
    prec = tp_t / (ends + 1)
    rec = tp_t / total_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


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


# ---------- 各現象の読み込み（G100・G102・G103・G104 と同じ段取り） ----------

def load_hb_formation():
    """水素結合の形成。DocD キャッシュ＋K11（第11〜14案）＋K05 の第22案。G80・G100 と同じ。"""
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
            test_protein, PROPOSALS, 3.5, {0: 1, 3: 30})


def load_salt_formation():
    """塩橋の形成。K15 キャッシュ＋G19 で作り直した系列（第11〜14案）。G102 と同じ。"""
    K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
    K15_DIR = REPO / "experiments" / "K15_saltbridge_expansion"
    GEN_DIR = REPO / "experiments" / "G19_salt_prefix_plus_generated" / "gen"
    ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
    ORIG_TEST = ("1mix_A", "2qt4_A")
    PROPOSALS = ("p11", "p12", "p13", "p14")

    def load(prots):
        real, gen, proteins = [], {tag: [] for tag in PROPOSALS}, []
        for p in prots:
            for r in (1, 2, 3):
                z = np.load(K15 / f"{p}_R{r}_salt.npz")
                d = {k: z[k] for k in z.files}
                g = np.load(GEN_DIR / f"{p}_R{r}.npz")
                assert np.array_equal(g["res_i"], d["res_i"]) and np.array_equal(g["res_j"], d["res_j"]), \
                    f"residue pair mismatch {p} R{r}"
                real.append(d)
                proteins.extend([p] * len(d["y"]))
                for j, tag in enumerate(PROPOSALS):
                    gen[tag].append(g[f"gen{11 + j}"].mean(axis=0).mean(axis=0).astype(np.float64))
        return real, gen, np.array(proteins)

    split = json.loads((K15_DIR / "split.json").read_text())
    train_p = ORIG_TRAIN + tuple(split["train_additions"])
    test_p = ORIG_TEST + tuple(split["test_additions"])
    train, gtr, _ = load(train_p)
    test, gte, test_protein = load(test_p)
    Gtr = np.column_stack([np.concatenate(gtr[t]) for t in PROPOSALS])
    Gte = np.column_stack([np.concatenate(gte[t]) for t in PROPOSALS])
    return (np.concatenate([d["X"] for d in train]).astype(np.float64),
            np.concatenate([d["X"] for d in test]).astype(np.float64),
            np.concatenate([d["hist"] for d in train]).astype(np.float64),
            np.concatenate([d["hist"] for d in test]).astype(np.float64),
            np.concatenate([d["y"] for d in train]),
            np.concatenate([d["y"] for d in test]),
            Gtr, Gte, test_protein, PROPOSALS, 4.0, {1: 5, 3: 30})


def load_hb_dissociation():
    """水素結合の解離。K46 キャッシュ＋K54（第11〜14案）＋G18 の第22案。G95・G103 と同じ。"""
    CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
    K54_DIR = Path("/home/hatch/workspace/kinescope/work/k54_gen_residue")
    GEN22_DIR = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
    TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
    TEST_P = ("1mix_A", "2qt4_A")
    PROPOSALS = ("p11", "p12", "p13", "p14", "p22")
    gen22 = {}
    for f in sorted(GEN22_DIR.glob("atlas-*.npz")):
        z = np.load(f)
        tid = f.stem
        pair, gd = z["pair"], z["gen_dist"]
        gen5 = np.column_stack([pair[:, 2].astype(np.float64), gd.astype(np.float64)])
        assert gen5.shape[1] == 5, gen5.shape
        for i in range(len(pair)):
            gen22[(tid, int(pair[i, 0]), int(pair[i, 1]))] = gen5[i]
    print("regenerated #22 pairs:", len(gen22), flush=True)

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
                              for j in range(4)], axis=1)
                g22 = np.array([gen22[(tid, int(da), int(aa))].mean()
                                for da, aa in zip(d["donor_atom"], d["acceptor_atom"])])
                d["genmean"] = np.column_stack([g, g22[:, None]])
                parts.append(d)
                proteins.extend([p] * len(d["y"]))
        return parts, np.array(proteins)

    train, _ = load(TRAIN_P)
    test, test_protein = load(TEST_P)
    return (cat(train, "X"), cat(test, "X"), cat(train, "hist"), cat(test, "hist"),
            cat(train, "y"), cat(test, "y"), cat(train, "genmean"), cat(test, "genmean"),
            test_protein, PROPOSALS, 3.5, {0: 1, 3: 30})


def load_contact_formation():
    """接触の形成。K13 キャッシュ＋K17（第11〜14案）。G84・G104 と同じ。"""
    K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")
    K17 = Path("/home/hatch/workspace/k17/out")
    TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
    TEST_P = ("1mix_A", "2qt4_A")
    PROPOSALS = (11, 12, 13, 14)

    def load(prots):
        real, gen, proteins = [], {n: [] for n in PROPOSALS}, []
        for p in prots:
            for r in (1, 2, 3):
                z = np.load(K13 / f"{p}_R{r}_contact.npz")
                k = np.load(K17 / f"{p}_R{r}.npz")
                assert np.array_equal(k["X"], z["X"]) and np.array_equal(k["hist"], z["hist"]) \
                    and np.array_equal(k["y"], z["y"]), f"real-side mismatch {p} R{r}"
                real.append({kk: z[kk] for kk in z.files})
                proteins.extend([p] * len(z["y"]))
                for n in PROPOSALS:
                    gen[n].append(k[f"gen{n}"].mean(axis=0).mean(axis=0).astype(np.float64))
        return real, gen, np.array(proteins)

    train, gtr, _ = load(TRAIN_P)
    test, gte, test_protein = load(TEST_P)
    Gtr = np.column_stack([np.concatenate(gtr[n]) for n in PROPOSALS])
    Gte = np.column_stack([np.concatenate(gte[n]) for n in PROPOSALS])
    return (np.concatenate([d["X"] for d in train]).astype(np.float64),
            np.concatenate([d["X"] for d in test]).astype(np.float64),
            np.concatenate([d["hist"] for d in train]),
            np.concatenate([d["hist"] for d in test]),
            np.concatenate([d["y"] for d in train]),
            np.concatenate([d["y"] for d in test]),
            Gtr, Gte, test_protein, tuple(f"p{n}" for n in PROPOSALS), 4.5, {0: 1, 3: 30})


PARTS = [
    ("hb_formation", "HB formation (G100)", load_hb_formation, "full_hist6"),
    ("salt_formation", "Salt-bridge formation (G102)", load_salt_formation, "full_hist6"),
    ("hb_dissociation", "HB dissociation (G103)", load_hb_dissociation, "full_hist6"),
    ("contact_formation", "Contact formation (G104)", load_contact_formation, "full_hist"),
]


def run_part(name, label, loader, full_key):
    """1つの現象について学習・評価し、予測確率を保存してブートストラップする。"""
    print(f"===== {label} =====", flush=True)
    Xtr, Xte, Htr, Hte, Ytr, Yte, Gtr, Gte, test_protein, PROPOSALS, BOUND, HORIZONS = loader()
    print("train", Xtr.shape, "test", Xte.shape,
          "test proteins:", {p: int((test_protein == p).sum()) for p in np.unique(test_protein)},
          flush=True)
    prot_idx = {p: np.where(test_protein == p)[0] for p in np.unique(test_protein)}
    test_prots = list(np.unique(test_protein))

    CONDS = ["static_only", full_key, "prefix_hist6_0.25ns", "prefix_mean_0.25ns"]
    for tag in PROPOSALS:
        CONDS.append(f"gen_mean_only_{tag}")
        CONDS.append(f"mixed_{tag}")

    def feats_for(cname):
        if cname == "static_only":
            return (Xtr, Xte)
        if cname == full_key:
            return (np.column_stack([Xtr, hist6(Htr, BOUND)]),
                    np.column_stack([Xte, hist6(Hte, BOUND)]))
        if cname == "prefix_hist6_0.25ns":
            return (np.column_stack([Xtr, hist6(Htr[:, PREFIX_FRAMES], BOUND)]),
                    np.column_stack([Xte, hist6(Hte[:, PREFIX_FRAMES], BOUND)]))
        if cname == "prefix_mean_0.25ns":
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
        gain = a[full_key] - a["static_only"]
        rec = {c: (a[c] - a["static_only"]) / gain for c in CONDS if c != "static_only"}
        diff = {tag: rec[f"mixed_{tag}"] - rec["prefix_hist6_0.25ns"]
                for tag in PROPOSALS}
        diff_auprc = {tag: a[f"mixed_{tag}"] - a["prefix_hist6_0.25ns"]
                      for tag in PROPOSALS}
        per_protein = {}
        for prot in test_prots:
            idx = prot_idx[prot]
            ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
            g = ap[full_key] - ap["static_only"]
            per_protein[prot] = {
                "n": int(len(idx)), "positives": int(yte[idx].sum()),
                "auprc": {c: ap[c] for c in CONDS},
                "full_gain": g,
                "recovery_prefix": (ap["prefix_hist6_0.25ns"] - ap["static_only"]) / g if g else float("nan"),
                "recovery_mixed": {tag: (ap[f"mixed_{tag}"] - ap["static_only"]) / g if g else float("nan")
                                   for tag in PROPOSALS},
                "mixed_minus_prefix_auprc": {tag: ap[f"mixed_{tag}"] - ap["prefix_hist6_0.25ns"]
                                             for tag in PROPOSALS},
                "mixed_minus_prefix_recovery": {tag: ((ap[f"mixed_{tag}"] - ap["prefix_hist6_0.25ns"]) / g
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
        exp = EXPECTED[name][hns]
        for c in ("static_only", full_key, "prefix_hist6_0.25ns"):
            assert abs(a[c] - exp[c]) < 5e-4, (name, hns, c, a[c], exp[c])
        for tag in PROPOSALS:
            assert abs(a[f"gen_mean_only_{tag}"] - exp[f"gen_mean_only_{tag}"]) < 5e-4, \
                (name, hns, tag, a[f"gen_mean_only_{tag}"])
            assert abs(a[f"mixed_{tag}"] - exp[f"mixed_{tag}"]) < 5e-4, (name, hns, tag, a[f"mixed_{tag}"])
            assert abs(diff[tag] - exp["mixed_minus_prefix"][tag]) < 5e-3, \
                (name, hns, tag, diff[tag])
        print(f"{label} {hns}ns: static {a['static_only']:.4f} full {a[full_key]:.4f} "
              f"prefix {a['prefix_hist6_0.25ns']:.4f} gain {gain:.4f} "
              f"diff " + ", ".join(f"{tag} {diff[tag]:+.3f}" for tag in PROPOSALS)
              + f" (record " + ", ".join(f"{tag} {exp['mixed_minus_prefix'][tag]:+.3f}" for tag in PROPOSALS) + ")"
              + f" auprc-diff " + ", ".join(f"{tag} {diff_auprc[tag]:+.4f}" for tag in PROPOSALS),
              flush=True)
        if len(test_prots) == 2:
            print(f"  per-protein mixed-prefix recovery: "
                  + "; ".join(f"{prot} " + ", ".join(
                      f"{tag} {per_protein[prot]['mixed_minus_prefix_recovery'][tag]:+.3f}"
                      for tag in PROPOSALS) for prot in test_prots), flush=True)

    # ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
    np.savez_compressed(
        OUT / f"test_predictions_{name}.npz",
        test_protein=test_protein,
        **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
        **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
    )
    print(f"checkpoint: saved test_predictions_{name}.npz", flush=True)

    # ---- ブートストラップ ----
    if name == "salt_formation":
        boot, boot_auprc = bootstrap_30prot(name, HORIZONS, Yte, preds, results,
                                            test_prots, prot_idx, full_key, PROPOSALS)
    else:
        boot, boot_auprc = bootstrap_2prot(name, HORIZONS, Yte, preds, results,
                                           test_prots, prot_idx, full_key, PROPOSALS)
    return {"results": results, "bootstrap": boot, "bootstrap_auprc": boot_auprc,
            "test_proteins": test_prots}


def bootstrap_2prot(name, HORIZONS, Yte, preds, results, test_prots, prot_idx, full_key, PROPOSALS):
    """2 タンパク質の場合: 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
    組み合わせへの写像として行う（G23 のやり方と数学的に同じ結果になる）。"""
    assert len(test_prots) == 2, test_prots
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
                    yte[idx], p["static_only"][idx], p[full_key][idx],
                    p["prefix_hist6_0.25ns"][idx], p[f"mixed_{tag}"][idx])
                three_a["+".join(combo)] = auprc_diff(
                    yte[idx], p["prefix_hist6_0.25ns"][idx], p[f"mixed_{tag}"][idx])
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
        print(f"{name} {hns}ns bootstrap: " + ", ".join(
            f"{tag} {boot[hns][tag]['point_estimate']:+.3f} "
            f"[{boot[hns][tag]['ci_lower']:+.3f}, {boot[hns][tag]['ci_upper']:+.3f}] "
            f"zero={boot[hns][tag]['contains_zero']}" for tag in PROPOSALS), flush=True)
        print(f"{name} {hns}ns bootstrap (AUPRC diff): " + ", ".join(
            f"{tag} {boot_auprc[hns][tag]['point_estimate']:+.4f} "
            f"[{boot_auprc[hns][tag]['ci_lower']:+.4f}, {boot_auprc[hns][tag]['ci_upper']:+.4f}] "
            f"zero={boot_auprc[hns][tag]['contains_zero']}" for tag in PROPOSALS), flush=True)
    return boot, boot_auprc


def bootstrap_30prot(name, HORIZONS, Yte, preds, results, test_prots, prot_idx, full_key, PROPOSALS):
    """30 タンパク質の場合: タンパク質単位の真の再標本化（G73・G96 と同じ）。
    1回の抽出につき行の集合を1回だけ作り、静的のみ・全部入り・切り取り
    6統計量だけの AUPRC は予測幅ごとに1回だけ測って4案で使い回す。"""
    # ap_fast が sklearn と一致することを、学習し直した条件で確かめてから使う
    max_err = 0.0
    for hns in HORIZONS.values():
        for c, p in preds[hns].items():
            yte = Yte[:, [i for i, h in HORIZONS.items() if h == hns][0]]
            err = abs(ap_fast(yte, p) - auprc(yte, p))
            max_err = max(max_err, err)
    print(f"{name}: ap_fast vs sklearn max abs error {max_err:.2e}", flush=True)
    assert max_err < 1e-12, max_err
    rng = np.random.default_rng(BOOT_SEED)
    prot_rows = [prot_idx[p] for p in test_prots]
    draws = rng.integers(0, len(test_prots), size=(N_BOOT, len(test_prots)))
    vals = {hns: {tag: np.empty(N_BOOT) for tag in PROPOSALS} for hns in HORIZONS.values()}
    vals_auprc = {hns: {tag: np.empty(N_BOOT) for tag in PROPOSALS} for hns in HORIZONS.values()}
    for b in range(N_BOOT):
        idx = np.concatenate([prot_rows[i] for i in draws[b]])
        for hidx, hns in HORIZONS.items():
            yte = Yte[:, hidx]
            p = preds[hns]
            y = yte[idx]
            ap_s = ap_fast(y, p["static_only"][idx])
            ap_f = ap_fast(y, p[full_key][idx])
            ap_p = ap_fast(y, p["prefix_hist6_0.25ns"][idx])
            gain = ap_f - ap_s
            for tag in PROPOSALS:
                ap_m = ap_fast(y, p[f"mixed_{tag}"][idx])
                vals[hns][tag][b] = (ap_m - ap_p) / gain if gain else float("nan")
                vals_auprc[hns][tag][b] = ap_m - ap_p
        if (b + 1) % 2000 == 0:
            print(f"{name} bootstrap {b + 1}/{N_BOOT} draws done", flush=True)
    boot, boot_auprc = {}, {}
    for hns in HORIZONS.values():
        boot[hns] = {}
        boot_auprc[hns] = {}
        for tag in PROPOSALS:
            v = vals[hns][tag]
            lo, hi = np.percentile(v, [2.5, 97.5])
            boot[hns][tag] = {
                "n_boot": N_BOOT, "seed": BOOT_SEED,
                "point_estimate": results[hns]["mixed_minus_prefix"][tag],
                "ci_lower": float(lo), "ci_upper": float(hi),
                "contains_zero": bool(lo <= 0 <= hi),
                "n_distinct_boot_values": int(len(np.unique(np.round(v, 12)))),
            }
            va = vals_auprc[hns][tag]
            loa, hia = np.percentile(va, [2.5, 97.5])
            boot_auprc[hns][tag] = {
                "n_boot": N_BOOT, "seed": BOOT_SEED,
                "point_estimate": results[hns]["mixed_minus_prefix_auprc"][tag],
                "ci_lower": float(loa), "ci_upper": float(hia),
                "contains_zero": bool(loa <= 0 <= hia),
                "n_distinct_boot_values": int(len(np.unique(np.round(va, 12)))),
            }
        print(f"{name} {hns}ns bootstrap: " + ", ".join(
            f"{tag} {boot[hns][tag]['point_estimate']:+.3f} "
            f"[{boot[hns][tag]['ci_lower']:+.3f}, {boot[hns][tag]['ci_upper']:+.3f}] "
            f"zero={boot[hns][tag]['contains_zero']}" for tag in PROPOSALS), flush=True)
        print(f"{name} {hns}ns bootstrap (AUPRC diff): " + ", ".join(
            f"{tag} {boot_auprc[hns][tag]['point_estimate']:+.4f} "
            f"[{boot_auprc[hns][tag]['ci_lower']:+.4f}, {boot_auprc[hns][tag]['ci_upper']:+.4f}] "
            f"zero={boot_auprc[hns][tag]['contains_zero']}" for tag in PROPOSALS), flush=True)
    return boot, boot_auprc


all_out = {}
for name, label, loader, full_key in PARTS:
    all_out[name] = run_part(name, label, loader, full_key)

# +0.05 を超えたセルの区間が 0 を含むかどうか（先に固定した読み方）
ABOVE = [("hb_formation", 30, "p13"), ("hb_formation", 30, "p22"),
         ("salt_formation", 5, "p12"),
         ("salt_formation", 30, "p11"), ("salt_formation", 30, "p12"), ("salt_formation", 30, "p14"),
         ("hb_dissociation", 1, "p11"), ("hb_dissociation", 1, "p13"),
         ("contact_formation", 1, "p13"), ("contact_formation", 30, "p11")]
above_check = {f"{name}_{h}ns_{tag}": all_out[name]["bootstrap"][h][tag]["contains_zero"]
               for name, h, tag in ABOVE}
print("above-0.05 cells contain zero:", above_check, flush=True)

out = {
    "label": "Hybrid 0.25ns-prefix 6-stats uncertainty (G100/G102/G103/G104)",
    "prefix_frames": PREFIX_FRAMES,
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein",
                  "note": "Salt-bridge formation uses a true protein-level resample "
                  "(30 test proteins, G73/G96 method with a numpy average_precision_score "
                  "verified to agree with sklearn to 1e-12). The other three phenomena "
                  "have only 2 test proteins, so only 3 resample combinations exist and "
                  "the bootstrap distribution takes only 3 distinct values; the 10,000 "
                  "draws are mapped onto the 3 exact combination values (G23 method)."},
    "parts": {name: {"horizons": {str(h): v["results"][h] for h in v["results"]},
                     "bootstrap_results": {str(h): v["bootstrap"][h] for h in v["bootstrap"]},
                     "bootstrap_results_auprc_diff": {str(h): v["bootstrap_auprc"][h]
                                                      for h in v["bootstrap_auprc"]},
                     "test_proteins": v["test_proteins"]}
              for name, v in all_out.items()},
    "above_0.05_cells_contain_zero": above_check,
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
    for name, v in all_out.items():
        for h, r in v["results"].items():
            for tag in v["bootstrap"][h]:
                b = v["bootstrap"][h][tag]
                ba = v["bootstrap_auprc"][h][tag]
                w.writerow([name, h, tag, r["test_positives"],
                            r["auprc"]["static_only"],
                            r["auprc"]["full_hist6" if "full_hist6" in r["auprc"] else "full_hist"],
                            r["auprc"]["prefix_hist6_0.25ns"],
                            r["auprc"][f"gen_mean_only_{tag}"],
                            r["auprc"][f"mixed_{tag}"], r["full_gain"],
                            r["recovery"]["prefix_hist6_0.25ns"],
                            r["recovery"][f"mixed_{tag}"],
                            r["mixed_minus_prefix"][tag],
                            b["ci_lower"], b["ci_upper"], b["contains_zero"],
                            r["mixed_minus_prefix_auprc"][tag],
                            ba["ci_lower"], ba["ci_upper"], ba["contains_zero"]])

print("saved", OUT / "results.json", OUT / "results.csv")
