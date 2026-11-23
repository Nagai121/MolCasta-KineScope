"""G117: 塩橋の形成の混合で、切り取り 0.5 ns の 6 個の統計量にした場合の差に、
区間を付ける.

G108 では、切り取り 0.5 ns の 6 個の統計量に生成平均を足した混合の差が、
5 ns先で第12案 +0.244・第13案 +0.140 でした。G112 では、10 ns先で
第12案 +0.052、50 ns先で第11案 +0.405・第12案 +0.378・第13案 +0.552・
第14案 +0.555 でした。5・10・50 ns先のうち +0.05 を超えた 7 セルについて、
混合と切り取り 6統計量だけの差（mixed_minus_prefix）のブートストラップ
95% 区間を出します（G73・G96 と同じやり方）。

再標本化の単位はタンパク質。テストは 30 タンパク質（K15 の広げた条件の
テスト分け）なので、2 タンパク質の土台（G49・G59・G62・G67）のような
「取りうる組み合わせは3通りしかない」という粗さはありません。
30 タンパク質から復元抽出で 30 タンパク質を選び直す、ふつうの
ブートストラップです。区間は 2 タンパク質の土台よりは細かい、という
留保を記録します（G73・G96 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G73・G49・G59・G62・G67・G23・G96 と同じです。

段取り:
- G108・G112 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
  生成平均は G19 で作り直した第11〜14案の生成系列を使う。生成器の
  チェックポイントは読み込まない。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り 6統計量だけの
  回復率。回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り 6統計量
  だけ・混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
  土台の利得が小さいので、AUPRC の差（混合 - 切り取り 6統計量だけ）の
  区間も同じ段取りで出す。50 ns先は土台の利得が +0.0274 と極端に小さく、
  回復率の差の区間が発散するかもしれない。その場合は AUPRC の差で評価する
  （G96 と同じ）。
- 計算の工夫: 1回の抽出で 30 タンパク質を復元抽出し、その行の集合に
  ついて、3つの予測幅 ×（静的のみ・全部入り・切り取り 6統計量だけ・
  混合4案）の AUPRC を測ります。AUPRC の計算は sklearn の
  average_precision_score と同じ段取り（しきい値で区切って精度と再現率
  を取り、再現率の増え分で重み付けする）を numpy で書き直したもので、
  実際の予測確率で sklearn の値と 1e-12 まで一致することを確かめて
  から使います。抽出の乱数 seed は G23 と同じ 42 です。

読み方（先に固定）:
- +0.05 を超えた 7 セル（5 ns先の第12・13案、10 ns先の第12案、
  50 ns先の第11・12・13・14案）の区間が 0 を含むかどうかを見る。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

K15_CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
GEN_DIR = Path(__file__).resolve().parent.parent / "G19_salt_prefix_plus_generated" / "gen"
OUT = Path(__file__).resolve().parent
SPLIT = json.loads((Path(__file__).resolve().parent.parent
                    / "K15_saltbridge_expansion" / "split.json").read_text())
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A") + tuple(SPLIT["train_additions"])
TEST_P = ("1mix_A", "2qt4_A") + tuple(SPLIT["test_additions"])
BOUND = 4.0
PROPOSALS = (11, 12, 13, 14)
PREFIX_FRAMES = [0, 1, 2, 3, 4, 5]  # 0.5 ns の切り取り（G108・G112 と同じ）
HORIZONS = {1: 5, 2: 10, 4: 50}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G73・G49・G59・G62・G67・G23 と同じ）

# G108・G112 の記録との照合用の期待値（合算 AUPRC）
EXPECTED = {
    5: {"static_only": 0.2513, "full_hist6": 0.3301, "prefix_hist6_0.5ns": 0.3029,
        "gen_mean_only_p11": 0.2799, "mixed_p11": 0.3000,
        "gen_mean_only_p12": 0.2788, "mixed_p12": 0.3221,
        "gen_mean_only_p13": 0.2434, "mixed_p13": 0.3139,
        "gen_mean_only_p14": 0.2601, "mixed_p14": 0.3027,
        "full_gain": 0.0788,
        "mixed_minus_prefix": {"p11": -0.037, "p12": 0.244, "p13": 0.140, "p14": -0.003},
        "mixed_minus_prefix_auprc": {"p11": -0.0029, "p12": 0.0193, "p13": 0.0110, "p14": -0.0002}},
    10: {"static_only": 0.2034, "full_hist6": 0.2627, "prefix_hist6_0.5ns": 0.2804,
         "gen_mean_only_p11": 0.1957, "mixed_p11": 0.2710,
         "gen_mean_only_p12": 0.2072, "mixed_p12": 0.2835,
         "gen_mean_only_p13": 0.1900, "mixed_p13": 0.2714,
         "gen_mean_only_p14": 0.1823, "mixed_p14": 0.2698,
         "full_gain": 0.0594,
         "mixed_minus_prefix": {"p11": -0.160, "p12": 0.052, "p13": -0.153, "p14": -0.180},
         "mixed_minus_prefix_auprc": {"p11": -0.0095, "p12": 0.0031, "p13": -0.0091, "p14": -0.0107}},
    50: {"static_only": 0.2295, "full_hist6": 0.2569, "prefix_hist6_0.5ns": 0.2771,
         "gen_mean_only_p11": 0.2163, "mixed_p11": 0.2882,
         "gen_mean_only_p12": 0.2191, "mixed_p12": 0.2875,
         "gen_mean_only_p13": 0.2238, "mixed_p13": 0.2922,
         "gen_mean_only_p14": 0.2178, "mixed_p14": 0.2923,
         "full_gain": 0.0274,
         "mixed_minus_prefix": {"p11": 0.405, "p12": 0.378, "p13": 0.552, "p14": 0.555},
         "mixed_minus_prefix_auprc": {"p11": 0.0111, "p12": 0.0104, "p13": 0.0151, "p14": 0.0152}},
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


def ap_fast(y, s):
    """sklearn の average_precision_score と同じ段取りの numpy 版。
    予測確率の同じ値（タイブ）をしきい値でひとまとめにして扱う。
    実際の予測確率で sklearn の値と 1e-12 まで一致することを確かめてから使う。"""
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


def load(prots):
    """実側は K15 キャッシュ、生成平均は G19 で作り直した系列から残基ペアで対応付ける（G73 と同じ）。"""
    real, gen = [], {n: [] for n in PROPOSALS}
    proteins = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15_CACHE / f"{p}_R{r}_salt.npz")
            d = {k: z[k] for k in z.files}
            g = np.load(GEN_DIR / f"{p}_R{r}.npz")
            assert np.array_equal(g["res_i"], d["res_i"]) and np.array_equal(g["res_j"], d["res_j"]), \
                f"residue pair mismatch {p} R{r}"
            real.append(d)
            proteins.extend([p] * len(d["y"]))
            for n in PROPOSALS:
                # 分岐平均の9点系列 -> 9点の平均（G13/G19 と同じ）
                gen[n].append(g[f"gen{n}"].mean(axis=0).mean(axis=0).astype(np.float64))
    return real, gen, np.array(proteins)


train, gtr, _ = load(TRAIN_P)
test, gte, test_protein = load(TEST_P)
Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
Htr = np.concatenate([d["hist"] for d in train]).astype(np.float64)
Hte = np.concatenate([d["hist"] for d in test]).astype(np.float64)
Ytr = np.concatenate([d["y"] for d in train])
Yte = np.concatenate([d["y"] for d in test])
print("train", Xtr.shape, "test", Xte.shape, "test proteins", len(set(test_protein)),
      flush=True)

prefix_hist6_tr = hist6(Htr[:, PREFIX_FRAMES])
prefix_hist6_te = hist6(Hte[:, PREFIX_FRAMES])
gen_tr = {n: np.concatenate(gtr[n]) for n in PROPOSALS}
gen_te = {n: np.concatenate(gte[n]) for n in PROPOSALS}

prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in list(prot_idx.items())[:5]}, "... total",
      sum(len(v) for v in prot_idx.values()), flush=True)

CONDS = ["static_only", "full_hist6", "prefix_hist6_0.5ns"]
for n in PROPOSALS:
    CONDS.append(f"gen_mean_only_p{n}")
    CONDS.append(f"mixed_p{n}")


def feats_for(cname):
    if cname == "static_only":
        return (Xtr, Xte)
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname == "prefix_hist6_0.5ns":
        return (np.column_stack([Xtr, prefix_hist6_tr]),
                np.column_stack([Xte, prefix_hist6_te]))
    if cname.startswith("gen_mean_only_p"):
        n = int(cname[len("gen_mean_only_p"):])
        return (np.column_stack([Xtr, gen_tr[n][:, None].astype(np.float32)]),
                np.column_stack([Xte, gen_te[n][:, None].astype(np.float32)]))
    n = int(cname[len("mixed_p"):])
    return (np.column_stack([Xtr, prefix_hist6_tr,
                             gen_tr[n][:, None].astype(np.float32)]),
            np.column_stack([Xte, prefix_hist6_te,
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
    # ap_fast が sklearn と一致することの確認（ブートストラップで使うため）
    for c in CONDS:
        assert abs(ap_fast(yte, p[c]) - a[c]) < 1e-12, (hns, c)
    gain = a["full_hist6"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain for c in CONDS if c != "static_only"}
    diff = {f"p{n}": rec[f"mixed_p{n}"] - rec["prefix_hist6_0.5ns"]
            for n in PROPOSALS}
    diff_auprc = {f"p{n}": a[f"mixed_p{n}"] - a["prefix_hist6_0.5ns"]
                  for n in PROPOSALS}
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        if len(idx) == 0:
            continue
        ap = {c: auprc(yte[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist6"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": int(yte[idx].sum()),
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "recovery_prefix": (ap["prefix_hist6_0.5ns"] - ap["static_only"]) / g if g else float("nan"),
            "recovery_mixed": {f"p{n}": (ap[f"mixed_p{n}"] - ap["static_only"]) / g if g else float("nan")
                               for n in PROPOSALS},
            "mixed_minus_prefix_auprc": {f"p{n}": ap[f"mixed_p{n}"] - ap["prefix_hist6_0.5ns"]
                                         for n in PROPOSALS},
            "mixed_minus_prefix_recovery": {f"p{n}": ((ap[f"mixed_p{n}"] - ap["prefix_hist6_0.5ns"]) / g
                                                     if g else float("nan"))
                                            for n in PROPOSALS},
        }
    results[hns] = {
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "mixed_minus_prefix_auprc": diff_auprc,
        "per_protein": per_protein,
    }
    # G108・G112 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist6", "prefix_hist6_0.5ns"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    for n in PROPOSALS:
        assert abs(a[f"gen_mean_only_p{n}"] - exp[f"gen_mean_only_p{n}"]) < 5e-4, \
            (hns, n, a[f"gen_mean_only_p{n}"])
        assert abs(a[f"mixed_p{n}"] - exp[f"mixed_p{n}"]) < 5e-4, (hns, n, a[f"mixed_p{n}"])
        assert abs(diff[f"p{n}"] - exp["mixed_minus_prefix"][f"p{n}"]) < 5e-3, \
            (hns, n, diff[f"p{n}"])
        assert abs(diff_auprc[f"p{n}"] - exp["mixed_minus_prefix_auprc"][f"p{n}"]) < 5e-4, \
            (hns, n, diff_auprc[f"p{n}"])
    print(f"{hns}ns: static {a['static_only']:.4f} full {a['full_hist6']:.4f} "
          f"prefix_hist6 {a['prefix_hist6_0.5ns']:.4f} gain {gain:.4f} "
          f"diff " + ", ".join(f"p{n} {diff[f'p{n}']:+.3f}" for n in PROPOSALS)
          + f" (record " + ", ".join(f"p{n} {exp['mixed_minus_prefix'][f'p{n}']:+.3f}" for n in PROPOSALS) + ")",
          flush=True)
    pos_count = {f"p{n}": sum(1 for prot in per_protein
                              if per_protein[prot]["mixed_minus_prefix_recovery"][f"p{n}"] > 0)
                 for n in PROPOSALS}
    print(f"  per-protein mixed-prefix > 0: " + ", ".join(
        f"p{n} {pos_count[f'p{n}']}/{len(per_protein)}" for n in PROPOSALS), flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
# 実行が途中で止まっても、学習し直しからやり直さずに済むようにする。
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("checkpoint: saved test_predictions.npz", flush=True)

# ---- タンパク質単位のブートストラップ（30 タンパク質からの復元抽出） ----
# 1回の抽出につき行の集合を1回だけ作り、静的のみ・全部入り・切り取り
# 6統計量だけの AUPRC は予測幅ごとに1回だけ測って4案で使い回す。
rng = np.random.default_rng(BOOT_SEED)
prot_list = list(TEST_P)
prot_rows = [prot_idx[p] for p in prot_list]
# 抽出はタンパク質の添字で行い、行はタンパク質ごとにまとめてつなげる
draws = rng.integers(0, len(prot_list), size=(N_BOOT, len(prot_list)))
vals = {hns: {f"p{n}": np.empty(N_BOOT) for n in PROPOSALS}
        for hns in HORIZONS.values()}
vals_auprc = {hns: {f"p{n}": np.empty(N_BOOT) for n in PROPOSALS}
              for hns in HORIZONS.values()}
for b in range(N_BOOT):
    idx = np.concatenate([prot_rows[i] for i in draws[b]])
    for hidx, hns in HORIZONS.items():
        yte = Yte[:, hidx]
        p = preds[hns]
        y = yte[idx]
        ap_s = ap_fast(y, p["static_only"][idx])
        ap_f = ap_fast(y, p["full_hist6"][idx])
        ap_p = ap_fast(y, p["prefix_hist6_0.5ns"][idx])
        gain = ap_f - ap_s
        for n in PROPOSALS:
            ap_m = ap_fast(y, p[f"mixed_p{n}"][idx])
            vals[hns][f"p{n}"][b] = (ap_m - ap_p) / gain if gain else float("nan")
            vals_auprc[hns][f"p{n}"][b] = ap_m - ap_p
    if (b + 1) % 2000 == 0:
        print(f"bootstrap {b + 1}/{N_BOOT} draws done", flush=True)
boot = {}
boot_auprc = {}
for hns in HORIZONS.values():
    boot[hns] = {}
    boot_auprc[hns] = {}
    for n in PROPOSALS:
        v = vals[hns][f"p{n}"]
        lo, hi = np.percentile(v, [2.5, 97.5])
        boot[hns][f"p{n}"] = {
            "n_boot": N_BOOT, "seed": BOOT_SEED,
            "point_estimate": results[hns]["mixed_minus_prefix"][f"p{n}"],
            "ci_lower": float(lo), "ci_upper": float(hi),
            "contains_zero": bool(lo <= 0 <= hi),
            "n_distinct_boot_values": int(len(np.unique(np.round(v, 12)))),
        }
        va = vals_auprc[hns][f"p{n}"]
        loa, hia = np.percentile(va, [2.5, 97.5])
        boot_auprc[hns][f"p{n}"] = {
            "n_boot": N_BOOT, "seed": BOOT_SEED,
            "point_estimate": results[hns]["mixed_minus_prefix_auprc"][f"p{n}"],
            "ci_lower": float(loa), "ci_upper": float(hia),
            "contains_zero": bool(loa <= 0 <= hia),
            "n_distinct_boot_values": int(len(np.unique(np.round(va, 12)))),
        }
    print(f"{hns}ns bootstrap: " + ", ".join(
        f"p{n} {boot[hns][f'p{n}']['point_estimate']:+.3f} "
        f"[{boot[hns][f'p{n}']['ci_lower']:+.3f}, {boot[hns][f'p{n}']['ci_upper']:+.3f}] "
        f"zero={boot[hns][f'p{n}']['contains_zero']}" for n in PROPOSALS), flush=True)
    print(f"{hns}ns bootstrap (AUPRC diff): " + ", ".join(
        f"p{n} {boot_auprc[hns][f'p{n}']['point_estimate']:+.4f} "
        f"[{boot_auprc[hns][f'p{n}']['ci_lower']:+.4f}, {boot_auprc[hns][f'p{n}']['ci_upper']:+.4f}] "
        f"zero={boot_auprc[hns][f'p{n}']['contains_zero']}" for n in PROPOSALS), flush=True)

# +0.05 を超えたセルの区間が 0 を含むかどうか（先に固定した読み方）
ABOVE = [(5, 12), (5, 13), (10, 12),
         (50, 11), (50, 12), (50, 13), (50, 14)]
above_check = {f"{h}ns_p{n}": boot[h][f"p{n}"]["contains_zero"]
               for h, n in ABOVE}
above_check_auprc = {f"{h}ns_p{n}": boot_auprc[h][f"p{n}"]["contains_zero"]
                     for h, n in ABOVE}
print("above-0.05 cells contain zero (recovery):", above_check, flush=True)
print("above-0.05 cells contain zero (AUPRC diff):", above_check_auprc, flush=True)

out = {
    "label": "Salt-bridge formation, 6-statistic-prefix hybrid uncertainty across horizons and proposals",
    "phenomenon": "salt-bridge formation (K15 expanded conditions + G19 rebuilt generated means)",
    "prefix_frames": [0, 1, 2, 3, 4, 5],
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (30 test proteins, sampled with "
                  "replacement; unlike the 2-protein bases, the bootstrap "
                  "distribution takes many distinct values)",
                  "note": "AUPRC inside the bootstrap uses a numpy reimplementation "
                  "of sklearn's average_precision_score (same threshold-grouping "
                  "treatment of ties); it was verified to agree with sklearn to "
                  "1e-12 on all fitted conditions before use. Models were retrained "
                  "per condition and horizon; pooled AUPRCs were verified against "
                  "G108/G112 records to 4 decimal places."},
    "horizons": {str(h): results[h] for h in HORIZONS.values()},
    "bootstrap_results": {str(h): boot[h] for h in HORIZONS.values()},
    "bootstrap_results_auprc": {str(h): boot_auprc[h] for h in HORIZONS.values()},
    "above_0.05_cells_contain_zero": above_check,
    "above_0.05_cells_contain_zero_auprc": above_check_auprc,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "proposal", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix_hist6", "auprc_gen_mean_only", "auprc_mixed",
                "full_gain", "recovery_prefix_hist6", "recovery_mixed",
                "mixed_minus_prefix", "boot_ci_lower", "boot_ci_upper",
                "ci_contains_zero",
                "mixed_minus_prefix_auprc", "boot_auprc_ci_lower", "boot_auprc_ci_upper",
                "auprc_ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        for n in PROPOSALS:
            b = boot[h][f"p{n}"]
            ba = boot_auprc[h][f"p{n}"]
            w.writerow([h, n, r["test_positives"],
                        r["auprc"]["static_only"], r["auprc"]["full_hist6"],
                        r["auprc"]["prefix_hist6_0.5ns"],
                        r["auprc"][f"gen_mean_only_p{n}"],
                        r["auprc"][f"mixed_p{n}"], r["full_gain"],
                        r["recovery"]["prefix_hist6_0.5ns"],
                        r["recovery"][f"mixed_p{n}"],
                        r["mixed_minus_prefix"][f"p{n}"],
                        b["ci_lower"], b["ci_upper"], b["contains_zero"],
                        r["mixed_minus_prefix_auprc"][f"p{n}"],
                        ba["ci_lower"], ba["ci_upper"], ba["contains_zero"]])

np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    **{f"y_{h}": Yte[:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
