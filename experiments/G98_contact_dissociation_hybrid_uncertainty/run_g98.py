"""G98: 接触の解離の混合の差に、区間を付ける.

G79・G85・G89 では、接触の解離の混合の差が +0.05 を超えたのは
90 ns先の +0.054 だけで、1 ns先 -0.040、5 ns先 +0.033、10 ns先 +0.035、
30 ns先 +0.025、50 ns先 -0.010 でした。全予測幅（1・5・10・30・50・90 ns先）
について、混合と切り取り平均だけの差（mixed_minus_prefix）の
ブートストラップ 95% 区間を出します（G59・G62・G67 と同じやり方）。

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
- G79・G85・G89 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で
  決定的なため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
  生成平均は、G18 の第22案の生成系列（ドナー・アクセプター原子ペアごと
  の 4点系列）から、接触の候補（残基ペア）に重なるものだけを取り、
  残基ペアごとに生成ペアの 4点系列の点ごとの最小値を取って 4点で
  平均したものを使う（G79 の定義）。生成器のチェックポイントは
  読み込まない。
- 測るのは、G18 の生成系列と重なる候補だけです（学習 4,489件、
  テスト 2,226件、テストの 76%）。重ならない候補は測れません。
  テストの陽性のうち重なるのは、1 ns先で 105件（全 207件のうち）、
  5 ns先で 110件（211件）、10 ns先で 122件、30 ns先で 124件（246件）、
  50 ns先で 149件、90 ns先で 160件です。土台（静的のみ・全部入り・
  切り取り平均だけ）も、重なる候補だけの集合で測り直しています。
  そのため回復率の分母は、重なる候補だけの集合での実履歴の利得です。
  全集合での土台は別に確かめます（下の「全集合での土台の確認」）。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り平均だけの
  回復率。回復率の分母は、その予測幅での実履歴の利得。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り平均
  だけ・混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42 です。

読み方（先に固定）:
- +0.05 を超えたセル（90 ns先だけ）の区間が 0 を含むかどうかを見る。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

K50 = Path("/home/hatch/workspace/kinescope/work/k50_cache")
MATCHED = Path(__file__).resolve().parent.parent / "G79_contact_dissociation_hybrid_1ns" / "matched_gen_mean.npz"
OUT = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
PREFIX_FRAMES = [0, 1]  # 0.1 ns の切り取り（G79・G85・G89 と同じ）
HORIZONS = {0: 1, 1: 5, 2: 10, 3: 30, 4: 50, 5: 90}  # y の列番号 -> ns
N_BOOT = 10000  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）

# G79・G85・G89 の記録との照合用の期待値（合算 AUPRC、重なる候補だけの集合）
EXPECTED = {
    1: {"static_only": 0.2879, "full_hist": 0.4567, "prefix_mean_0.1ns": 0.4689,
        "gen_mean_only": 0.2901, "mixed": 0.4622,
        "full_gain": 0.1688, "mixed_minus_prefix": -0.040,
        "test_positives_matched": 105},
    5: {"static_only": 0.2705, "full_hist": 0.5918, "prefix_mean_0.1ns": 0.5054,
        "gen_mean_only": 0.3075, "mixed": 0.5161,
        "full_gain": 0.3213, "mixed_minus_prefix": 0.033,
        "test_positives_matched": 110},
    10: {"static_only": 0.3065, "full_hist": 0.5095, "prefix_mean_0.1ns": 0.4160,
         "gen_mean_only": 0.3048, "mixed": 0.4231,
         "full_gain": 0.2030, "mixed_minus_prefix": 0.035,
         "test_positives_matched": 122},
    30: {"static_only": 0.2488, "full_hist": 0.4395, "prefix_mean_0.1ns": 0.3857,
         "gen_mean_only": 0.2509, "mixed": 0.3905,
         "full_gain": 0.1907, "mixed_minus_prefix": 0.025,
         "test_positives_matched": 124},
    50: {"static_only": 0.2653, "full_hist": 0.4490, "prefix_mean_0.1ns": 0.3664,
         "gen_mean_only": 0.2711, "mixed": 0.3645,
         "full_gain": 0.1837, "mixed_minus_prefix": -0.010,
         "test_positives_matched": 149},
    90: {"static_only": 0.3294, "full_hist": 0.4818, "prefix_mean_0.1ns": 0.4527,
         "gen_mean_only": 0.3452, "mixed": 0.4610,
         "full_gain": 0.1524, "mixed_minus_prefix": 0.054,
         "test_positives_matched": 160},
}

# 全集合での土台の記録値（K50 の静的のみ・全部入りと G06 の切り取り平均の回復率）
FULL_SET_VERIFY = {
    1: {"static_auprc": 0.3105, "full_auprc": 0.5246, "prefix_0.1ns_recovery": 0.919},
    5: {"static_auprc": 0.3022, "full_auprc": 0.5528, "prefix_0.1ns_recovery": 0.8023},
    10: {"static_auprc": 0.2772, "full_auprc": 0.4794, "prefix_0.1ns_recovery": 0.5778},
    30: {"static_auprc": 0.2756, "full_auprc": 0.5124, "prefix_0.1ns_recovery": 0.7964},
    50: {"static_auprc": 0.2927, "full_auprc": 0.4533, "prefix_0.1ns_recovery": 0.6117},
    90: {"static_auprc": 0.2886, "full_auprc": 0.4750, "prefix_0.1ns_recovery": 0.9346},
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
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K50 / f"{p}_R{r}_contact_diss.npz")
            parts.append({k: z[k] for k in z.files})
            proteins.extend([p] * len(z["y"]))
    return parts, np.array(proteins)


# 重なる候補のマスクと生成平均は、G79・G85・G89 で同じもの（3つのファイルが
# 配列単位で一致することを確かめた）を使う。G79 のファイルから読む。
mz = np.load(MATCHED)
mtr_full, mte_full = mz["train_matched"], mz["test_matched"]
gmtr_full, gmte_full = mz["train_gen_mean"], mz["test_gen_mean"]
print("matched train", int(mtr_full.sum()), "of", len(mtr_full),
      "test", int(mte_full.sum()), "of", len(mte_full), flush=True)

train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
Xtr = np.concatenate([d["X"] for d in train]).astype(np.float64)
Xte = np.concatenate([d["X"] for d in test]).astype(np.float64)
Htr = np.concatenate([d["hist"] for d in train])
Hte = np.concatenate([d["hist"] for d in test])
Ytr = np.concatenate([d["y"] for d in train])
Yte = np.concatenate([d["y"] for d in test])
print("train", Xtr.shape, "test", Xte.shape, flush=True)

prefix_tr = Htr[:, PREFIX_FRAMES].mean(1)
prefix_te = Hte[:, PREFIX_FRAMES].mean(1)
prot_idx_full = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx_full.items()}, flush=True)

# 重なる候補だけの集合（G79・G85・G89 と同じ）
Xtr_m, Xte_m = Xtr[mtr_full], Xte[mte_full]
Htr_m, Hte_m = Htr[mtr_full], Hte[mte_full]
pf_tr_m, pf_te_m = prefix_tr[mtr_full], prefix_te[mte_full]
gm_tr_m = gmtr_full[mtr_full][:, None].astype(np.float32)
gm_te_m = gmte_full[mte_full][:, None].astype(np.float32)
test_protein_m = test_protein[mte_full]
prot_idx = {p: np.where(test_protein_m == p)[0] for p in TEST_P}
print("matched test proteins:", {p: len(v) for p, v in prot_idx.items()}, flush=True)

CONDS = ["static_only", "full_hist", "prefix_mean_0.1ns", "gen_mean_only", "mixed"]


def feats_for(cname, matched=True):
    if matched:
        xtr, xte = Xtr_m, Xte_m
        htr, hte = Htr_m, Hte_m
        pftr, pfte = pf_tr_m, pf_te_m
        gmtr, gmte = gm_tr_m, gm_te_m
    else:
        xtr, xte = Xtr, Xte
        htr, hte = Htr, Hte
        pftr, pfte = prefix_tr, prefix_te
        gmtr, gmte = None, None
    if cname == "static_only":
        return (xtr, xte)
    if cname == "full_hist":
        return (np.column_stack([xtr, hist6(htr)]),
                np.column_stack([xte, hist6(hte)]))
    if cname == "prefix_mean_0.1ns":
        return (np.column_stack([xtr, pftr[:, None].astype(np.float32)]),
                np.column_stack([xte, pfte[:, None].astype(np.float32)]))
    if cname == "gen_mean_only":
        return (np.column_stack([xtr, gmtr]), np.column_stack([xte, gmte]))
    return (np.column_stack([xtr, pftr[:, None].astype(np.float32), gmtr]),
            np.column_stack([xte, pfte[:, None].astype(np.float32), gmte]))


results = {}
preds = {}  # horizon_ns -> condition -> test 予測確率（重なる候補だけの集合）
for hidx, hns in HORIZONS.items():
    ytr, yte = Ytr[:, hidx], Yte[:, hidx]
    # --- 全集合での土台の確認（K50・G06 の記録値と照合） ---
    ytr_f, yte_f = ytr, yte
    # 全集合の 3条件を学習し直す（生成平均は使わない）
    xtr_s, xte_s = feats_for("static_only", matched=False)
    xtr_f_, xte_f_ = feats_for("full_hist", matched=False)
    xtr_p, xte_p = feats_for("prefix_mean_0.1ns", matched=False)
    ps = predict(xtr_s, ytr_f, xte_s)
    pf_ = predict(xtr_f_, ytr_f, xte_f_)
    pp = predict(xtr_p, ytr_f, xte_p)
    s_f, f_f = auprc(yte_f, ps), auprc(yte_f, pf_)
    rec_f = (auprc(yte_f, pp) - s_f) / (f_f - s_f)
    exp_f = FULL_SET_VERIFY[hns]
    assert abs(s_f - exp_f["static_auprc"]) < 5e-4, (hns, s_f, exp_f)
    assert abs(f_f - exp_f["full_auprc"]) < 5e-4, (hns, f_f, exp_f)
    assert abs(rec_f - exp_f["prefix_0.1ns_recovery"]) < 0.005, (hns, rec_f, exp_f)
    print(f"{hns}ns FULL SET verify: static {s_f:.4f} (record {exp_f['static_auprc']}) "
          f"full {f_f:.4f} (record {exp_f['full_auprc']}) "
          f"prefix recovery {rec_f:.3f} (record {exp_f['prefix_0.1ns_recovery']})", flush=True)

    # --- 重なる候補だけの集合での測定 ---
    ytr_m, yte_m = ytr[mtr_full], yte[mte_full]
    p = {}
    for cname in CONDS:
        xtr, xte = feats_for(cname, matched=True)
        p[cname] = predict(xtr, ytr_m, xte)
    preds[hns] = p
    a = {c: auprc(yte_m, p[c]) for c in CONDS}
    gain = a["full_hist"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain for c in CONDS if c != "static_only"}
    diff = rec["mixed"] - rec["prefix_mean_0.1ns"]
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        ap = {c: auprc(yte_m[idx], p[c][idx]) for c in CONDS}
        g = ap["full_hist"] - ap["static_only"]
        per_protein[prot] = {
            "n": int(len(idx)), "positives": int(yte_m[idx].sum()),
            "auprc": {c: ap[c] for c in CONDS},
            "full_gain": g,
            "recovery_prefix": (ap["prefix_mean_0.1ns"] - ap["static_only"]) / g if g else float("nan"),
            "recovery_mixed": (ap["mixed"] - ap["static_only"]) / g if g else float("nan"),
            "mixed_minus_prefix_auprc": ap["mixed"] - ap["prefix_mean_0.1ns"],
            "mixed_minus_prefix_recovery": ((ap["mixed"] - ap["prefix_mean_0.1ns"]) / g
                                            if g else float("nan")),
        }
    results[hns] = {
        "test_positives_matched": int(yte_m.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "per_protein": per_protein,
        "full_set_verify": {"static_auprc": s_f, "full_auprc": f_f,
                            "prefix_0.1ns_recovery": float(rec_f)},
    }
    # G79・G85・G89 の記録との照合
    exp = EXPECTED[hns]
    for c in CONDS:
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(diff - exp["mixed_minus_prefix"]) < 5e-3, (hns, diff)
    assert int(yte_m.sum()) == exp["test_positives_matched"], (hns, int(yte_m.sum()))
    print(f"{hns}ns matched: static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"prefix {a['prefix_mean_0.1ns']:.4f} gen {a['gen_mean_only']:.4f} "
          f"mixed {a['mixed']:.4f} gain {gain:.4f} diff {diff:+.3f} "
          f"(record {exp['mixed_minus_prefix']:+.3f}) pos {int(yte_m.sum())}", flush=True)
    print(f"  per-protein mixed-prefix recovery: "
          + "; ".join(f"{prot} {per_protein[prot]['mixed_minus_prefix_recovery']:+.3f}"
                      for prot in TEST_P), flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein_matched=test_protein_m,
    **{f"y_{h}": Yte[mte_full][:, i] for i, h in HORIZONS.items()},
    **{f"p_{h}ns_{c}": preds[h][c] for h in HORIZONS.values() for c in CONDS},
)
print("checkpoint: saved test_predictions.npz", flush=True)

# ---- タンパク質単位のブートストラップ（G23・G49・G59・G62・G67 と同じ回数・seed・パーセンタイル法） ----
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))
rng = np.random.default_rng(BOOT_SEED)
boot = {}
for hidx, hns in HORIZONS.items():
    yte_m = Yte[mte_full][:, hidx]
    p = preds[hns]
    three = {}
    for combo in COMBOS:
        idx = np.concatenate([prot_idx[prot] for prot in combo])
        three["+".join(combo)] = recovery_diff(
            yte_m[idx], p["static_only"][idx], p["full_hist"][idx],
            p["prefix_mean_0.1ns"][idx], p["mixed"][idx])
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
    print(f"{hns}ns bootstrap: {boot[hns]['point_estimate']:+.3f} "
          f"[{boot[hns]['ci_lower']:+.3f}, {boot[hns]['ci_upper']:+.3f}] "
          f"zero={boot[hns]['contains_zero']}", flush=True)

# +0.05 を超えたセルの区間が 0 を含むかどうか（先に固定した読み方）
above_check = {"90ns": boot[90]["contains_zero"]}
print("above-0.05 cells contain zero:", above_check, flush=True)

out = {
    "label": "Contact dissociation, hybrid uncertainty at 1/5/10/30/50/90ns",
    "phenomenon": "contact dissociation (DocD, K50 cache + G18 regenerated proposal-22 series, matched subset only)",
    "prefix_frames": PREFIX_FRAMES,
    "matched_subset": {"train_n_matched": int(mtr_full.sum()), "test_n_matched": int(mte_full.sum()),
                       "train_n_full": 6383, "test_n_full": 2915,
                       "note": "Only candidates overlapping G18's regenerated series "
                       "(76% of test candidates) are measured; the base is re-measured "
                       "on the same matched subset, so the recovery denominator is the "
                       "matched-subset full gain."},
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
    w.writerow(["horizon_ns", "test_positives_matched", "auprc_static",
                "auprc_full", "auprc_prefix_mean", "auprc_gen_mean_only", "auprc_mixed",
                "full_gain", "recovery_prefix", "recovery_mixed",
                "mixed_minus_prefix", "boot_ci_lower", "boot_ci_upper",
                "ci_contains_zero"])
    for h in HORIZONS.values():
        r = results[h]
        b = boot[h]
        w.writerow([h, r["test_positives_matched"],
                    r["auprc"]["static_only"], r["auprc"]["full_hist"],
                    r["auprc"]["prefix_mean_0.1ns"],
                    r["auprc"]["gen_mean_only"],
                    r["auprc"]["mixed"], r["full_gain"],
                    r["recovery"]["prefix_mean_0.1ns"],
                    r["recovery"]["mixed"],
                    r["mixed_minus_prefix"],
                    b["ci_lower"], b["ci_upper"], b["contains_zero"]])

print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
