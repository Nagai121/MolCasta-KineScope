"""G135: 接触の解離の混合で、切り取り 1.0 ns の 6 個の統計量にした場合の差に、
区間を付ける。

対象は G134 で +0.05 を超えた 90 ns先（+0.059）: 切り取り 1.0 ns
（フレーム [0,1,2,3,4,5,6,7,8,9]、実際は 0.9 ns の範囲）の 6統計量に
方法22の生成平均。G18 の生成系列と重なる候補だけ（テストの 76%）で
測っている。G59・G62・G67 と同じ、タンパク質単位の再標本化による
ブートストラップ 95% 区間（10,000回、seed 42、パーセンタイル法）。

再標本化の単位はタンパク質。テストのタンパク質は 1mix_A と 2qt4_A の
2つだけなので、タンパク質単位の再標本化で取りうる組み合わせは
{1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通りしかない。そのため
ブートストラップ分布も3つの値しか取らず、95% 区間はその3値の
最小値〜最大値になる。区間が粗いことは留保として記録する
（G23・G49・G59・G62・G67・G101 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（2.5%点と 97.5%点）。
G23・G49・G59・G62・G67・G101 と同じ。

段取り:
- G99・G121・G89 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で
  決定的なため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
  生成平均は G18 で作り直した方法22の系列から、G79 と同じやり方で作る
  （残基ペアに属する生成ペアの 4点系列の点ごとの最小値、4点で平均）。
  生成器のチェックポイントは読み込まない。残基の添え字は
  /tmp/g79_atoms/trajectories が消えたため、元の置き場の atoms.parquet
  を使う（G121 と同じ。重なる候補の数は G99・G115 と同じ）。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取りだけの回復率。
  回復率の分母は、その予測幅での実履歴の利得。各ブートストラップ標本の
  中でも、静的のみ・全部入り・切り取りだけ・混合の AUPRC を測り直し、
  利得と回復率をその標本の中で計算し直す。
- 90 ns先は土台の利得が小さい（重なる候補だけでは +0.1524）ため、
  回復率の差だけでは読みを誤る。90 ns先については、混合と切り取りだけの
  AUPRC の差の絶対値も記録する（点推定、タンパク質ごとの値、
  ブートストラップ 95% 区間）。AUPRC の差の区間は、全セルについて
  同じやり方で出す。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42。

読み方（先に固定）:
- +0.05 を超えた 90 ns先のセルの区間が 0 を含むかどうかを見る。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

G18 = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
K50 = Path("/home/hatch/workspace/kinescope/work/k50_cache")
ATOMS = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
OUT = Path(__file__).resolve().parent
BOUND = 4.5
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
N_BOOT = 10000  # 結果を見る前に固定（G23・G49・G59・G62・G67・G101 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23・G49・G59・G62・G67・G101 と同じ）

# セル: (予測幅 ns, y の列, 切り取りの種類, 切り取りのフレーム)
CELLS = [
    (90, 5, "hist6_1.0ns", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
]

# G134 の記録との照合用の期待値（重なる候補だけの合算 AUPRC）
EXPECTED = {
    90: {"static_only": 0.3294, "full_hist": 0.4818, "prefix": 0.4697,
         "gen_mean_only": 0.3452, "mixed": 0.4787,
         "full_gain": 0.1524, "mixed_minus_prefix": 0.0591,
         "mixed_minus_prefix_auprc": 0.0090},
}


def hist6(series, bound=BOUND):
    return np.stack([series[:, 0], series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0], (series <= bound).mean(1)],
                    axis=1).astype(np.float32)


def predict(xtr, ytr, xte):
    if np.unique(ytr).size < 2:
        return np.full(len(xte), float(ytr.mean()))
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                       min_samples_leaf=15, l2_regularization=.02, random_state=42)
    m.fit(xtr, ytr)
    return m.predict_proba(xte)[:, 1]


def auprc(y, p):
    return float(average_precision_score(y, p))


def load_with_gen(prots):
    """K50 キャッシュに、候補ごとの生成平均を付ける。G99 と同じ作り方。

    X, H, Y、重なるかどうかのマスク、生成平均の配列（重ならない候補は
    NaN）、タンパク質のラベルを返す。並びはタンパク質の順、レプリカ 1〜3
    の順で、キャッシュのファイルと同じ。生成平均は G79 と同じ作り方
    （残基ペアに属する生成ペアの 4点系列の点ごとの最小値、4点で平均）。
    """
    Xs, Hs, Ys, matched, gen_mean, proteins = [], [], [], [], [], []
    for p in prots:
        a = pd.read_parquet(ATOMS / f"atlas-{p}-analysis-R1" / "atoms.parquet")
        res_idx = a["residue_index"].to_numpy()
        for r in (1, 2, 3):
            g = np.load(G18 / f"atlas-{p}-analysis-R{r}.npz")
            dres = res_idx[g["pair"][:, 0].astype(int)]
            ares = res_idx[g["pair"][:, 1].astype(int)]
            series = {}
            for di, ai, gd in zip(dres.tolist(), ares.tolist(), g["gen_dist"]):
                key = (min(di, ai), max(di, ai))
                if key in series:
                    series[key] = np.minimum(series[key], gd)
                else:
                    series[key] = gd.astype(np.float64)
            k = np.load(K50 / f"{p}_R{r}_contact_diss.npz")
            Xs.append(k["X"].astype(np.float64))
            Hs.append(k["hist"])
            Ys.append(k["y"])
            m, gm = [], []
            for ri, rj in zip(k["res_i"].tolist(), k["res_j"].tolist()):
                key = (min(ri, rj), max(ri, rj))
                if key in series:
                    m.append(True)
                    gm.append(float(series[key].mean()))
                else:
                    m.append(False)
                    gm.append(float("nan"))
            matched.append(np.array(m))
            gen_mean.append(np.array(gm))
            proteins.extend([p] * len(k["y"]))
    return (np.concatenate(Xs), np.concatenate(Hs), np.concatenate(Ys),
            np.concatenate(matched), np.concatenate(gen_mean), np.array(proteins))


def recovery_diff(y, p_static, p_full, p_prefix, p_mixed):
    """その標本の中での mixed_minus_prefix（回復率の差）。分母は標本内の実履歴の利得。"""
    gain = auprc(y, p_full) - auprc(y, p_static)
    if gain == 0:
        return float("nan")
    return (auprc(y, p_mixed) - auprc(y, p_prefix)) / gain


def auprc_diff(y, p_prefix, p_mixed):
    """その標本の中での混合と切り取りだけの AUPRC の差の絶対値。"""
    return auprc(y, p_mixed) - auprc(y, p_prefix)


Xtr, Htr, Ytr, mtr, gmtr, _ = load_with_gen(TRAIN_P)
Xte, Hte, Yte, mte, gmte, test_protein = load_with_gen(TEST_P)
print(f"train n: {len(Ytr)} matched {int(mtr.sum())} "
      f"test n: {len(Yte)} matched {int(mte.sum())}", flush=True)
assert int(mtr.sum()) == 4489 and int(mte.sum()) == 2226, "matched counts differ from G99/G115/G121"

# 重なる候補だけに絞る（G99・G121・G89 と同じ）
Xtr_m, Xte_m = Xtr[mtr], Xte[mte]
Htr_m, Hte_m = Htr[mtr], Hte[mte]
gm_tr_m = gmtr[mtr][:, None]
gm_te_m = gmte[mte][:, None]
prot_te = test_protein[mte]
prot_idx = {p: np.where(prot_te == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

results = {}
preds = {}  # horizon_ns -> condition -> test 予測確率（重なる候補だけ）
for hns, hi, kind, frames in CELLS:
    ytr, yte = Ytr[mtr][:, hi], Yte[mte][:, hi]
    if kind.startswith("hist6"):
        pf_tr = hist6(Htr_m[:, frames])
        pf_te = hist6(Hte_m[:, frames])
    else:  # mean_0.1ns
        pf_tr = Htr_m[:, frames].mean(1)[:, None]
        pf_te = Hte_m[:, frames].mean(1)[:, None]
    conds = {
        "static_only": (Xtr_m, Xte_m),
        "full_hist": (np.column_stack([Xtr_m, hist6(Htr_m)]),
                      np.column_stack([Xte_m, hist6(Hte_m)])),
        "prefix": (np.column_stack([Xtr_m, pf_tr]),
                   np.column_stack([Xte_m, pf_te])),
        "gen_mean_only": (np.column_stack([Xtr_m, gm_tr_m]),
                          np.column_stack([Xte_m, gm_te_m])),
        "mixed": (np.column_stack([Xtr_m, pf_tr, gm_tr_m]),
                  np.column_stack([Xte_m, pf_te, gm_te_m])),
    }
    p = {}
    for cname, (xtr, xte) in conds.items():
        p[cname] = predict(xtr, ytr, xte)
    preds[hns] = p
    a = {c: auprc(yte, p[c]) for c in conds}
    gain = a["full_hist"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain for c in conds if c != "static_only"}
    diff = rec["mixed"] - rec["prefix"]
    diff_auprc = a["mixed"] - a["prefix"]
    per_protein = {}
    for prot in TEST_P:
        idx = prot_idx[prot]
        ap = {c: auprc(yte[idx], p[c][idx]) for c in conds}
        g = ap["full_hist"] - ap["static_only"]
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
    results[hns] = {
        "horizon_ns": hns, "kind": kind, "prefix_frames": frames,
        "test_n_matched": int(mte.sum()), "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "mixed_minus_prefix_auprc": diff_auprc,
        "per_protein": per_protein,
    }
    # G134 の記録との照合
    exp = EXPECTED[hns]
    for c in ("static_only", "full_hist", "prefix", "gen_mean_only", "mixed"):
        assert abs(a[c] - exp[c]) < 5e-4, (hns, c, a[c], exp[c])
    assert abs(diff - exp["mixed_minus_prefix"]) < 5e-3, (hns, diff)
    assert abs(diff_auprc - exp["mixed_minus_prefix_auprc"]) < 5e-3, (hns, diff_auprc)
    print(f"{hns}ns ({kind}): static {a['static_only']:.4f} full {a['full_hist']:.4f} "
          f"prefix {a['prefix']:.4f} mixed {a['mixed']:.4f} gain {gain:.4f} "
          f"diff {diff:+.3f} (record {exp['mixed_minus_prefix']:+.3f}) "
          f"auprc-diff {diff_auprc:+.4f} (record {exp['mixed_minus_prefix_auprc']:+.4f})",
          flush=True)
    print(f"  per-protein mixed-prefix recovery: "
          + "; ".join(f"{prot} {per_protein[prot]['mixed_minus_prefix_recovery']:+.3f}"
                      for prot in TEST_P), flush=True)

# ---- 途中経過の保存（学習し直した予測確率を先に保存しておく） ----
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=prot_te,
    **{f"y_{h}": Yte[mte][:, hi] for h, hi, _, _ in CELLS},
    **{f"p_{h}ns_{c}": preds[h][c] for h, _, _, _ in CELLS for c in preds[h]},
)
print("checkpoint: saved test_predictions.npz", flush=True)

# ---- タンパク質単位のブートストラップ（G23・G49・G59・G62・G67・G101 と同じ回数・seed・パーセンタイル法） ----
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))
rng = np.random.default_rng(BOOT_SEED)
boot = {}
boot_auprc = {}
for hns, hi, kind, frames in CELLS:
    yte = Yte[mte][:, hi]
    p = preds[hns]
    three = {}
    three_a = {}
    for combo in COMBOS:
        idx = np.concatenate([prot_idx[prot] for prot in combo])
        three["+".join(combo)] = recovery_diff(
            yte[idx], p["static_only"][idx], p["full_hist"][idx],
            p["prefix"][idx], p["mixed"][idx])
        three_a["+".join(combo)] = auprc_diff(
            yte[idx], p["prefix"][idx], p["mixed"][idx])
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
    vals_a = np.array([three_a["+".join(COMBOS[i])] for i in combo_idx])
    lo_a, hi_a = np.percentile(vals_a, [2.5, 97.5])
    boot_auprc[hns] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[hns]["mixed_minus_prefix_auprc"],
        "ci_lower": float(lo_a), "ci_upper": float(hi_a),
        "contains_zero": bool(lo_a <= 0 <= hi_a),
        "three_resample_values": three_a,
        "n_distinct_boot_values": int(len(np.unique(np.round(vals_a, 12)))),
    }
    print(f"{hns}ns bootstrap: diff {boot[hns]['point_estimate']:+.3f} "
          f"[{boot[hns]['ci_lower']:+.3f}, {boot[hns]['ci_upper']:+.3f}] "
          f"zero={boot[hns]['contains_zero']}", flush=True)
    print(f"{hns}ns bootstrap (AUPRC diff): {boot_auprc[hns]['point_estimate']:+.4f} "
          f"[{boot_auprc[hns]['ci_lower']:+.4f}, {boot_auprc[hns]['ci_upper']:+.4f}] "
          f"zero={boot_auprc[hns]['contains_zero']}", flush=True)

ABOVE = [(90, "G134")]
above_check = {f"{h}ns_{src}": boot[h]["contains_zero"] for h, src in ABOVE}
print("above-0.05 cells contain zero:", above_check, flush=True)

out = {
    "label": "Contact dissociation hybrid uncertainty at 90ns (G134)",
    "phenomenon": "contact dissociation (DocD, K50 cache + G18 regenerated proposal-22 series, overlapping candidates only)",
    "cells": [{"horizon_ns": hns, "source": src, "kind": kind, "prefix_frames": frames}
              for (hns, _, kind, frames), (_, src) in zip(CELLS, ABOVE)],
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (1mix_A, 2qt4_A; 2 proteins only, "
                  "so only 3 resample combinations exist and the bootstrap "
                  "distribution takes only 3 distinct values)",
                  "note": "The 10,000 draws are mapped onto the 3 exact combination "
                  "values; this is mathematically identical to recomputing AUPRC "
                  "on the resampled rows each time, because the difference depends "
                  "only on the chosen combination."},
    "horizons": {str(h): results[h] for h, _, _, _ in CELLS},
    "bootstrap_results": {str(h): boot[h] for h, _, _, _ in CELLS},
    "bootstrap_results_auprc_diff": {str(h): boot_auprc[h] for h, _, _, _ in CELLS},
    "above_0.05_cells_contain_zero": above_check,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon_ns", "source", "kind", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix", "auprc_gen_mean_only", "auprc_mixed",
                "full_gain", "recovery_prefix", "recovery_mixed",
                "mixed_minus_prefix", "boot_ci_lower", "boot_ci_upper",
                "ci_contains_zero",
                "mixed_minus_prefix_auprc", "auprc_boot_ci_lower",
                "auprc_boot_ci_upper", "auprc_ci_contains_zero"])
    for (hns, _, kind, _), (_, src) in zip(CELLS, ABOVE):
        r = results[hns]
        b = boot[hns]
        ba = boot_auprc[hns]
        w.writerow([hns, src, kind, r["test_positives"],
                    r["auprc"]["static_only"], r["auprc"]["full_hist"],
                    r["auprc"]["prefix"],
                    r["auprc"]["gen_mean_only"],
                    r["auprc"]["mixed"], r["full_gain"],
                    r["recovery"]["prefix"],
                    r["recovery"]["mixed"],
                    r["mixed_minus_prefix"],
                    b["ci_lower"], b["ci_upper"], b["contains_zero"],
                    r["mixed_minus_prefix_auprc"],
                    ba["ci_lower"], ba["ci_upper"], ba["contains_zero"]])

print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
