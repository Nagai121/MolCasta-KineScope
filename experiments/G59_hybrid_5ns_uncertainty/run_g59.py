"""G59: 5 ns先での混合の差について、0.25 ns の切り取りだけが上がるのは本物か.

G26 では 0.1 ns の切り取りで 5 ns先の混合の差が -0.169、
G56 では 0.25 ns の切り取りで +0.082、
G35 では 0.5 ns の切り取りで -0.143 でした。
5 ns先の 0.1・0.25・0.5 ns の切り取りの 3 つについて、
混合と切り取り平均だけの差（mixed_minus_prefix）の
ブートストラップ 95% 区間を出します。

再標本化の単位はタンパク質。テストのタンパク質は 1mix_A と 2qt4_A の
2つだけなので、タンパク質単位の再標本化で取りうる組み合わせは
{1mix,1mix}・{1mix,2qt4}・{2qt4,2qt4} の3通りしかありません。
そのためブートストラップ分布も3つの値しか取らず、95% 区間は
その3値の最小値〜最大値になります。区間が粗いことは留保として記録する
（G23 と同じ留保）。

ブートストラップの回数と乱数 seed は結果を見る前に固定した:
10,000回、seed 42。区間はパーセンタイル法（G23 と同じ）。

段取り:
- G26・G56・G35 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で
  決定的なため、合算の AUPRC は記録と一致するはずで、一致を確認する）。
  候補ごとの予測確率を保存し、そこからブートストラップする。
  生成器のチェックポイント .pt ファイルは読み込まない（保存済みの
  生成系列を使う）。
- 差の定義: mixed_minus_prefix = 混合の回復率 - 切り取り平均だけの回復率。
  回復率の分母は 5 ns先での実履歴の利得（+0.1155）。
  各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り平均だけ・
  混合の AUPRC を測り直し、利得と回復率をその標本の中で計算し直す。
- 計算の工夫: タンパク質が2つだけのため、再標本化の標本は3通りの
  組み合わせのどれかでしかなく、差の値は組み合わせだけで決まる。
  そのため 3通りの組み合わせの値を正確に計算し、10,000回の抽出は
  組み合わせへの写像として行う。行を並べ替えて AUPRC を 1万回
  計算し直す G23 のやり方と数学的に同じ結果になる（差が組み合わせの
  関数であるため）。抽出の乱数 seed は G23 と同じ 42。

判定（先に固定）:
- 0.25 ns の切り取りの差の区間が、0.1 ns と 0.5 ns の切り取りの差の
  区間と重ならなければ、真ん中だけ上がるのは切り取りの長さによる
  本物の形だということになる。
- 区間が重なれば、真ん中の上がりは揺れの範囲だということになる。
- 「重なる」は 0.25 ns の区間と 0.1 ns の区間の重なり、0.25 ns の区間と
  0.5 ns の区間の重なりの両方で読む。片方だけ重なる場合も、そのまま記録する。
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
H_IDX = 1  # 5 ns先 -> y の列番号
PREFIXES = {
    "0.1ns": [0, 1],              # G02/G26 と同じ
    "0.25ns": [0, 1, 2],          # G02/G56 と同じ
    "0.5ns": [0, 1, 2, 3, 4, 5],  # G02/G35 と同じ
}
N_BOOT = 10000  # 結果を見る前に固定（G23 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23 と同じ）

# G26・G56・G35 の記録との照合用の期待値（合算 AUPRC、5 ns先）
EXPECTED = {
    "0.1ns": {"static_only": 0.3493, "full": 0.4648, "prefix": 0.4470, "mixed": 0.4274,
              "full_gain": 0.1155, "mixed_minus_prefix": -0.169},
    "0.25ns": {"static_only": 0.3493, "full": 0.4648, "prefix": 0.4682, "mixed": 0.4776,
               "full_gain": 0.1155, "mixed_minus_prefix": 0.082},
    "0.5ns": {"static_only": 0.3493, "full": 0.4648, "prefix": 0.4691, "mixed": 0.4526,
              "full_gain": 0.1155, "mixed_minus_prefix": -0.143},
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

gen_mean_tr = Gtr.mean(1)
gen_mean_te = Gte.mean(1)

# タンパク質ごとのテスト候補の行番号
prot_idx = {p: np.where(test_protein == p)[0] for p in TEST_P}
print({p: len(v) for p, v in prot_idx.items()}, flush=True)

results = {}
preds = {}  # prefix -> condition -> test 予測確率
for pname, frames in PREFIXES.items():
    prefix_mean_tr = Htr[:, frames].mean(1)
    prefix_mean_te = Hte[:, frames].mean(1)
    ytr, yte = Ytr[:, H_IDX], Yte[:, H_IDX]
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
    preds[pname] = p
    a = {c: auprc(yte, p[c]) for c in conds}
    gain = a["full"] - a["static_only"]
    rec = {c: (a[c] - a["static_only"]) / gain for c in ("full", "prefix", "mixed")}
    diff = rec["mixed"] - rec["prefix"]
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
    results[pname] = {
        "prefix_frames": frames,
        "test_positives": int(yte.sum()),
        "auprc": a, "full_gain": gain, "recovery": rec,
        "mixed_minus_prefix": diff, "per_protein": per_protein,
    }
    # G26・G56・G35 の記録との照合
    exp = EXPECTED[pname]
    for c in ("static_only", "full", "prefix", "mixed"):
        assert abs(a[c] - exp[c]) < 5e-4, (pname, c, a[c], exp[c])
    assert abs(diff - exp["mixed_minus_prefix"]) < 5e-3, (pname, diff)
    print(f"{pname}: static {a['static_only']:.4f} full {a['full']:.4f} "
          f"prefix {a['prefix']:.4f} mixed {a['mixed']:.4f} gain {gain:.4f} "
          f"diff {diff:.4f} (record {exp['mixed_minus_prefix']:+.3f})", flush=True)
    print(f"  per-protein mixed-prefix recovery: "
          + ", ".join(f"{prot} {per_protein[prot]['mixed_minus_prefix_recovery']:+.4f}"
                      for prot in TEST_P), flush=True)

# ---- タンパク質単位のブートストラップ（G23 と同じ回数・seed・パーセンタイル法） ----
# タンパク質が2つだけのため、再標本化の標本は3通りの組み合わせのどれかで
# しかなく、差の値は組み合わせだけで決まる。3通りの値を正確に計算し、
# 10,000回の抽出は組み合わせへの写像として行う（行を並べ替えて AUPRC を
# 1万回計算し直す G23 のやり方と数学的に同じ結果になる）。
COMBOS = (("1mix_A", "1mix_A"), ("1mix_A", "2qt4_A"), ("2qt4_A", "2qt4_A"))
rng = np.random.default_rng(BOOT_SEED)
boot = {}
for pname in PREFIXES:
    yte = Yte[:, H_IDX]
    p = preds[pname]
    three = {}
    for combo in COMBOS:
        idx = np.concatenate([prot_idx[prot] for prot in combo])
        three["+".join(combo)] = recovery_diff(
            yte[idx], p["static_only"][idx], p["full"][idx],
            p["prefix"][idx], p["mixed"][idx])
    # 10,000回の抽出（組み合わせへの写像）
    combo_idx = rng.integers(0, len(COMBOS), size=N_BOOT)
    vals = np.array([three["+".join(COMBOS[i])] for i in combo_idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    boot[pname] = {
        "n_boot": N_BOOT, "seed": BOOT_SEED,
        "point_estimate": results[pname]["mixed_minus_prefix"],
        "ci_lower": float(lo), "ci_upper": float(hi),
        "contains_zero": bool(lo <= 0 <= hi),
        "three_resample_values": three,
        "n_distinct_boot_values": int(len(np.unique(np.round(vals, 12)))),
    }
    print(f"{pname} bootstrap: point {results[pname]['mixed_minus_prefix']:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] contains_zero={lo <= 0 <= hi} "
          f"distinct={boot[pname]['n_distinct_boot_values']}", flush=True)
    print(f"  three values: " + ", ".join(f"{k} {v:+.4f}" for k, v in three.items()), flush=True)

# ---- 区間の重なり（先に固定した読み方） ----
def overlap(a, b):
    return not (a["ci_upper"] < b["ci_lower"] or b["ci_upper"] < a["ci_lower"])


overlap_025_01 = overlap(boot["0.25ns"], boot["0.1ns"])
overlap_025_05 = overlap(boot["0.25ns"], boot["0.5ns"])
print(f"overlap 0.25ns vs 0.1ns: {overlap_025_01}", flush=True)
print(f"overlap 0.25ns vs 0.5ns: {overlap_025_05}", flush=True)

out = {
    "label": "HB formation 5ns, hybrid uncertainty across prefix lengths",
    "horizon_ns": 5,
    "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile",
                  "resampling_unit": "protein (1mix_A, 2qt4_A; 2 proteins only, "
                  "so only 3 resample combinations exist and the bootstrap "
                  "distribution takes only 3 distinct values)",
                  "note": "The 10,000 draws are mapped onto the 3 exact combination "
                  "values; this is mathematically identical to recomputing AUPRC "
                  "on the resampled rows each time, because the difference depends "
                  "only on the chosen combination."},
    "full_gain_5ns": 0.1155,
    "prefixes": {p: results[p] for p in PREFIXES},
    "bootstrap_results": boot,
    "ci_overlap": {
        "0.25ns_vs_0.1ns": bool(overlap_025_01),
        "0.25ns_vs_0.5ns": bool(overlap_025_05),
        "both_overlap": bool(overlap_025_01 and overlap_025_05),
    },
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

import csv
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["prefix", "prefix_frames", "test_positives", "auprc_static",
                "auprc_full", "auprc_prefix", "auprc_mixed", "full_gain",
                "recovery_prefix", "recovery_mixed", "mixed_minus_prefix",
                "boot_ci_lower", "boot_ci_upper", "ci_contains_zero"])
    for pname in PREFIXES:
        r, b = results[pname], boot[pname]
        w.writerow([pname, ";".join(map(str, r["prefix_frames"])), r["test_positives"],
                    r["auprc"]["static_only"], r["auprc"]["full"],
                    r["auprc"]["prefix"], r["auprc"]["mixed"], r["full_gain"],
                    r["recovery"]["prefix"], r["recovery"]["mixed"],
                    r["mixed_minus_prefix"], b["ci_lower"], b["ci_upper"],
                    b["contains_zero"]])

# 候補ごとの予測確率は再利用のために保存する（ブートストラップの再現用）
np.savez_compressed(
    OUT / "test_predictions.npz",
    test_protein=test_protein,
    y_5=Yte[:, H_IDX],
    **{f"p_{pname}_{c}": preds[pname][c] for pname in PREFIXES for c in preds[pname]},
)
print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")
