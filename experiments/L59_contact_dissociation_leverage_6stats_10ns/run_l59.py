"""L59: 短いMDの長さに対して、指導力がどれだけ戻るかの曲がり具合を
統計量6種で測る（資料Dの条件、接触の解離、10 ns先）.

L 系列（短い実MDの覆う長さに対して回復率がどう伸びるかを測る系列）。
L23（接触の解離、10 ns先、切り取りの平均だけ）と同じ土台・同じ
覆う長さで、統計量を平均だけから資料Dの6種（最初・平均・標準偏差・
最小・最後−最初・4.5 Å以下割合）に変えて曲がりを測る。あわせて、
同じ実行の中で平均だけの曲がりも測り直し、L23 の記録と一致するか
確かめる。

長さの呼び方（L01・L02 と同じ、結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 長さは「覆う長さ」で統一する。覆う長さ L ns の切り取りは
  フレーム 0〜(L/0.1) を使う（フレーム数は L/0.1 + 1）。
  - 覆う長さ 0 ns   -> フレーム 0 のみ（1フレーム）
  - 覆う長さ 0.1 ns -> フレーム 0, 1（2フレーム）
  - ...
  - 覆う長さ 1.0 ns -> フレーム 0〜10（11フレーム。0〜1 ns の全部）
- 覆う長さ 1.0 ns の統計量6種は全部入り（full_hist6）と同値になる。
- 起点フレーム10（1 ns）は変えない。ラベルは 10 ns先。

条件:
- 静的のみ: 静的特徴5個だけ。
- 全部入り（full_hist6）: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム、
  境界 4.5 Å）。回復率の分母（実履歴の利得）の参照。
- 切り取りの統計量6種: 静的特徴＋切り取り（フレーム 0〜k）の6統計量。
  覆う長さ 0.0〜1.0 ns の 11通り。
- 切り取りの平均だけ（照合用）: 静的特徴＋切り取りの平均だけ。
  覆う長さ 0.0〜1.0 ns の 11通り。L23 の記録と照らす。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。

特徴量の型は G71 と同じ作り方にそろえる（静的特徴を float64、
切り取りの平均を float32、6統計量は float32）。L23〜L26 で、型の
違いが覆う長さ 0.5 ns の回復率を G71 とずらすことが分かっているため。

読み方（結果を見る前に固定）:
1. 回復率 0.8 と 0.9 に届く最小の覆う長さを、全体（統合値）で読む。
2. 隣り合う長さの間の増え幅がいちばん大きい場所と大きさ（段の有無）。
3. 信頼区間は資料Dの流儀で、3つの値（全体・1mix_A・2qt4_A）の
   回復率の最小から最大。テストは 2タンパク質だけなので区間は粗い。
4. 段らしい場所（増え幅が最大の区間）があれば、その前後で
   タンパク質ごとの値も記録する。
あわせて、平均だけと比べて必要な長さが縮んだかどうかを記録する。

土台の再現の期待値（L23・G71・G140 の記録、10 ns先）:
- 静的のみ AUPRC 0.2772、全部入り 0.4794、利得 0.2022、
  テスト 2,915件・陽性 228 件。

実行はローカルの仮想マシンで行う（24条件の学習で数分。L 系列では
重い計算は Colab に逃がす決め事があるが、この規模は L21〜L26 と
同じく重い計算にはあたらないと判断した）。sklearn の学習は
スレッド数を 1 に絞る。
"""
import csv
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
OUT = Path(__file__).resolve().parent
REF_RESULTS = OUT.parent / "L23_contact_dissociation_leverage_10ns" / "results.json"
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
HIDX = 2  # 10 ns先（予測幅は 1/5/10/30/50/90 ns で 0〜5）
HORIZON_NS = 10
EXP_NAME = "L59"
REF_NAME = "L23"

EXPECTED = {"static_only": 0.27719749486883505, "full_hist6": 0.4794242789590083,
            "gain": 0.20222678409017325, "test_n": 2915, "test_positives": 228}

# 覆う長さ（ns）-> 使うフレーム数。フレーム 0〜(n-1) を使う。
LENGTHS = {round(0.1 * k, 1): k + 1 for k in range(11)}


def load(prots):
    parts, proteins = [], []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
            d = {k: z[k] for k in z.files}
            parts.append(d)
            proteins.extend([p] * len(d["y"]))
    return parts, np.array(proteins)


def cat(parts, key):
    # K50 のキャッシュの hist は (N, 11) で保存されているので転置しない
    return np.concatenate([p[key] for p in parts])


def hist6(series, bound=BOUND):
    """series は (N, F) の距離系列。資料Dの6統計量（float32 で返す）。"""
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


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


train, _ = load(TRAIN_P)
test, test_protein = load(TEST_P)
# 特徴量の型は G71 と同じ作り方（静的特徴は float64）
Xtr, Xte = (cat(train, "X").astype(np.float64),
            cat(test, "X").astype(np.float64))
# H はキャッシュの型のまま使う（G71 と同じ）。6統計量は内部で float32 に
# 落とし、切り取りの平均は float32 に落としてから静的特徴と並べる。
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HIDX], Yte[:, HIDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 6383 and Xte.shape[0] == 2915, (Xtr.shape, Xte.shape)

CONDS = (["static_only", "full_hist6"]
         + [f"hist6_len_{L:.1f}" for L in LENGTHS]
         + [f"mean_len_{L:.1f}" for L in LENGTHS])


def feats_for(cname):
    if cname == "static_only":
        return Xtr, Xte
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    if cname.startswith("hist6_len_"):
        n = LENGTHS[float(cname.replace("hist6_len_", ""))]
        return (np.column_stack([Xtr, hist6(Htr[:, :n])]),
                np.column_stack([Xte, hist6(Hte[:, :n])]))
    if cname.startswith("mean_len_"):
        n = LENGTHS[float(cname.replace("mean_len_", ""))]
        return (np.column_stack([Xtr, Htr[:, :n].mean(1)[:, None].astype(np.float32)]),
                np.column_stack([Xte, Hte[:, :n].mean(1)[:, None].astype(np.float32)]))
    raise ValueError(cname)


res, preds, per_protein = {}, {}, {}
for cname in CONDS:
    xtr, xte = feats_for(cname)
    p = predict(xtr, ytr, xte)
    preds[cname] = p
    res[cname] = metrics(yte, p)
    per_protein[cname] = {prot: metrics(yte[test_protein == prot],
                                         p[test_protein == prot])
                          for prot in TEST_P}
    print(cname, round(res[cname]["auprc"], 4), flush=True)

# ---- 土台の再現の確認（L23・G71・G140 の記録と照合） ----
a_static = res["static_only"]["auprc"]
a_full = res["full_hist6"]["auprc"]
gain_full = a_full - a_static
assert res["static_only"]["n"] == EXPECTED["test_n"]
assert res["static_only"]["positives"] == EXPECTED["test_positives"], res["static_only"]
assert abs(a_static - EXPECTED["static_only"]) < 5e-4, a_static
assert abs(a_full - EXPECTED["full_hist6"]) < 5e-4, a_full
assert abs(gain_full - EXPECTED["gain"]) < 5e-4, gain_full
print("base reproduction vs %s/G71/G140: OK (static %.4f / full %.4f / gain %+.4f)"
      % (REF_NAME, a_static, a_full, gain_full), flush=True)

# ---- 回復率・区間・増え分 ----
def recovery(auprc, static, full):
    return (auprc - static) / (full - static)


def curve(prefix):
    out_c, prev = {}, None
    for L in sorted(LENGTHS):
        cname = f"{prefix}_len_{L:.1f}"
        rec_pool = recovery(res[cname]["auprc"], a_static, a_full)
        rec_pp = {}
        for prot in TEST_P:
            pp = per_protein[cname][prot]
            ps = per_protein["static_only"][prot]
            pf = per_protein["full_hist6"][prot]
            rec_pp[prot] = recovery(pp["auprc"], ps["auprc"], pf["auprc"])
        three = [rec_pool, rec_pp["1mix_A"], rec_pp["2qt4_A"]]
        out_c[f"{L:.1f}"] = {
            "covered_length_ns": L, "n_frames": LENGTHS[L],
            "metrics": res[cname], "recovery_pooled": rec_pool,
            "recovery_per_protein": rec_pp,
            "ci_min": min(three), "ci_max": max(three),
            "increment_from_previous": None if prev is None else rec_pool - prev,
        }
        prev = rec_pool
    return out_c


def readings_of(curve_d):
    def smallest(thr):
        for L in sorted(LENGTHS):
            if curve_d[f"{L:.1f}"]["recovery_pooled"] >= thr:
                return L
        return None
    incs = [(f"{L:.1f}", curve_d[f"{L:.1f}"]["increment_from_previous"])
            for L in sorted(LENGTHS)
            if curve_d[f"{L:.1f}"]["increment_from_previous"] is not None]
    step_to, step_val = max(incs, key=lambda t: t[1])
    step_from = round(float(step_to) - 0.1, 1)
    spp = {}
    for prot in TEST_P:
        a = curve_d[f"{step_from:.1f}"]["recovery_per_protein"][prot]
        b = curve_d[f"{step_to}"]["recovery_per_protein"][prot]
        spp[prot] = {"before": a, "after": b, "increment": b - a}
    return {"smallest_length_recovery_0.8_pooled": smallest(0.8),
            "smallest_length_recovery_0.9_pooled": smallest(0.9),
            "largest_increment": {"from_ns": step_from, "to_ns": float(step_to),
                                  "increment_pooled": step_val},
            "step_per_protein": spp}


hist6_curve = curve("hist6")
mean_curve = curve("mean")
readings = readings_of(hist6_curve)
mean_readings = readings_of(mean_curve)

# ---- 平均だけの曲がりの L23 記録との照合 ----
ref = json.loads(REF_RESULTS.read_text())
mean_diffs = {}
for L in sorted(LENGTHS):
    ref_rec = ref["lengths"][f"{L:.1f}"]["recovery_pooled"]
    mean_diffs[f"{L:.1f}"] = mean_curve[f"{L:.1f}"]["recovery_pooled"] - ref_rec
max_mean_diff = max(abs(v) for v in mean_diffs.values())
print("mean-curve cross-check vs %s: max |diff| = %.6f" % (REF_NAME, max_mean_diff), flush=True)
print("readings (hist6):", readings, flush=True)
print("readings (mean, re-measured):", mean_readings, flush=True)

def shrink(a, b):
    """a = 統計量6種の最小長さ、b = 平均だけの最小長さ。"""
    if a is None and b is None:
        return "both_not_reached"
    if a is None:
        return "hist6_not_reached_but_mean_reached"
    if b is None:
        return "mean_not_reached_but_hist6_reached"
    if a < b:
        return "shorter_with_hist6"
    if a > b:
        return "longer_with_hist6"
    return "same"

comparison = {
    "recovery_0.8": {"hist6": readings["smallest_length_recovery_0.8_pooled"],
                     "mean_only": mean_readings["smallest_length_recovery_0.8_pooled"],
                     "verdict": shrink(readings["smallest_length_recovery_0.8_pooled"],
                                       mean_readings["smallest_length_recovery_0.8_pooled"])},
    "recovery_0.9": {"hist6": readings["smallest_length_recovery_0.9_pooled"],
                     "mean_only": mean_readings["smallest_length_recovery_0.9_pooled"],
                     "verdict": shrink(readings["smallest_length_recovery_0.9_pooled"],
                                       mean_readings["smallest_length_recovery_0.9_pooled"])},
}

out = {
    "label": f"Contact dissociation {HORIZON_NS}ns: recovery curve over covered length (hist6, Doc D)",
    "series": "L (leverage of short real MD)",
    "experiment": EXP_NAME,
    "phenomenon": "Contact dissociation (Doc D, k50_cache)",
    "horizon_ns": HORIZON_NS,
    "statistics": "Doc D hist6 of the prefix frames: [first, mean, std, min, last-first, frac<=4.5A]",
    "length_definition": {
        "covered_length_ns": "prefix frames 0..(L/0.1); L=0 -> frame 0 only; L=1.0 -> frames 0..10",
        "note": "Same definition as L01/L02/L21-L26. hist6 at covered 1.0 ns equals full_hist6.",
    },
    "feature_dtypes": "static features float64; prefix mean float32; hist6 float32 (G71 construction)",
    "base_reproduction": {
        "static_only_auprc": a_static, "full_hist6_auprc": a_full,
        "full_gain": gain_full,
        "expected_from_L23_G71_G140": EXPECTED,
        "match": "static/full/gain/positives within 5e-4",
    },
    "metrics": res,
    "per_protein_metrics": per_protein,
    "lengths_hist6": hist6_curve,
    "lengths_mean_recheck": mean_curve,
    "mean_curve_diff_vs_L23": mean_diffs,
    "mean_curve_max_abs_diff_vs_L23": max_mean_diff,
    "reading_rule_fixed_in_advance": {
        "1": "smallest covered length reaching recovery 0.8 and 0.9 (pooled)",
        "2": "largest increment between adjacent lengths and its size",
        "3": "CI = min to max of the three recoveries (pooled, 1mix_A, 2qt4_A); coarse, 2 test proteins",
        "4": "per-protein values around the largest-increment interval",
        "plus": "whether the needed length shrank compared with mean only (same run)",
    },
    "readings": readings,
    "readings_mean_recheck": mean_readings,
    "comparison_vs_mean_only": comparison,
}


def clean_nan(x):
    if isinstance(x, dict):
        return {k: clean_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_nan(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


(OUT / "results.json").write_text(json.dumps(clean_nan(out), indent=2, ensure_ascii=False))

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["covered_length_ns", "n_frames",
                "hist6_auprc", "hist6_recovery_pooled", "hist6_recovery_1mix_A",
                "hist6_recovery_2qt4_A", "hist6_ci_min", "hist6_ci_max",
                "hist6_increment_from_previous",
                "mean_auprc", "mean_recovery_pooled", "mean_increment_from_previous"])
    for L in sorted(LENGTHS):
        d6 = hist6_curve[f"{L:.1f}"]
        dm = mean_curve[f"{L:.1f}"]
        w.writerow([L, d6["n_frames"], round(d6["metrics"]["auprc"], 4),
                    round(d6["recovery_pooled"], 4),
                    round(d6["recovery_per_protein"]["1mix_A"], 4),
                    round(d6["recovery_per_protein"]["2qt4_A"], 4),
                    round(d6["ci_min"], 4), round(d6["ci_max"], 4),
                    "" if d6["increment_from_previous"] is None else round(d6["increment_from_previous"], 4),
                    round(dm["metrics"]["auprc"], 4), round(dm["recovery_pooled"], 4),
                    "" if dm["increment_from_previous"] is None else round(dm["increment_from_previous"], 4)])

with open(OUT / "per_protein.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "protein", "n", "positives", "auprc", "auc", "brier", "logloss"])
    for cname in CONDS:
        for prot in TEST_P:
            m = per_protein[cname][prot]
            w.writerow([cname, prot, m["n"], m["positives"], round(m["auprc"], 4),
                        round(m["auc"], 4), round(m["brier"], 4), round(m["logloss"], 4)])

np.savez_compressed(OUT / "test_predictions.npz",
                    **{f"pred_{c}": preds[c].astype(np.float32) for c in CONDS},
                    y_test=yte, protein=test_protein)
print("saved", OUT, flush=True)
