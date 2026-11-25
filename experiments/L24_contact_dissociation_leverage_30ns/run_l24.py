"""L24: 短いMDの長さに対して、指導力がどれだけ戻るかの曲がり具合
（資料Dの条件、接触の解離、30 ns先）.

L 系列（短い実MDの覆う長さに対して回復率がどう伸びるかを測る系列）。
L01（水素結合の形成、1 ns先、資料D）と同じ曲がりの測り方を、接触の
解離の 30 ns先で測る。実の 0〜1 ns の履歴を始めの短い部分だけに切って
使い、切り取りの長さを 0.1 ns 刻みで細かく変えたときの、平均だけの
回復率の曲がり具合を測る。

長さの呼び方（L01・L02 と同じ、結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 長さは「覆う長さ」で統一する。覆う長さ L ns の切り取りは
  フレーム 0〜(L/0.1) を使う（フレーム数は L/0.1 + 1）。
  - 覆う長さ 0 ns   -> フレーム 0 のみ（1フレーム）
  - 覆う長さ 0.1 ns -> フレーム 0, 1（2フレーム）
  - ...
  - 覆う長さ 1.0 ns -> フレーム 0〜10（11フレーム。0〜1 ns の全部）
- G71 の呼び方との対応: G71 の 0.1 ns（フレーム 0, 1）= 覆う長さ 0.1 ns、
  G71 の 0.25 ns（フレーム 0〜2）= 覆う長さ 0.2 ns、
  G71 の 0.5 ns（フレーム 0〜5）= 覆う長さ 0.5 ns。
- 起点フレーム10（1 ns）は変えない。ラベルは 30 ns先。

条件:
- 静的のみ: 静的特徴5個だけ。
- 全部入り（full_hist6）: 静的特徴＋実履歴6統計量（0〜1 ns の11フレーム、
  境界 4.5 Å）。回復率の分母（実履歴の利得）の参照。
- 切り取りの平均だけ: 静的特徴＋切り取り（フレーム 0〜k）の距離の平均だけ。
  覆う長さ 0.0〜1.0 ns の 11通り。

回復率 = (その長さの AUPRC - 静的のみ) / (実履歴の利得)。

読み方（結果を見る前に固定）:
1. 回復率 0.8 と 0.9 に届く最小の覆う長さを、統合値で読む。
2. 隣り合う長さの間の増え幅がいちばん大きい場所と大きさ（段の有無）。
3. 信頼区間は資料Dの流儀で、3つの値（統合・1mix_A・2qt4_A）の
   回復率の最小から最大。テストは 2タンパク質だけなので区間は粗い。
4. 段らしい場所（増え幅が最大の区間）があれば、その前後で
   タンパク質ごとの値も記録する。

土台の再現の期待値（G71・G140 の記録、30 ns先）:
- 静的のみ AUPRC 0.2756、全部入り 0.5124、利得 0.2368、
  テスト 2,915件・陽性 246 件。
- G71 と重なる長さの回復率: 覆う長さ 0.1 ns で 0.7964、
  0.2 ns で 0.7953、0.5 ns で 0.8275（5e-3 以内で照合する）。

作り直しの経緯: 最初の実行では静的特徴をキャッシュの型のまま使ったと
ころ、土台の再現（静的のみ・全部入り）は G71・G140 と合ったが、覆う長さ
0.5 ns の回復率が G71 と合わなかった（10 ns先で 0.847 に対して G71 は
0.802、30 ns先で 0.784 に対して 0.827、50 ns先で 0.748 に対して 0.827、
90 ns先で 0.907 に対して 0.930）。原因は入力の型の違いで、G71 は静的
特徴を float64 にそろえ、切り取りの平均を float32 に落としてから並べ
ている。この実行では入力の作り方を G71 と同じにして回し直す。

実行はローカルの仮想マシンで行う（13条件の学習で数十秒。L 系列では
重い計算は Colab に逃がす決め事があるが、この規模は L01・L02 と
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
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
BOUND = 4.5
HIDX = 3  # 30 ns先（予測幅は 1/5/10/30/50/90 ns で 0〜5）
HORIZON_NS = 30
EXP_NAME = "L24"

EXPECTED = {"static_only": 0.2756011407090635, "full_hist6": 0.512394342139979,
            "gain": 0.23679320143091553, "test_n": 2915, "test_positives": 246}
EXPECTED_G71 = {0.1: 0.7963620511410819, 0.2: 0.795308289784015, 0.5: 0.8274939396562517}

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
Xtr, Xte = (cat(train, "X").astype(np.float64),
            cat(test, "X").astype(np.float64))
# H はキャッシュの型のまま使う（G71 と同じ）。hist6 は内部で float32 に
# 落とし、切り取りの平均は float32 に落としてから静的特徴と並べる。
Htr, Hte = cat(train, "hist"), cat(test, "hist")
Ytr, Yte = cat(train, "y"), cat(test, "y")
ytr, yte = Ytr[:, HIDX], Yte[:, HIDX]
print("train", Xtr.shape, "test", Xte.shape, "test pos", int(yte.sum()), flush=True)
assert Xtr.shape[0] == 6383 and Xte.shape[0] == 2915, (Xtr.shape, Xte.shape)

CONDS = ["static_only", "full_hist6"] + [f"mean_len_{L:.1f}" for L in LENGTHS]


def feats_for(cname):
    if cname == "static_only":
        return Xtr, Xte
    if cname == "full_hist6":
        return (np.column_stack([Xtr, hist6(Htr)]),
                np.column_stack([Xte, hist6(Hte)]))
    L = float(cname.replace("mean_len_", ""))
    n = LENGTHS[L]
    return (np.column_stack([Xtr, Htr[:, :n].mean(1)[:, None].astype(np.float32)]),
            np.column_stack([Xte, Hte[:, :n].mean(1)[:, None].astype(np.float32)]))


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

# ---- 土台の再現の確認（G71・G140 の記録と照合） ----
a_static = res["static_only"]["auprc"]
a_full = res["full_hist6"]["auprc"]
gain_full = a_full - a_static
assert res["static_only"]["n"] == EXPECTED["test_n"]
assert res["static_only"]["positives"] == EXPECTED["test_positives"], res["static_only"]
assert abs(a_static - EXPECTED["static_only"]) < 5e-4, a_static
assert abs(a_full - EXPECTED["full_hist6"]) < 5e-4, a_full
assert abs(gain_full - EXPECTED["gain"]) < 5e-4, gain_full
print("base reproduction vs G71/G140: OK (static %.4f / full %.4f / gain %+.4f)"
      % (a_static, a_full, gain_full), flush=True)

# ---- 回復率・区間・増え分 ----
def recovery(auprc, static, full):
    return (auprc - static) / (full - static)


lengths_out = {}
prev_rec = None
for L in sorted(LENGTHS):
    cname = f"mean_len_{L:.1f}"
    rec_pool = recovery(res[cname]["auprc"], a_static, a_full)
    rec_pp = {}
    for prot in TEST_P:
        pp = per_protein[cname][prot]
        ps = per_protein["static_only"][prot]
        pf = per_protein["full_hist6"][prot]
        rec_pp[prot] = recovery(pp["auprc"], ps["auprc"], pf["auprc"])
    three = [rec_pool, rec_pp["1mix_A"], rec_pp["2qt4_A"]]
    inc = None if prev_rec is None else rec_pool - prev_rec
    lengths_out[f"{L:.1f}"] = {
        "covered_length_ns": L,
        "n_frames": LENGTHS[L],
        "metrics": res[cname],
        "recovery_pooled": rec_pool,
        "recovery_per_protein": rec_pp,
        "ci_min": min(three), "ci_max": max(three),
        "increment_from_previous": inc,
    }
    prev_rec = rec_pool

# G71 と重なる長さの照合
for L, exp_rec in EXPECTED_G71.items():
    got = lengths_out[f"{L:.1f}"]["recovery_pooled"]
    assert abs(got - exp_rec) < 5e-3, (L, got, exp_rec)
print("G71 prefix cross-check: OK", flush=True)

# ---- 先に固定した読み方への答え ----
def smallest_length(thr):
    for L in sorted(LENGTHS):
        if lengths_out[f"{L:.1f}"]["recovery_pooled"] >= thr:
            return L
    return None


incs = [(f"{L:.1f}", lengths_out[f"{L:.1f}"]["increment_from_previous"])
        for L in sorted(LENGTHS) if lengths_out[f"{L:.1f}"]["increment_from_previous"] is not None]
step_to, step_val = max(incs, key=lambda t: t[1])
step_from = round(float(step_to) - 0.1, 1)
step_per_protein = {}
for prot in TEST_P:
    a = lengths_out[f"{step_from:.1f}"]["recovery_per_protein"][prot]
    b = lengths_out[f"{step_to}"]["recovery_per_protein"][prot]
    step_per_protein[prot] = {"before": a, "after": b, "increment": b - a}
readings = {
    "smallest_length_recovery_0.8_pooled": smallest_length(0.8),
    "smallest_length_recovery_0.9_pooled": smallest_length(0.9),
    "largest_increment": {"from_ns": step_from, "to_ns": float(step_to),
                          "increment_pooled": step_val},
    "step_per_protein": step_per_protein,
}
print("readings:", readings, flush=True)

out = {
    "label": f"Contact dissociation {HORIZON_NS}ns: recovery curve over covered length (mean only, Doc D)",
    "series": "L (leverage of short real MD)",
    "experiment": EXP_NAME,
    "phenomenon": "Contact dissociation (Doc D, k50_cache)",
    "horizon_ns": HORIZON_NS,
    "length_definition": {
        "covered_length_ns": "prefix frames 0..(L/0.1); L=0 -> frame 0 only; L=1.0 -> frames 0..10",
        "note": "Same definition as L01/L02. G71 0.1ns = covered 0.1, G71 0.25ns = covered 0.2, G71 0.5ns = covered 0.5.",
    },
    "base_reproduction": {
        "static_only_auprc": a_static, "full_hist6_auprc": a_full,
        "full_gain": gain_full,
        "expected_from_G71_G140": EXPECTED,
        "g71_prefix_recovery_expected": {str(k): v for k, v in EXPECTED_G71.items()},
        "match": "static/full/gain/positives within 5e-4; G71 prefix recoveries within 5e-3",
    },
    "metrics": res,
    "per_protein_metrics": per_protein,
    "lengths": lengths_out,
    "reading_rule_fixed_in_advance": {
        "1": "smallest covered length reaching recovery 0.8 and 0.9 (pooled)",
        "2": "largest increment between adjacent lengths and its size",
        "3": "CI = min to max of the three recoveries (pooled, 1mix_A, 2qt4_A); coarse, 2 test proteins",
        "4": "per-protein values around the largest-increment interval",
    },
    "readings": readings,
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
    w.writerow(["covered_length_ns", "n_frames", "auprc", "auc", "brier", "logloss",
                "recovery_pooled", "recovery_1mix_A", "recovery_2qt4_A",
                "ci_min", "ci_max", "increment_from_previous"])
    for L in sorted(LENGTHS):
        d = lengths_out[f"{L:.1f}"]
        m = d["metrics"]
        w.writerow([L, d["n_frames"], round(m["auprc"], 4), round(m["auc"], 4),
                    round(m["brier"], 4), round(m["logloss"], 4),
                    round(d["recovery_pooled"], 4),
                    round(d["recovery_per_protein"]["1mix_A"], 4),
                    round(d["recovery_per_protein"]["2qt4_A"], 4),
                    round(d["ci_min"], 4), round(d["ci_max"], 4),
                    "" if d["increment_from_previous"] is None else round(d["increment_from_previous"], 4)])

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
