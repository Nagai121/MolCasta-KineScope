"""G99: 接触の解離で、切り取りを 6 個の統計量にした場合も測る
（資料Dの条件、接触の解離、1 ns先・30 ns先）。

G79・G85・G89 と同じ生成平均で、切り取りだけを 6 個の統計量に
変えるのが目的。生成平均は G18 で作り直した方法22の系列から、
G79 と同じやり方で作る：1つの残基ペアに生成ペアが複数ある場合は、
生成ペアの 4点系列の点ごとの最小値をその残基ペアの生成系列とし、
4点で平均する。K58 の残基レベルの第11〜14案は使わない。案の
分け方はない。単一の生成平均で測る。

候補は G18 の生成系列と重なる候補だけ（G79 の count_overlap.py:
学習の 70%、テストの 76%）。土台（静的のみ・全部入り・切り取り
だけ）も同じ重なる候補だけで測り直すので、回復率の分母は重なる
候補だけの全部入りの利得。全部の候補での土台は別に確かめ、G71 の
記録と比べる（1 ns先: 静的 0.3105 / 全部入り 0.5246 / 利得 +0.2142 /
切り取り 0.1 ns の回復率 0.919。30 ns先: 静的 0.2756 / 全部入り
0.5124 / 利得 +0.2368 / 回復率 0.796）。

平均だけの混合では、1 ns先は −0.040、30 ns先は +0.025 だった。

モデルは HistGradientBoosting（反復120回、学習率0.075、葉31、
葉あたり最小15件、L2 0.02、seed 42）で条件と予測幅ごとに学習し直す。
距離はフレームごとの箱を軸ごとに扱う周期境界で計算している
（キャッシュをそのまま使う）。6 統計量の閾値は 4.5 Å（K50）。

読み方（先に固定）: 混合の回復率が 6 統計量の切り取りだけの回復率
より +0.05 を超えれば混合は足しになる。超えなければならない。
"""
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).resolve().parent
G18 = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
K50 = Path("/home/hatch/workspace/kinescope/work/k50_cache")
ATOMS = Path("/tmp/g79_atoms/trajectories")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")


def hist6(series, bound):
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


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))


def load_with_gen(prots):
    """K50 キャッシュに、候補ごとの生成平均を付ける。

    X, H, Y、重なるかどうかのマスク、生成平均の配列（重ならない候補は
    NaN）を返す。並びはタンパク質の順、レプリカ 1〜3 の順で、キャッシュ
    のファイルと同じ。生成平均は G79 と同じ作り方（残基ペアに属する
    生成ペアの 4点系列の点ごとの最小値、4点で平均）。
    """
    Xs, Hs, Ys, matched, gen_mean = [], [], [], [], []
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
    return (np.concatenate(Xs), np.concatenate(Hs), np.concatenate(Ys),
            np.concatenate(matched), np.concatenate(gen_mean))


HORIZONS = {1: 0, 30: 3}  # 予測幅 ns -> y の列
# 予測幅 ns -> (静的のみの AUPRC、全部入りの AUPRC、切り取り 0.1 ns の回復率)、G71 の記録
G71_REF = {1: (0.3105, 0.5246, 0.919), 30: (0.2756, 0.5124, 0.796)}

Xtr, Htr, Ytr, mtr, gmtr = load_with_gen(TRAIN_P)
Xte, Hte, Yte, mte, gmte = load_with_gen(TEST_P)
print(f"train n: {len(Ytr)} matched {int(mtr.sum())} "
      f"test n: {len(Yte)} matched {int(mte.sum())}", flush=True)
pf_tr = Htr[:, [0, 1]].mean(1)[:, None]
pf_te = Hte[:, [0, 1]].mean(1)[:, None]
pf6_tr = hist6(Htr[:, [0, 1]], 4.5)
pf6_te = hist6(Hte[:, [0, 1]], 4.5)

results = {"train_n_matched": int(mtr.sum()), "test_n_matched": int(mte.sum()),
           "train_n_full": int(len(Ytr)), "test_n_full": int(len(Yte)),
           "horizons": {}}
rows = []
for hns, hi in HORIZONS.items():
    ytr_full, yte_full = Ytr[:, hi], Yte[:, hi]

    # --- 全部の候補での検証（G71 の土台＋切り取りの回復率） ---
    p_static = predict(Xtr, ytr_full, Xte)
    p_full = predict(np.column_stack([Xtr, hist6(Htr, 4.5)]), ytr_full,
                     np.column_stack([Xte, hist6(Hte, 4.5)]))
    p_pref = predict(np.column_stack([Xtr, pf_tr]), ytr_full,
                     np.column_stack([Xte, pf_te]))
    s_f, f_f = metrics(yte_full, p_static)["auprc"], metrics(yte_full, p_full)["auprc"]
    rec_f = (metrics(yte_full, p_pref)["auprc"] - s_f) / (f_f - s_f)
    ref_s, ref_f, ref_rec = G71_REF[hns]
    print(f"{hns} ns FULL SET verify vs G71: static {s_f:.4f} (recorded {ref_s}) "
          f"full {f_f:.4f} (recorded {ref_f}) gain {f_f - s_f:+.4f} "
          f"prefix 0.1ns recovery {rec_f:.3f} (recorded {ref_rec})", flush=True)
    assert abs(s_f - ref_s) < 1e-4 and abs(f_f - ref_f) < 1e-4
    assert abs(rec_f - ref_rec) < 0.005

    # --- 重なる候補だけでの測定 ---
    Xtr_m, Xte_m = Xtr[mtr], Xte[mte]
    Htr_m, Hte_m = Htr[mtr], Hte[mte]
    ytr, yte = ytr_full[mtr], yte_full[mte]
    pf_tr_m, pf_te_m = pf_tr[mtr], pf_te[mte]
    pf6_tr_m, pf6_te_m = pf6_tr[mtr], pf6_te[mte]
    gm_tr_m = gmtr[mtr][:, None]
    gm_te_m = gmte[mte][:, None]
    conds = {
        "static_only": (Xtr_m, Xte_m),
        "full_hist": (np.column_stack([Xtr_m, hist6(Htr_m, 4.5)]),
                      np.column_stack([Xte_m, hist6(Hte_m, 4.5)])),
        "prefix_mean_0.1ns": (np.column_stack([Xtr_m, pf_tr_m]),
                              np.column_stack([Xte_m, pf_te_m])),
        "prefix_hist6_0.1ns": (np.column_stack([Xtr_m, pf6_tr_m]),
                               np.column_stack([Xte_m, pf6_te_m])),
        "gen_mean_only": (np.column_stack([Xtr_m, gm_tr_m]),
                          np.column_stack([Xte_m, gm_te_m])),
        "mixed": (np.column_stack([Xtr_m, pf6_tr_m, gm_tr_m]),
                  np.column_stack([Xte_m, pf6_te_m, gm_te_m])),
    }
    entry = {"horizon_ns": hns, "test_n_matched": int(mte.sum()),
             "test_positives_matched": int(yte.sum()),
             "test_positives_full": int(yte_full.sum()),
             "full_set_verify": {"static_auprc": float(s_f), "full_auprc": float(f_f),
                                 "prefix_0.1ns_recovery": float(rec_f)},
             "conditions": {}}
    for name, (xtr, xte) in conds.items():
        p = predict(xtr, ytr, xte)
        m = metrics(yte, p)
        entry["conditions"][name] = m
        print(f"{hns} ns matched {name}: AUPRC {m['auprc']:.4f} AUC {m['auc']:.4f}", flush=True)
        rows.append({"horizon_ns": hns, "condition": name, **m})
    s = entry["conditions"]["static_only"]["auprc"]
    f = entry["conditions"]["full_hist"]["auprc"]
    entry["matched_base_gain"] = float(f - s)
    for name in ("prefix_mean_0.1ns", "prefix_hist6_0.1ns", "gen_mean_only", "mixed"):
        pm = entry["conditions"][name]["auprc"]
        entry["conditions"][name]["gain"] = float(pm - s)
        entry["conditions"][name]["recovery"] = float((pm - s) / (f - s))
        print(f"{hns} ns matched {name}: gain {pm - s:+.4f} recovery {(pm - s) / (f - s):.3f}",
              flush=True)
    entry["mixed_minus_prefix_hist6"] = float(
        entry["conditions"]["mixed"]["recovery"]
        - entry["conditions"]["prefix_hist6_0.1ns"]["recovery"])
    entry["mixed_minus_prefix_hist6_auprc"] = float(
        entry["conditions"]["mixed"]["auprc"]
        - entry["conditions"]["prefix_hist6_0.1ns"]["auprc"])
    print(f"{hns} ns mixed - prefix_hist6 recovery diff: {entry['mixed_minus_prefix_hist6']:+.3f} "
          f"AUPRC diff: {entry['mixed_minus_prefix_hist6_auprc']:+.4f} (line +0.05)", flush=True)
    results["horizons"][str(hns)] = entry

(HERE / "results.json").write_text(json.dumps(results, indent=2))
with (HERE / "results.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["horizon_ns", "condition", "n", "positives", "auc",
                                       "auprc", "brier", "logloss", "gain", "recovery"])
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in
                    ["horizon_ns", "condition", "n", "positives", "auc", "auprc", "brier",
                     "logloss", "gain", "recovery"]})
np.savez_compressed(HERE / "matched_gen_mean.npz",
                    train_matched=mtr, test_matched=mte,
                    train_gen_mean=gmtr, test_gen_mean=gmte)
print("saved to", HERE)
