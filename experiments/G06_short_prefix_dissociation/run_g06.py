"""G06: 短い実MDの切り取りは、解離でも指導力を戻すか.

G02（水素結合の形成）では、0.1 ns の切り取りで回復率 0.570、0.5 ns で 0.770
だった。同じ切り取りを解離の2現象で確かめる。
  (a) 水素結合の解離（資料Dの条件、K46 のキャッシュ）
  (b) 接触の解離（K50 の条件、K50 のキャッシュ）

切り取りの定義（G02 と同じ、結果を見る前に固定）:
- 軌道の保存間隔は 100 ps（0.1 ns）。フレーム k は時刻 k*0.1 ns。
- 0.1 ns -> フレーム 0, 1（2フレーム）
- 0.5 ns -> フレーム 0〜5（6フレーム）
- 起点フレーム10（1 ns）は変えない。ラベルは 1 ns先（フレーム20）。

測るもの（1 ns先）:
- 土台の再現: 静的のみ・全部入りが K46・K50 の記録と小数第4位まで一致すること。
- (b) 切った平均だけ（静的特徴＋切った履歴の平均だけ）の利得の回復率。
  回復率 = (その条件の AUPRC - 静的のみ) / (実履歴の利得)。
- 参考: 切った6統計量全部の条件、切った平均と 0〜1 ns の平均との残りのばらつき
  （解離の候補について。G02 の形成の値と並べる）。

参照: 水素結合の解離の実履歴の利得 +0.1785（K46、1 ns先）、
      接触の解離の実履歴の利得 +0.2142（K50、1 ns先）。
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).resolve().parent
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZON_IDX = 0  # 1 ns先

# 切り取り: 名前 -> フレームの列
CUTS = {"0.1ns": [0, 1], "0.5ns": [0, 1, 2, 3, 4, 5]}

PHENOMENA = {
    "hb_dissociation": dict(
        cache=Path("/home/hatch/workspace/kinescope/work/data_cache_k46"),
        suffix="", bound=3.5, hist_transpose=True,
        ref_static_auprc=0.4160, ref_full_auprc=0.5945, ref_gain=0.1785,
    ),
    "contact_dissociation": dict(
        cache=Path("/home/hatch/workspace/kinescope/work/k50_cache"),
        suffix="_contact_diss", bound=4.5, hist_transpose=False,
        ref_static_auprc=0.3105, ref_full_auprc=0.5246, ref_gain=0.2142,
    ),
}


def load(cache, prots, suffix, hist_transpose):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(cache / f"{p}_R{r}{suffix}.npz")
            parts.append({k: z[k] for k in z.files})
    X = np.concatenate([p["X"] for p in parts])
    H = np.concatenate([p["hist"].T if hist_transpose else p["hist"] for p in parts])
    Y = np.concatenate([p["y"] for p in parts])
    return X, H, Y


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


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


results = {}
for tag, cfg in PHENOMENA.items():
    Xtr, Htr, Ytr = load(cfg["cache"], TRAIN_P, cfg["suffix"], cfg["hist_transpose"])
    Xte, Hte, Yte = load(cfg["cache"], TEST_P, cfg["suffix"], cfg["hist_transpose"])
    ytr, yte = Ytr[:, HORIZON_IDX], Yte[:, HORIZON_IDX]
    print(tag, "train", Xtr.shape, "test", Xte.shape,
          "test pos 1ns", int(yte.sum()), flush=True)

    full_mean_tr = Htr.mean(1)
    full_mean_te = Hte.mean(1)

    # (a) 残りのばらつき（切った平均 - 0〜1 ns の平均）
    residual = {}
    for name, idx in CUTS.items():
        pm_tr = Htr[:, idx].mean(1)
        pm_te = Hte[:, idx].mean(1)
        residual[name] = dict(
            frames=idx,
            train_residual_std=float(np.std(pm_tr - full_mean_tr)),
            test_residual_std=float(np.std(pm_te - full_mean_te)),
            train_residual_mean=float(np.mean(pm_tr - full_mean_tr)),
            test_residual_mean=float(np.mean(pm_te - full_mean_te)),
        )
        print(tag, name, "residual std test",
              round(residual[name]["test_residual_std"], 4), "train",
              round(residual[name]["train_residual_std"], 4), flush=True)

    conditions = {"static_only": None}
    conditions["full_hist6"] = (hist6(Htr, cfg["bound"]), hist6(Hte, cfg["bound"]))
    conditions["mean_only_full"] = (
        full_mean_tr[:, None].astype(np.float32),
        full_mean_te[:, None].astype(np.float32))
    for name, idx in CUTS.items():
        conditions[f"prefix_mean_{name}"] = (
            Htr[:, idx].mean(1)[:, None].astype(np.float32),
            Hte[:, idx].mean(1)[:, None].astype(np.float32))
        conditions[f"prefix_hist6_{name}"] = (
            hist6(Htr[:, idx], cfg["bound"]), hist6(Hte[:, idx], cfg["bound"]))

    mets = {}
    for cname, feats in conditions.items():
        if feats is None:
            p = predict(Xtr, ytr, Xte)
        else:
            ftr, fte = feats
            p = predict(np.concatenate([Xtr, ftr], axis=1), ytr,
                        np.concatenate([Xte, fte], axis=1))
        mets[cname] = metrics(yte, p)
        print(tag, cname, {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in mets[cname].items()}, flush=True)

    base_gain = mets["full_hist6"]["auprc"] - mets["static_only"]["auprc"]
    print(tag, "base gain (ref %.4f):" % cfg["ref_gain"], round(base_gain, 4), flush=True)
    assert abs(mets["static_only"]["auprc"] - cfg["ref_static_auprc"]) < 5e-4, mets["static_only"]
    assert abs(mets["full_hist6"]["auprc"] - cfg["ref_full_auprc"]) < 5e-4, mets["full_hist6"]

    recovery = {c: float((m["auprc"] - mets["static_only"]["auprc"]) / base_gain)
                for c, m in mets.items() if c != "static_only"}
    results[tag] = dict(
        cuts={k: v for k, v in CUTS.items()},
        residual_spread_vs_full_mean=residual,
        metrics_1ns=mets,
        base_gain_1ns=float(base_gain),
        reference_gain_1ns=cfg["ref_gain"],
        recovery_vs_real_history_gain=recovery,
    )
    print(tag, "recovery", {k: round(v, 3) for k, v in recovery.items()}, flush=True)

out = {
    "references": {
        "hb_dissociation_real_history_gain_1ns": 0.1785,
        "contact_dissociation_real_history_gain_1ns": 0.2142,
        "g02_formation_recovery_prefix_mean": {"0.1ns": 0.570, "0.5ns": 0.770},
    },
    "phenomena": results,
}
(HERE / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

rows = []
for tag, res in results.items():
    for cname, m in res["metrics_1ns"].items():
        rows.append(dict(phenomenon=tag, condition=cname, horizon_ns=1,
                         recovery=res["recovery_vs_real_history_gain"].get(cname, ""),
                         **m))
with open(HERE / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved", HERE / "results.json")
