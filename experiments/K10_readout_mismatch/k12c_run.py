"""K12c: 尺度合わせの強い土台での確認

K12b で、強い土台（凍結した本体 T1I2006＋補正モデル、未見100タンパク質）
でも、実履歴で学習した補正を生成特徴に当てると静的補正を下回った
（1 ns先 AUPRC 0.4004、利得 -0.0587）。K10b の尺度合わせが強い土台
でも同じように働くかを確かめる。

データ: K06(b)/K12b と同じ計算済み特徴
  ~/workspace/k06b/run/k06b_out/<protein>.npz （108タンパク質分）
  較正8タンパク質（13,808件）で学習、未見100タンパク質×3複製
  （163,074件）で評価。補正モデルの学習器は K06(b) の eval_b.py と同一
  （HistGradientBoosting、160本、学習率0.065、葉15、葉あたり最小20件、
  L2 0.15、seed 20261121）。

方法（K10b と同じ段取り）: 生成側の6統計量を、統計量ごとに実側の分布へ
線形に合わせる。gen_scaled = (gen - mu_gen)/sd_gen * sd_real + mu_real で、
mu・sd は較正タンパク質だけで求める（較正側の実6と生成6の各統計量の
平均・標準偏差）。評価側のデータでは変換のパラメータを決めない（漏れなし）。
変換後の生成統計量を、K12b の (b) と同じ実6学習モデルで評価する。
モデルは学習し直さない。

実6 = DocA history の列 [0,1,2,3,5,7]（K12b と同じ）。生成側は保存済みの
gen6。静的5特徴は全条件で共通。

条件:
  b            実6→生成6（尺度合わせなし、K12b から引用し本実行で再現確認）
  b_scaled     実6→生成6（尺度合わせあり）  <- K12c の主条件
比較先: 静的補正・(a6)・(c) は K12b / K06(b) から引用。

読み方（先に固定）:
  尺度合わせ後の (b) が静的補正に近づくか上回れば、尺度合わせは強い土台
  でも働く。ほとんど変わらなければ、強い土台でのずれは尺度の問題ではない。

途中結果は results_k12c_partial.json に予測幅ごとに書き出す。
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

OUTD = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_k12c_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED = 20261121
REAL6_IDX = [0, 1, 2, 3, 5, 7]
FEAT_NAMES = ["first", "mean", "std", "min", "last_minus_first", "frac_le_3.5"]

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(OUTD / f"{p}.npz") as z:
            for r in (1, 2, 3):
                rows.append({k: z[f"r{r}_{k}"] for k in
                             ("static", "history", "frozen", "label", "gen8", "gen6")})
    return rows

def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED).fit(X, y)

def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

cal, test = load(CAL), load(TEST)
y_cal = np.concatenate([r["label"] for r in cal])
y_test = np.concatenate([r["label"] for r in test])
print("n_cal", len(y_cal), "n_test", len(y_test), flush=True)

def blocks(rows):
    st = np.concatenate([r["static"] for r in rows])
    real6 = np.concatenate([r["history"][:, REAL6_IDX] for r in rows])
    gen6 = np.concatenate([r["gen6"] for r in rows])
    return st, real6, gen6

Str, Rtr, Gtr = blocks(cal)
Ste, Rte, Gte = blocks(test)

# --- affine fit on CALIBRATION proteins only ---
mu_real = Rtr.mean(0); sd_real = Rtr.std(0)
mu_gen = Gtr.mean(0); sd_gen = Gtr.std(0)
assert (sd_gen > 0).all() and (sd_real > 0).all()
scale = sd_real / sd_gen
shift = mu_real - scale * mu_gen
Gte_scaled = (Gte * scale + shift).astype(np.float32)
params = {name: {"mu_real_cal": float(mu_real[j]), "sd_real_cal": float(sd_real[j]),
                 "mu_gen_cal": float(mu_gen[j]), "sd_gen_cal": float(sd_gen[j]),
                 "scale": float(scale[j]), "shift": float(shift[j])}
          for j, name in enumerate(FEAT_NAMES)}
for name, v in params.items():
    print(name, {k: round(x, 4) for k, x in v.items()}, flush=True)
print("scaled test-gen mean", np.round(Gte_scaled.mean(0), 4),
      "real test mean", np.round(Rte.mean(0), 4), flush=True)
print("scaled test-gen std ", np.round(Gte_scaled.std(0), 4),
      "real test std ", np.round(Rte.std(0), 4), flush=True)

# spans for macro AUPRC
spans, cur = [], 0
for p in TEST:
    with np.load(OUTD / f"{p}.npz") as z:
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
    spans.append((p, cur, cur + n)); cur += n

def macro_auprc(y, p, j):
    vals = [average_precision_score(y[s:e, j], p[s:e])
            for _, s, e in spans if 0 < y[s:e, j].sum() < e - s]
    return float(np.mean(vals)), len(vals)

Xtr = np.concatenate([Str, Rtr], axis=1)
Xte_scaled = np.concatenate([Ste, Gte_scaled], axis=1)
Xte_unscaled = np.concatenate([Ste, Gte], axis=1)

k12b = json.loads((OUT / "results_k12b.json").read_text())
results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())["b_scaled"]
    print("resumed from partial:", sorted(results), flush=True)

for j, h in enumerate(HORIZONS):
    if str(h) in results:
        print("b_scaled", h, "already done, skipped", flush=True)
        continue
    m = learner(Xtr, y_cal[:, j])
    p = m.predict_proba(Xte_scaled)[:, 1]
    row = metrics(y_test[:, j], p)
    macro, nprot = macro_auprc(y_test, p, j)
    row["macro_auprc"] = macro; row["macro_n_proteins"] = nprot
    results[str(h)] = row
    print("b_scaled", h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    PARTIAL.write_text(json.dumps({"b_scaled": results, "affine_params_cal_only": params}, indent=2))

# checks: unscaled (b) reproduction via a fresh fit on one horizon (1ns)
m1 = learner(Xtr, y_cal[:, 0])
p1 = m1.predict_proba(Xte_unscaled)[:, 1]
repro = metrics(y_test[:, 0], p1)
checks = {"b_unscaled_1ns_repro_vs_K12b_auprc_abs_diff":
          abs(repro["auprc"] - k12b["conditions"]["b_train_real6_eval_gen6"]["1"]["auprc"])}
print("check (b) unscaled 1ns repro:", {k: round(v, 4) for k, v in repro.items()}, "vs K12b",
      round(k12b["conditions"]["b_train_real6_eval_gen6"]["1"]["auprc"], 4), "diff", checks, flush=True)

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())
summary = {}
for h in HORIZONS:
    s = k06b["by_horizon"][str(h)]["static"]["auprc"]
    summary[str(h)] = {
        "static_auprc": s,
        "a6_auprc": k12b["conditions"]["a6_train_real6_eval_real6"][str(h)]["auprc"],
        "b_unscaled_auprc": k12b["conditions"]["b_train_real6_eval_gen6"][str(h)]["auprc"],
        "b_scaled_auprc": results[str(h)]["auprc"],
        "c_auprc": k12b["conditions"]["c_train_gen6_eval_gen6"][str(h)]["auprc"],
        "b_scaled_gain_vs_static_pooled": float(results[str(h)]["auprc"] - s),
        "b_scaled_minus_b_unscaled_auprc": float(results[str(h)]["auprc"] - k12b["conditions"]["b_train_real6_eval_gen6"][str(h)]["auprc"]),
    }
s1 = summary["1"]
summary["horizon_1ns_fractions"] = {
    "b_scaled_recovers_fraction_of_c_gain": float((s1["b_scaled_auprc"] - s1["static_auprc"]) / (s1["c_auprc"] - s1["static_auprc"])) if s1["c_auprc"] != s1["static_auprc"] else None,
    "b_scaled_recovers_fraction_of_a6_gain": float((s1["b_scaled_auprc"] - s1["static_auprc"]) / (s1["a6_auprc"] - s1["static_auprc"])),
}
print("summary:", json.dumps(summary, indent=1), flush=True)

payload = {"conditions": {"b_scaled": results,
                          "b_unscaled_K12b_reference": k12b["conditions"]["b_train_real6_eval_gen6"]},
           "affine_params_cal_only": params,
           "checks": checks, "summary": summary, "horizons": list(HORIZONS),
           "note": "affine gen6->real6 per-feature, fitted on CALIBRATION proteins only; "
                   "model is K12b (b) real6-trained model, unchanged; "
                   "(b) unscaled values quoted from K12b results_k12b.json (1ns repro checked in-run)"}
(OUT / "results_k12c.json").write_text(json.dumps(payload, indent=2))
with open(OUT / "results_k12c.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for name, cond in (("b_scaled", results), ("b_unscaled_K12b_reference", k12b["conditions"]["b_train_real6_eval_gen6"])):
        for h in HORIZONS:
            r = cond[str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
