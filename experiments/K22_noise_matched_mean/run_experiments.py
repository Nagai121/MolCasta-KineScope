"""K22: 生成の失敗は平均の誤差の大きさで説明できるか（強い土台、水素結合）

K11b/K08 の測定では、生成側の平均の誤差（生成の平均 − 実の平均）は
候補平均で -0.155 A、候補ごとのばらつきは標準偏差で約 0.48 A だった。
そこで、実系列の平均に生成側と同じ大きさの誤差を人為的に乗せたら、
指導力が生成側の水準まで落ちるかを測る。

方法: K06(b) の条件（凍結した本体 T1I2006＋補正モデル、較正8タンパク質
で学習、評価100タンパク質×3複製、候補163,074件）。実系列（0〜1 ns の
11フレーム、K22 用に再抽出したもの。抽出時に K06(b) の保存済み履歴
統計10個と小数第4位まで一致することを確認済み）の各候補の系列全体を
平行移動して、その候補の平均が「実の平均 - 0.155 A + 候補ごとに引いた
乱数（標準偏差 0.48 A）」になるようにする。平行移動なので系列の形
（標準偏差、増減のパターン）は変わらない。変わるのは水準に依存する
統計量（最初の値・平均・最小値・最大値と、閾値以下の割合）だけ。
乱数は候補ごとに独立に引き、学習側・テスト側の両方に当てる。
乱数の種を変えて3回（seed 101, 202, 303）行い、ばらつきも記録する。
補正モデルは条件ごとに学習し直す（学習器は K06(b) の eval_b.py と同一:
HistGradientBoosting、160本、学習率0.065、葉15、葉あたり最小20件、
L2 0.15、seed 20261121）。

参照（両端）: 全部入りの実履歴条件 (a10) と、生成条件（K06(b) の
生成6特徴・生成8特徴）、静的補正は K06(b) の k06b_results.json から引用。
利得は静的補正との AUPRC の差（全体とタンパク質平均の両方）。

読み方（先に固定）:
 誤差を乗せた実履歴の利得が生成条件の水準まで落ちれば、生成の失敗は
 平均の誤差の大きさで説明がつく。利得が生成条件よりずっと上に残れば、
 生成の失敗は平均の誤差だけでは説明できない。

途中結果は results_partial.json に seed×予測幅ごとに書き出す。
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/k22_cache")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED_MODEL = 20261121
NOISE_SEEDS = (101, 202, 303)
BIAS = -0.155
NOISE_SD = 0.48

def load(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            saved = {k: z[k] for k in z.files}
        with np.load(SERIES / f"{p}.npz") as z:
            series = {k: z[k] for k in z.files}
        for r in (1, 2, 3):
            rows.append({"static": saved[f"r{r}_static"], "label": saved[f"r{r}_label"],
                         "series": series[f"r{r}_series"]})
    return rows

def history10(series):
    """series: (n, 11) rows -> (n, 10), same formula as K06(b) score_b.py"""
    s = series.T
    return np.column_stack((s[0], s.mean(axis=0), s.std(axis=0), s.min(axis=0),
                            s.max(axis=0), s[-1] - s[0], s[-1] - s[-2],
                            (s <= 3.5).mean(axis=0), (s <= 4.5).mean(axis=0),
                            (np.diff(s, axis=0) > 0).mean(axis=0))).astype(np.float32)

def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED_MODEL).fit(X, y)

def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

cal = load(CAL); test = load(TEST)
Str = np.concatenate([r["static"] for r in cal]); Ste = np.concatenate([r["static"] for r in test])
Ytr = np.concatenate([r["label"] for r in cal]); Yte = np.concatenate([r["label"] for r in test])
SRtr = np.concatenate([r["series"] for r in cal]); SRte = np.concatenate([r["series"] for r in test])
print("n_cal", len(Ytr), "n_test", len(Yte), flush=True)

spans, cur = [], 0
for p in TEST:
    with np.load(SAVED / f"{p}.npz") as z:
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
    spans.append((p, cur, cur + n)); cur += n

def macro_auprc(y, p, j):
    vals = [average_precision_score(y[s:e, j], p[s:e])
            for _, s, e in spans if 0 < y[s:e, j].sum() < e - s]
    return float(np.mean(vals))

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed:", {k: sorted(v) for k, v in results.items() if isinstance(v, dict)}, flush=True)

for seed in NOISE_SEEDS:
    name = f"shifted_seed{seed}"
    results.setdefault(name, {})
    rng = np.random.default_rng(seed)
    d_tr = BIAS + rng.normal(0.0, NOISE_SD, len(SRtr))
    d_te = BIAS + rng.normal(0.0, NOISE_SD, len(SRte))
    Htr = history10(SRtr + d_tr[:, None].astype(np.float32))
    Hte = history10(SRte + d_te[:, None].astype(np.float32))
    Xtr = np.concatenate([Str, Htr], axis=1); Xte = np.concatenate([Ste, Hte], axis=1)
    results[f"_noise_realized_seed{seed}"] = {
        "cal_delta_mean": float(d_tr.mean()), "cal_delta_sd": float(d_tr.std()),
        "test_delta_mean": float(d_te.mean()), "test_delta_sd": float(d_te.std())}
    for j, h in enumerate(HORIZONS):
        if str(h) in results[name]:
            print(name, h, "already done, skipped", flush=True); continue
        m = learner(Xtr, Ytr[:, j])
        p = m.predict_proba(Xte)[:, 1]
        row = metrics(Yte[:, j], p)
        row["macro_auprc"] = macro_auprc(Yte, p, j)
        results[name][str(h)] = row
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        PARTIAL.write_text(json.dumps(results, indent=2))

k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())
summary = {}
for h in HORIZONS:
    s = k06b["by_horizon"][str(h)]["static"]
    row = {"static_auprc": s["auprc"], "static_macro_auprc": s["macro_auprc"],
           "a10_real_auprc": k06b["by_horizon"][str(h)]["history"]["auprc"],
           "a10_gain_pooled": float(k06b["by_horizon"][str(h)]["history"]["auprc"] - s["auprc"]),
           "a10_gain_macro": float(k06b["by_horizon"][str(h)]["history"]["macro_auprc"] - s["macro_auprc"]),
           "gen6_gain_pooled": float(k06b["by_horizon"][str(h)]["gen6"]["auprc"] - s["auprc"]),
           "gen8_gain_pooled": float(k06b["by_horizon"][str(h)]["gen8"]["auprc"] - s["auprc"]),
           "gen6_gain_macro": float(k06b["by_horizon"][str(h)]["gen6"]["macro_auprc"] - s["macro_auprc"]),
           "gen8_gain_macro": float(k06b["by_horizon"][str(h)]["gen8"]["macro_auprc"] - s["macro_auprc"])}
    gains = []
    for seed in NOISE_SEEDS:
        r = results[f"shifted_seed{seed}"][str(h)]
        row[f"shifted_seed{seed}_gain_pooled"] = float(r["auprc"] - s["auprc"])
        row[f"shifted_seed{seed}_gain_macro"] = float(r["macro_auprc"] - s["macro_auprc"])
        gains.append(r["auprc"] - s["auprc"])
    row["shifted_gain_pooled_mean"] = float(np.mean(gains))
    row["shifted_gain_pooled_sd_across_seeds"] = float(np.std(gains))
    row["shifted_gain_pooled_min"] = float(np.min(gains))
    row["shifted_gain_pooled_max"] = float(np.max(gains))
    summary[str(h)] = row
print("summary:", json.dumps(summary, indent=1), flush=True)
results["_summary"] = summary
results["_references_K06b"] = {str(h): {k: k06b["by_horizon"][str(h)][k] for k in ("static", "history", "gen6", "gen8", "frozen")} for h in HORIZONS}

(OUT / "results.json").write_text(json.dumps(results, indent=2))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for seed in NOISE_SEEDS:
        name = f"shifted_seed{seed}"
        for h in HORIZONS:
            r = results[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
