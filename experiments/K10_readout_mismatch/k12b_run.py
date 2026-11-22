"""K12b: 読み出しのずれの強い土台での確認

K06(b)（凍結した本体 T1I2006＋補正モデル、未見100タンパク質）では、
生成側で学習し直した条件 (c) でも生成特徴は補正に何も足さなかった。
K10（弱い土台）で「実履歴で学習した予測器を生成履歴に当てると利得が
ほぼ消える」という結論が、強い土台でも成り立つかを確かめる。

データ: K06(b) の計算済み特徴をそのまま使う
  ~/workspace/k06b/run/k06b_out/<protein>.npz （108タンパク質分）
  各ファイルに r{1,2,3}_static(5) / history(10) / frozen / label(6) /
  gen8(8) / gen6(6) が入っている。軌道の取り直しはしない。

次元の問題と対応: 実履歴は DocA の10特徴、生成は6特徴で、そのままでは
学習と評価の出所を入れ替えられない。そこで実履歴から K10 と同じ6統計量
だけを取り出す（history の列 [0,1,2,3,5,7] = 最初の値・平均・標準偏差・
最小値・最後−最初・3.5 A以下の割合。K10 の hist6 と同一の定義）。
生成側は保存済みの gen6 をそのまま使う。静的5特徴は全条件で共通。
このため交差条件の (a) の比較相手は (a6)（実6→実6）になる。
K06(b) の (a10)（実10→実10）は引用で併記する。

条件（補正モデルの学習器は K06(b) の eval_b.py と同一:
 HistGradientBoosting max_iter=160, lr=.065, leaf15, min_samples_leaf20,
 L2=.15, seed=20261121。学習は較正8タンパク質、評価は未見100タンパク質
 x3複製の合算163,074件。指標は全体（pooled）の AUC/AUPRC/Brier/LogLoss
 とタンパク質平均（macro）の AUPRC）:
  static   静的5のみの補正（K06(b) から引用）
  a10      実10→実10（K06(b) の history から引用）
  a6       実6→実6（新規）
  b        実6→生成6（新規、主条件）
  c        生成6→生成6（K06(b) の gen6 から引用し、本実行で再現確認）
  d        生成6→実6（新規）
  frozen   凍結した本体だけ（K06(b) から引用）

読み方（先に固定）:
 強い土台でも (b) が静的補正と同水準かそれ以下なら、読み出しのずれの
 結論は強い土台でも成り立つ。(b) が K10 の弱い土台での結果よりずっと
 良ければ、ずれは土台の強さに依存する。

途中結果は results_k12b_partial.json に条件×予測幅ごとに書き出す。
"""
import numpy as np, json, csv
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

OUTD = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
OUT = Path(__file__).resolve().parent
PARTIAL = OUT / "results_k12b_partial.json"
HORIZONS = (1, 5, 10, 30, 50, 90)
SEED = 20261121
REAL6_IDX = [0, 1, 2, 3, 5, 7]

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
print("n_cal", len(y_cal), "n_test", len(y_test), "test pos", y_test.sum(0), flush=True)

# feature blocks: static(5) + real6 / gen6 (6 each)
def block(rows, kind):
    if kind == "real6":
        return [np.column_stack([r["static"], r["history"][:, REAL6_IDX]]) for r in rows]
    if kind == "gen6":
        return [np.column_stack([r["static"], r["gen6"]]) for r in rows]
    raise ValueError(kind)

Rtr = np.concatenate(block(cal, "real6")); Rte = np.concatenate(block(test, "real6"))
Gtr = np.concatenate(block(cal, "gen6")); Gte = np.concatenate(block(test, "gen6"))
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

CONDITIONS = {
    "a6_train_real6_eval_real6": ("real6", "real6"),
    "b_train_real6_eval_gen6": ("real6", "gen6"),
    "c_train_gen6_eval_gen6": ("gen6", "gen6"),
    "d_train_gen6_eval_real6": ("gen6", "real6"),
}
FEATS = {"real6": (Rtr, Rte), "gen6": (Gtr, Gte)}

results = {}
if PARTIAL.exists():
    results = json.loads(PARTIAL.read_text())
    print("resumed from partial:", {k: sorted(v) for k, v in results.items()}, flush=True)

for name, (tr_src, ev_src) in CONDITIONS.items():
    results.setdefault(name, {})
    for j, h in enumerate(HORIZONS):
        if str(h) in results[name]:
            print(name, h, "already done, skipped", flush=True)
            continue
        m = learner(FEATS[tr_src][0], y_cal[:, j])
        p = m.predict_proba(FEATS[ev_src][1])[:, 1]
        row = metrics(y_test[:, j], p)
        macro, nprot = macro_auprc(y_test, p, j)
        row["macro_auprc"] = macro; row["macro_n_proteins"] = nprot
        results[name][str(h)] = row
        print(name, h, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
        PARTIAL.write_text(json.dumps(results, indent=2))

# checks: (c) must reproduce K06(b) gen6
k06b = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K06_aligned_recomparison/k06b_results.json").read_text())
checks = {
    "c_vs_K06b_gen6_auprc_max_abs_diff":
        max(abs(results["c_train_gen6_eval_gen6"][str(h)]["auprc"] - k06b["by_horizon"][str(h)]["gen6"]["auprc"]) for h in HORIZONS),
    "c_vs_K06b_gen6_macro_auprc_max_abs_diff":
        max(abs(results["c_train_gen6_eval_gen6"][str(h)]["macro_auprc"] - k06b["by_horizon"][str(h)]["gen6"]["macro_auprc"]) for h in HORIZONS),
}
print("checks vs K06(b) gen6:", checks, flush=True)

# per-protein mean gain vs static (DocA style) for the new conditions
static_macro = {h: k06b["by_horizon"][str(h)]["static"]["macro_auprc"] for h in HORIZONS}
# recompute per-protein AUPRCs for static from K06b? Not stored per protein; use macro from K06b
# instead report macro gains using K06b static macro as reference
summary = {}
for name in CONDITIONS:
    summary[name] = {}
    for h in HORIZONS:
        s_pooled = k06b["by_horizon"][str(h)]["static"]["auprc"]
        summary[name][str(h)] = {
            "auprc_gain_vs_static_pooled": float(results[name][str(h)]["auprc"] - s_pooled),
            "auprc_gain_vs_static_macro": float(results[name][str(h)]["macro_auprc"] - static_macro[h]),
        }
print("summary gains vs static:", json.dumps(summary, indent=1), flush=True)

payload = {"conditions": results, "checks_vs_K06b_gen6": checks,
           "gains_vs_static_K06b": summary, "horizons": list(HORIZONS),
           "real6_history_columns": REAL6_IDX,
           "note": "real6 = DocA history cols [0,1,2,3,5,7] == K10 hist6; gen6 as saved in K06(b); "
                   "learner identical to K06(b) eval_b.py; (a10)/static/frozen quoted from k06b_results.json"}
(OUT / "results_k12b.json").write_text(json.dumps(payload, indent=2))
with open(OUT / "results_k12b.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "horizon_ns", "n", "positives", "auc", "auprc", "macro_auprc", "brier", "logloss"])
    for name in CONDITIONS:
        for h in HORIZONS:
            r = results[name][str(h)]
            w.writerow([name, h, r["n"], r["positives"], r["auc"], r["auprc"], r["macro_auprc"], r["brier"], r["logloss"]])
print("saved to", OUT)
