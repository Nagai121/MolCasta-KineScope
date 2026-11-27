"""L2-04 第1段階: 全予測幅の学習・予測と前提確認。予測をチェックポイントに保存する。

再起動で止まっても、予測を失わないための分割版。L2-03 の第1段階と同じ段取りを、
塩橋の解離 (K15 の土台、L33〜L38 の条件) で行う。
終了時に以下を保存する (コミット対象外の作業ファイル):
- _checkpoint_phase1.npz: 予測幅ごとのラベルと各条件の予測確率 (float64)
- _checkpoint_phase1.json: 予測幅ごとの metrics とタンパク質ごとの AUPRC
- premise_check.json: 前提確認の結果 (48項目)

L2-03 (塩橋の形成、K15 の土台) との違い:
- データは K53 のキャッシュ (~/workspace/kinescope/work/k53_cache/、学習 735件、
  テスト 360件、分割は experiments/K15_saltbridge_expansion/split.json)。
  静的特徴5個、履歴は 11フレームの最小距離の系列。
- 全部入りは静的＋履歴6統計量 (最初、平均、標準偏差、最小、最後−最初、
  4.0 Å以下割合)。予測器は HistGradientBoosting (120回、学習率 0.075、
  葉 31、葉あたり最小 15、L2 0.02、種 42)。
- 区間の平均は L33 と同じく float32 で入れる (L33 の作り方)。
- 照合先は L33〜L38 (塩橋の解離、平均だけ、K15 の土台)。
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
SPLIT = Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json")
REPO_EXP = Path("/home/hatch/workspace/kinescope/repo/experiments")
OUT = Path(__file__).resolve().parent
SEED = 42
BOUND = 4.0
HORIZONS = [1, 5, 10, 30, 50, 90]
L_REF_DIR = {
    1: "L33_salt_dissociation_leverage_1ns",
    5: "L34_salt_dissociation_leverage_5ns",
    10: "L35_salt_dissociation_leverage_10ns",
    30: "L36_salt_dissociation_leverage_30ns",
    50: "L37_salt_dissociation_leverage_50ns",
    90: "L38_salt_dissociation_leverage_90ns",
}

LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(10 - int(round(L / 0.1)), 11)) for L in LENGTHS}
COND_ADJ = {L: f"adjacent_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = ["static_only", "full_history"] + [COND_ADJ[L] for L in LENGTHS]

sp = json.loads(SPLIT.read_text())
TRAIN_P = sp["orig_train"] + sp["train_additions"]
TEST_P = sp["orig_test"] + sp["test_additions"]


def load(proteins):
    X, H, Y = [], [], []
    spans = []
    cur = 0
    for p in proteins:
        xs, hs, ys = [], [], []
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt_diss.npz")
            xs.append(z["X"]); hs.append(z["hist"]); ys.append(z["y"])
        x = np.concatenate(xs); h = np.concatenate(hs); y = np.concatenate(ys)
        X.append(x); H.append(h); Y.append(y)
        spans.append((p, cur, cur + len(y))); cur += len(y)
    return (np.concatenate(X), np.concatenate(H), np.concatenate(Y), spans)


def hist6(series):
    return np.column_stack([
        series[:, 0], series.mean(1), series.std(1), series.min(1),
        series[:, -1] - series[:, 0], (series <= BOUND).mean(1)])


def learner(X, y):
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.075, max_leaf_nodes=31,
        min_samples_leaf=15, l2_regularization=0.02,
        random_state=SEED).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def main():
    t0 = time.time()
    Xtr, Htr, Ytr, _ = load(TRAIN_P)
    Xte, Hte, Yte, spans = load(TEST_P)
    print("train", len(Ytr), "test", len(Yte), flush=True)
    assert len(Ytr) == 735 and len(Yte) == 360, (len(Ytr), len(Yte))
    assert len(spans) == 30, len(spans)

    feats = {}
    feats["static_only"] = (Xtr, Xte)
    feats["full_history"] = (np.column_stack([Xtr, hist6(Htr)]),
                             np.column_stack([Xte, hist6(Hte)]))
    for L in LENGTHS:
        idx = FRAMES[L]
        feats[COND_ADJ[L]] = (
            np.column_stack([Xtr, Htr[:, idx].mean(1, keepdims=True).astype(np.float32)]),
            np.column_stack([Xte, Hte[:, idx].mean(1, keepdims=True).astype(np.float32)]))

    all_checks = []
    ckpt_npz_payload = dict(
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]))
    ckpt_json = {}
    for hi, h in enumerate(HORIZONS):
        y_tr = Ytr[:, hi].astype(int)
        y_test = Yte[:, hi].astype(int)
        preds, results = {}, {}
        per_protein_ap = {c: {} for c in CONDITIONS}
        for cname in CONDITIONS:
            xtr, xte = feats[cname]
            pred = learner(xtr, y_tr).predict_proba(xte)[:, 1]
            preds[cname] = pred
            m = metrics(y_test, pred)
            aps = []
            for p, s, e in spans:
                yy = y_test[s:e]
                if 0 < yy.sum() < e - s:
                    ap = float(average_precision_score(yy, pred[s:e]))
                    aps.append(ap)
                    per_protein_ap[cname][p] = ap
                else:
                    per_protein_ap[cname][p] = None
            m["macro_auprc"] = float(np.mean(aps)) if aps else None
            m["n_proteins_macro"] = len(aps)
            results[cname] = m
        print(f"[{h} ns] static pooled %.4f macro %.4f | full pooled %.4f macro %.4f (%.0fs)"
              % (results["static_only"]["auprc"], results["static_only"]["macro_auprc"],
                 results["full_history"]["auprc"], results["full_history"]["macro_auprc"],
                 time.time() - t0), flush=True)

        lref = json.loads((REPO_EXP / L_REF_DIR[h] / "results.json").read_text())
        lm = lref["metrics"]
        lg = lref["base_gain"]
        lrec = lref["recovery_vs_full_history_gain"]["prefix_mean_1.0ns"]
        assert results["static_only"]["positives"] == lm["static_only"]["positives"]
        gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
        gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]
        rec10_pooled = (results[COND_ADJ[1.0]]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        rec10_macro = (results[COND_ADJ[1.0]]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

        def chk(name, got, ref, tol):
            ok = abs(got - ref) < tol
            all_checks.append(dict(horizon_ns=h, check=name, got=float(got),
                                   reference=float(ref), tol=tol, ok=bool(ok),
                                   reference_source=f"experiments/{L_REF_DIR[h]}/results.json"))
            print(f"check [{h} ns]", name, "got %.4f ref %.4f tol %g -> %s" % (got, ref, tol, ok), flush=True)
            return ok

        chk("static_pooled_vs_L", results["static_only"]["auprc"], lm["static_only"]["auprc"], 5e-4)
        chk("static_macro_vs_L", results["static_only"]["macro_auprc"], lm["static_only"]["macro_auprc"], 5e-4)
        chk("full_pooled_vs_L", results["full_history"]["auprc"], lm["full_history"]["auprc"], 5e-4)
        chk("full_macro_vs_L", results["full_history"]["macro_auprc"], lm["full_history"]["macro_auprc"], 5e-4)
        chk("gain_pooled_vs_L", gain_pooled, lg["pooled"], 1e-3)
        chk("gain_macro_vs_L", gain_macro, lg["macro"], 1e-3)
        chk("recovery_1.0_pooled_vs_L", rec10_pooled, lrec["pooled"], 5e-3)
        chk("recovery_1.0_macro_vs_L", rec10_macro, lrec["macro"], 5e-3)

        ckpt_npz_payload[f"h{h}_label"] = y_test.astype(np.int8)
        for cname in CONDITIONS:
            ckpt_npz_payload[f"h{h}_{cname}"] = preds[cname].astype(np.float64)
        ckpt_json[str(h)] = dict(results=results, per_protein_ap=per_protein_ap)

    premise_ok = all(c["ok"] for c in all_checks)
    (OUT / "premise_check.json").write_text(json.dumps(
        dict(checks=all_checks, all_ok=bool(premise_ok)), indent=2, ensure_ascii=False))
    if not premise_ok:
        print("PREMISE FAILED: 長さ 1.0 ns の点が L の記録と一致しないため、ここで止める。", flush=True)
        sys.exit(2)
    print("premise OK for all 6 horizons (%.0fs)" % (time.time() - t0), flush=True)

    # np.savez はファイル名に .npz を足すため、パスをそのまま渡すと
    # "_checkpoint_phase1.npz.tmp.npz" ができて os.replace が失敗する。
    # ファイルハンドルを渡すと指定したパスそのままに書ける (L2-01 の教訓)。
    tmp_npz = OUT / "_checkpoint_phase1.npz.tmp"
    with open(tmp_npz, "wb") as fh:
        np.savez_compressed(fh, **ckpt_npz_payload)
    os.replace(tmp_npz, OUT / "_checkpoint_phase1.npz")
    tmp_json = OUT / "_checkpoint_phase1.json.tmp"
    tmp_json.write_text(json.dumps(ckpt_json, indent=2, ensure_ascii=False))
    os.replace(tmp_json, OUT / "_checkpoint_phase1.json")
    print("saved _checkpoint_phase1.npz/.json (%.0fs total)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
