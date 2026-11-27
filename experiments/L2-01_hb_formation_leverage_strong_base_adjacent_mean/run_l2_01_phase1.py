"""L2-01 第1段階: 全予測幅の学習・予測と前提確認。予測をチェックポイントに保存する。

再起動で止まっても、予測を失わないための分割版。run_l2_01.py の第1段階と同じ計算をする。
終了時に以下を保存する (コミット対象外の作業ファイル):
- _checkpoint_phase1.npz: 予測幅ごとのラベルと各条件の予測確率 (float64)
- _checkpoint_phase1.json: 予測幅ごとの metrics とタンパク質ごとの AUPRC
- premise_check.json: 前提確認の結果 (48項目)
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/g05_series")
REPO_EXP = Path("/home/hatch/workspace/kinescope/repo/experiments")
OUT = Path(__file__).resolve().parent
SEED = 20261121
HORIZONS = [1, 5, 10, 30, 50, 90]
L_REF_DIR = {
    1: "L02_hb_formation_leverage_strong_base_1ns",
    5: "L04_hb_formation_leverage_strong_base_5ns",
    10: "L05_hb_formation_leverage_strong_base_10ns",
    30: "L06_hb_formation_leverage_strong_base_30ns",
    50: "L07_hb_formation_leverage_strong_base_50ns",
    90: "L08_hb_formation_leverage_strong_base_90ns",
}

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(10 - int(round(L / 0.1)), 11)) for L in LENGTHS}
COND_ADJ = {L: f"adjacent_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = ["static_only", "full_history"] + [COND_ADJ[L] for L in LENGTHS]


def load(proteins):
    static, hist, label, series = [], [], [], []
    for p in proteins:
        z = np.load(K06B / f"{p}.npz")
        s = np.load(SERIES / f"{p}.npz")
        for r in (1, 2, 3):
            static.append(z[f"r{r}_static"])
            hist.append(z[f"r{r}_history"])
            label.append(z[f"r{r}_label"])
            series.append(s[f"r{r}_series"])
    return dict(static=np.concatenate(static), hist=np.concatenate(hist),
                label=np.concatenate(label), series=np.concatenate(series, axis=1))


def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
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
    cal, test = load(CAL), load(TEST)
    print("cal", cal["label"].shape, "test", test["label"].shape, flush=True)

    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n
    assert cur == len(test["label"])

    feats = {}
    feats["static_only"] = (cal["static"], test["static"])
    feats["full_history"] = (np.column_stack([cal["static"], cal["hist"]]),
                             np.column_stack([test["static"], test["hist"]]))
    for L in LENGTHS:
        idx = FRAMES[L]
        feats[COND_ADJ[L]] = (
            np.column_stack([cal["static"], cal["series"][idx].mean(axis=0)[:, None].astype(np.float32)]),
            np.column_stack([test["static"], test["series"][idx].mean(axis=0)[:, None].astype(np.float32)]))

    all_checks = []
    ckpt_npz_payload = dict(
        proteins=np.concatenate([np.full(e - s, p) for p, s, e in spans]))
    ckpt_json = {}
    for hi, h in enumerate(HORIZONS):
        y_cal = cal["label"][:, hi]
        y_test = test["label"][:, hi]
        preds, results = {}, {}
        per_protein_ap = {c: {} for c in CONDITIONS}
        for cname in CONDITIONS:
            xtr, xte = feats[cname]
            pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
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
            m["macro_auprc"] = float(np.mean(aps))
            m["n_proteins_macro"] = len(aps)
            results[cname] = m
        print(f"[{h} ns] static pooled %.4f macro %.4f | full pooled %.4f macro %.4f (%.0fs)"
              % (results["static_only"]["auprc"], results["static_only"]["macro_auprc"],
                 results["full_history"]["auprc"], results["full_history"]["macro_auprc"],
                 time.time() - t0), flush=True)

        lref = json.loads((REPO_EXP / L_REF_DIR[h] / "results.json").read_text())
        lm = lref[f"metrics_{h}ns"]
        lg = lref[f"base_gain_{h}ns"]
        lrec = lref["recovery_vs_real_history_gain"]["prefix_mean_1.0ns"]
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
    # ファイルハンドルを渡すと指定したパスそのままに書ける。
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
