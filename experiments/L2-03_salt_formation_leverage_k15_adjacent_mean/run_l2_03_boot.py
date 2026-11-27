"""L2-03 第2段階: 予測幅1つ分の回復率・読み・ブートストラップ。

使い方: python run_l2_03_boot.py <horizon_ns>
チェックポイント (_checkpoint_phase1.npz/.json) から予測を読み、
その予測幅の結果を _checkpoint_h<h>.json に保存する (一時書き込み＋置き換え)。
完了した予測幅は飛ばせるので、再起動で止まってもその予測幅だけやり直せばよい。

L2-02 との違い:
- 照合先は L27〜L32 (塩橋の形成、平均だけ、K15 の土台)。
- テストは 30タンパク質 (L2-02 は 100タンパク質)。区間はその分だけ粗い。
- ブートストラップで標本内の利得が 0 以下になった抽出は、L27 と同じく
  区間の計算から除く (件数を n_draws_with_nonpositive_gain に記録)。
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

REPO_EXP = Path("/home/hatch/workspace/kinescope/repo/experiments")
OUT = Path(__file__).resolve().parent
BOOT_SEED = 42
N_BOOT = 10000
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(10 - int(round(L / 0.1)), 11)) for L in LENGTHS}
COND_ADJ = {L: f"adjacent_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = ["static_only", "full_history"] + [COND_ADJ[L] for L in LENGTHS]
L_REF_DIR = {
    1: "L27_salt_formation_leverage_1ns",
    5: "L28_salt_formation_leverage_5ns",
    10: "L29_salt_formation_leverage_10ns",
    30: "L30_salt_formation_leverage_30ns",
    50: "L31_salt_formation_leverage_50ns",
    90: "L32_salt_formation_leverage_90ns",
}


def ap_numpy(y, score):
    """sklearn の average_precision_score と同じ段取り (同点はしきい値でまとめる)。"""
    order = np.argsort(-score, kind="stable")
    ys = y[order].astype(np.float64)
    ss = score[order]
    ends = np.flatnonzero(np.r_[ss[:-1] != ss[1:], True])
    tp = np.cumsum(ys)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall_step = np.diff(np.r_[0.0, tp]) / tp[-1]
    return float(np.sum(recall_step * precision))


def main():
    h = int(sys.argv[1])
    assert h in L_REF_DIR, h
    t0 = time.time()
    ckpt = np.load(OUT / "_checkpoint_phase1.npz")
    ckpt_json = json.loads((OUT / "_checkpoint_phase1.json").read_text())
    y_test = ckpt[f"h{h}_label"].astype(np.int64)
    preds = {c: ckpt[f"h{h}_{c}"].astype(np.float64) for c in CONDITIONS}
    results = ckpt_json[str(h)]["results"]
    per_protein_ap = ckpt_json[str(h)]["per_protein_ap"]
    lref = json.loads((REPO_EXP / L_REF_DIR[h] / "results.json").read_text())
    lrec_all = lref["recovery_vs_full_history_gain"]

    prots = ckpt["proteins"].astype(str)
    spans, cur = [], 0
    for p in dict.fromkeys(prots.tolist()):
        n = int((prots == p).sum())
        spans.append((p, cur, cur + n)); cur += n
    assert cur == len(y_test) and len(spans) == 30
    prot_of_row = np.concatenate([np.full(e - s, i) for i, (_, s, e) in enumerate(spans)])

    gain_pooled = results["full_history"]["auprc"] - results["static_only"]["auprc"]
    gain_macro = results["full_history"]["macro_auprc"] - results["static_only"]["macro_auprc"]

    def rec(cname, kind):
        if kind == "pooled":
            return (results[cname]["auprc"] - results["static_only"]["auprc"]) / gain_pooled
        return (results[cname]["macro_auprc"] - results["static_only"]["macro_auprc"]) / gain_macro

    recovery = {c: dict(pooled=float(rec(c, "pooled")), macro=float(rec(c, "macro")))
                for c in CONDITIONS if c != "static_only"}
    increments = {}
    for kind in ("pooled", "macro"):
        increments[kind] = {}
        prev = None
        for L in LENGTHS:
            r = recovery[COND_ADJ[L]][kind]
            if prev is not None:
                increments[kind][f"{L:.1f}"] = float(r - prev)
            prev = r

    def min_length(kind, thr):
        for L in LENGTHS:
            if recovery[COND_ADJ[L]][kind] >= thr:
                return L
        return None

    def largest_step(kind):
        inc = increments[kind]
        key = max(inc, key=lambda k: inc[k])
        Lto = float(key)
        return dict(from_ns=Lto - 0.1, to_ns=Lto, increment=inc[key])

    per_protein = {}
    for p, s, e in spans:
        yy = y_test[s:e]
        entry = dict(n=int(e - s), positives=int(yy.sum()))
        ap_s = per_protein_ap["static_only"][p]
        ap_f = per_protein_ap["full_history"][p]
        entry["auprc_static"] = ap_s
        entry["auprc_full"] = ap_f
        entry["full_gain"] = (ap_f - ap_s) if (ap_s is not None and ap_f is not None) else None
        entry["auprc_adjacent"] = {}
        entry["recovery_adjacent"] = {}
        for L in LENGTHS:
            ap = per_protein_ap[COND_ADJ[L]][p]
            entry["auprc_adjacent"][f"{L:.1f}"] = ap
            if ap is not None and entry["full_gain"] is not None and entry["full_gain"] > 0:
                entry["recovery_adjacent"][f"{L:.1f}"] = float((ap - ap_s) / entry["full_gain"])
            else:
                entry["recovery_adjacent"][f"{L:.1f}"] = None
        per_protein[p] = entry

    # ブートストラップ (L27 と同じ段取り)
    for cname in CONDITIONS:
        ap_np = ap_numpy(y_test, preds[cname].astype(np.float64))
        assert abs(ap_np - results[cname]["auprc"]) < 1e-9, (h, cname, ap_np, results[cname]["auprc"])
    boot_prep = {}
    for cname in CONDITIONS:
        score = preds[cname].astype(np.float64)
        order = np.argsort(-score, kind="stable")
        ss = score[order]
        starts = np.flatnonzero(np.r_[True, ss[:-1] != ss[1:]])
        boot_prep[cname] = dict(
            y_sorted=y_test[order].astype(np.float64),
            prot_sorted=prot_of_row[order],
            starts=starts)

    def pooled_ap_boot(prep, mult):
        w = mult[prep["prot_sorted"]]
        wp = w * prep["y_sorted"]
        tp_g = np.add.reduceat(wp, prep["starts"])
        tot_g = np.add.reduceat(w, prep["starts"])
        if tp_g.sum() <= 0:
            return None
        cum_tp = np.cumsum(tp_g)
        cum_tot = np.cumsum(tot_g)
        precision = np.divide(cum_tp, cum_tot, out=np.zeros_like(cum_tp), where=cum_tot > 0)
        return float(np.sum((tp_g / cum_tp[-1]) * precision))

    defined = [p for p, _, _ in spans if per_protein_ap["static_only"][p] is not None]
    ap_mat = np.array([[per_protein_ap[c][p] for c in CONDITIONS] for p in defined])
    cond_idx = {c: j for j, c in enumerate(CONDITIONS)}
    draw_to_defined = np.full(len(spans), -1, dtype=int)
    pos = 0
    for i, (p, _, _) in enumerate(spans):
        if p in defined:
            draw_to_defined[i] = pos
            pos += 1

    rng = np.random.default_rng(BOOT_SEED)
    nP = len(spans)
    boot_rec = {kind: {f"{L:.1f}": [] for L in LENGTHS} for kind in ("pooled", "macro")}
    n_bad_gain = dict(pooled=0, macro=0)
    for b in range(N_BOOT):
        draw = rng.integers(0, nP, nP)
        mult = np.bincount(draw, minlength=nP).astype(np.float64)
        ap_s = pooled_ap_boot(boot_prep["static_only"], mult)
        ap_f = pooled_ap_boot(boot_prep["full_history"], mult)
        if ap_s is None or ap_f is None or ap_f - ap_s <= 0:
            n_bad_gain["pooled"] += 1
        else:
            g = ap_f - ap_s
            for L in LENGTHS:
                ap = pooled_ap_boot(boot_prep[COND_ADJ[L]], mult)
                boot_rec["pooled"][f"{L:.1f}"].append((ap - ap_s) / g)
        dpos = draw_to_defined[draw]
        wsel = dpos[dpos >= 0]
        if len(wsel) == 0:
            n_bad_gain["macro"] += 1
        else:
            m = ap_mat[wsel].mean(axis=0)
            gm = m[cond_idx["full_history"]] - m[cond_idx["static_only"]]
            if gm <= 0:
                n_bad_gain["macro"] += 1
            else:
                for L in LENGTHS:
                    boot_rec["macro"][f"{L:.1f}"].append(
                        (m[cond_idx[COND_ADJ[L]]] - m[cond_idx["static_only"]]) / gm)
        if (b + 1) % 2000 == 0:
            print(f"[{h} ns] bootstrap %d/%d (%.0fs)" % (b + 1, N_BOOT, time.time() - t0), flush=True)

    bootstrap = {}
    for kind in ("pooled", "macro"):
        bootstrap[kind] = {}
        for L in LENGTHS:
            arr = np.array(boot_rec[kind][f"{L:.1f}"])
            if len(arr) == 0 or np.isnan(arr).any():
                raise RuntimeError(f"bootstrap {h} {kind} {L}: invalid values (n={len(arr)}, nan={int(np.isnan(arr).sum())})")
            bootstrap[kind][f"{L:.1f}"] = dict(
                n_boot=int(len(arr)), seed=BOOT_SEED,
                point_estimate=recovery[COND_ADJ[L]][kind],
                ci_lower=float(np.percentile(arr, 2.5)),
                ci_upper=float(np.percentile(arr, 97.5)),
                boot_mean=float(arr.mean()), boot_std=float(arr.std()))
    print(f"[{h} ns] bootstrap done (%.0fs), bad gains: %s" % (time.time() - t0, n_bad_gain), flush=True)

    step = largest_step("pooled")
    Lfrom, Lto = step["from_ns"], step["to_ns"]
    step_inc_protein = []
    for p, entry in per_protein.items():
        r_from = entry["recovery_adjacent"].get(f"{Lfrom:.1f}")
        r_to = entry["recovery_adjacent"].get(f"{Lto:.1f}")
        if r_from is not None and r_to is not None:
            step_inc_protein.append(r_to - r_from)
    step_inc_protein = np.array(step_inc_protein)
    step_per_protein = dict(
        from_ns=Lfrom, to_ns=Lto, n_proteins=int(len(step_inc_protein)),
        median=float(np.median(step_inc_protein)),
        q25=float(np.percentile(step_inc_protein, 25)),
        q75=float(np.percentile(step_inc_protein, 75)),
        min=float(step_inc_protein.min()), max=float(step_inc_protein.max()),
        frac_positive=float(np.mean(step_inc_protein > 0)),
        frac_ge_0p1=float(np.mean(step_inc_protein >= 0.1)))

    readings = dict(
        min_length_recovery_0_8=dict(pooled=min_length("pooled", 0.8), macro=min_length("macro", 0.8)),
        min_length_recovery_0_9=dict(pooled=min_length("pooled", 0.9), macro=min_length("macro", 0.9)),
        largest_step=dict(pooled=step, macro=largest_step("macro")),
        step_per_protein=step_per_protein)

    comparison_vs_L = {}
    for L in LENGTHS:
        lr = lrec_all[f"prefix_mean_{L:.1f}ns"]
        comparison_vs_L[f"{L:.1f}"] = dict(
            L_pooled=float(lr["pooled"]), L_macro=float(lr["macro"]),
            diff_pooled=float(recovery[COND_ADJ[L]]["pooled"] - lr["pooled"]),
            diff_macro=float(recovery[COND_ADJ[L]]["macro"] - lr["macro"]))
    l_readings = lref["readings"]
    comparison_readings = dict(
        L_min_length_0_8=l_readings["min_length_recovery_0_8"],
        L_min_length_0_9=l_readings["min_length_recovery_0_9"],
        L_largest_step=l_readings["largest_step"])

    frag = dict(
        horizon_ns=h,
        reference_experiment=L_REF_DIR[h],
        metrics=results,
        base_gain=dict(pooled=float(gain_pooled), macro=float(gain_macro)),
        recovery_vs_real_history_gain=recovery,
        recovery_increment_per_0p1ns=increments,
        readings=readings,
        bootstrap=dict(n_boot=N_BOOT, seed=BOOT_SEED, method="percentile",
                       resampling_unit="protein (30 test proteins, resampled with replacement)",
                       n_draws_with_nonpositive_gain=n_bad_gain,
                       results=bootstrap),
        comparison_vs_L=dict(recovery_by_length=comparison_vs_L, readings=comparison_readings),
        per_protein=per_protein)

    tmp = OUT / f"_checkpoint_h{h}.json.tmp"
    tmp.write_text(json.dumps(frag, indent=2, ensure_ascii=False))
    os.replace(tmp, OUT / f"_checkpoint_h{h}.json")
    print(f"[{h} ns] done: 0.8 pooled {readings['min_length_recovery_0_8']['pooled']}, "
          f"0.9 pooled {readings['min_length_recovery_0_9']['pooled']}, "
          f"largest step {step['from_ns']}->{step['to_ns']} {step['increment']:.3f} (%.0fs)"
          % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
