"""L2-02 第3段階: 6予測幅分の断片を最終成果物にまとめる。

_checkpoint_phase1.npz/.json と _checkpoint_h<h>.json (6予測幅分) から
results.json / results.csv / per_protein.csv / test_predictions.npz を作る。
チェックポイント自体はコミットしない。
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
HORIZONS = [1, 5, 10, 30, 50, 90]
LENGTHS = [round(0.1 * k, 1) for k in range(11)]
FRAMES = {L: list(range(10 - int(round(L / 0.1)), 11)) for L in LENGTHS}
COND_ADJ = {L: f"adjacent_mean_{L:.1f}ns" for L in LENGTHS}
CONDITIONS = ["static_only", "full_history"] + [COND_ADJ[L] for L in LENGTHS]

premise = json.loads((OUT / "premise_check.json").read_text())
ckpt = np.load(OUT / "_checkpoint_phase1.npz")
ckpt_json = json.loads((OUT / "_checkpoint_phase1.json").read_text())
frags = {h: json.loads((OUT / f"_checkpoint_h{h}.json").read_text()) for h in HORIZONS}
assert premise["all_ok"], "premise not ok"

horizons_out = {str(h): frags[h] for h in HORIZONS}

csv_lines = ["horizon_ns,covered_ns,n_frames,frames,auprc_pooled,auprc_macro,recovery_pooled,recovery_macro,"
             "recovery_pooled_ci_lower,recovery_pooled_ci_upper,recovery_macro_ci_lower,recovery_macro_ci_upper,"
             "increment_pooled,increment_macro,auc,brier,logloss,L_recovery_pooled,L_recovery_macro"]
pp_lines = ["horizon_ns,protein,n,positives,auprc_static,auprc_full,full_gain,"
            + ",".join(f"auprc_{L:.1f}ns" for L in LENGTHS) + ","
            + ",".join(f"recovery_{L:.1f}ns" for L in LENGTHS)]
npz_payload = dict(proteins=ckpt["proteins"])

for h in HORIZONS:
    frag = frags[h]
    results = frag["metrics"]
    recovery = frag["recovery_vs_real_history_gain"]
    increments = frag["recovery_increment_per_0p1ns"]
    bootstrap = frag["bootstrap"]["results"]
    comp = frag["comparison_vs_L"]["recovery_by_length"]
    per_protein = frag["per_protein"]
    for L in LENGTHS:
        c = COND_ADJ[L]
        m = results[c]
        csv_lines.append(",".join([
            str(h), f"{L:.1f}", str(len(FRAMES[L])),
            '"' + " ".join(str(f) for f in FRAMES[L]) + '"',
            f"{m['auprc']:.6f}", f"{m['macro_auprc']:.6f}",
            f"{recovery[c]['pooled']:.6f}", f"{recovery[c]['macro']:.6f}",
            f"{bootstrap['pooled'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['pooled'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{bootstrap['macro'][f'{L:.1f}']['ci_lower']:.6f}",
            f"{bootstrap['macro'][f'{L:.1f}']['ci_upper']:.6f}",
            f"{increments['pooled'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{increments['macro'].get(f'{L:.1f}', float('nan')):.6f}",
            f"{m['auc']:.6f}", f"{m['brier']:.6f}", f"{m['logloss']:.6f}",
            f"{comp[f'{L:.1f}']['L_pooled']:.6f}",
            f"{comp[f'{L:.1f}']['L_macro']:.6f}"]))
    for p, entry in per_protein.items():
        row = [str(h), p, str(entry["n"]), str(entry["positives"]),
               "" if entry["auprc_static"] is None else f"{entry['auprc_static']:.6f}",
               "" if entry["auprc_full"] is None else f"{entry['auprc_full']:.6f}",
               "" if entry["full_gain"] is None else f"{entry['full_gain']:.6f}"]
        row += ["" if entry["auprc_adjacent"][f"{L:.1f}"] is None else f"{entry['auprc_adjacent'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        row += ["" if entry["recovery_adjacent"][f"{L:.1f}"] is None else f"{entry['recovery_adjacent'][f'{L:.1f}']:.6f}" for L in LENGTHS]
        pp_lines.append(",".join(row))
    npz_payload[f"h{h}_label"] = ckpt[f"h{h}_label"].astype(np.int8)
    for cname in CONDITIONS:
        npz_payload[f"h{h}_{cname}"] = ckpt[f"h{h}_{cname}"].astype(np.float32)

out = {
    "label": "HB dissociation, guidance recovery vs adjacent-window covered length, strong base (K75), horizons 1/5/10/30/50/90 ns",
    "series": "L2 (revision of L: window placed immediately before the origin frame 10, E01 Protocol A)",
    "length_definition": {
        "unit": "covered length (ns); frame count same as L01/L02, but the window is the last L/0.1+1 frames ending at the origin frame 10",
        "lengths": {f"{L:.1f}": dict(covered_ns=L, n_frames=len(FRAMES[L]), frames=FRAMES[L]) for L in LENGTHS},
        "note": "Length 0 = frame 10 only; 0.1 ns = frames 9-10; ...; 1.0 ns = frames 0-10 (identical to the L condition, used for the premise check). In L the window was frames 0..(L/0.1) paired with origin frame 10, leaving a time gap for short lengths.",
    },
    "base": {
        "name": "K75 strong base for HB dissociation: HistGBM correction model (160 iters, lr 0.065, 15 leaves, min_samples_leaf 20, L2 0.15, seed 20261121), trained on 8 calibration proteins, evaluated on 100 unseen proteins x 3 replicas (input bundle from K2-18)",
        "static_features": 5, "history_features": 10,
        "n_calibration": 6144, "n_test": int(len(ckpt["h1_label"])),
    },
    "premise_checks": premise["checks"],
    "premise_all_ok": True,
    "bootstrap_note": "Within each resample, static-only, full-history and adjacent-mean AUPRCs are recomputed and the gain and recovery are recomputed inside the sample. Models are not refit; per-candidate predicted probabilities are resampled. Pooled AUPRC uses a numpy implementation of sklearn's average_precision_score (threshold-grouped), verified against sklearn to <1e-9 on the full test set for every condition and horizon. Macro AUPRC within a resample is the multiplicity-weighted mean of per-protein AUPRCs over proteins whose AUPRC is defined.",
    "horizons": horizons_out,
}
(OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
(OUT / "results.csv").write_text("\n".join(csv_lines) + "\n")
(OUT / "per_protein.csv").write_text("\n".join(pp_lines) + "\n")
np.savez_compressed(OUT / "test_predictions.npz", **npz_payload)
print("saved results.json / results.csv / per_protein.csv / test_predictions.npz")
for h in HORIZONS:
    r = frags[h]["readings"]
    print(f"[{h} ns] 0.8 pooled {r['min_length_recovery_0_8']['pooled']}, "
          f"0.9 pooled {r['min_length_recovery_0_9']['pooled']}, "
          f"largest step {r['largest_step']['pooled']['from_ns']}->{r['largest_step']['pooled']['to_ns']} "
          f"{r['largest_step']['pooled']['increment']:.3f}")
