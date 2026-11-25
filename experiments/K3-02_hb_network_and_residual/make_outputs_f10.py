"""Build the K3-02 record outputs in the experiment directory:
results.json (frame-10 analysis results + premise audit), results.csv,
per_protein.csv, premise_check_f10.json."""
import csv, json
from pathlib import Path
import numpy as np

WORK = Path("/home/hatch/workspace/k3_02")
EXP = Path("/home/hatch/workspace/kinescope/repo/experiments/K3-02_hb_network_and_residual")
analysis = json.loads((WORK / "analysis_results_f10.json").read_text())

# premise audit from the saved counts files
worst0 = worst100 = 0.0
n_rep = 0
fails = []
for f in sorted((WORK / "counts_f10").glob("*.npz")):
    z = np.load(f)
    prem = json.loads(z["premises"].item())
    for r, d in prem.items():
        n_rep += 1
        worst0 = max(worst0, d["frame0_maxdiff"])
        worst100 = max(worst100, d["frame100_maxdiff"])
        if not (d["frame0_maxdiff"] < 0.1 and d["frame100_maxdiff"] < 0.1
                and d["elem_a_ok"] and d["elem_b_ok"]):
            fails.append({"protein": f.stem, "replica": r, **d})
premise = {
    "n_proteins": 108,
    "n_replicas_checked": n_rep,
    "worst_frame0_maxdiff_A": worst0,
    "worst_frame100_maxdiff_A": worst100,
    "failures": fails,
    "all_ok": len(fails) == 0,
    "note": "frame0/frame100 are total-format frames 0 and 100 (= history frames 0 and 10); "
            "pair distances recomputed from the total trajectory vs the stored k06b history "
            "column 0 distances; element checks compare pair atom atomic numbers.",
}
(EXP / "premise_check_f10.json").write_text(json.dumps(premise, indent=1))

results = dict(analysis)
results["premise_check"] = premise
(EXP / "results_f10.json").write_text(json.dumps(results, indent=1))

names = list(analysis["correlations"].keys())
with open(EXP / "results_f10.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["quantity", "pearson_gbm", "spearman_gbm", "pearson_linear",
                "spearman_linear", "added_value_resid_std_gbm", "mean_test", "std_test", "max_test"])
    for q in names:
        c = analysis["correlations"][q]
        s = analysis["quantity_stats_test"][q]
        w.writerow([q, c["gbm"]["pearson"], c["gbm"]["spearman"],
                    c["linear"]["pearson"], c["linear"]["spearman"],
                    analysis["added_value_gbm"][q]["resid_std_test"],
                    s["mean"], s["std"], s["max"]])
    w.writerow(["ALL", "", "", "", "",
                analysis["added_value_gbm"]["ALL"]["resid_std_test"], "", "", ""])
    w.writerow(["ALL_permuted", "", "", "", "",
                analysis["permutation_control_all"]["resid_std_test"], "", "", ""])
    w.writerow(["baseline_none", "", "", "", "",
                analysis["baseline"]["gbm"]["resid_std_test"], "", "", ""])

# per-protein Pearson correlations (replicas pooled within protein)
z = np.load(WORK / "analysis_arrays_f10.npz")
resid = z["residual_gbm"]
Q = z["quantities_test"]
groups = np.array([g.split("|")[0] for g in z["groups"]])
proteins = sorted(set(groups.tolist()))
with open(EXP / "per_protein_f10.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["protein", "n_candidates"] + [f"pearson_{q}" for q in names])
    for p in proteins:
        m = groups == p
        row = [p, int(m.sum())]
        for j in range(Q.shape[1]):
            x = Q[m, j]
            if x.std() == 0 or resid[m].std() == 0:
                row.append("")
            else:
                row.append(float(np.corrcoef(x, resid[m])[0, 1]))
        w.writerow(row)
print("wrote outputs to", EXP)
