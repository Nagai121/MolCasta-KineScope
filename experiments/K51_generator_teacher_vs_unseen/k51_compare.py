"""K51 compare: generated-mean error on the generator's 8 teacher
proteins vs unseen proteins (DocD 6, from K05 + K08 cache).
gen mean = mean of [origin, +0.1/+0.3/+0.5/+1.0 ns generated] (5 points);
real mean = mean of the 11-point backward history (K08/K23 definition)."""
import json
from pathlib import Path
import numpy as np

K51OUT = Path("/home/hatch/workspace/k51/out")
TEACHER = ("1u84_A", "4u5h_G", "5hub_A", "1x6j_A", "2y1y_A", "1krn_A", "2y4x_B", "4xqm_A")
DOCD = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")

def err_stats(err):
    return dict(n=int(len(err)), mean=float(err.mean()), std=float(err.std()),
                mae=float(np.abs(err).mean()))

def main():
    per_protein = {}
    # teacher: K51 outputs
    for p in TEACHER:
        errs, gms, rms = [], [], []
        for r in (1, 2, 3):
            f = K51OUT / f"atlas-{p}-analysis-R{r}.npz"
            assert f.is_file(), f"missing {f}"
            with np.load(f) as z:
                gen5 = np.column_stack([z["pair"][:, 2].astype(np.float64),
                                        z["gen_dist"].astype(np.float64)])
                gm = gen5.mean(axis=1)
                rm = z["real_hist"].mean(axis=0).astype(np.float64)
            errs.append(gm - rm); gms.append(gm); rms.append(rm)
        err = np.concatenate(errs)
        per_protein[p] = dict(group="teacher", **err_stats(err),
                              corr=float(np.corrcoef(np.concatenate(gms), np.concatenate(rms))[0, 1]))
        print(p, "teacher", {k: round(v, 4) for k, v in per_protein[p].items() if k != "group"}, flush=True)
    # unseen DocD 6: K05 generated series + K08 cache
    z = np.load(GEN)
    gen5 = np.column_stack([z["pair"][:, 2].astype(np.float64), z["gen"].astype(np.float64)])
    gen_means = {(str(t), int(z["pair"][i, 0]), int(z["pair"][i, 1])): float(gen5[i].mean())
                 for i, t in enumerate(z["tid"])}
    for p in DOCD:
        errs, gms, rms = [], [], []
        for r in (1, 2, 3):
            with np.load(CACHE / f"{p}_R{r}.npz") as c:
                tid = f"atlas-{p}-analysis-R{r}"
                gm = np.array([gen_means[(tid, int(da), int(aa))]
                               for da, aa in zip(c["donor_atom"], c["acceptor_atom"])])
                rm = c["hist"].mean(axis=0).astype(np.float64)
            errs.append(gm - rm); gms.append(gm); rms.append(rm)
        err = np.concatenate(errs)
        per_protein[p] = dict(group="unseen_docd", **err_stats(err),
                              corr=float(np.corrcoef(np.concatenate(gms), np.concatenate(rms))[0, 1]))
        print(p, "unseen", {k: round(v, 4) for k, v in per_protein[p].items() if k != "group"}, flush=True)
    res = {"per_protein": per_protein}
    for g in ("teacher", "unseen_docd"):
        errs = []
        for p, d in per_protein.items():
            if d["group"] != g: continue
            # recompute pooled from files is overkill; use per-protein stats via n-weighted combination
        # pooled: reload errors
        pooled = []
        if g == "teacher":
            for p in TEACHER:
                for r in (1, 2, 3):
                    with np.load(K51OUT / f"atlas-{p}-analysis-R{r}.npz") as zz:
                        gen5 = np.column_stack([zz["pair"][:, 2].astype(np.float64),
                                                zz["gen_dist"].astype(np.float64)])
                        pooled.append(gen5.mean(axis=1) - zz["real_hist"].mean(axis=0).astype(np.float64))
        else:
            for p in DOCD:
                for r in (1, 2, 3):
                    with np.load(CACHE / f"{p}_R{r}.npz") as c:
                        tid = f"atlas-{p}-analysis-R{r}"
                        gm = np.array([gen_means[(tid, int(da), int(aa))]
                                       for da, aa in zip(c["donor_atom"], c["acceptor_atom"])])
                        pooled.append(gm - c["hist"].mean(axis=0).astype(np.float64))
        res[g] = err_stats(np.concatenate(pooled))
        print(g, res[g], flush=True)
    res["ratio_teacher_unseen_std"] = res["teacher"]["std"] / res["unseen_docd"]["std"]
    res["ratio_teacher_unseen_absmean"] = abs(res["teacher"]["mean"]) / abs(res["unseen_docd"]["mean"])
    print("std ratio teacher/unseen:", round(res["ratio_teacher_unseen_std"], 3))
    Path("/home/hatch/workspace/k51/k51_results.json").write_text(json.dumps(res, indent=1))
    print("WROTE k51_results.json")

if __name__ == "__main__":
    main()
