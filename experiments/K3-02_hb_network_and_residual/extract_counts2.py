"""K3-02 v2 local extraction: range-fetch total data (TPR + XTC prefixes),
count frame-0 H-bonds, premise-gate, save counts npz. Checkpointed per
protein; fetched files are deleted after success.
Usage: extract_counts2.py PROTEIN [PROTEIN...]"""
import json, shutil, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/k3_02")
from k3lib import build_protein_topology, count_replica, FramesUnavailable, COLUMNS
from fetch_range import fetch_protein

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
OUT = Path("/home/hatch/workspace/k3_02/counts")
TMP = Path("/home/hatch/workspace/k3_02/fetched")

def one_protein(protein):
    out_path = OUT / f"{protein}.npz"
    if out_path.exists():
        print("SKIP", protein, flush=True); return
    import MDAnalysis as mda
    z = np.load(K06B / f"{protein}.npz")
    last_err = None
    for target_mb in (48, 160, None):
        dest, got = fetch_protein(protein, TMP, xtc_raw_target=(target_mb * 1024 * 1024) if target_mb else 1 << 62)
        try:
            topo = None; results, premises = {}, {}
            for r in (1, 2, 3):
                u = mda.Universe(str(dest / f"{protein}_prod_R{r}.tpr"),
                                 str(dest / f"{protein}_prod_R{r}.xtc"))
                if topo is None:
                    topo = build_protein_topology(u)
                counts, prem = count_replica(u, topo, z[f"r{r}_pair"],
                                             z[f"r{r}_history"][:, 0].astype(np.float64))
                if not (prem["frame0_maxdiff"] < 0.1 and prem["frame100_maxdiff"] < 0.1
                        and prem["elem_a_ok"] and prem["elem_b_ok"]):
                    raise RuntimeError(f"premise failed {protein} R{r}: {prem}")
                results[f"r{r}_counts"] = counts; premises[f"r{r}"] = prem
            np.savez_compressed(out_path, columns=np.array(COLUMNS),
                                premises=json.dumps(premises), **results)
            print("DONE", protein, "target_mb", target_mb,
                  {r: round(p["frame100_maxdiff"], 4) for r, p in premises.items()}, flush=True)
            shutil.rmtree(dest, ignore_errors=True)
            return
        except FramesUnavailable as e:
            last_err = e
            print("frames unavailable", protein, "target_mb", target_mb, str(e)[:100], flush=True)
            continue
    raise RuntimeError(f"all targets failed {protein}: {last_err}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        t0 = time.time()
        try:
            one_protein(p)
        except Exception as e:
            print("FAILED", p, type(e).__name__, str(e)[:250], flush=True)
        print("elapsed", p, round(time.time() - t0, 1), flush=True)
