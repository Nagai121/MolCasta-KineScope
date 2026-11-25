"""K3-02 frame-10 re-measurement: range-fetch total data (TPR + XTC
prefixes), count H-bonds at history frame 10 (total frame 100, the
origin frame the static prediction takes its input from), premise-gate,
save counts npz to counts_f10/. As a validation, frame-0 counts are
recomputed in the same pass and must match the saved counts/*.npz
exactly (same code, same inputs); a mismatch aborts before saving.
Checkpointed per protein; fetched files are deleted after success.
Usage: extract_counts_f10.py PROTEIN [PROTEIN...]"""
import json, shutil, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/k3_02")
from k3lib import build_protein_topology, count_replica, FramesUnavailable, COLUMNS_F10
from fetch_range import fetch_protein

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
OUT = Path("/home/hatch/workspace/k3_02/counts_f10")
OLD = Path("/home/hatch/workspace/k3_02/counts")
TMP = Path("/home/hatch/workspace/k3_02/fetched")

def one_protein(protein):
    out_path = OUT / f"{protein}.npz"
    if out_path.exists():
        print("SKIP", protein, flush=True); return
    old_path = OLD / f"{protein}.npz"
    if not old_path.exists():
        raise RuntimeError(f"no frame-0 counts to validate against: {protein}")
    old = np.load(old_path)
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
                counts0, prem = count_replica(u, topo, z[f"r{r}_pair"],
                                              z[f"r{r}_history"][:, 0].astype(np.float64),
                                              count_frame=0)
                if not (prem["frame0_maxdiff"] < 0.1 and prem["frame100_maxdiff"] < 0.1
                        and prem["elem_a_ok"] and prem["elem_b_ok"]):
                    raise RuntimeError(f"premise failed {protein} R{r}: {prem}")
                old_counts = old[f"r{r}_counts"]
                if not np.array_equal(counts0, old_counts):
                    md = float(np.max(np.abs(counts0 - old_counts)))
                    raise RuntimeError(f"frame-0 recount mismatch {protein} R{r}: maxdiff {md}")
                counts10, _ = count_replica(u, topo, z[f"r{r}_pair"],
                                            z[f"r{r}_history"][:, 0].astype(np.float64),
                                            count_frame=100)
                results[f"r{r}_counts"] = counts10; premises[f"r{r}"] = prem
            np.savez_compressed(out_path, columns=np.array(COLUMNS_F10),
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
