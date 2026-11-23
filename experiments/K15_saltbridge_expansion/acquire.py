"""K15 acquisition: reuse K06(b)'s tseries.data.acquire (AtlasClient download
+ MiniTopol import_atlas) into a K15-dedicated workspace.
Usage: acquire.py <worker_index> <n_workers>  (proteins taken from proteins108.txt in order)
Already-acquired proteins (A.zarr present for all 3 replicas) are skipped by acquire().
"""
import sys, time
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
from pathlib import Path
from tseries.data import acquire

ROOT = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
PROTEINS = [l.strip() for l in open("/home/hatch/workspace/k06b/run/proteins108.txt") if l.strip()]
w, nw = int(sys.argv[1]), int(sys.argv[2])
mine = [p for i, p in enumerate(PROTEINS) if i % nw == w]
for p in mine:
    t0 = time.time()
    try:
        acquire(ROOT, p)
        print(f"[{w}] OK {p} {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[{w}] FAIL {p} {type(e).__name__}: {e}", flush=True)
print(f"[{w}] DONE", flush=True)
