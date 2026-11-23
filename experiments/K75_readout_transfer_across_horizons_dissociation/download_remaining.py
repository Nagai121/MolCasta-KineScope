"""K75: pre-download the ATLAS trajectories still missing from the
K75 workspace, in parallel (6 workers). The sequential extraction was
stuck on ~40 KB/s ATLAS downloads (26 of the 108 proteins are not in
K15's workspace, and 2 have broken K15 copies). Each worker calls
tseries acquire() for one protein into the K75 workspace; different
proteins use different directories and locks, as K22 did.
Usage: python download_remaining.py proteins_remaining.txt
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tseries.data import acquire

OWN_ROOT = Path("/home/hatch/workspace/kinescope/work/k75_atlas")


def one(p):
    try:
        acquire(OWN_ROOT, p)
        return p, "OK"
    except Exception as e:
        return p, f"FAILED {type(e).__name__}: {str(e)[:150]}"


def main():
    proteins = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, p): p for p in proteins}
        for fut in as_completed(futs):
            p, status = fut.result()
            print(p, status, flush=True)


if __name__ == "__main__":
    main()
