"""K22 preparation: re-extract the 11-frame real distance series for the
K06(b) candidates. K06(b) saved only the 10 history statistics, but K22
shifts each candidate's whole series, which changes frac<=3.5 / frac<=4.5,
so the series itself is needed.

Source of trajectories, in order:
 1. K15's workspace (~/workspace/kinescope/work/k15_atlas), which holds
    the same 108 ATLAS proteins for the salt-bridge expansion. Read-only:
    nothing is acquired or deleted there.
 2. For proteins missing there, acquire into a K22-dedicated workspace
    (~/workspace/kinescope/work/k22_atlas) with the same tseries acquire
    K06(b) used, and delete the converted trajectory after extraction.

Follows score_b.py of K06(b): frames 0..10, donor-acceptor distance per
frame with the per-frame box (tseries cell + _mic). The recomputed 10
history statistics are verified against the saved K06(b) values before
the series is kept. Checkpointed: skips proteins whose series file exists.
"""
import sys, shutil, time
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
from pathlib import Path
import numpy as np
from tseries.data import registry, trajectory, cell, acquire
from tseries.features import _mic

K15_ROOT = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
import os
OWN_ROOT = Path(os.environ.get("K22_OWN_ROOT", "/home/hatch/workspace/kinescope/work/k22_atlas"))
SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
OUTD = Path("/home/hatch/workspace/kinescope/work/k22_cache")
OUTD.mkdir(parents=True, exist_ok=True)

def history10(series):
    """series: (11, n) -> (n, 10), same formula as K06(b) score_b.py"""
    return np.column_stack((series[0], series.mean(axis=0), series.std(axis=0), series.min(axis=0),
                            series.max(axis=0), series[-1] - series[0], series[-1] - series[-2],
                            (series <= 3.5).mean(axis=0), (series <= 4.5).mean(axis=0),
                            (np.diff(series, axis=0) > 0).mean(axis=0))).astype(np.float32)

def has_all(root, protein):
    reg = registry(root, protein)
    return all((Path(reg.workspace.trajectory_path(trajectory(protein, r))) / "A.zarr").is_dir()
               for r in (1, 2, 3))

def one_replica(root, protein, replica, saved):
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    blocks = list(reg.blocks(tid, start=0, stop=11, chunk_frames=128))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    boxes = np.concatenate([b.box for b in blocks], axis=0)
    assert pos.shape[0] == 11, pos.shape
    pair = saved[f"r{replica}_pair"]
    a, b = pair[:, 0].astype(np.intp), pair[:, 1].astype(np.intp)
    series = np.stack([np.linalg.norm(_mic(pos[k, b] - pos[k, a], cell(boxes[k])), axis=1)
                       for k in range(11)], axis=0)  # (11, n)
    return series.T.astype(np.float32)  # (n, 11)

def main():
    proteins = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    for p in proteins:
        out = OUTD / f"{p}.npz"
        if out.is_file():
            print("SKIP", p, flush=True); continue
        t0 = time.time()
        own = False
        try:
            with np.load(SAVED / f"{p}.npz") as z:
                saved = {k: z[k] for k in z.files}
            payload = None
            roots = []
            if has_all(K15_ROOT, p):
                roots.append(K15_ROOT)
            roots.append(OWN_ROOT)
            last_err = None
            for root in roots:
                try:
                    if root is OWN_ROOT:
                        acquire(OWN_ROOT, p); own = True
                    payload = {}
                    worst = 0.0
                    for r in (1, 2, 3):
                        s = one_replica(root, p, r, saved)
                        h = history10(s.T)
                        diff = float(np.abs(h - saved[f"r{r}_history"]).max())
                        worst = max(worst, diff)
                        payload[f"r{r}_series"] = s
                    assert worst < 2e-3, (p, worst)
                    break
                except Exception as e:
                    last_err = e
                    payload = None
                    print("SOURCE FAILED", p, root.name, type(e).__name__, str(e)[:120], flush=True)
            if payload is None:
                raise last_err
            np.savez_compressed(out, **payload)
            print("EXTRACTED", p, "from", root.name, "max history diff", round(worst, 6),
                  "sec", round(time.time() - t0, 1), flush=True)
        except Exception as e:
            print("FAILED", p, type(e).__name__, str(e)[:200], flush=True)
        finally:
            if own:
                shutil.rmtree(OWN_ROOT / p, ignore_errors=True)

if __name__ == "__main__":
    main()
