"""K64 extraction: 0.1 ns-ahead labels (frame 11) for the Doc D base.

The K01 cache (~/workspace/kinescope/work/data_cache/) stores, per
protein x replica, the candidate donor/acceptor atom indices, the
static features X, the 11-frame history series (frames 0..10), and
labels y for horizons 1/5/10/30/50/90 ns (frames 20/60/110/310/510/910).
It does not store frame 11, so the 0.1 ns label (frame 11 <= 3.5 A)
is computed here from the converted trajectories, with the same
per-frame-box periodic-boundary distance as K01's extract.py.

Sanity checks per replica (printed, and asserted):
  - distance recomputed at frame 10 matches the cached hist[-1]
    (max abs diff; should be ~0, same frame read twice).
  - label recomputed at frame 20 matches the cached y[:, 0]
    (mismatch count; should be 0).
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # registers the minitopol_agx3 zarr codec
import numpy as np
import zarr
from pathlib import Path

BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path("/home/hatch/workspace/kinescope/work/k64_cache")
OUT.mkdir(parents=True, exist_ok=True)
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
BOUND = 3.5


def distance(v, box):
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        v = v - lengths * np.rint(v / lengths)
    else:
        inv = np.linalg.inv(box)
        fr = v @ inv
        fr -= np.rint(fr)
        v = fr @ box
    return np.linalg.norm(v, axis=-1)


def build(prot, rep):
    tid = f"atlas-{prot}-analysis-R{rep}"
    z = np.load(CACHE / f"{prot}_R{rep}.npz")
    a, b = z["donor_atom"], z["acceptor_atom"]
    hist, y = z["hist"], z["y"]
    zg = zarr.open_group(str(BASE / tid / "A.zarr"), mode="r")
    pos = zg["positions"][[10, 11, 20]].astype(np.float64)
    box = zg["box"][[10, 11, 20]].astype(np.float64)
    d10 = distance(pos[0, a, :] - pos[0, b, :], box[0])
    d11 = distance(pos[1, a, :] - pos[1, b, :], box[1])
    d20 = distance(pos[2, a, :] - pos[2, b, :], box[2])
    maxdiff10 = float(np.max(np.abs(d10 - hist[-1]))) if len(a) else 0.0
    mism20 = int(((d20 <= BOUND).astype(np.uint8) != y[:, 0]).sum())
    assert maxdiff10 < 1e-6, (prot, rep, maxdiff10)
    assert mism20 == 0, (prot, rep, mism20)
    y01 = (d11 <= BOUND).astype(np.uint8)
    np.savez_compressed(OUT / f"{prot}_R{rep}_y01.npz", y01=y01, d11=d11)
    print(tid, "cand", len(a), "pos@0.1ns", int(y01.sum()),
          "frame10 maxdiff", f"{maxdiff10:.2e}", "frame20 mismatches", mism20, flush=True)


for p in PROTEINS:
    for r in (1, 2, 3):
        if (OUT / f"{p}_R{r}_y01.npz").exists():
            print(p, r, "already done, skipped", flush=True)
            continue
        build(p, r)
print("done")
