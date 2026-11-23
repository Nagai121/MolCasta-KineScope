"""K29 preparation: future donor-acceptor distances for the K05/K08
candidates (weak base, data D).

The data cache holds only the 11-frame history (0-1 ns) and the binary
labels. The future distances (at the horizon frames 20/60/110/310/510/910,
i.e. 1/5/10/30/50/90 ns after the 1 ns origin) are read from the full
1001-frame trajectories in the GPT base (read-only), with the same MIC
distance function K13 used. Candidate atom pairs come from the cache.

Verification per file: frame 10 distance vs cache hist[10] and frame 0
vs hist[0] (must match), and (future distance <= 3.5) vs the cache label
(agreement rate, printed).
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # registers the minitopol_agx3 zarr codec
import numpy as np, zarr
from pathlib import Path

BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUTD = Path("/home/hatch/workspace/kinescope/work/k29_cache")
OUTD.mkdir(parents=True, exist_ok=True)
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
HORIZON_FRAMES = (20, 60, 110, 310, 510, 910)
FRAMES = list(range(11)) + list(HORIZON_FRAMES)

def distance(v, box):
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box); v = v - lengths * np.rint(v / lengths)
    else:
        inv = np.linalg.inv(box); fr = v @ inv; fr -= np.rint(fr); v = fr @ box
    return np.linalg.norm(v, axis=-1)

for p in PROTEINS:
    for r in (1, 2, 3):
        out = OUTD / f"{p}_R{r}_future.npz"
        if out.is_file():
            print("SKIP", p, r, flush=True); continue
        d = np.load(CACHE / f"{p}_R{r}.npz")
        da, aa, hist, y = d["donor_atom"], d["acceptor_atom"], d["hist"], d["y"]
        z = zarr.open_group(str(BASE / f"atlas-{p}-analysis-R{r}" / "A.zarr"), mode='r')
        pos = z['positions'][:].astype(np.float64)
        box = z['box'][:].astype(np.float64)
        assert pos.shape[0] == 1001, pos.shape
        series = np.stack([distance(pos[f, aa, :] - pos[f, da, :], box[f]) for f in FRAMES])
        assert series.shape == (17, len(da)), series.shape
        v10 = float(np.abs(series[10] - hist[10]).max())
        v0 = float(np.abs(series[0] - hist[0]).max())
        future = series[11:]  # (6, N) at frames 20/60/110/310/510/910
        agree = float(((future <= 3.5).astype(np.uint8) == y.T).mean())
        np.savez_compressed(out, future=future.astype(np.float32),
                            frame0=series[0].astype(np.float32), frame10=series[10].astype(np.float32))
        print(f"{p} R{r} max|frame0-hist0|={v0:.2e} max|frame10-hist10|={v10:.2e} label agreement={(future <= 3.5).astype(np.uint8) == y.T} {agree:.4f}", flush=True)
