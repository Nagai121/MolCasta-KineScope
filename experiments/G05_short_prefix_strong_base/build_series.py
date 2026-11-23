"""G05 build: rebuild the raw 11-frame distance series for the
K06(b) candidates (strong base, HB formation), so the short-prefix
means can be computed.

Why this exists: k06b_out/*.npz saves only the 10 history
statistics of the 11-frame series (first, mean, std, min, max,
last-first, last-prev, frac<=3.5, frac<=4.5, frac(diff>0)). The
prefix means (frames 0..1 for 0.1 ns, frames 0..5 for 0.5 ns)
cannot be derived from those statistics. The raw series is rebuilt
from the trajectories, the same way score_b.py built it.

Method (mirrors score_b.py's one_replica, minus the model call):
- Candidate atom pairs (a, b) come from the saved r{r}_pair array
  (first two columns are the atom indices) -- no need to run
  predict_snapshot again.
- Positions/box for frames 0..10 come from the k15_atlas
  trajectories (atoms.parquet + A.zarr; import minitopol first for
  the codec). Only frames 0..10 are read (zarr partial read).
- Distances use the per-frame box, axis-wise (cell + _mic), the
  same as score_b.py.
- Verification: the 10 history statistics derived from the rebuilt
  series must match the saved r{r}_history to < 1e-5 (float32
  rounding; the single-trajectory check gave 1.4e-06). Any larger
  mismatch fails loudly -- the trajectories would not be the ones
  K06 used.
- Fallback: the k15_atlas copy is incomplete -- 26 of the 108
  proteins have no trajectories there at all, and 4cv4_A R2 /
  2hng_A R3 have an all-zero box in frames 0..10 there (a
  conversion artifact: the Registry copy of 4cv4_A has valid boxes
  for all replicas). For those, the series is rebuilt via the
  minitopol Registry (the path score_b.py used), which downloads
  the protein if needed. The history check verifies either source
  against K06's saved history.

Output: ~/workspace/kinescope/work/g05_series/<protein>.npz with
r1_series, r2_series, r3_series (11, N) float32, plus
build_summary.json (per-protein max diffs and row counts).
"""
import sys, json, time
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
import minitopol  # registers the minitopol_agx3 zarr codec
from pathlib import Path
import numpy as np, zarr
from tseries.data import cell, registry, trajectory, acquire
from tseries.features import _mic

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
ATLAS = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
REG_ROOT = Path("/home/hatch/workspace/kinescope/work/g05_atlas_workspace")
OUT = Path("/home/hatch/workspace/kinescope/work/g05_series")
OUT.mkdir(parents=True, exist_ok=True)

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
PROTS = orig["calibration_proteins"] + orig["test_proteins"]


def history10(series):
    return np.column_stack((series[0], series.mean(axis=0), series.std(axis=0), series.min(axis=0),
                            series.max(axis=0), series[-1] - series[0], series[-1] - series[-2],
                            (series <= 3.5).mean(axis=0), (series <= 4.5).mean(axis=0),
                            (np.diff(series, axis=0) > 0).mean(axis=0))).astype(np.float32)


def frames_from_k15_atlas(protein, replica):
    """Positions/box for frames 0..10 from the k15_atlas A.zarr copy."""
    tid = f"atlas-{protein}-analysis-R{replica}"
    zz = zarr.open_group(str(ATLAS / protein / "trajectories" / tid / "A.zarr"), mode="r")
    pos = zz["positions"][0:11].astype(np.float64)
    box = zz["box"][0:11].astype(np.float64)
    if any(np.all(box[k] == 0) for k in range(11)):
        raise ValueError(f"all-zero box in {tid} frames 0..10")
    return pos, box


def frames_from_registry(protein, replica):
    """Positions/box for frames 0..10 via the minitopol Registry
    (the path score_b.py used). Downloads the protein if needed."""
    acquire(REG_ROOT, protein)
    reg = registry(REG_ROOT, protein)
    tid = trajectory(protein, replica)
    blocks = list(reg.blocks(tid, start=0, stop=11, chunk_frames=11))
    pos = np.concatenate([b.positions for b in blocks], axis=0).astype(np.float64)
    box = np.concatenate([b.box for b in blocks], axis=0).astype(np.float64)
    assert pos.shape[0] == 11 and box.shape[0] == 11, (pos.shape, box.shape)
    return pos, box


def run_one(protein):
    z = np.load(K06B / f"{protein}.npz")
    out, maxdiff, n, sources = {}, 0.0, 0, []
    for r in (1, 2, 3):
        pair = z[f"r{r}_pair"]
        saved_hist = z[f"r{r}_history"]
        a, b = pair[:, :2].astype(np.intp).T
        try:
            pos, box = frames_from_k15_atlas(protein, r)
            src = "k15_atlas"
        except (FileNotFoundError, ValueError) as e:
            # k15_atlas copy is missing or has an all-zero box
            # (4cv4_A R2, 2hng_A R3, and 26 proteins with no
            # trajectories there at all). Fall back to the Registry.
            print(f"{protein} R{r}: k15_atlas unusable ({type(e).__name__}), "
                  f"falling back to Registry", flush=True)
            pos, box = frames_from_registry(protein, r)
            src = "registry"
        sources.append(src)
        series = np.stack([np.linalg.norm(_mic(pos[k, b] - pos[k, a], cell(box[k])), axis=1)
                           for k in range(11)], axis=0)
        d = float(np.abs(history10(series) - saved_hist).max())
        assert d < 1e-5, (protein, r, d)
        maxdiff = max(maxdiff, d)
        n += len(a)
        out[f"r{r}_series"] = series.astype(np.float32)
    np.savez_compressed(OUT / f"{protein}.npz", **out)
    return dict(protein=protein, n=int(n), max_hist_diff=maxdiff, sources=sources)


def main():
    results = []
    for p in PROTS:
        if (OUT / f"{p}.npz").is_file():
            print("SKIP", p, flush=True)
            continue
        t0 = time.time()
        r = run_one(p)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(json.dumps(r), flush=True)
    (OUT / "build_summary.json").write_text(json.dumps(results, indent=1))
    print("saved to", OUT)


if __name__ == "__main__":
    main()
