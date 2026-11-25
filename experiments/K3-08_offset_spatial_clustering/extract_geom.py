"""K3-08 extraction: per protein x replica, at the origin frame (frame 10)
of the converted trajectory (MiniTopol A.zarr):
  - midpoint of each K06(b) candidate pair (minimum-image midpoint, Angstrom)
  - residue ordinal of the donor atom and of the acceptor atom
    (ordinal = order of first appearance of (segment, residue_index)
     in atoms.parquet)
  - verification: frame-10 pair distance vs the stored static feature
    (k06b static column 1); the protein is saved only if maxdiff < 0.1 A.
Workspaces already on disk are reused; missing proteins are acquired
through the MiniTopol ATLAS client into tmpws/ and deleted after
extraction (same pattern as K3-03/K3-04's k3_extract.py).
Checkpointed: one output npz per protein under geom/; existing outputs
are skipped.
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # noqa: F401  (registers the zarr codec)
import numpy as np
import pandas as pd
import zarr

WORK = Path("/home/hatch/workspace/k3_08")
GEOM = WORK / "geom"
TMPWS = WORK / "tmpws"
K06B_OUT = Path("/home/hatch/workspace/k06b/run/k06b_out")
EXISTING_WS = [
    Path("/home/hatch/workspace/kinescope/work/g05_atlas_workspace"),
    Path("/home/hatch/workspace/kinescope/work/k63_atlas"),
    Path("/home/hatch/workspace/kinescope/work/k75_atlas"),
    Path("/home/hatch/workspace/kinescope/gam/atlas-workspace"),
    Path("/home/hatch/workspace/k06b/run/atlas-workspace"),
]
DOCD_TRAJ = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")


def mic(v, box):
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        return v - lengths * np.rint(v / lengths)
    inv = np.linalg.inv(box)
    fr = v @ inv
    fr -= np.rint(fr)
    return fr @ box


def traj_dir(protein, rep):
    name = f"atlas-{protein}-analysis-R{rep}"
    for ws in EXISTING_WS:
        d = ws / protein / "trajectories" / name
        if (d / "A.zarr").is_dir():
            return d
    d = DOCD_TRAJ / name
    if (d / "A.zarr").is_dir():
        return d
    d = TMPWS / protein / "trajectories" / name
    if (d / "A.zarr").is_dir():
        return d
    return None


def acquire(protein):
    from minitopol import Registry, Workspace
    from minitopol.atlas import AtlasClient, import_atlas
    reg = Registry(Workspace(TMPWS / protein))
    if all(traj_dir(protein, r) is not None for r in (1, 2, 3)):
        return
    client = AtlasClient(timeout=180)
    client.download(reg.workspace, protein, "analysis")
    result = import_atlas(reg.workspace, protein, "analysis")
    if len(result["replicas"]) != 3 or any(r["A"] != "complete" for r in result["replicas"]):
        raise RuntimeError(result)


def process(protein):
    out_path = GEOM / f"{protein}.npz"
    if out_path.exists():
        return "skip"
    acquired = False
    if any(traj_dir(protein, r) is None for r in (1, 2, 3)):
        acquire(protein)
        acquired = True
    k = np.load(K06B_OUT / f"{protein}.npz")
    store = {}
    for rep in (1, 2, 3):
        d = traj_dir(protein, rep)
        df = pd.read_parquet(d / "atoms.parquet")
        z = zarr.open_group(str(d / "A.zarr"), mode="r")
        pos = z["positions"][10].astype(np.float64)
        box = z["box"][10].astype(np.float64)
        pair = k[f"r{rep}_pair"]
        static = k[f"r{rep}_static"]
        da = pair[:, 0].astype(int)
        ac = pair[:, 1].astype(int)
        disp = mic(pos[ac] - pos[da], box)
        d10 = np.linalg.norm(disp, axis=1)
        maxdiff = float(np.abs(d10 - static[:, 1]).max())
        if not maxdiff < 0.1:
            raise RuntimeError(f"{protein} R{rep}: verify failed maxdiff={maxdiff}")
        store[f"r{rep}_verify_d10_maxdiff"] = np.array([maxdiff])
        mid = pos[da] + 0.5 * disp
        store[f"r{rep}_mid"] = mid.astype(np.float32)
        store[f"r{rep}_box"] = box
        seg = df["segment"].values
        ridx = df["residue_index"].values
        ordinal = {}
        atom_res = np.empty(len(df), dtype=np.int32)
        for i in range(len(df)):
            key = (str(seg[i]), int(ridx[i]))
            if key not in ordinal:
                ordinal[key] = len(ordinal)
            atom_res[i] = ordinal[key]
        store[f"r{rep}_dres"] = atom_res[da]
        store[f"r{rep}_ares"] = atom_res[ac]
        store[f"r{rep}_n_residues"] = np.array([len(ordinal)])
    np.savez_compressed(out_path, **store)
    if acquired:
        shutil.rmtree(TMPWS / protein, ignore_errors=True)
    return "done"


def main():
    GEOM.mkdir(exist_ok=True)
    proteins = sorted(p.stem for p in K06B_OUT.glob("*.npz"))
    only = sys.argv[1:] or None
    if only:
        proteins = [p for p in proteins if p in only]
    print(f"{len(proteins)} proteins", flush=True)
    for p in proteins:
        t0 = time.time()
        try:
            status = process(p)
            print(f"{p}: {status} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"{p}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
