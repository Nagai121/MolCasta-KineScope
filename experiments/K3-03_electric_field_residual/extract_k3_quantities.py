"""K3-03 / K3-04 extraction.

For every protein in the K06(b) output set (8 calibration + 100 test), for each
replica, at the origin frame (frame 10 of the 0-1 ns history):
  - electric field at the pair midpoint and its component along the
    donor->acceptor direction, from CHARMM36 (Jul2022 GROMACS port) partial
    charges of the heavy atoms, donor/acceptor atoms themselves excluded.
    Primary: HIS = HSE. Sensitivity: HIS = HSD.
  - backbone phi/psi of the donor and acceptor residues.
Also verifies the layout: frame-10 donor-acceptor distance vs the stored
static feature, and the 11-frame mean distance vs the stored history mean.

Checkpointed: one output npz per protein under work/k3_data/; existing outputs
are skipped. Workspaces already on disk are reused; missing proteins are
acquired through the MiniTopol ATLAS client into work/k3_atlas_tmp/ and
deleted right after extraction.
"""
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # noqa: F401  (registers the zarr codec)
import numpy as np
import pandas as pd
import zarr

WORK = Path("/home/hatch/workspace/kinescope/work")
DATA = WORK / "k3_data"
import os
TMPWS = Path(os.environ.get("K3_TMPWS", WORK / "k3_atlas_tmp"))
K06B_OUT = Path("/home/hatch/workspace/k06b/run/k06b_out")
COULOMB = 14.3996454784255  # V Angstrom / e

EXISTING_WS = [
    WORK / "g05_atlas_workspace",
    WORK / "k63_atlas",
    WORK / "k75_atlas",
    Path("/home/hatch/workspace/kinescope/gam/atlas-workspace"),
    Path("/home/hatch/workspace/k06b/run/atlas-workspace"),
]
DOCD_TRAJ = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")


# ---------------------------------------------------------------- charges
def parse_rtp(path):
    txt = Path(path).read_text()
    charges = {}
    cur, in_atoms = None, False
    sections = {"atoms", "bonds", "impropers", "cmap", "dihedrals", "angles",
                "exclusions", "pairs"}
    for line in txt.splitlines():
        m = re.match(r"\s*\[\s*([A-Za-z0-9_]+)\s*\]", line)
        if m:
            name = m.group(1)
            if name in sections:
                in_atoms = name == "atoms"
            else:
                cur, in_atoms = name, False
            continue
        if in_atoms and cur:
            parts = line.split(";")[0].split()
            if len(parts) >= 3:
                try:
                    charges[(cur, parts[0])] = float(parts[2])
                except ValueError:
                    pass
    return charges


RTP = parse_rtp(DATA / "aminoacids_charmm36_jul2022.rtp")


def build_charges(df, his_variant):
    """Charge per atom row of atoms.parquet, with termini patches.

    Returns (charges float64 array, list of (residue, atom) without a charge).
    """
    res_names = df["residue_name"].values
    atom_names = df["name"].values
    seg = df["segment"].values
    ridx = df["residue_index"].values
    q = np.zeros(len(df), dtype=np.float64)
    missing = []
    # ordered residues per segment, in file order
    residues = []
    seen = set()
    for i in range(len(df)):
        key = (seg[i], int(ridx[i]))
        if key not in seen:
            seen.add(key)
            residues.append(key)
    first_res = {}
    last_res = {}
    for s in dict.fromkeys(seg.tolist()):
        rs = [k for k in residues if k[0] == s]
        first_res[s] = rs[0]
        last_res[s] = rs[-1]
    for i in range(len(df)):
        rn, an = str(res_names[i]), str(atom_names[i])
        key = (seg[i], int(ridx[i]))
        table_res = rn
        if rn == "HIS":
            table_res = his_variant
        val = RTP.get((table_res, an))
        # termini patches (CHARMM36 GROMACS aminoacids.n/c.tdb)
        if key == first_res[seg[i]]:
            if an == "N":
                val = -0.07 if rn == "PRO" else -0.30
            elif an == "CA":
                val = 0.13 if rn == "GLY" else (0.16 if rn == "PRO" else 0.21)
            elif an == "CD" and rn == "PRO":
                val = 0.16
        if key == last_res[seg[i]]:
            if an == "C":
                val = 0.34
            elif an in ("OT1", "OT2", "OXT"):
                val = -0.67
        if val is None:
            missing.append((rn, an))
        else:
            q[i] = val
    return q, missing, residues


# ---------------------------------------------------------------- geometry
def mic(v, box):
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        return v - lengths * np.rint(v / lengths)
    inv = np.linalg.inv(box)
    fr = v @ inv
    fr -= np.rint(fr)
    return fr @ box


def dihedral(p0, p1, p2, p3, box):
    # standard formulation (praxeolitic); sign verified against the known
    # Ramachandran distribution (phi mostly negative) on 4cv4_A frame 10.
    b0 = -mic(p1 - p0, box)
    b1 = mic(p2 - p1, box)
    b2 = mic(p3 - p2, box)
    nb1 = np.linalg.norm(b1)
    if nb1 == 0:
        return np.nan
    b1n = b1 / nb1
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    if np.linalg.norm(v) == 0 or np.linalg.norm(w) == 0:
        return np.nan
    return float(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def field_at_pairs(pos, box, q, da, ac):
    """E at each pair midpoint (donor/acceptor atoms excluded).

    Returns (e_parallel, e_magnitude) in V/Angstrom.
    """
    D = pos[da]
    A = pos[ac]
    disp = mic(A - D, box)
    dist = np.linalg.norm(disp, axis=1)
    u = disp / dist[:, None]
    M = D + 0.5 * disp
    out_par = np.empty(len(da))
    out_mag = np.empty(len(da))
    for s in range(0, len(da), 256):
        e = slice(s, min(s + 256, len(da)))
        r = mic(pos[None, :, :] - M[e, None, :], box)          # (b, atoms, 3)
        r2 = (r * r).sum(axis=-1)
        inv_r3 = r2 ** -1.5
        inv_r3[np.arange(r.shape[0]), da[e]] = 0.0
        inv_r3[np.arange(r.shape[0]), ac[e]] = 0.0
        E = COULOMB * (r * (inv_r3 * q[None, :])[:, :, None]).sum(axis=1)
        out_par[e] = (E * u[e]).sum(axis=1)
        out_mag[e] = np.linalg.norm(E, axis=1)
    return out_par, out_mag


def phi_psi(pos, box, df, residues):
    """phi/psi per residue (ordered list), radians, NaN where undefined."""
    names = df["name"].values
    ridx = df["residue_index"].values
    seg = df["segment"].values
    by_res = {}
    for (s, r) in residues:
        idx = np.flatnonzero((ridx == r) & (seg == s))
        by_res[(s, r)] = {str(names[i]): int(i) for i in idx}
    phi = {}
    psi = {}
    for k, key in enumerate(residues):
        m = by_res[key]
        if k > 0 and residues[k - 1][0] == key[0]:
            pm = by_res[residues[k - 1]]
            if "C" in pm and {"N", "CA", "C"} <= m.keys():
                phi[key] = dihedral(pos[pm["C"]], pos[m["N"]], pos[m["CA"]],
                                    pos[m["C"]], box)
        if k < len(residues) - 1 and residues[k + 1][0] == key[0]:
            nm = by_res[residues[k + 1]]
            if "N" in nm and {"N", "CA", "C"} <= m.keys():
                psi[key] = dihedral(pos[m["N"]], pos[m["CA"]], pos[m["C"]],
                                    pos[nm["N"]], box)
    return phi, psi


# ---------------------------------------------------------------- locations
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


# ---------------------------------------------------------------- main loop
def process(protein):
    out_path = DATA / f"{protein}.npz"
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
        pos11 = z["positions"][:11].astype(np.float64)
        box11 = z["box"][:11].astype(np.float64)
        pair = k[f"r{rep}_pair"]
        static = k[f"r{rep}_static"]
        hist = k[f"r{rep}_history"]
        da = pair[:, 0].astype(int)
        ac = pair[:, 1].astype(int)
        # verification of the layout
        d10 = np.linalg.norm(mic(pos11[10, da] - pos11[10, ac], box11[10]), axis=1)
        dall = np.stack([np.linalg.norm(mic(pos11[f, da] - pos11[f, ac], box11[f]), axis=1)
                         for f in range(11)])
        store[f"r{rep}_verify_d10_maxdiff"] = np.array([np.abs(d10 - static[:, 1]).max()])
        store[f"r{rep}_verify_dmean_maxdiff"] = np.array([np.abs(dall.mean(axis=0) - hist[:, 1]).max()])
        # charges and field at frame 10
        pos = pos11[10]
        box = box11[10]
        for variant, suffix in (("HSE", ""), ("HSD", "_hsd")):
            q, missing, residues = build_charges(df, variant)
            if missing:
                raise RuntimeError(f"{protein} R{rep} {variant}: missing charges {sorted(set(missing))}")
            ep, em = field_at_pairs(pos, box, q, da, ac)
            store[f"r{rep}_e_par{suffix}"] = ep.astype(np.float32)
            store[f"r{rep}_e_mag{suffix}"] = em.astype(np.float32)
        # phi / psi at frame 10
        q, missing, residues = build_charges(df, "HSE")
        phi, psi = phi_psi(pos, box, df, residues)
        ridx = df["residue_index"].values
        seg = df["segment"].values
        dres = [(str(seg[i]), int(ridx[i])) for i in da]
        ares = [(str(seg[i]), int(ridx[i])) for i in ac]
        store[f"r{rep}_phi_d"] = np.array([phi.get(r, np.nan) for r in dres], dtype=np.float32)
        store[f"r{rep}_psi_d"] = np.array([psi.get(r, np.nan) for r in dres], dtype=np.float32)
        store[f"r{rep}_phi_a"] = np.array([phi.get(r, np.nan) for r in ares], dtype=np.float32)
        store[f"r{rep}_psi_a"] = np.array([psi.get(r, np.nan) for r in ares], dtype=np.float32)
        store[f"r{rep}_n_atoms"] = np.array([len(df)])
        store[f"r{rep}_n_residues"] = np.array([len(residues)])
    np.savez_compressed(out_path, **store)
    if acquired:
        shutil.rmtree(TMPWS / protein, ignore_errors=True)
    return "done"


def main():
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
