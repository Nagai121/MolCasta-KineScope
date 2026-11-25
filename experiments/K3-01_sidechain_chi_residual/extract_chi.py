"""K3-01 step 2: chi angles for the 6 residues around each K06(b) pair.

For every protein (100 test + 8 calibration), every replica (1..3), every
candidate pair in k06b_out: donor residue D, acceptor residue A, and their
sequence neighbours D-1, D+1, A-1, A+1. Chi1..chi4 (where defined for the
residue type) are computed at frame 10 (origin frame; primary) and frame 0
(secondary) from the ATLAS analysis trajectory, with MIC bond vectors.
Premise check per replica: distances recomputed from the coordinates must
match pair[:,2] (frame 10) and history[:,1] (mean of frames 0..10).
Proteins already present in the G05 workspace are read there and left in
place; others are acquired into k3_01/atlas and deleted after processing.
Checkpointed: skips proteins whose chi npz exists.
"""
import sys, json, shutil, time
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
import numpy as np
from pathlib import Path
from tseries.data import registry, trajectory, cell, acquire
from tseries.features import _mic

ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
PROTEINS = ORIG["test_proteins"] + ORIG["calibration_proteins"]
G05 = Path("/home/hatch/workspace/kinescope/work/g05_atlas_workspace")
OWN = Path("/home/hatch/workspace/kinescope/work/k3_01/atlas")
OUTD = Path("/home/hatch/workspace/kinescope/work/k3_01/chi")
OUTD.mkdir(parents=True, exist_ok=True)
SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")

CHI = {
 'ARG': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','NE'), ('CG','CD','NE','CZ')],
 'ASN': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
 'ASP': [('N','CA','CB','CG'), ('CA','CB','CG','OD1')],
 'CYS': [('N','CA','CB','SG')],
 'GLN': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
 'GLU': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','OE1')],
 'HIS': [('N','CA','CB','CG'), ('CA','CB','CG','ND1')],
 'ILE': [('N','CA','CB','CG1'), ('CA','CB','CG1','CD1')],
 'LEU': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
 'LYS': [('N','CA','CB','CG'), ('CA','CB','CG','CD'), ('CB','CG','CD','CE'), ('CG','CD','CE','NZ')],
 'MET': [('N','CA','CB','CG'), ('CA','CB','CG','SD'), ('CB','CG','SD','CE')],
 'PHE': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
 'PRO': [('N','CA','CB','CG'), ('CA','CB','CG','CD')],
 'SER': [('N','CA','CB','OG')],
 'THR': [('N','CA','CB','OG1')],
 'TRP': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
 'TYR': [('N','CA','CB','CG'), ('CA','CB','CG','CD1')],
 'VAL': [('N','CA','CB','CG1')],
}

def dihedral(p1, p2, p3, p4, box):
    b1 = _mic(p2 - p1, box); b2 = _mic(p3 - p2, box); b3 = _mic(p4 - p3, box)
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    b2h = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    m1 = np.cross(n1, b2h)
    return np.degrees(np.arctan2(np.einsum("...i,...i->...", m1, n2),
                                 np.einsum("...i,...i->...", n1, n2)))

def residue_structure(atoms):
    names = list(atoms["name"]); res_idx = list(atoms["residue_index"]); res_name = list(atoms["residue_name"])
    order = []  # residue_index in order of first appearance
    seen = {}
    for i, r in enumerate(res_idx):
        if r not in seen:
            seen[r] = len(order); order.append(r)
    atom_res_pos = np.array([seen[r] for r in res_idx])  # atom -> position in residue order
    res_names = []
    name2idx = []
    for r in order:
        idxs = [i for i in range(len(names)) if res_idx[i] == r]
        res_names.append(res_name[idxs[0]])
        name2idx.append({names[i]: i for i in idxs})
    return order, atom_res_pos, res_names, name2idx

def one_protein(p):
    src = G05 if (G05 / p).is_dir() else OWN
    if src is OWN:
        acquire(OWN, p)
    reg = registry(src, p)
    out = {}
    ref_struct = None
    for r in (1, 2, 3):
        tid = trajectory(p, r)
        atoms = reg.atoms(tid).to_pydict()
        order, atom_res_pos, res_names, name2idx = residue_structure(atoms)
        if ref_struct is None:
            ref_struct = (list(atoms["name"]), list(atoms["residue_index"]))
        else:
            assert list(atoms["name"]) == ref_struct[0] and list(atoms["residue_index"]) == ref_struct[1], (p, r, "topology differs across replicas")
        blocks = list(reg.blocks(tid, start=0, stop=11, chunk_frames=11))
        pos = np.concatenate([b.positions for b in blocks], axis=0)
        boxes = np.concatenate([b.box for b in blocks], axis=0)
        with np.load(SAVED / f"{p}.npz") as z:
            pair = z[f"r{r}_pair"]; hist = z[f"r{r}_history"]
        a, b = pair[:, 0].astype(np.intp), pair[:, 1].astype(np.intp)
        series = np.stack([np.linalg.norm(_mic(pos[k, b] - pos[k, a], cell(boxes[k])), axis=1) for k in range(11)], axis=0)
        d_d10 = float(np.abs(series[10] - pair[:, 2]).max())
        d_mean = float(np.abs(series.mean(axis=0) - hist[:, 1]).max())
        assert d_d10 < 1e-3 and d_mean < 1e-3, (p, r, "premise mismatch", d_d10, d_mean)
        dpos = atom_res_pos[a]; apos = atom_res_pos[b]
        nres = len(order)
        slots = np.column_stack([dpos, apos, dpos - 1, dpos + 1, apos - 1, apos + 1])  # D, A, D-1, D+1, A-1, A+1
        valid_slot = (slots >= 0) & (slots < nres)
        chi = np.full((len(pair), 6, 4, 2), np.nan, dtype=np.float32)  # pair, slot, chi_k, frame(0->frame0, 1->frame10)
        slot_resname = np.empty((len(pair), 6), dtype=object)
        for s in range(6):
            sp = np.where(valid_slot[:, s], slots[:, s], 0)
            slot_resname[:, s] = [res_names[i] if valid_slot[j, s] else "" for j, i in enumerate(sp)]
            for j, i in enumerate(sp):
                if not valid_slot[j, s]:
                    continue
                rn = res_names[i]
                for k, quad in enumerate(CHI.get(rn, [])):
                    try:
                        idx = [name2idx[i][nm] for nm in quad]
                    except KeyError:
                        continue
                    for f, fr in ((0, 0), (1, 10)):
                        chi[j, s, k, f] = dihedral(pos[fr, idx[0]], pos[fr, idx[1]], pos[fr, idx[2]], pos[fr, idx[3]], cell(boxes[fr]))
        out[f"r{r}_chi"] = chi
        out[f"r{r}_slot_resname"] = slot_resname
        out[f"r{r}_donor_res"] = np.array([order[i] for i in dpos])
        out[f"r{r}_acceptor_res"] = np.array([order[i] for i in apos])
        out[f"r{r}_check"] = np.array([d_d10, d_mean])
    np.savez_compressed(OUTD / f"{p}.npz", **out)
    if src is OWN:
        shutil.rmtree(OWN / p, ignore_errors=True)
    return out[f"r1_check"]

def main():
    only = sys.argv[1:] or None
    for p in PROTEINS:
        if only and p not in only:
            continue
        if (OUTD / f"{p}.npz").exists():
            print("SKIP", p, flush=True); continue
        t0 = time.time()
        try:
            chk = one_protein(p)
            print("DONE", p, "check", chk, "sec", round(time.time() - t0, 1), flush=True)
        except Exception as e:
            print("FAILED", p, type(e).__name__, str(e)[:300], flush=True)

if __name__ == "__main__":
    main()
