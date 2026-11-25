"""K3-01 step 2 (vectorized): chi angles for the 6 residues around each pair.
Same outputs as extract_chi.py (per-protein npz in k3_01/chi), but the
dihedral is computed for all pairs at once per (slot, chi, frame).
Checkpointed: skips proteins whose chi npz exists. Proteins present in the
G05 workspace are read there and left in place; others are acquired into
k3_01/atlas and deleted after processing. Use: extract_chi2.py <log> <protein...>
"""
import sys, json, shutil, time
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
import numpy as np
from pathlib import Path
from tseries.data import registry, trajectory, cell, acquire
from tseries.features import _mic

ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
G05 = Path("/home/hatch/workspace/kinescope/work/g05_atlas_workspace")
K15 = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
OWN = Path("/home/hatch/workspace/kinescope/work/k3_01/atlas")
OUTD = Path("/home/hatch/workspace/kinescope/work/k3_01/chi")
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
    order = []; seen = {}
    for i, r in enumerate(res_idx):
        if r not in seen:
            seen[r] = len(order); order.append(r)
    atom_res_pos = np.array([seen[r] for r in res_idx])
    res_names = []; name2idx = []
    for r in order:
        idxs = [i for i in range(len(names)) if res_idx[i] == r]
        res_names.append(res_name[idxs[0]])
        name2idx.append({names[i]: i for i in idxs})
    return order, atom_res_pos, res_names, name2idx

def one_protein(p):
    # Read-only sources first (other experiments' workspaces: never delete).
    if (G05 / p).is_dir():
        src = G05
    elif (K15 / p).is_dir():
        src = K15
    else:
        src = OWN
    if src is OWN:
        acquire(OWN, p)
    reg = registry(src, p)
    out = {}
    ref_names = None
    for r in (1, 2, 3):
        tid = trajectory(p, r)
        atoms = reg.atoms(tid).to_pydict()
        if ref_names is None:
            ref_names = list(atoms["name"])
        else:
            assert list(atoms["name"]) == ref_names, (p, r, "topology differs across replicas")
        order, atom_res_pos, res_names, name2idx = residue_structure(atoms)
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
        slots = np.column_stack([dpos, apos, dpos - 1, dpos + 1, apos - 1, apos + 1])
        valid_slot = (slots >= 0) & (slots < nres)
        chi = np.full((len(pair), 6, 4, 2), np.nan, dtype=np.float32)
        slot_resname = np.empty((len(pair), 6), dtype=object)
        for s in range(6):
            sp = np.where(valid_slot[:, s], slots[:, s], 0)
            slot_resname[:, s] = np.array([res_names[i] if valid_slot[j, s] else "" for j, i in enumerate(sp)], dtype=object)
            for k in range(4):
                # pairs whose slot residue defines chi k and has all 4 atoms
                rows = []
                for j, i in enumerate(sp):
                    if not valid_slot[j, s]:
                        continue
                    quad = CHI.get(res_names[i], [])
                    if k >= len(quad):
                        continue
                    try:
                        rows.append((j, tuple(name2idx[i][nm] for nm in quad[k])))
                    except KeyError:
                        continue
                if not rows:
                    continue
                js = np.array([x[0] for x in rows])
                idx = np.array([x[1] for x in rows])  # (m, 4) atom indices
                for f, fr in ((0, 0), (1, 10)):
                    chi[js, s, k, f] = dihedral(pos[fr, idx[:, 0]], pos[fr, idx[:, 1]],
                                               pos[fr, idx[:, 2]], pos[fr, idx[:, 3]], cell(boxes[fr]))
        out[f"r{r}_chi"] = chi
        out[f"r{r}_slot_resname"] = slot_resname
        out[f"r{r}_donor_res"] = np.array([order[i] for i in dpos])
        out[f"r{r}_acceptor_res"] = np.array([order[i] for i in apos])
        out[f"r{r}_check"] = np.array([d_d10, d_mean])
    np.savez_compressed(OUTD / f"{p}.npz", **out)
    if src is OWN:
        shutil.rmtree(OWN / p, ignore_errors=True)  # only our own downloads are deleted
    return out["r1_check"]

def main():
    log = Path(sys.argv[1]); prots = sys.argv[2:]
    for p in prots:
        if (OUTD / f"{p}.npz").exists():
            print("SKIP", p, flush=True); continue
        t0 = time.time()
        try:
            chk = one_protein(p)
            with open(log, "a") as f:
                f.write(f"DONE {p} check {chk} sec {round(time.time() - t0, 1)}\n")
            print("DONE", p, "sec", round(time.time() - t0, 1), flush=True)
        except Exception as e:
            with open(log, "a") as f:
                f.write(f"FAILED {p} {type(e).__name__} {str(e)[:300]}\n")
            print("FAILED", p, type(e).__name__, str(e)[:300], flush=True)

if __name__ == "__main__":
    main()
