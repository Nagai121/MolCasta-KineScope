"""L3 payload builder (runs on this VM): per protein, write
  payload/<prot>_f10.pdb   frame-10 heavy-atom PDB (E05's exact procedure)
  payload/<prot>_meta.npz  pair keys for the OpenMM runner:
    hb_{mode}_ka_res/ka_name/kb_res/kb_name : (residue index in the original
      PDB's residue order, atom name) for each hb candidate pair atom
    topo_res_to_u : topo residue index -> original residue index
    topo_resnames : residue names in topo index space
The extracted cache npz (l3work/extracted/<prot>.npz) is shipped alongside.
"""
import sys
from pathlib import Path
import MDAnalysis as mda
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/kinescope/repo/experiments/E01_true_short_md_guidance")
import extract_e01_e02 as ext_mod

ATLAS_ROOT = Path("/home/hatch/workspace/kinescope/l3work/atlas")
EXTRACTED = Path("/home/hatch/workspace/kinescope/l3work/extracted")
PAYLOAD = Path("/home/hatch/workspace/kinescope/l3work/payload")
PROTS = ['1a62_A', '1dd3_B', '1ikt_A', '1m8s_A', '1sf9_A', '1uha_A', '1xmk_A',
         '2e5y_B', '2gpi_A', '2i9x_A', '2nwv_A', '2ysk_A', '3c9p_A', '3fpn_A',
         '3kz7_A', '3pes_A', '3t92_A', '3v3l_A', '4cv4_A', '4myy_A', '4zds_B',
         '5ii7_A', '6bk4_A', '6in7_A', '7ead_A']


def build_one(prot):
    out_pdb = PAYLOAD / f"{prot}_f10.pdb"
    out_meta = PAYLOAD / f"{prot}_meta.npz"
    if out_pdb.exists() and out_meta.exists():
        return (prot, "cached")
    pdir = ATLAS_ROOT / prot
    import json
    meta_in = json.loads((pdir / "done.json").read_text(encoding="utf-8"))
    pdb_path = pdir / meta_in["pdb"]
    r1_xtc = pdir / sorted(meta_in["xtc_files"])[0]
    topo = ext_mod.parse_topology(pdb_path)

    u = mda.Universe(str(pdb_path), str(r1_xtc))
    # original residue index -> backbone-complete? (E05's topo_res_to_u)
    topo_res_to_u = []
    orig_atom_to_key = {}
    for u_ri, res in enumerate(u.residues):
        nm = {a.name.strip(): a for a in res.atoms}
        if {"N", "CA", "C", "O"}.issubset(nm):
            topo_res_to_u.append(u_ri)
        for a in res.atoms:
            orig_atom_to_key[int(a.index)] = (u_ri, a.name.strip())
    assert len(topo_res_to_u) == topo["n_res"], (prot, len(topo_res_to_u), topo["n_res"])

    u.trajectory[10]
    heavy_atoms = u.select_atoms("not name H* and not name [0-9]H*")
    heavy_atoms.write(str(out_pdb))

    ext = np.load(EXTRACTED / f"{prot}.npz")
    out = {
        "topo_res_to_u": np.array(topo_res_to_u, dtype=np.int32),
        "topo_resnames": np.array(topo["resnames"]),
    }
    for mode in ("hb_form", "hb_diss"):
        pairs = ext[f"{mode}_pair"]
        ka_res = np.empty(len(pairs), dtype=np.int32)
        ka_name = np.empty(len(pairs), dtype="<U4")
        kb_res = np.empty(len(pairs), dtype=np.int32)
        kb_name = np.empty(len(pairs), dtype="<U4")
        for i in range(len(pairs)):
            ra, na = orig_atom_to_key[int(pairs[i, 0])]
            rb, nb = orig_atom_to_key[int(pairs[i, 1])]
            ka_res[i], ka_name[i] = ra, na
            kb_res[i], kb_name[i] = rb, nb
        out[f"{mode}_ka_res"] = ka_res
        out[f"{mode}_ka_name"] = ka_name
        out[f"{mode}_kb_res"] = kb_res
        out[f"{mode}_kb_name"] = kb_name
    np.savez_compressed(out_meta, **out)
    return (prot, "ok")


def main():
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(PROTS, 1):
        prot, status = build_one(p)
        print(f"[{i}/{len(PROTS)}] {prot}: {status}", flush=True)


if __name__ == "__main__":
    main()
