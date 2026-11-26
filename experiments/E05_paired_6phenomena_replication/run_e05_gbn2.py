"""E05 Step 3: Run 4 x 100 ps OpenMM GBn2 GPU implicit-solvent MD on all 49 unseen atlas_test proteins,
extracting all 6 phenomena (hb_form, hb_diss, ct_form, ct_diss, sb_form, sb_diss) from ONE pass per protein.
"""
import json
import sys
import time
from pathlib import Path
import MDAnalysis as mda
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

sys.path.insert(0, r"d:\KineScope\experiments\E01_true_short_md_guidance")
sys.path.insert(0, r"d:\KineScope\experiments\E05_paired_6phenomena_replication")
import extract_e01_e02 as ext_mod
import fetch_e05_atlas

ATLAS_ROOT = Path(r"d:\KineScope\experiments\shared_data\atlas")
REAL_CACHE = Path(r"d:\KineScope\experiments\E05_paired_6phenomena_replication\cache_real")
OUT_CACHE = Path(r"d:\KineScope\experiments\E05_paired_6phenomena_replication\cache_gbn2")

N_SEEDS = 4
SIM_PS = 100.0
SNAP_PS = 10.0  # 10 snapshots per seed = 40 post-origin samples
THRESH = {
    "hb_form": 3.5, "hb_diss": 3.5,
    "ct_form": 4.5, "ct_diss": 4.5,
    "sb_form": 4.0, "sb_diss": 4.0,
}


def compute_min_pair_dists(pos: np.ndarray, pair_idx: np.ndarray, groups_a: list, groups_b: list) -> np.ndarray:
    """Compute min Euclidean distance between atom index groups for each candidate residue pair."""
    n = len(pair_idx)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        ia = groups_a[pair_idx[i, 0]]
        ib = groups_b[pair_idx[i, 1]]
        diff = pos[ia][:, None, :] - pos[ib][None, :, :]
        out[i] = float(np.sqrt((diff * diff).sum(axis=-1)).min())
    return out


def summarize_7stats(all_dists: np.ndarray, d0_ref: np.ndarray, thresh: float) -> tuple:
    """all_dists: (N_SEEDS, n_snaps+1, N) -> (N, 7) stats and (N_SEEDS, N) seed_means."""
    n = all_dists.shape[2]
    if n == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((all_dists.shape[0], 0), dtype=np.float32)
    post = all_dists[:, 1:, :].reshape(-1, n)
    stats = np.column_stack([
        post.mean(axis=0),
        post.std(axis=0),
        post.min(axis=0),
        post.max(axis=0),
        all_dists[:, -1, :].mean(axis=0) - d0_ref,
        (post <= thresh).mean(axis=0),
        (post <= thresh + 1.0).mean(axis=0),
    ]).astype(np.float32)
    seed_means = all_dists[:, 1:, :].mean(axis=1).astype(np.float32)
    return stats, seed_means


def run_one(prot: str):
    out_npz = OUT_CACHE / f"{prot}.npz"
    if out_npz.exists():
        return (prot, "cached", 0.0)

    t0 = time.time()
    real_f = REAL_CACHE / f"{prot}.npz"
    if not real_f.exists():
        return (prot, "skip_no_real", 0.0)
    ext = np.load(real_f, allow_pickle=True)

    pdir = ATLAS_ROOT / prot
    meta = json.loads((pdir / "done.json").read_text(encoding="utf-8"))
    pdb_path = pdir / meta["pdb"]
    r1_xtc = pdir / sorted(meta["xtc_files"])[0]
    topo = ext_mod.parse_topology(pdb_path)

    u = mda.Universe(str(pdb_path), str(r1_xtc))
    u.trajectory[10]

    # Write ALL protein heavy atoms (including C-terminal OT1/OT2) so OpenMM recognizes C-terminus
    heavy_atoms = u.select_atoms("not name H* and not name [0-9]H*")
    tmp_pdb = OUT_CACHE / f"_tmp_{prot}_f10.pdb"
    heavy_atoms.write(str(tmp_pdb))

    # Map original atom index -> (u_res_idx, atom_name), and topo residue index -> u_res_idx
    orig_atom_to_key = {}
    topo_res_to_u = []
    for u_ri, res in enumerate(u.residues):
        nm = {a.name.strip(): a for a in res.atoms}
        if {"N", "CA", "C", "O"}.issubset(nm):
            topo_res_to_u.append(u_ri)
        for a in res.atoms:
            orig_atom_to_key[int(a.index)] = (u_ri, a.name.strip())
    assert len(topo_res_to_u) == topo["n_res"], f"{prot}: res count mismatch {len(topo_res_to_u)} vs {topo['n_res']}"

    pdb_omm = app.PDBFile(str(tmp_pdb))
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = app.Modeller(pdb_omm.topology, pdb_omm.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element == app.element.hydrogen])
    modeller.addHydrogens(ff, pH=7.0)

    omm_map = {}
    omm_heavy_by_u = {}
    omm_atoms_by_u = {}
    for res_i, res in enumerate(modeller.topology.residues()):
        h_list = []
        an_map = {}
        for a in res.atoms():
            an = a.name.strip()
            omm_map[(res_i, an)] = a.index
            an_map[an] = a.index
            if a.element != app.element.hydrogen:
                h_list.append(a.index)
        omm_heavy_by_u[res_i] = np.array(h_list, dtype=np.int32)
        omm_atoms_by_u[res_i] = an_map

    # Build topo-indexed heavy/pos/neg atom arrays matching ext's ct_pair & sb_pair indices
    omm_heavy_by_res = []
    omm_pos_by_res = {}
    omm_neg_by_res = {}
    for topo_ri, u_ri in enumerate(topo_res_to_u):
        omm_heavy_by_res.append(omm_heavy_by_u[u_ri])
        rname = topo["resnames"][topo_ri]
        an_map = omm_atoms_by_u[u_ri]
        p_list = [an_map[nm] for nm in sorted(ext_mod.POS_RES.get(rname, ())) if nm in an_map]
        n_list = [an_map[nm] for nm in sorted(ext_mod.NEG_RES.get(rname, ())) if nm in an_map]
        if p_list:
            omm_pos_by_res[topo_ri] = np.array(p_list, dtype=np.int32)
        if n_list:
            omm_neg_by_res[topo_ri] = np.array(n_list, dtype=np.int32)

    init_pos_A = np.array([[p.x, p.y, p.z] for p in modeller.positions.value_in_unit(unit.angstrom)], dtype=np.float64)

    # Map atom pairs for hb_form and hb_diss
    hb_maps = {}
    max_err_all = 0.0
    for mode in ("hb_form", "hb_diss"):
        pairs = ext[f"{mode}_pair"]
        oa = np.zeros(len(pairs), dtype=np.int32)
        ob = np.zeros(len(pairs), dtype=np.int32)
        vm = np.ones(len(pairs), dtype=bool)
        for i in range(len(pairs)):
            ka = orig_atom_to_key.get(int(pairs[i, 0]))
            kb = orig_atom_to_key.get(int(pairs[i, 1]))
            ia = omm_map.get(ka) if ka else None
            ib = omm_map.get(kb) if kb else None
            if ib is None and kb and kb[1] in ("OT1", "OT2", "O", "OXT"):
                for alt in ("O", "OXT", "OT1", "OT2"):
                    if (kb[0], alt) in omm_map:
                        ib = omm_map[(kb[0], alt)]
                        break
            if ia is None or ib is None:
                vm[i] = False
            else:
                oa[i], ob[i] = ia, ib
        if vm.any():
            d0_omm = np.linalg.norm(init_pos_A[oa[vm]] - init_pos_A[ob[vm]], axis=1)
            d0_ref = ext[f"{mode}_series"][vm, 10].astype(np.float64)
            err = float(np.abs(d0_omm - d0_ref).max())
            max_err_all = max(max_err_all, err)
            assert err < 0.05, f"{prot} {mode}: origin mismatch {err:.4f}A"
        hb_maps[mode] = (oa, ob, vm)

    # Verify residue-level origin distances for ct & sb
    ct_pairs = {m: ext[f"{m}_pair"] for m in ("ct_form", "ct_diss")}
    sb_pairs = {m: ext[f"{m}_pair"] for m in ("sb_form", "sb_diss")}
    for mode in ("ct_form", "ct_diss"):
        p = ct_pairs[mode]
        if len(p):
            d0_omm = compute_min_pair_dists(init_pos_A, p, omm_heavy_by_res, omm_heavy_by_res)
            d0_ref = ext[f"{mode}_series"][:, 10].astype(np.float64)
            err = float(np.abs(d0_omm - d0_ref).max())
            max_err_all = max(max_err_all, err)
            assert err < 0.05, f"{prot} {mode}: origin mismatch {err:.4f}A"
    for mode in ("sb_form", "sb_diss"):
        p = sb_pairs[mode]
        if len(p):
            d0_omm = compute_min_pair_dists(init_pos_A, p, omm_pos_by_res, omm_neg_by_res)
            d0_ref = ext[f"{mode}_series"][:, 10].astype(np.float64)
            err = float(np.abs(d0_omm - d0_ref).max())
            max_err_all = max(max_err_all, err)
            assert err < 0.05, f"{prot} {mode}: origin mismatch {err:.4f}A"

    # Build OpenMM simulation
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName("OpenCL")
    sim = app.Simulation(modeller.topology, system, integrator, platform)

    n_snaps = int(SIM_PS / SNAP_PS)
    steps_per_snap = int(SNAP_PS / 0.002)
    dists = {
        m: np.empty((N_SEEDS, n_snaps + 1, len(ext[f"{m}_pair"])), dtype=np.float32)
        for m in ("hb_form", "hb_diss", "ct_form", "ct_diss", "sb_form", "sb_diss")
    }

    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=200)
    min_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    def record_frame(pos_A, s_i, t_i):
        for m in ("hb_form", "hb_diss"):
            oa, ob, _ = hb_maps[m]
            if len(oa):
                dists[m][s_i, t_i] = np.linalg.norm(pos_A[oa] - pos_A[ob], axis=1)
        for m in ("ct_form", "ct_diss"):
            dists[m][s_i, t_i] = compute_min_pair_dists(pos_A, ct_pairs[m], omm_heavy_by_res, omm_heavy_by_res)
        for m in ("sb_form", "sb_diss"):
            dists[m][s_i, t_i] = compute_min_pair_dists(pos_A, sb_pairs[m], omm_pos_by_res, omm_neg_by_res)

    for s_i in range(N_SEEDS):
        sim.context.setPositions(min_pos * unit.angstrom)
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 20261126 + s_i * 997)
        record_frame(min_pos, s_i, 0)
        for t_i in range(1, n_snaps + 1):
            sim.step(steps_per_snap)
            pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            record_frame(pos, s_i, t_i)

    if tmp_pdb.exists():
        tmp_pdb.unlink()

    out = {"protein": prot, "max_d0_err": np.float32(max_err_all), "elapsed_s": np.float32(time.time() - t0)}
    for m in ("hb_form", "hb_diss", "ct_form", "ct_diss", "sb_form", "sb_diss"):
        d0_ref = ext[f"{m}_series"][:, 10].astype(np.float32) if len(ext[f"{m}_series"]) else np.zeros(0, dtype=np.float32)
        st, sm = summarize_7stats(dists[m], d0_ref, THRESH[m])
        out[f"{m}_stats7"] = st
        out[f"{m}_seed_means"] = sm
        if m in hb_maps:
            out[f"{m}_valid"] = hb_maps[m][2]
        else:
            out[f"{m}_valid"] = np.ones(len(ext[f"{m}_pair"]), dtype=bool)

    np.savez_compressed(out_npz, **out)
    return (prot, f"ok (L={topo['n_res']}, d0_err={max_err_all:.4f}A)", time.time() - t0)


def main():
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    chosen = [p for p, _ in fetch_e05_atlas.select_e05_proteins(250)]
    print(f"E05 GBn2: running 4x100ps implicit MD (all 6 phenomena) on {len(chosen)} unseen atlas_test proteins...", flush=True)
    t0 = time.time()
    ok = cached = fail = 0
    for i, p in enumerate(chosen, 1):
        try:
            prot, status, dt = run_one(p)
            if status == "cached":
                cached += 1
            else:
                ok += 1
            print(f"[{i}/{len(chosen)}] {prot}: {status} ({dt:.1f}s)", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(chosen)}] {p}: FAIL {e}", flush=True)
    print(f"E05 GBn2 finished in {time.time() - t0:.1f}s | ok={ok} cached={cached} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
