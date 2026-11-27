"""L3 GBn2 runner (Colab CPU session or local): implicit-solvent short MD
from ATLAS R1 frame 10, E03/E05 protocol, recording RAW pair distances for
all 6 phenomena at every 10 ps snapshot so that all length conditions can
be derived in analysis without re-running.

Runs per protein (checkpointed, skip/extend if present):
  <prot>_batch.npz : seeds 20261126 + i*997 (i = 0..n_seeds-1), 100 ps each,
                     snapshots at 0,10,...,100 ps -> (n_seeds, 11, N) per mode
  <prot>_long.npz  : seed 20261126, 400 ps, snapshots every 10 ps
                     -> (1, 41, N) per mode
Snapshot 0 is the minimized origin (excluded from condition means, as E03).
Protocol identical to E05 run_e05_gbn2.py: amber14-all + implicit/gbn2,
NoCutoff, HBonds constraints, LangevinMiddle 300 K, 1/ps, 2 fs,
minimizeEnergy(200) once, per-seed velocities from the minimized positions.
Platform is given explicitly (default CPU).
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

MODES = ("hb_form", "hb_diss", "ct_form", "ct_diss", "sb_form", "sb_diss")
POS_RES = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}}
NEG_RES = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
BASE_SEED = 20261126
SNAP_PS = 10.0
DT_PS = 0.002


def compute_min_pair_dists(pos, pair_idx, groups_a, groups_b):
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


def build_simulation(prot, payload):
    meta = np.load(payload / f"{prot}_meta.npz", allow_pickle=True)
    ext = np.load(payload / f"{prot}.npz", allow_pickle=True)
    pdb_omm = app.PDBFile(str(payload / f"{prot}_f10.pdb"))
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

    topo_res_to_u = meta["topo_res_to_u"]
    topo_resnames = [str(x) for x in meta["topo_resnames"]]
    omm_heavy_by_res = []
    omm_pos_by_res = {}
    omm_neg_by_res = {}
    for topo_ri, u_ri in enumerate(topo_res_to_u):
        u_ri = int(u_ri)
        omm_heavy_by_res.append(omm_heavy_by_u[u_ri])
        rname = topo_resnames[topo_ri]
        an_map = omm_atoms_by_u[u_ri]
        p_list = [an_map[nm] for nm in sorted(POS_RES.get(rname, ())) if nm in an_map]
        n_list = [an_map[nm] for nm in sorted(NEG_RES.get(rname, ())) if nm in an_map]
        if p_list:
            omm_pos_by_res[topo_ri] = np.array(p_list, dtype=np.int32)
        if n_list:
            omm_neg_by_res[topo_ri] = np.array(n_list, dtype=np.int32)

    init_pos_A = np.array([[p.x, p.y, p.z] for p in modeller.positions.value_in_unit(unit.angstrom)], dtype=np.float64)

    hb_maps = {}
    max_err_all = 0.0
    for mode in ("hb_form", "hb_diss"):
        pairs = ext[f"{mode}_pair"]
        oa = np.zeros(len(pairs), dtype=np.int32)
        ob = np.zeros(len(pairs), dtype=np.int32)
        vm = np.ones(len(pairs), dtype=bool)
        ka_res, ka_name = meta[f"{mode}_ka_res"], meta[f"{mode}_ka_name"]
        kb_res, kb_name = meta[f"{mode}_kb_res"], meta[f"{mode}_kb_name"]
        for i in range(len(pairs)):
            ka = (int(ka_res[i]), str(ka_name[i]))
            kb = (int(kb_res[i]), str(kb_name[i]))
            ia = omm_map.get(ka)
            ib = omm_map.get(kb)
            if ib is None and kb[1] in ("OT1", "OT2", "O", "OXT"):
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

    return modeller, ff, hb_maps, ct_pairs, sb_pairs, omm_heavy_by_res, omm_pos_by_res, omm_neg_by_res, max_err_all


def run_protein(prot, payload, outdir, platform_name, threads, n_seeds_target, do_long, long_ps):
    batch_f = outdir / f"{prot}_batch.npz"
    long_f = outdir / f"{prot}_long.npz"
    have_batch_seeds = 0
    batch_data = None
    if batch_f.exists():
        z = np.load(batch_f)
        have_batch_seeds = int(z["n_seeds"])
        batch_data = {k: z[k] for k in z.files}
    need_batch = have_batch_seeds < n_seeds_target
    need_long = do_long and not long_f.exists()
    if not need_batch and not need_long:
        return (prot, "cached", 0.0)

    t0 = time.time()
    modeller, ff, hb_maps, ct_pairs, sb_pairs, heavy_by_res, pos_by_res, neg_by_res, max_err = \
        build_simulation(prot, payload)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, DT_PS * unit.picoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    props = {}
    if platform_name == "CPU" and threads:
        props = {"Threads": str(threads)}
    sim = app.Simulation(modeller.topology, system, integrator, platform, props)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=200)
    min_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    pair_counts = {}
    ext = np.load(payload / f"{prot}.npz", allow_pickle=True)
    for m in MODES:
        pair_counts[m] = len(ext[f"{m}_pair"])

    def record_frame(pos_A, dists, s_i, t_i):
        for m in ("hb_form", "hb_diss"):
            oa, ob, _ = hb_maps[m]
            if len(oa):
                dists[m][s_i, t_i] = np.linalg.norm(pos_A[oa] - pos_A[ob], axis=1)
        for m in ("ct_form", "ct_diss"):
            dists[m][s_i, t_i] = compute_min_pair_dists(pos_A, ct_pairs[m], heavy_by_res, heavy_by_res)
        for m in ("sb_form", "sb_diss"):
            dists[m][s_i, t_i] = compute_min_pair_dists(pos_A, sb_pairs[m], pos_by_res, neg_by_res)

    steps_per_snap = int(SNAP_PS / DT_PS)
    status = []
    if need_batch:
        n_new = n_seeds_target - have_batch_seeds
        dists = {m: np.empty((n_new, 11, pair_counts[m]), dtype=np.float32) for m in MODES}
        for s_new in range(n_new):
            s_i = have_batch_seeds + s_new
            sim.context.setPositions(min_pos * unit.angstrom)
            sim.context.setVelocitiesToTemperature(300 * unit.kelvin, BASE_SEED + s_i * 997)
            record_frame(min_pos, dists, s_new, 0)
            for t_i in range(1, 11):
                sim.step(steps_per_snap)
                pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
                record_frame(pos, dists, s_new, t_i)
        out = {"protein": prot, "n_seeds": np.int32(n_seeds_target),
               "max_d0_err": np.float32(max_err), "platform": platform_name}
        for m in MODES:
            if batch_data is not None:
                out[f"{m}_dists"] = np.concatenate([batch_data[f"{m}_dists"], dists[m]], axis=0)
            else:
                out[f"{m}_dists"] = dists[m]
        for m in ("hb_form", "hb_diss"):
            out[f"{m}_valid"] = hb_maps[m][2]
        prev_elapsed = float(batch_data["elapsed_batch_s"]) if batch_data is not None else 0.0
        out["elapsed_batch_s"] = np.float32(prev_elapsed + (time.time() - t0))
        np.savez_compressed(batch_f, **out)
        status.append(f"batch seeds {have_batch_seeds}..{n_seeds_target - 1}")
    if need_long:
        n_snaps = int(long_ps / SNAP_PS)
        dists = {m: np.empty((1, n_snaps + 1, pair_counts[m]), dtype=np.float32) for m in MODES}
        sim.context.setPositions(min_pos * unit.angstrom)
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin, BASE_SEED)
        record_frame(min_pos, dists, 0, 0)
        for t_i in range(1, n_snaps + 1):
            sim.step(steps_per_snap)
            pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            record_frame(pos, dists, 0, t_i)
        out = {"protein": prot, "long_ps": np.float32(long_ps),
               "max_d0_err": np.float32(max_err), "platform": platform_name,
               "elapsed_long_s": np.float32(time.time() - t0)}
        for m in MODES:
            out[f"{m}_dists"] = dists[m]
        for m in ("hb_form", "hb_diss"):
            out[f"{m}_valid"] = hb_maps[m][2]
        np.savez_compressed(long_f, **out)
        status.append(f"long {long_ps:.0f} ps")
    return (prot, "ok (" + ", ".join(status) + f", d0_err={max_err:.5f}A)", time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proteins", required=True, help="comma-separated list")
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--long-ps", type=float, default=400.0)
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--no-batch", action="store_true")
    args = ap.parse_args()
    payload = Path(args.payload)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    prots = [p.strip() for p in args.proteins.split(",") if p.strip()]
    print(f"L3 GBn2 runner: {len(prots)} proteins, platform={args.platform}, "
          f"seeds->{args.seeds}, long={0 if args.no_long else args.long_ps} ps", flush=True)
    t_all = time.time()
    for i, p in enumerate(prots, 1):
        try:
            prot, status, dt = run_protein(p, payload, outdir, args.platform, args.threads,
                                           0 if args.no_batch else args.seeds,
                                           not args.no_long, args.long_ps)
            print(f"[{i}/{len(prots)}] {prot}: {status} ({dt:.1f}s)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{i}/{len(prots)}] {p}: FAIL {e}", flush=True)
    print(f"runner done in {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
