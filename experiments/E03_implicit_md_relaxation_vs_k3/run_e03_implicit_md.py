"""E03: K3 (Static Elimination) x 100 ps Implicit-Solvent MD Relaxation (OpenMM GPU) — End-to-End Public Verification.

Zero-leakage protocol on unseen ATLAS test proteins (100% T1I2006-free):
  - Input: ONLY the single static origin frame (frame 10, t = 1.0 ns) of ATLAS R1.
    ZERO access to real history frames [0..9] or future frames (> 10).
  - For each protein, run 4 independent seeds x 100 ps of AMBER14 + GBn2 implicit-solvent MD
    on the RTX 4060 Ti GPU (OpenMM OpenCL), recording snapshots every 10 ps (10 frames/seed).
  - Measure on the exact same HB-formation candidate pairs (3.5 < d(t0) <= 6.0 A, diff residues):
      1. Finding 4 diagnostic: what fraction (R^2) of implicit-MD pair-mean distances is predictable
         from the 20 public static features (vs real explicit MD ~0.63 and KineScope Proposal 22 ~0.97)?
      2. Offset recovery: Pearson r between (implicit-MD residual) and (real 0-1 ns explicit-MD residual),
         and residual SD reduction when adding implicit-MD stats to the public static baseline (5-fold protein CV).
      3. Downstream guidance: AUPRC gain and recovery fraction (vs full real 1 ns explicit MD) at
         1, 5, 10, 30, 50, 90 ns horizons (5-fold protein CV, 100% public baseline).
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
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import GroupKFold

ATLAS_ROOT = Path(r"d:\KineScope\experiments\shared_data\atlas")
CACHE = Path(r"d:\KineScope\experiments\shared_data\extracted_100")
OUT = Path(r"d:\KineScope\experiments\E03_implicit_md_relaxation_vs_k3")
MD_CACHE = OUT / "implicit_md_cache"

HORIZON_NS = [1, 5, 10, 30, 50, 90]
N_SEEDS = 4
SIM_PS = 100.0
SNAP_PS = 10.0  # 10 snapshots per seed

REG_KW = dict(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=25,
    l2_regularization=0.2,
    random_state=20261126,
)
CLF_KW = dict(
    max_iter=160,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=25,
    l2_regularization=0.15,
    random_state=20261126,
)


def run_implicit_md_for_protein(prot: str):
    """Run 4 seeds x 100 ps GBn2 implicit-solvent MD from ATLAS R1 frame 10 ONLY."""
    out_npz = MD_CACHE / f"{prot}.npz"
    if out_npz.exists():
        return (prot, "cached", 0.0)

    t0 = time.time()
    pdir = ATLAS_ROOT / prot
    meta = json.loads((pdir / "done.json").read_text(encoding="utf-8"))
    pdb_path = pdir / meta["pdb"]
    r1_xtc = pdir / sorted(meta["xtc_files"])[0]

    # Load extracted candidate atom pairs for R1
    ext = np.load(CACHE / f"{prot}.npz", allow_pickle=True)
    pairs_mda = ext["r1_hb_form_pair"]  # (N, 2) atom indices in original PDB
    if len(pairs_mda) == 0:
        return (prot, "skip_empty", 0.0)

    # Extract frame 10 coordinates (t = 1.0 ns) to a temporary PDB with clean heavy atoms only
    u = mda.Universe(str(pdb_path), str(r1_xtc))
    u.trajectory[10]
    # Build mapping from original PDB atom index -> (resseq, icode, atom_name)
    # Use sequential residue index (0-based among protein residues) + atom name to match OpenMM Modeller
    heavy_atoms = u.select_atoms("not name H* and not name [0-9]H*")
    # Write frame-10 heavy-atom PDB
    tmp_pdb = MD_CACHE / f"_tmp_{prot}_f10.pdb"
    heavy_atoms.write(str(tmp_pdb))

    # Map original mda atom index -> (resindex_in_heavy, atom_name)
    # Build map from original u atom index to (resindex, name)
    orig_to_key = {}
    for res_i, res in enumerate(u.residues):
        for a in res.atoms:
            orig_to_key[int(a.index)] = (res_i, a.name.strip())

    # Prepare OpenMM system (AMBER14 + GBn2 implicit solvent)
    pdb_omm = app.PDBFile(str(tmp_pdb))
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = app.Modeller(pdb_omm.topology, pdb_omm.positions)
    # Strip any lingering hydrogens and add clean pH 7.0 hydrogens
    modeller.delete([a for a in modeller.topology.atoms() if a.element == app.element.hydrogen])
    modeller.addHydrogens(ff, pH=7.0)

    # Map (resindex, atom_name) -> OpenMM atom index
    omm_map = {}
    for res_i, res in enumerate(modeller.topology.residues()):
        for a in res.atoms():
            omm_map[(res_i, a.name.strip())] = a.index

    # Match every candidate pair to OpenMM atom indices
    omm_a = np.empty(len(pairs_mda), dtype=np.int32)
    omm_b = np.empty(len(pairs_mda), dtype=np.int32)
    valid_mask = np.ones(len(pairs_mda), dtype=bool)
    for idx in range(len(pairs_mda)):
        ka = orig_to_key.get(int(pairs_mda[idx, 0]))
        kb = orig_to_key.get(int(pairs_mda[idx, 1]))
        ia = omm_map.get(ka) if ka is not None else None
        ib = omm_map.get(kb) if kb is not None else None
        # Handle C-terminal O/OXT naming variation
        if ib is None and kb is not None and kb[1] in ("OT1", "OT2", "O", "OXT"):
            for alt in ("O", "OXT", "OT1", "OT2"):
                if (kb[0], alt) in omm_map:
                    ib = omm_map[(kb[0], alt)]
                    break
        if ia is None or ib is None:
            valid_mask[idx] = False
            omm_a[idx] = 0
            omm_b[idx] = 0
        else:
            omm_a[idx] = ia
            omm_b[idx] = ib

    # Sanity check: origin distance match between OpenMM starting positions and ATLAS frame 10
    init_pos_A = np.array([[p.x, p.y, p.z] for p in modeller.positions.value_in_unit(unit.angstrom)], dtype=np.float64)
    d0_omm = np.linalg.norm(init_pos_A[omm_a[valid_mask]] - init_pos_A[omm_b[valid_mask]], axis=1)
    d0_ref = ext["r1_hb_form_series"][valid_mask, 10].astype(np.float64)
    max_d0_err = float(np.abs(d0_omm - d0_ref).max())
    assert max_d0_err < 0.05, f"{prot}: origin distance mismatch {max_d0_err:.4f} A"

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName("OpenCL")
    sim = app.Simulation(modeller.topology, system, integrator, platform)

    n_snaps = int(SIM_PS / SNAP_PS)  # 10 snapshots
    steps_per_snap = int((SNAP_PS / 0.002))  # 5000 steps per 10 ps
    # Store pair distances: (N_SEEDS, n_snaps + 1, N_pairs) including frame 0 (minimized/origin)
    all_dists = np.empty((N_SEEDS, n_snaps + 1, len(pairs_mda)), dtype=np.float32)

    # First minimize once from frame 10 to relax any hydrogen placement clashes
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=200)
    min_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    for s_i in range(N_SEEDS):
        sim.context.setPositions(min_pos * unit.angstrom)
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 20261126 + s_i * 997)
        all_dists[s_i, 0] = np.linalg.norm(min_pos[omm_a] - min_pos[omm_b], axis=1)
        for t_i in range(1, n_snaps + 1):
            sim.step(steps_per_snap)
            pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            all_dists[s_i, t_i] = np.linalg.norm(pos[omm_a] - pos[omm_b], axis=1)

    if tmp_pdb.exists():
        tmp_pdb.unlink()

    # Summarize per-pair across seeds & snapshots (excluding frame 0 or including all)
    # Post-origin snapshots (10..100 ps across 4 seeds = 40 samples per pair)
    post = all_dists[:, 1:, :].reshape(N_SEEDS * n_snaps, len(pairs_mda))
    gen_mean = post.mean(axis=0).astype(np.float32)
    gen_std = post.std(axis=0).astype(np.float32)
    gen_min = post.min(axis=0).astype(np.float32)
    gen_max = post.max(axis=0).astype(np.float32)
    gen_frac35 = (post <= 3.5).mean(axis=0).astype(np.float32)
    gen_frac45 = (post <= 4.5).mean(axis=0).astype(np.float32)
    gen_end_minus_start = (all_dists[:, -1, :].mean(axis=0) - d0_ref).astype(np.float32)
    # Also compute mean per seed to measure inter-seed variance of implicit MD
    seed_means = all_dists[:, 1:, :].mean(axis=1).astype(np.float32)  # (N_SEEDS, N)

    np.savez_compressed(
        out_npz,
        protein=prot,
        valid_mask=valid_mask,
        max_d0_err=np.float32(max_d0_err),
        gen_mean=gen_mean,
        gen_std=gen_std,
        gen_min=gen_min,
        gen_max=gen_max,
        gen_frac35=gen_frac35,
        gen_frac45=gen_frac45,
        gen_end_minus_start=gen_end_minus_start,
        seed_means=seed_means,
        elapsed_s=np.float32(time.time() - t0),
    )
    return (prot, f"ok (N={valid_mask.sum()}/{len(valid_mask)}, d0_err={max_d0_err:.4f}A)", time.time() - t0)


def oof_reg(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(**REG_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def oof_clf(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingClassifier(**CLF_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def analyze(prots):
    Xs, K302s, S_real, Y_real, G_feats, Ps = [], [], [], [], [], []
    times = []
    for p in prots:
        md_f = MD_CACHE / f"{p}.npz"
        if not md_f.exists():
            continue
        zm = np.load(md_f, allow_pickle=True)
        ze = np.load(CACHE / f"{p}.npz", allow_pickle=True)
        vm = zm["valid_mask"].astype(bool)
        if vm.sum() == 0:
            continue
        X30 = ze["r1_hb_form_X"][vm]
        Xs.append(X30[:, :20])
        K302s.append(X30[:, 20:30])
        S_real.append(ze["r1_hb_form_series"][vm])
        Y_real.append(ze["r1_hb_form_y"][vm])
        gf = np.column_stack([
            zm["gen_mean"][vm],
            zm["gen_std"][vm],
            zm["gen_min"][vm],
            zm["gen_max"][vm],
            zm["gen_end_minus_start"][vm],
            zm["gen_frac35"][vm],
            zm["gen_frac45"][vm],
        ])
        G_feats.append(gf)
        Ps.append(np.full(vm.sum(), p, dtype=object))
        times.append(float(zm["elapsed_s"]))

    X_pub = np.concatenate(Xs).astype(np.float64)
    X_k302 = np.concatenate(K302s).astype(np.float64)
    S = np.concatenate(S_real).astype(np.float64)
    Y = np.concatenate(Y_real).astype(np.uint8)
    G = np.concatenate(G_feats).astype(np.float64)
    P = np.concatenate(Ps)

    real_mean = S.mean(axis=1)
    gen_mean = G[:, 0]

    # 1. Static explainability (Finding 4 diagnostic on real explicit MD vs 100ps implicit MD)
    pred_real_from_static = oof_reg(X_pub, real_mean, P)
    pred_gen_from_static = oof_reg(X_pub, gen_mean, P)
    real_resid = real_mean - pred_real_from_static
    gen_resid = gen_mean - pred_gen_from_static

    r2_real_static = float(r2_score(real_mean, pred_real_from_static))
    r2_gen_static = float(r2_score(gen_mean, pred_gen_from_static))
    resid_corr_p, _ = pearsonr(gen_resid, real_resid)
    resid_corr_s, _ = spearmanr(gen_resid, real_resid)

    # 2. Predicting real_mean from static + implicit MD (and vs static + K3-02)
    pred_real_k302 = oof_reg(np.column_stack([X_pub, X_k302]), real_mean, P)
    pred_real_gmean = oof_reg(np.column_stack([X_pub, gen_mean[:, None]]), real_mean, P)
    pred_real_gall = oof_reg(np.column_stack([X_pub, G]), real_mean, P)

    # Within-protein shuffle control for G
    rng = np.random.default_rng(20261126)
    G_sh = G.copy()
    for pr in np.unique(P):
        idx = np.where(P == pr)[0]
        G_sh[idx] = G_sh[idx][rng.permutation(len(idx))]
    pred_real_gshuf = oof_reg(np.column_stack([X_pub, G_sh]), real_mean, P)

    # 3. Downstream H-bond formation AUPRC (1..90 ns)
    real_h6 = np.column_stack([
        S[:, 0], S.mean(1), S.std(1), S.min(1), S[:, -1] - S[:, 0], (S <= 3.5).mean(1)
    ])
    downstream = {}
    print(f"\n=== E03 Analysis ({len(np.unique(P))} unseen ATLAS proteins, N={len(real_mean):,} R1 pairs) ===", flush=True)
    print(f"Static explainability R^2: real explicit MD = {r2_real_static:.4f} | 100ps implicit MD = {r2_gen_static:.4f}", flush=True)
    print(f"Residual correlation (implicit MD resid vs real explicit MD resid): Pearson={resid_corr_p:.4f}, Spearman={resid_corr_s:.4f}", flush=True)
    print(f"Real-mean residual SD: static20={real_resid.std():.4f} A | +K3-02={(real_mean-pred_real_k302).std():.4f} A | "
          f"+impMD_mean={(real_mean-pred_real_gmean).std():.4f} A | +impMD_all7={(real_mean-pred_real_gall).std():.4f} A | "
          f"shuffle={(real_mean-pred_real_gshuf).std():.4f} A", flush=True)

    for hi, h_ns in enumerate(HORIZON_NS):
        y = Y[:, hi]
        p_s = oof_clf(X_pub, y, P)
        p_real = oof_clf(np.column_stack([X_pub, real_h6]), y, P)
        p_gm = oof_clf(np.column_stack([X_pub, gen_mean[:, None]]), y, P)
        p_ga = oof_clf(np.column_stack([X_pub, G]), y, P)
        p_sh = oof_clf(np.column_stack([X_pub, G_sh]), y, P)
        ap_s = float(average_precision_score(y, p_s))
        ap_r = float(average_precision_score(y, p_real))
        ap_gm = float(average_precision_score(y, p_gm))
        ap_ga = float(average_precision_score(y, p_ga))
        ap_sh = float(average_precision_score(y, p_sh))
        g_real = ap_r - ap_s
        g_gm = ap_gm - ap_s
        g_ga = ap_ga - ap_s
        r_resid_y = float(pearsonr(real_resid, y)[0])
        g_resid_y = float(pearsonr(gen_resid, y)[0])
        downstream[str(h_ns)] = {
            "ap_static20": ap_s,
            "ap_real_1ns_6stats": ap_r,
            "ap_impMD_mean": ap_gm,
            "ap_impMD_all7": ap_ga,
            "ap_impMD_shuffle": ap_sh,
            "gain_real": g_real,
            "gain_impMD_mean": g_gm,
            "gain_impMD_all7": g_ga,
            "recovery_impMD_mean": float(g_gm / g_real) if g_real > 1e-6 else float("nan"),
            "recovery_impMD_all7": float(g_ga / g_real) if g_real > 1e-6 else float("nan"),
            "corr_real_resid_vs_label": r_resid_y,
            "corr_impMD_resid_vs_label": g_resid_y,
        }
        print(f"  H={h_ns:2d}ns | static={ap_s:.4f} real1ns={ap_r:.4f} (+{g_real:.4f}) | "
              f"impMD_all7={ap_ga:.4f} (+{g_ga:.4f}, rec={g_ga/g_real:.3f}) | shuffle={ap_sh:.4f} | "
              f"resid_y_r: real={r_resid_y:.4f} vs impMD={g_resid_y:.4f}", flush=True)

    res = {
        "n_proteins": int(len(np.unique(P))),
        "proteins": sorted(np.unique(P).tolist()),
        "n_candidates": int(len(real_mean)),
        "n_seeds": N_SEEDS,
        "sim_ps_per_seed": SIM_PS,
        "mean_gpu_sec_per_protein": float(np.mean(times)),
        "total_gpu_sec": float(np.sum(times)),
        "static_explainability_r2": {
            "real_explicit_1ns": r2_real_static,
            "implicit_md_100ps": r2_gen_static,
        },
        "offset_recovery": {
            "residual_pearson_r": float(resid_corr_p),
            "residual_spearman_r": float(resid_corr_s),
            "static20_resid_sd": float(real_resid.std()),
            "static20_plus_k302_resid_sd": float((real_mean - pred_real_k302).std()),
            "static20_plus_impMD_mean_resid_sd": float((real_mean - pred_real_gmean).std()),
            "static20_plus_impMD_all7_resid_sd": float((real_mean - pred_real_gall).std()),
            "static20_plus_impMD_shuffle_resid_sd": float((real_mean - pred_real_gshuf).std()),
            "r2_real_mean_with_impMD_all7": float(r2_score(real_mean, pred_real_gall)),
        },
        "downstream_hb_formation": downstream,
    }
    (OUT / "e03_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nSaved e03_results.json", flush=True)


def main():
    MD_CACHE.mkdir(parents=True, exist_ok=True)
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    files = sorted(CACHE.glob("*.npz"))
    # Select n_target proteins evenly spaced across sorted 100 unseen ATLAS proteins
    idxs = np.linspace(0, len(files) - 1, min(n_target, len(files)), dtype=int)
    prots = [files[i].stem for i in idxs]
    print(f"E03: running 4x100ps OpenMM GBn2 GPU implicit MD on {len(prots)} unseen ATLAS proteins...", flush=True)
    t_all = time.time()
    ok = fail = 0
    done_prots = []
    for i, p in enumerate(prots, 1):
        try:
            prot, status, dt = run_implicit_md_for_protein(p)
            print(f"[{i}/{len(prots)}] {prot}: {status} ({dt:.1f}s)", flush=True)
            ok += 1
            done_prots.append(prot)
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(prots)}] {p}: FAIL {e}", flush=True)
    print(f"Implicit MD generation done in {time.time() - t_all:.1f}s | ok={ok} fail={fail}", flush=True)
    assert len(done_prots) >= 15, f"Too few completed proteins: {len(done_prots)}"
    analyze(done_prots)


if __name__ == "__main__":
    main()
