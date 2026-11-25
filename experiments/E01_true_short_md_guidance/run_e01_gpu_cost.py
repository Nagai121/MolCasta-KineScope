"""E01 Part 2 (GPU MD Cost Benchmark — fixing G04):
Measures real wall-clock time on NVIDIA RTX 4060 Ti (OpenMM OpenCL GPU) for explicit-solvent all-atom MD:
  - Prep (addHydrogens + addSolvent TIP3P + 150 mM NaCl + createSystem, AMBER14)
  - Energy minimization
  - Standard equilibration: 100 ps NVT (50,000 steps) + 100 ps NPT (50,000 steps)
    (also records 10 ps NVT + 10 ps NPT intermediate timings to compare directly with G04)
  - Production: actual 0.1 ns (50,000 steps) AND actual 1.0 ns (500,000 steps) on 3 protein sizes:
      1) Small (~80 res, from ATLAS 100 set)
      2) Medium (~140 res, from ATLAS 100 set)
      3) 1mix_A (206 res — exact G04 reference protein)
"""
import json
import time
from pathlib import Path
import openmm as mm
import openmm.app as app
import openmm.unit as unit

ATLAS_ROOT = Path(r"d:\KineScope\experiments\shared_data\atlas")
OUT = Path(r"d:\KineScope\experiments\E01_true_short_md_guidance")


def pick_three_proteins():
    """Pick 1mix_A plus one small (~80 res) and one medium (~135-150 res) protein from downloaded ATLAS."""
    candidates = []
    for d in sorted(ATLAS_ROOT.iterdir()):
        if not d.is_dir() or not (d / "done.json").exists():
            continue
        meta = json.loads((d / "done.json").read_text(encoding="utf-8"))
        pdb_path = d / meta["pdb"]
        # Quick residue count via CA lines
        ca_count = sum(1 for ln in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                       if ln.startswith("ATOM") and ln[12:16].strip() == "CA")
        candidates.append((d.name, ca_count, pdb_path))
    by_name = {c[0]: c for c in candidates}
    chosen = []
    # Small closest to 80 res
    small = min((c for c in candidates if c[0] != "1mix_A" and 65 <= c[1] <= 95), key=lambda x: abs(x[1] - 80), default=None)
    if small:
        chosen.append(("small", small))
    # Medium closest to 140 res
    med = min((c for c in candidates if c[0] != "1mix_A" and 125 <= c[1] <= 155), key=lambda x: abs(x[1] - 140), default=None)
    if med:
        chosen.append(("medium", med))
    if "1mix_A" in by_name:
        chosen.append(("G04_ref_1mix_A", by_name["1mix_A"]))
    return chosen


def benchmark_protein(tag: str, prot: str, n_res: int, pdb_path: Path):
    print(f"\n--- GPU MD Cost Benchmark [{tag}] {prot} ({n_res} res) ---", flush=True)
    t_prep0 = time.time()
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    # Strip existing hydrogens/water if any, then add clean standard hydrogens + explicit solvent
    modeller.deleteWater()
    modeller.delete([a for a in modeller.topology.atoms() if a.element == app.element.hydrogen])
    modeller.addHydrogens(ff, pH=7.0)
    modeller.addSolvent(
        ff,
        model="tip3p",
        padding=1.0 * unit.nanometer,
        boxShape="dodecahedron",
        ionicStrength=0.15 * unit.molar,
    )
    n_atoms = modeller.topology.getNumAtoms()
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,  # standard HMR? Keep standard 1.0 amu (2 fs) for exact G04 comparability!
    )
    # Re-create with standard masses (no HMR, 2 fs exact match to G04/ATLAS)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName("OpenCL")
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    dev_name = platform.getPropertyValue(sim.context, "DeviceName")
    t_prep = time.time() - t_prep0
    print(f"  Prep: {t_prep:.2f}s | atoms={n_atoms:,} | GPU={dev_name}", flush=True)

    # Energy minimization
    t0 = time.time()
    sim.minimizeEnergy(maxIterations=2000)
    t_em = time.time() - t0
    print(f"  EM:   {t_em:.2f}s", flush=True)

    # NVT equilibration: measure 10 ps first, then remaining 90 ps (total 100 ps)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 20261126)
    t0 = time.time()
    sim.step(5000)  # 10 ps
    t_nvt_10 = time.time() - t0
    t0 = time.time()
    sim.step(45000)  # +90 ps = 100 ps
    t_nvt_100 = t_nvt_10 + (time.time() - t0)
    print(f"  NVT:  10ps={t_nvt_10:.2f}s, 100ps={t_nvt_100:.2f}s", flush=True)

    # NPT equilibration: add barostat, measure 10 ps + remaining 90 ps (total 100 ps)
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin))
    sim.context.reinitialize(preserveState=True)
    t0 = time.time()
    sim.step(5000)  # 10 ps
    t_npt_10 = time.time() - t0
    t0 = time.time()
    sim.step(45000)  # +90 ps = 100 ps
    t_npt_100 = t_npt_10 + (time.time() - t0)
    print(f"  NPT:  10ps={t_npt_10:.2f}s, 100ps={t_npt_100:.2f}s", flush=True)

    # Production: actual 0.1 ns (50,000 steps) AND actual 1.0 ns (500,000 steps)
    t0 = time.time()
    sim.step(50000)  # 0.1 ns
    t_prod_01 = time.time() - t0
    ns_per_day_01 = 0.1 / (t_prod_01 / 86400.0)
    print(f"  Prod 0.1 ns: {t_prod_01:.2f}s ({ns_per_day_01:.1f} ns/day)", flush=True)

    t0 = time.time()
    sim.step(450000)  # +0.9 ns = 1.0 ns total production
    t_prod_10 = t_prod_01 + (time.time() - t0)
    ns_per_day_10 = 1.0 / (t_prod_10 / 86400.0)
    print(f"  Prod 1.0 ns: {t_prod_10:.2f}s ({ns_per_day_10:.1f} ns/day)", flush=True)

    fixed_std = t_prep + t_em + t_nvt_100 + t_npt_100
    fixed_g04 = t_prep + t_em + t_nvt_10 + t_npt_10
    tot_01_std = fixed_std + t_prod_01
    tot_10_std = fixed_std + t_prod_10
    tot_01_g04 = fixed_g04 + t_prod_01
    tot_10_g04 = fixed_g04 + t_prod_10

    rec = {
        "tag": tag,
        "protein": prot,
        "n_res": n_res,
        "n_atoms": int(n_atoms),
        "device": dev_name,
        "t_prep_s": t_prep,
        "t_em_s": t_em,
        "t_nvt_10ps_s": t_nvt_10,
        "t_nvt_100ps_s": t_nvt_100,
        "t_npt_10ps_s": t_npt_10,
        "t_npt_100ps_s": t_npt_100,
        "t_prod_0.1ns_s": t_prod_01,
        "t_prod_1.0ns_s": t_prod_10,
        "ns_per_day_1.0ns": ns_per_day_10,
        "standard_200ps_eq": {
            "fixed_s": fixed_std,
            "total_0.1ns_s": tot_01_std,
            "total_1.0ns_s": tot_10_std,
            "ratio_0.1_over_1.0": tot_01_std / tot_10_std,
            "fixed_frac_at_0.1ns": fixed_std / tot_01_std,
        },
        "g04_20ps_eq": {
            "fixed_s": fixed_g04,
            "total_0.1ns_s": tot_01_g04,
            "total_1.0ns_s": tot_10_g04,
            "ratio_0.1_over_1.0": tot_01_g04 / tot_10_g04,
            "fixed_frac_at_0.1ns": fixed_g04 / tot_01_g04,
        },
    }
    print(f"  => Standard 200ps eq: 0.1ns={tot_01_std:.1f}s vs 1.0ns={tot_10_std:.1f}s | ratio={rec['standard_200ps_eq']['ratio_0.1_over_1.0']:.3f} (fixed share at 0.1ns = {rec['standard_200ps_eq']['fixed_frac_at_0.1ns']:.1%})", flush=True)
    return rec


def main():
    chosen = pick_three_proteins()
    assert chosen, "No downloaded ATLAS proteins found"
    results = []
    for tag, (prot, n_res, pdb_path) in chosen:
        try:
            results.append(benchmark_protein(tag, prot, n_res, pdb_path))
        except Exception as e:
            print(f"  [ERROR] {prot}: {e}", flush=True)
            results.append({"tag": tag, "protein": prot, "n_res": n_res, "error": str(e)})
    (OUT / "e01_gpu_cost_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved e01_gpu_cost_results.json", flush=True)


if __name__ == "__main__":
    main()
