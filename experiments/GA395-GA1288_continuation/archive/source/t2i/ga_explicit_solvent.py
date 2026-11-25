"""GA1236: test a small explicit solvent box from the same static solute."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import openmm as mm
from openmm import app,unit

from .ga333 import WINDOW
from .ga_openmm_pilot import build
from .ga_replica_teacher import trajectory_rows
from .ga_static_offset import VALID,estimates

R=Path("runs")
OUT=R/"ga1236-explicit-solvent"


def prepare(name,origin,seed_solvent=False):
    # Only the static solute coordinates are used. Water and velocities are generated.
    raw,heavy,source,topology,_,_,base_sim,_,_=build(name,1)
    del base_sim
    xyz=raw.trajectory[origin].positions
    force=app.ForceField("amber14-all.xml","amber14/tip3pfb.xml")
    cpu=mm.Platform.getPlatformByName("CPU")
    model=app.Modeller(topology,xyz[heavy]*unit.angstrom)
    if origin!=49 or seed_solvent:
        import random
        random.seed(1010+origin)
    model.addHydrogens(force,platform=cpu)
    derived=np.searchsorted(heavy,source)
    np.testing.assert_array_equal(heavy[derived],source)
    heavy_index=[a.index for a in model.topology.atoms() if a.element!=app.element.hydrogen]
    mapping=np.asarray(heavy_index)[derived]
    model.addSolvent(force,model="tip3p",padding=1.*unit.nanometer, ionicStrength=.15*unit.molar)
    system=force.createSystem(model.topology,nonbondedMethod=app.PME,
                              nonbondedCutoff=.8*unit.nanometer,constraints=app.HBonds,
                              hydrogenMass=3.*unit.amu)
    first=mm.LangevinMiddleIntegrator(300.*unit.kelvin,1./unit.picosecond,4.*unit.femtoseconds)
    sim=app.Simulation(model.topology,system,first,mm.Platform.getPlatformByName("OpenCL"))
    sim.context.setPositions(model.positions)
    sim.minimizeEnergy(maxIterations=80)
    positions=sim.context.getState(positions=True).getPositions()
    return sim,positions,mapping


def simulate(name,origin=49,out=OUT,seed_solvent=False):
    f=out/f"{name}-{origin}.npz"
    if f.exists():
        return
    start=perf_counter()
    rows,t=trajectory_rows(name,1)
    pair=rows[t==origin]
    sim,pose,indices=prepare(name,origin,seed_solvent)
    a,b=indices[pair[:,:2].astype(np.intp)].T
    tracks=[]
    for branch in range(2):
        seed=1200+origin*3+branch
        sim.integrator.setRandomNumberSeed(seed)
        sim.context.reinitialize(preserveState=False)
        sim.context.setPositions(pose)
        sim.context.setVelocitiesToTemperature(sim.integrator.getTemperature(),seed)
        previous=0
        saved=[]
        for end in WINDOW:
            sim.step((end-previous)*1000//4)
            xyz=sim.context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            saved.append(np.linalg.norm(xyz[b]-xyz[a],axis=1))
            previous=end
        tracks.append(saved)
    array=np.asarray(tracks,np.float32)
    if not np.isfinite(array).all():
        raise RuntimeError(f"Nonfinite explicit solvent positions for {name}")
    np.savez_compressed(f,pair=pair,tracks=array,seconds=perf_counter()-start)
    print(name,origin,"explicit elapsed",perf_counter()-start,flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    for name in VALID:
        simulate(name)
    delta=[]
    for name in VALID:
        filename=f"{name}-49.npz"
        with np.load(OUT/filename) as actual,np.load(R/"ga360/amber"/filename) as control:
            np.testing.assert_array_equal(actual["pair"],control["pair"])
            delta.append(actual["tracks"].mean((0,1))-control["tracks"][:2,:len(WINDOW)].mean((0,1)))
    with np.load(R/"ga360/validation.npz") as z:
        selected=z["origin"]==49
        target,base=z["target"][selected],z["baseline"][selected]
    with np.load(R/"ga1077-reliability-width/GA1086.npz") as z:
        parent=z["prediction"][selected]
    offset=np.concatenate(delta)
    report={str(w):estimates(parent+w*offset,target,base) for w in (0.,.1,.25,.5,1.)}
    (OUT/"development.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":
    main()


