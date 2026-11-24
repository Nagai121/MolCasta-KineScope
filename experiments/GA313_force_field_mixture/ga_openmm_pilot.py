"""Matched-origin pilot for standard implicit-solvent MD (speed and ranking)."""
import argparse
from pathlib import Path
import time

import MDAnalysis as mda
import numpy as np
import openmm as mm
from openmm import app,unit

from tseries.data import TEST, registry, trajectory
from tseries.hb_explicit_hydrogen import source_plans
from tseries.hb_single_structure import ROOT
from .frontends import data
from .explore import record, score


def build(protein,replica, *, force_files=("amber14-all.xml","implicit/obc2.xml"), platform="OpenCL"):
    plan=source_plans(protein)[f"R{replica}"]
    reg=registry(ROOT,protein)
    tid=trajectory(protein,replica)
    atoms=reg.atoms(tid).to_pydict()
    src=np.asarray(atoms["source_atom"],dtype=np.intp)
    pdb=app.PDBFile(str(plan["topology"]))
    raw=mda.Universe(str(plan["topology"]),str(plan["xtc"]))
    if len(list(pdb.topology.atoms()))!=len(raw.atoms):
        raise ValueError("OpenMM and trajectory topology differ")
    field=app.ForceField(*force_files)
    model=app.Modeller(pdb.topology,pdb.positions)
    heavy=np.array([a.index for a in pdb.topology.atoms() if a.element != app.element.hydrogen])
    mapped=np.searchsorted(heavy,src)
    if not np.array_equal(heavy[mapped],src):
        raise ValueError("Heavy atom mapping differs")
    cpu=mm.Platform.getPlatformByName("CPU")
    # Rebuild topology hydrogens once, and rebuild their coordinates at each origin.
    model.delete([a for a in model.topology.atoms() if a.element==app.element.hydrogen])
    heavy_topology=model.topology
    model.addHydrogens(field,platform=cpu)
    system=field.createSystem(model.topology,nonbondedMethod=app.CutoffNonPeriodic,
                              nonbondedCutoff=1.6*unit.nanometers,constraints=app.HBonds,hydrogenMass=3*unit.amu)
    read_indices=np.array([a.index for a in model.topology.atoms() if a.element!=app.element.hydrogen])[mapped]
    integ=mm.LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,.004*unit.picoseconds)
    sim=app.Simulation(model.topology,system,integ,mm.Platform.getPlatformByName(platform))
    return raw,heavy,src,heavy_topology,read_indices,field,sim,reg,tid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--hybrid",action="store_true")
    args=parser.parse_args()
    dest=Path("runs/ga_hybrid300")
    dest.mkdir(parents=True,exist_ok=True)
    proteins=TEST if args.test else ("1yu5_X","1knt_A","3txs_C")
    parts,views,base=data(proteins)
    splits=np.split(base,np.cumsum([len(p["y"]) for p in parts])[:-1])
    responses=np.split(views["combined"],np.cumsum([len(p["y"]) for p in parts])[:-1])
    chosen=[]
    predictions={k:[] for k in ("frozen","closing","occupancy","pack")}
    costs=[]
    errors=[]
    for part,p0,static in zip(parts,splits,responses):
        if part["replica"]!=1:
            continue
        protein=part["protein"]
        try:
            raw,heavy,src,topo,indices,field,sim,reg,tid=build(protein,1)
        except Exception as e:
            errors.append(dict(protein=protein,error=str(e)))
            continue
        for a,b in list(zip(np.r_[0,part["ends"][:-1]],part["ends"]))[:3]:
            origin=int(part["origin"][a])
            block=next(reg.blocks(tid,start=origin,stop=origin+1))
            rp=raw.trajectory[origin].positions
            if not np.allclose(rp[src],block.positions[0],atol=.002,rtol=0):
                errors.append(dict(protein=protein,origin=origin,error="OpenMM source frame mismatch"))
                continue
            started=time.perf_counter()
            model=app.Modeller(topo,rp[heavy]*unit.angstrom)
            model.addHydrogens(field,platform=mm.Platform.getPlatformByName("CPU"))
            sim.context.setPositions(model.positions)
            sim.minimizeEnergy(maxIterations=25)
            sim.context.setVelocitiesToTemperature(300*unit.kelvin,500+origin)
            warmup=time.perf_counter()-started
            started=time.perf_counter()
            pair=part["pair"][a:b]
            left,right=indices[pair[:,:2].astype(np.intp)].T
            d=[]
            for _ in range(10):
                sim.step(2500)
                x=sim.context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
                d.append(np.linalg.norm(x[right]-x[left],axis=1))
            d=np.asarray(d)
            if not np.isfinite(d).all():
                raise RuntimeError("Non-finite production trajectory")
            duration=time.perf_counter()-started
            costs.append(dict(protein=protein,origin=origin,production_seconds=duration,setup_seconds=warmup))
            chosen.append(dict(protein=protein,y=part["y"][a:b],pair=pair))
            d0=pair[:,2]
            from scipy.special import expit,logit
            clipped=p0[a:b].clip(1e-6,1-1e-6)
            predictions["frozen"].append(clipped)
            # Fixed weights from the design, without pilot fitting.
            predictions["closing"].append(expit(logit(clipped)+.25*np.tanh((d0-d.mean(0))/.25)))
            predictions["occupancy"].append(expit(logit(clipped)+.25*((d<=3.5).mean(0)-.05)))
            predictions["pack"].append(d0-d.mean(0))
            if args.hybrid:
                from .ga_hybrid300 import frame_features
                _,_,_,r,_,seeded,_=frame_features(pair,static[a:b],d.T[:,None,:])
                for name,variance in (("distance_rod",r),("native_seeded_rod",seeded)):
                    p_enter=variance[:,10]
                    predictions.setdefault(name,[]).append(expit(logit(clipped)+.35*(p_enter-.05)))
                predictions.setdefault("native_plus_rod",[]).append(
                    expit(logit(clipped)+.25*np.tanh((d0-d.mean(0))/.25)
                          +.25*(seeded[:,10]-r[:,10])))
            print(protein,origin,duration,flush=True)
    from .gc_assist import matched
    result=dict(costs=costs,errors=errors,integrated_ps_each=100)
    if chosen:
        m={k:np.concatenate(v) for k,v in predictions.items()}
        cases=("frozen","closing") if args.test else ("frozen","closing","occupancy")
        if args.hybrid:
            cases+=("distance_rod","native_seeded_rod","native_plus_rod")
        result["scores"]={k:score(chosen,m[k]) for k in cases}
        signals=dict(frozen_t1=m["frozen"],closing=m["closing"])
        if not args.test:
            signals.update(occupancy=m["occupancy"],md_delta=m["pack"])
        result["conditional"]=matched(chosen,m["frozen"],signals)
    name="openmm_test.json" if args.test else ("openmm_hybrid.json" if args.hybrid else "openmm_pilot.json")
    record(dest/name,result)


if __name__=="__main__":
    main()
