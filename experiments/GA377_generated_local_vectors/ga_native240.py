"""Native implicit-solvent MD window, branching, and distance-ROD sweep."""
import json
import random
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit
from scipy.special import expit, logit

from tseries.data import TEST
from .explore import record, score
from .frontends import data
from .ga_hybrid300 import rod
from .ga_openmm_pilot import build

DEST=Path("runs/ga_native240")
DEV=("1yu5_X","1knt_A","3txs_C")
CONFIRM=("3mx7_A","3t92_A","4wrp_A")
WINDOWS=(5,10,25,50,100)
SAMPLES=(2,5,10,15,20,25,30,40,50,60,75,90,100)
MODES=("closing","occupancy","soft_occupancy","endpoint","mixed",
       "rod_seed","md_plus_rod","md_selective")
STRENGTHS=(.12,.25,.5)


def simulate(names,max_origins=3,replica=1, *, destination=DEST, samples=SAMPLES, branches=2, step_fs=4, minimize_iterations=25, setup_fn=build, preserve_hydrogens=False, parts_fn=data, velocity_hook=None, coordinate_cache=False):
    parts,sources,base=parts_fn(tuple(names))
    limits=np.cumsum([0,*[len(p["y"]) for p in parts]])
    out=[]
    phase=[]
    for i,p in enumerate(parts):
        if p["replica"]!=replica:
            continue
        protein=p["protein"]
        slots=list(zip(np.r_[0,p["ends"][:-1]],p["ends"]))[:max_origins]
        cache_name=protein if replica==1 else f"{protein}-R{replica}"
        missing=any(not (destination/f"{cache_name}-{p['origin'][a]}.npz").exists() for a,_ in slots)
        setup=setup_fn(protein,replica) if missing else None
        for a,b in slots:
            origin=int(p["origin"][a])
            f=destination/f"{cache_name}-{origin}.npz"
            pair=p["pair"][a:b]
            if not f.exists():
                raw,heavy,src,topo,indices,force,sim,reg,tid=setup
                block=next(reg.blocks(tid,start=origin,stop=origin+1))
                xyz=raw.trajectory[origin].positions
                if not np.allclose(xyz[src],block.positions[0],rtol=0,atol=.002):
                    raise RuntimeError("Source alignment changed")
                left,right=indices[pair[:,:2].astype(np.intp)].T
                begun=time.perf_counter()
                random.seed(1010+origin)
                model=app.Modeller(topo,xyz[heavy]*unit.angstrom)
                model.addHydrogens(force,platform=mm.Platform.getPlatformByName("CPU"))
                initial_coordinates = model.positions
                if preserve_hydrogens:
                    actual = {(int(i), name): k for k, (i, name) in enumerate(
                        zip(raw.atoms.resindices, raw.atoms.names))}
                    coords = np.array(model.positions.value_in_unit(unit.angstrom))
                    count = 0
                    for atom in model.topology.atoms():
                        if atom.element != app.element.hydrogen:
                            continue
                        k = actual.get((atom.residue.index, atom.name))
                        if k is not None:
                            coords[atom.index] = xyz[k]
                            count += 1
                    if not count:
                        raise RuntimeError("No measured hydrogens matched the generated topology")
                    print(protein,origin,"restored_hydrogens",count,flush=True)
                    initial_coordinates = coords * unit.angstrom
                sim.context.setPositions(initial_coordinates)
                for restraint in sim.system.getForces():
                    if restraint.getName() == "ga_origin_ca_position":
                        coordinates = initial_coordinates.value_in_unit(unit.nanometer)
                        for ri in range(restraint.getNumParticles()):
                            atom_id, _ = restraint.getParticleParameters(ri)
                            restraint.setParticleParameters(ri, atom_id, coordinates[atom_id])
                        restraint.updateParametersInContext(sim.context)
                sim.minimizeEnergy(maxIterations=minimize_iterations)
                initial=sim.context.getState(positions=True).getPositions()
                prep=time.perf_counter()-begun
                kick = velocity_hook(model.topology, initial) if velocity_hook else None
                tracks=[]
                coordinates=[]
                costs=[]
                contexts=[]
                for branch in range(branches):
                    started=time.perf_counter()
                    seed=1200+origin*3+branch+(replica-1)*100000
                    sim.integrator.setStepSize(step_fs*unit.femtoseconds)
                    sim.integrator.setRandomNumberSeed(seed)
                    sim.context.reinitialize(preserveState=False)
                    sim.context.setPositions(initial)
                    sim.context.setVelocitiesToTemperature(sim.integrator.getTemperature(),seed)
                    if kick is not None:
                        sim.context.setVelocities(kick(
                            sim.context.getState(velocities=True).getVelocities(asNumpy=True), branch))
                    contexts.append(time.perf_counter()-started)
                    d=[]
                    frames=[]
                    prior=0
                    elapsed=[]
                    for t in samples:
                        start=time.perf_counter()
                        sim.integrator.step((t-prior)*1000//step_fs)
                        x=sim.context.getState(positions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
                        d.append(np.linalg.norm(x[right]-x[left],axis=1))
                        if coordinate_cache:
                            frames.append(x[indices].astype(np.float32))
                        elapsed.append(time.perf_counter()-start)
                        prior=t
                    tracks.append(d)
                    if coordinate_cache:
                        coordinates.append(frames)
                    costs.append(np.cumsum(elapsed))
                tracks=np.asarray(tracks,dtype=np.float32)
                if not np.isfinite(tracks).all():
                    raise RuntimeError("Nonfinite native trajectory")
                with f.with_suffix(".tmp").open("wb") as tmp:
                    fields={"generated_xyz": np.asarray(coordinates), "origin_xyz": xyz[src].copy()} if coordinate_cache else {}
                    np.savez(tmp,pair=pair,tracks=tracks,costs=costs,prep=prep,contexts=contexts,**fields)
                f.with_suffix(".tmp").replace(f)
                print(protein,origin,"native-seconds",round(sum(c[-1] for c in costs),2),flush=True)
            with np.load(f) as z:
                if not np.array_equal(z["pair"],pair):
                    raise RuntimeError("Cached pairs differ")
                track=z["tracks"]
                runtime=z["costs"]
                preparation=float(z["prep"])
                contexts=z["contexts"].tolist()
            out.append(dict(protein=protein,origin=origin,pair=pair,y=p["y"][a:b],
                            frozen=base[limits[i]+a:limits[i]+b],tracks=track,
                            static=sources["combined"][limits[i]+a:limits[i]+b]))
            phase.append(dict(protein=protein,origin=origin,prep=preparation,
                              contexts=contexts,integrated_seconds=runtime.tolist()))
    return out,phase


def signals(row,window,branches):
    d=row["pair"][:,2]
    subset=row["tracks"][:branches,np.array(SAMPLES)<=window]
    closing=np.tanh((d-subset.mean((0,1)))/.25)
    occupy=4*((subset<=3.5).mean((0,1))-.05)
    soft=2*((expit((3.5-subset)/.20)).mean((0,1))-.10)
    end=np.tanh((d-subset[:,-1].mean(0))/.25)
    initial=row["static"][:,8:14]
    drift=(initial[:,3]-initial[:,0]).clip(-.5,.5)
    spread=(initial[:,4]*.25).clip(.003,.2)
    endpoint=row["pair"][:,:2].astype(np.intp).T
    r,_=rod(d,drift,spread,endpoint)
    rs,_=rod(d,.5*(drift+subset[:,-1].mean(0)-d),
             (.5*spread+.5*subset.var((0,1))).clip(.003,.25),endpoint)
    change=rs[:,10]-r[:,10]
    return dict(closing=closing,occupancy=occupy,soft_occupancy=soft,endpoint=end,
                mixed=.65*closing+.35*soft,rod_seed=4*(rs[:,10]-.05),
                md_plus_rod=closing+change,md_selective=closing*np.exp(-((d-4.2)/1.2)**2))


def evaluate(rows,seconds):
    baseline=np.concatenate([p["frozen"] for p in rows])
    anchor=logit(baseline.clip(1e-6,1-1e-6))
    base=score(rows,baseline)
    trials={}
    for window in WINDOWS:
        for branches in (1,2):
            fields=[signals(p,window,branches) for p in rows]
            for mode in MODES:
                value=np.concatenate([p[mode] for p in fields])
                for strength in STRENGTHS:
                    name=f"{window}ps/{branches}seed/{mode}/{strength}"
                    trials[name]=score(rows,expit(anchor+strength*value))
    record(DEST/"development.json",dict(baseline=base,trials=trials,timing=seconds))
    return sorted(trials,key=lambda n:(-trials[n]["auprc"],trials[n]["r80"]))


def check(rows,timing,selected):
    baseline=np.concatenate([p["frozen"] for p in rows])
    anchor=logit(baseline.clip(1e-6,1-1e-6))
    scores={}
    for name in selected:
        t,s,m,w=name.split("/")
        x=np.concatenate([signals(p,int(t[:-2]),int(s[0]))[m] for p in rows])
        scores[name]=score(rows,expit(anchor+float(w)*x))
    return dict(proteins=list(dict.fromkeys(p["protein"] for p in rows)),
                baseline=score(rows,baseline),selected=scores,timing=timing)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    dev,t=simulate(DEV)
    order=evaluate(dev,t)
    selected=list(dict.fromkeys((order[0],next(n for n in order if "/closing/" in n),
                                "100ps/1seed/closing/0.25",
                                next(n for n in order if "/md_plus_rod/" in n))))
    print("Development completed",len(order),"selected",selected,flush=True)
    record(DEST/"locked.json",dict(selection=selected))
    for proteins,label in ((CONFIRM,"new_native_panel"),(TEST,"previous_test_panel")):
        rows,timing=simulate(proteins)
        record(DEST/f"{label}.json",check(rows,timing,selected))
        print("Confirmation completed",label,flush=True)


if __name__=="__main__":
    main()

