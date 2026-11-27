"""CPU-only full 400 ps dual-probe engine benchmark; no model or weights."""
import argparse, concurrent.futures, io, json, os, platform, random, time, urllib.request
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import app, unit

BACKBONE=set("N CA C O OXT OT1 OT2 H HA HA2 HA3 H1 H2 H3".split())
CHANNELS={"A":(300.,1.,.35,1.,0),"B":(340.,.15,.15,.75,5000)}
def prepare(pdb_id):
    url=f"https://files.rcsb.org/download/{pdb_id}.pdb"
    raw=urllib.request.urlopen(url,timeout=60).read().decode()
    lines=[]
    for line in raw.splitlines():
        if line.startswith("ENDMDL"): break
        if line.startswith(("ATOM  ","HETATM")):
            if line[17:20]=="MSE":
                line="ATOM  "+line[6:17]+"MET"+line[20:]
                if line[12:16].strip()=="SE":
                    line=line[:12]+" SD "+line[16:76]+" S"+line[78:]
            elif not line.startswith("ATOM  "): continue
            lines.append(line)
        elif line.startswith("TER"): lines.append(line)
    pdb=app.PDBFile(io.StringIO("\n".join(lines)+"\nEND\n"))
    ff=app.ForceField("amber14-all.xml","implicit/gbn2.xml")
    m=app.Modeller(pdb.topology,pdb.positions)
    m.delete([a for a in m.topology.atoms() if a.element==app.element.hydrogen])
    random.seed(20261127)
    m.addHydrogens(ff,pH=7.0,platform=mm.Platform.getPlatformByName("CPU"))
    return m,ff

def make_system(m,ff,ch,dt):
    temp,friction,k,scale,_=CHANNELS[ch]
    kw=dict(nonbondedMethod=app.NoCutoff,constraints=app.HBonds)
    if dt>=3: kw["hydrogenMass"]=3*unit.amu
    sys=ff.createSystem(m.topology,**kw)
    kappa=[]
    for f in sys.getForces():
        if isinstance(f,mm.CustomGBForce):
            for i in range(f.getNumGlobalParameters()):
                if f.getGlobalParameterName(i)=="kappa":
                    f.setGlobalParameterDefaultValue(i,1.27)
                    kappa.append(1.27)
    restraint=mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k",k*418.4)
    for n in ("x0","y0","z0"): restraint.addPerParticleParameter(n)
    xyz=m.positions.value_in_unit(unit.nanometer)
    for a in m.topology.atoms():
        if a.name=="CA": restraint.addParticle(a.index,list(xyz[a.index]))
    sys.addForce(restraint)
    names=[a.name for a in m.topology.atoms()]
    if scale!=1:
        for f in sys.getForces():
            if isinstance(f,mm.PeriodicTorsionForce):
                for i in range(f.getNumTorsions()):
                    a,b,c,d,p,ph,kk=f.getTorsionParameters(i)
                    if not {names[j] for j in (a,b,c,d)}<=BACKBONE:
                        f.setTorsionParameters(i,a,b,c,d,p,ph,kk*scale)
    return sys,kappa

def context(system,ch,threads,seed,dt):
    temp,friction,_,_,_=CHANNELS[ch]
    integ=mm.LangevinMiddleIntegrator(temp*unit.kelvin,friction/unit.picosecond,dt*.001*unit.picoseconds)
    integ.setRandomNumberSeed(seed)
    ctx=mm.Context(system,integ,mm.Platform.getPlatformByName("CPU"),{"Threads":str(threads)})
    return ctx,integ

def seed_for(ch,s):
    return 20261127+CHANNELS[ch][-1]+101*s

def minimize(system,xyz,ch,threads,dt):
    t=time.perf_counter()
    ctx,integ=context(system,ch,threads,seed_for(ch,0),dt)
    ctx.setPositions(xyz*unit.nanometer)
    mm.LocalEnergyMinimizer.minimize(ctx,maxIterations=120)
    st=ctx.getState(getPositions=True,getEnergy=True)
    pos=st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    energy=st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del ctx,integ
    return pos,float(energy),time.perf_counter()-t

def integrate(ctx,integ,pos,ch,s,dt):
    temp=CHANNELS[ch][0]; seed=seed_for(ch,s)
    ctx.setPositions(pos*unit.nanometer)
    ctx.setVelocitiesToTemperature(temp*unit.kelvin,seed+7)
    frames=[]; t=time.perf_counter(); step_s=0.; read_s=0.
    for _ in range(10):
        u=time.perf_counter(); integ.step(round(10000/dt)); step_s+=time.perf_counter()-u
        u=time.perf_counter()
        arr=ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        read_s+=time.perf_counter()-u
        if not np.isfinite(arr).all(): raise ValueError("nonfinite coordinates")
        frames.append(arr.copy())
    st=ctx.getState(getEnergy=True)
    e=float(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    if not np.isfinite(e): raise ValueError("nonfinite energy")
    return np.asarray(frames),dict(channel=ch,seed=s,steps=round(100000/dt),elapsed_s=time.perf_counter()-t,step_s=step_s,read_s=read_s,final_energy=e)

def one_group(task):
    xml,ch,seeds,pos,threads,dt=task
    sys=mm.XmlSerializer.deserialize(xml)
    t=time.perf_counter()
    ctx,integ=context(sys,ch,threads,seed_for(ch,seeds[0]),dt)
    init=time.perf_counter()-t
    out={}
    for s in seeds:
        # Reusing a context preserves the sequential RNG stream; independent
        # contexts use a distinct seed at construction. Not bitwise equivalent.
        fr,stat=integrate(ctx,integ,pos,ch,s,dt)
        stat["context_init_s"]=init if s==seeds[0] else 0.
        out[(ch,s)]=(fr,stat)
    del ctx,integ
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pdb",required=True);p.add_argument("--workers",type=int,required=True)
    p.add_argument("--threads",type=int,required=True);p.add_argument("--out",required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    os.environ["OPENMM_CPU_THREADS"]=str(a.threads)
    start=time.perf_counter()
    m,ff=prepare(a.pdb)
    xyz=np.asarray(m.positions.value_in_unit(unit.nanometer))
    prep=time.perf_counter()-start
    results={};times={};mins={};energies={};kappa={};dt_used={}
    # Keep actual production-like four 100 ps trajectories. On failed 4 fs
    # channel, retry whole channel with 2 fs and record the fallback.
    for dt in (4.,2.):
        active=[ch for ch in CHANNELS if ch not in results]
        if not active: break
        tasks=[]; t=time.perf_counter()
        for ch in active:
            sys,k=make_system(m,ff,ch,dt)
            pos,e,ms=minimize(sys,xyz,ch,a.threads,dt)
            mins[ch]=ms;energies[ch]=e;kappa[ch]=k
            xml=mm.XmlSerializer.serialize(sys)
            groups=[[0],[1]] if a.workers==4 else [[0,1]]
            for group in groups: tasks.append((xml,ch,group,pos,a.threads,dt))
        t_sim=time.perf_counter(); current={ch:{} for ch in active}; errors={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            futures={ex.submit(one_group,task):task[1] for task in tasks}
            for f in concurrent.futures.as_completed(futures):
                ch=futures[f]
                try: current[ch].update(f.result())
                except Exception as e: errors[ch]=repr(e)
        times[str(dt)]={"build_min_s":t_sim-t,"trajectory_wall_s":time.perf_counter()-t_sim,"errors":errors}
        for ch in active:
            if ch not in errors and len(current[ch])==2:
                results[ch]=current[ch];dt_used[ch]=dt
        print(json.dumps(times[str(dt)]),flush=True)
    if len(results)!=2: raise RuntimeError(f"MD failed: {times}")
    frames=np.stack([results[ch][(ch,s)][0] for ch in CHANNELS for s in (0,1)])
    stats=[results[ch][(ch,s)][1] for ch in CHANNELS for s in (0,1)]
    ca=[x.index for x in m.topology.atoms() if x.name=="CA"]
    ca_frames=frames[:,:,ca,:]
    rmsf=np.sqrt(((ca_frames-ca_frames.mean(axis=1,keepdims=True))**2).sum(axis=-1).mean(axis=1))
    # Translation-invariant pair distance statistics for cross-condition checks.
    ii,jj=np.triu_indices(len(ca),1)
    dd=np.linalg.norm(ca_frames[:,:,ii,:]-ca_frames[:,:,jj,:],axis=-1)
    elapsed=time.perf_counter()-start
    summary=dict(pdb=a.pdb,atoms=len(xyz),residues=m.topology.getNumResidues(),cpu=platform.processor(),cpu_count=os.cpu_count(),
        openmm=mm.__version__,workers=a.workers,threads_per_context=a.threads,prep_s=prep,
        elapsed_s=elapsed,elapsed_ms=elapsed*1000,dt_used_fs=dt_used,kappa_globals=kappa,
        minimize_s=mins,min_energy=energies,phase_times=times,trajectories=stats,ca_rmsf_nm_mean=float(rmsf.mean()),
        ca_pair_distance_mean_nm=float(dd.mean()),snapshots=40)
    np.savez_compressed(out/"trajectory.npz",positions_nm=frames,ca_indices=ca,ca_rmsf_nm=rmsf,ca_pair_dist_mean_nm=dd.mean(axis=1),initial_positions_nm=xyz)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("RESULT "+json.dumps(summary),flush=True)
if __name__=="__main__":main()
