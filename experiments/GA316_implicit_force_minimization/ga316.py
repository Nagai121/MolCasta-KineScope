"""GA316: implicit-solvent energy gradients and equilibrium displacements only."""
from pathlib import Path
import random

import numpy as np
import openmm as mm
from openmm import app, unit

from .explore import record
from .ga314 import load
from .ga_openmm_pilot import build
from .ga_static_offset import VALID, cohort, estimates, ridge


def dynamics(rows):
    seq=[]
    for r in rows:
        raw,heavy,src,top,index,field,sim,reg,tid=build(r["protein"],1,platform="CPU")
        origin=r["origin"]
        xyz=raw.trajectory[origin].positions
        reference=next(reg.blocks(tid,start=origin,stop=origin+1)).positions[0]
        np.testing.assert_allclose(xyz[src],reference,rtol=0,atol=.002)
        random.seed(1010+origin)
        model=app.Modeller(top,xyz[heavy]*unit.angstrom)
        model.addHydrogens(field,platform=mm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(model.positions)
        i,j=index[r["pair"][:,:2].astype(int)].T
        d=r["pair"][:,2]
        axis=r["pair"][:,3:6]/d[:,None]
        cols=[]
        for n in (0,10,25,60):
            if n:
                sim.minimizeEnergy(maxIterations=n)
            state=sim.context.getState(positions=True,forces=True)
            x=state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            f=state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
            projected=(axis*(f[j]-f[i])).sum(1)
            cols.extend((np.linalg.norm(x[j]-x[i],axis=1)-d,
                         np.arcsinh(projected/200.)))
        seq.append(np.column_stack(cols))
    return np.concatenate(seq)


def main():
    dest=Path("runs/ga316")
    dest.mkdir(parents=True,exist_ok=True)
    rows=cohort(VALID,1)
    a,y,b,p,f,g=load(VALID,Path("runs/ga310"))
    z=dynamics(rows)
    np.savez(dest/"fields.npz",fields=z,proteins=g)
    ref=.4*(a.mean((0,1))-p)+.525*f
    result=dict(base=estimates(b+ref,y,b),results={})
    for stop in (2,4,6,8):
        h=np.zeros_like(y)
        for n in VALID:
            q=g==n
            model=ridge(z[~q,:stop], (y-b-ref)[~q])
            h[q]=model.predict(z[q,:stop])
        result["results"][str(stop)]=estimates(b+ref+h,y,b)
    record(dest/"development.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
