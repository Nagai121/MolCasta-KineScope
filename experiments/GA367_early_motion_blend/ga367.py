"""GA367 tests whether early simulated response helps at later observed origins."""
from itertools import product
from pathlib import Path
import numpy as np

from .explore import record
from .ga314 import field
from .ga_native240 import SAMPLES
from .ga_static_offset import VALID, cohort, estimates

R=Path("runs")

def main():
    rows=cohort(VALID,3)
    packets={}
    for ff in ("amber","charmm"):
        parts=[]
        for row in rows:
            with np.load(R/"ga360"/ff/f"{row['protein']}-{row['origin']}.npz") as z:
                np.testing.assert_array_equal(row["pair"],z["pair"])
                parts.append(z["tracks"]-row["pair"][:,2])
        packets[ff]=np.concatenate(parts,axis=2)
    with np.load(R/"ga360/validation.npz") as z:
        y,b,parent,o=z["target"],z["baseline"],z["parent"],z["origin"]
    a,c=packets["amber"],packets["charmm"]
    grid=np.array(SAMPLES)
    old_a,old_c=field(a,10,"stable"),c.mean((0,1))
    pieces={}
    for horizon in (5,10,20,25,40,60):
        use=grid<=horizon
        pieces[horizon]=(a[:,use].mean((0,1))-old_a,c[:,use].mean((0,1))-old_c)
    masks={"all":np.ones(len(o),bool),"first":o==49,"later":o!=49}
    trials={}
    for horizon,target,weight in product(pieces,("amber","charmm","both"),(.2,.5,1.)):
        x,z=pieces[horizon]
        change={"amber":x,"charmm":z,"both":x+z}[target]
        p=parent+.16*weight*change
        trials[f"{horizon}/{target}/{weight}"]={k:estimates(p[m],y[m],b[m]) for k,m in masks.items()}
    reference={k:estimates(parent[m],y[m],b[m]) for k,m in masks.items()}
    best=max(trials,key=lambda k:trials[k]["all"]["residual_correlation"])
    dest=R/"ga367"
    dest.mkdir(exist_ok=True)
    record(dest/"development.json",dict(best=best,trials=trials,parent=reference))
    print(best,trials[best],reference,flush=True)

if __name__=="__main__":
    main()
