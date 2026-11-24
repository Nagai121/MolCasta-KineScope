"""Surrogate-compatible projection of the final predicted distance change."""
from itertools import product
from pathlib import Path
import numpy as np

from .explore import record
from .ga331 import project
from .ga_static_offset import VALID, cohort, estimates

R=Path("runs")

def main():
    rows=cohort(VALID,3)
    with np.load(R/"ga360/validation.npz") as z:
        y,b,parent,o,p=(z[k] for k in ("target","baseline","parent","origin","proteins"))
    np.testing.assert_array_equal(o,np.concatenate([np.repeat(r["origin"],len(r["pair"])) for r in rows]))
    np.testing.assert_array_equal(p,np.concatenate([np.repeat(r["protein"],len(r["pair"])) for r in rows]))
    distance=np.concatenate([r["pair"][:,2] for r in rows])
    masks={"all":np.ones(len(y),bool),"first":o==49,"later":o!=49}
    trials={}
    for mode,damping in product(("full_motion","residual"),(.03,.15,.4,1.)):
        origin=distance if mode=="full_motion" else b
        source=parent-origin
        start=0
        values=[]
        for row in rows:
            stop=start+len(row["pair"])
            values.append(project(row,source[start:stop],np.ones(stop-start),damping))
            start=stop
        pred=origin+np.concatenate(values)
        for weight in (.1,.25,.5,1.):
            q=(1-weight)*parent+weight*pred
            trials[f"{mode}/{damping}/{weight}"]={k:estimates(q[m],y[m],b[m]) for k,m in masks.items()}
    d=R/"ga382"
    d.mkdir(exist_ok=True)
    best=max(trials,key=lambda k:trials[k]["all"]["residual_correlation"])
    record(d/"development.json",dict(best=best,trials=trials,parent={k:estimates(parent[m],y[m],b[m])
                                                  for k,m in masks.items()}))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()
