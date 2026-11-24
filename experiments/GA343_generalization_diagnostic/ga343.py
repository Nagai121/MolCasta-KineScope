"""GA343 locates the generalization loss; these panels do not select weights."""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST, registry, trajectory
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga314 import field, load
from .ga_force import build
from .ga_native_offset import HOLD
from .ga_static_offset import cohort, estimates


def panel(names,amber,charmm):
    a,y,b,p,_,g=load(names,amber)
    c,*_=load(names,charmm)
    x=np.concatenate([build((n,1,1))[0] for n in names])
    priors=[]
    for file in ("runs/ga318/all-8.ubj","runs/ga330/all-9.ubj"):
        head=XGBRegressor(device="cpu")
        head.load_model(file)
        priors.append(head.predict(x)+x[:,0]-b)
    amber=field(a,10,"stable")-p
    charmm=c.mean((0,1))-p
    static=np.mean(priors,axis=0)
    z=.2*(amber+charmm)+.525*static
    groups={}
    for n in names:
        i=g==n
        groups[n]=dict(n=int(i.sum()),
                       amber=estimates(b[i]+.4*amber[i],y[i],b[i]),
                       charmm=estimates(b[i]+.4*charmm[i],y[i],b[i]),
                       static=estimates(b[i]+static[i],y[i],b[i]),
                       blend=estimates(b[i]+z[i],y[i],b[i]))
    for r in cohort(names,1):
        name=r["protein"]
        reg=registry(ROOT,name)
        origin=r["origin"]
        p0=next(reg.blocks(trajectory(name,1),start=origin,stop=origin+1)).positions[0]
        i,j=r["pair"][:,:2].astype(int).T
        error=np.abs(np.linalg.norm(p0[j]-p0[i],axis=1)-r["pair"][:,2])
        groups[name]["max_direct_coordinate_distance_difference"]=float(error.max())
        groups[name]["large_coordinate_distance_difference"]=int((error>.02).sum())
    return dict(proteins=groups,artifact_scope="analysis only",
                pooled={k:estimates(b+s,y,b) for k,s in (
                    ("amber",.4*amber),("charmm",.4*charmm),
                    ("motion",.2*(amber+charmm)),("static",static),("blend",z))})


def main():
    root=Path("runs")
    path=root/"ga321"
    reserved=tuple(json.loads((path/"reserve.json").read_bytes())["proteins"])
    result=dict(previous=panel(TEST+HOLD,root/"ga312",path),
                reserve=panel(reserved,path/"reserve/amber",path/"reserve"))
    dest=root/"ga343"
    dest.mkdir(parents=True,exist_ok=True)
    record(dest/"diagnostics.json",result)
    for k,v in result.items():
        print(k,{n:m["residual_correlation"] for n,m in v["pooled"].items()},flush=True)


if __name__=="__main__":
    main()
