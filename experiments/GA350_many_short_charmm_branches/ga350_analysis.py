"""GA350 compares force-field branch counts under matched conditions."""
from itertools import combinations, product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga314 import field, load
from .ga_force import build
from .ga_static_offset import VALID, estimates


def main():
    root=Path("runs")
    dest=root/"ga350"
    a,y,b,p,_,g=load(VALID,root/"ga310")
    c,*_=load(VALID,root/"ga313")
    s,*_=load(VALID,root/"ga325")
    many,*_=load(VALID,dest)
    x=np.concatenate([build((n,1,1))[0] for n in VALID])
    static=[]
    for path in (root/"ga318/all-8.ubj",root/"ga330/all-9.ubj"):
        head=XGBRegressor(device="cpu")
        head.load_model(path)
        static.append(head.predict(x)+x[:,0]-b)
    prior=.525*np.mean(static,axis=0)
    amber=field(a,10,"stable")-p
    short=s.mean((0,1))-p
    charmm=c.mean((0,1))-p
    report={}
    for seed_count,variant,weight in product((2,4,8,16),("long-amber","short-amber"),(.25,.5,.75)):
        first=amber if variant=="long-amber" else short
        n=2 if seed_count==2 else min(seed_count,16)
        samples=np.asarray(list(combinations(range(16),n)))
        if len(samples)>100:
            rng=np.random.default_rng(350)
            samples=samples[rng.choice(len(samples),100,replace=False)]
        val=[]
        for ids in samples:
            other=many[ids].mean((0,1))-p
            pred=b+.4*((1-weight)*first+weight*other)+prior
            val.append(estimates(pred,y,b)["residual_correlation"])
        report[f"{seed_count}/{variant}/{weight}"]=dict(
            mean=float(np.mean(val)),minimum=float(np.min(val)),maximum=float(np.max(val)),
            subsets=len(val))
    base=estimates(b+.2*(amber+charmm)+prior,y,b)
    best=sorted(report,key=lambda k:-report[k]["mean"])
    record(dest/"development.json",dict(reference=base,selected=best[:6],trials=report))
    print([(k,report[k]) for k in best[:6]],flush=True)


if __name__=="__main__":
    main()
