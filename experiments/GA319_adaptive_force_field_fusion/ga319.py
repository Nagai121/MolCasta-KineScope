"""GA319: cross-protein validation of adaptive force-field fusion."""
import itertools
from pathlib import Path

import numpy as np

from .explore import record
from .ga314 import field, load
from .ga_static_offset import VALID, cohort, estimates, ridge


def inputs(names, amber_dir, charmm_dir):
    a,y,base,native_base,force,group=load(names,amber_dir)
    c,y2,b2,p2,f2,g2=load(names,charmm_dir)
    for v,w in ((y,y2),(base,b2),(native_base,p2),(force,f2),(group,g2)):
        np.testing.assert_array_equal(v,w)
    d=np.concatenate([r["pair"][:,2] for r in cohort(names,1)])
    amber=field(a,10,"stable")-native_base
    charmm=c.mean((0,1))-native_base
    spread=np.c_[a.mean(1).std(0),c.mean(1).std(0),np.abs(amber-charmm)]
    x=np.c_[amber,charmm,force]
    return x,spread,d,y,base,group


def design(x,spread,d,kind):
    if kind=="simple":
        return x
    if kind=="distance":
        z=(d-4.75)[:,None]
        return np.c_[x,x*z,x*z*z]
    if kind=="uncertainty":
        return np.c_[x,x/(.3+spread),x*spread]
    if kind=="union":
        return np.c_[design(x,spread,d,"distance"),design(x,spread,d,"uncertainty")[:,3:]]
    raise ValueError(kind)


def main():
    dest=Path("runs/ga319")
    dest.mkdir(parents=True,exist_ok=True)
    x,s,d,y,base,group=inputs(VALID,Path("runs/ga310"),Path("runs/ga313"))
    result={}
    y0=y-base
    for kind,alpha in itertools.product(("simple","distance","uncertainty","union"),(32.,128.,512.)):
        t=design(x,s,d,kind)
        out=np.empty_like(y)
        for n in VALID:
            q=group==n
            model=ridge(t[~q],y0[~q])
            model.steps[-1][1].set_params(alpha=alpha)
            model.fit(t[~q],y0[~q])
            out[q]=model.predict(t[q])
        metric=estimates(base+out,y,base)
        result[f"{kind}/{alpha}"]=metric
    result["fixed_equal_dynamic"]=estimates(base+.2*(x[:,0]+x[:,1])+.525*x[:,2],y,base)
    record(dest/"development.json",result)
    print(sorted(result.items(),key=lambda item:-item[1]["residual_correlation"])[:5],flush=True)


if __name__=="__main__":
    main()
