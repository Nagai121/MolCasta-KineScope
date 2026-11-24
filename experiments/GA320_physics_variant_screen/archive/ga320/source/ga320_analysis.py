"""Compare refined and damped short simulations with the two-family mixture."""
from itertools import product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga314 import field, load
from .ga_force import build
from .ga_static_offset import VALID, estimates


def main():
    paths=Path("runs")
    a,y,b,p,f,g=load(VALID,paths/"ga310")
    c,v,base,proj,_,group=load(VALID,paths/"ga313")
    r,v2,b2,p2,_,g2=load(VALID,paths/"ga320")
    for first,second,third in ((y,v,v2),(b,base,b2),(p,proj,p2),(g,group,g2)):
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first,third)
    f_input=np.concatenate([build((n,1,1))[0] for n in VALID])
    s=XGBRegressor(device="cpu")
    s.load_model(paths/"ga318/all-8.ubj")
    force=s.predict(f_input)+f_input[:,0]-b
    dynamics={"amber4fs":field(a,10,"stable")-p,"charmm2fs":c.mean((0,1))-p,
              "amber2fs":r.mean((0,1))-p}
    if all((paths/"ga323"/f"{n}-49.npz").exists() for n in VALID):
        damp,*_=load(VALID,paths/"ga323")
        dynamics["damped2fs"]=damp.mean((0,1))-p
    if all((paths/"ga325"/f"{n}-49.npz").exists() for n in VALID):
        short,*_=load(VALID,paths/"ga325")
        dynamics["short16"]=short.mean((0,1))-p
    if all((paths/"ga326"/f"{n}-49.npz").exists() for n in VALID):
        restrained,*_=load(VALID,paths/"ga326")
        dynamics["restrained"]=restrained.mean((0,1))-p
    if all((paths/"ga328"/f"{n}-49.npz").exists() for n in VALID):
        third,*_=load(VALID,paths/"ga328")
        dynamics["amber19-gbn2"]=third.mean((0,1))-p
    if all((paths/"ga345"/f"{n}-49.npz").exists() for n in VALID):
        mode,*_=load(VALID,paths/"ga345")
        dynamics["elastic_mode"]=mode.mean((0,1))-p
    names=tuple(dynamics)
    out={}
    for weights in product((0.,.25,.5,.75,1.),repeat=len(names)):
        if sum(weights)!=1.:
            continue
        h=sum(w*dynamics[n] for n,w in zip(names,weights))
        for q in (.25,.4,.525,.75):
            guess=b+.4*h+q*force
            score=estimates(guess,y,b)
            score["macro"]=float(np.mean([estimates(guess[g==n],y[g==n],b[g==n])["residual_correlation"]
                                         for n in VALID]))
            out[f"{weights}/{q}"]=score
    order=sorted(out,key=lambda k:-out[k]["macro"])
    record(paths/"ga320/development.json",dict(fields=names,selected=order[:8],trials=out))
    print([(key,out[key]) for key in order[:5]],flush=True)


if __name__=="__main__":
    main()
