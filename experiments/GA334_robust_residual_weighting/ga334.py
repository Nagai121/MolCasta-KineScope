"""GA334 compares robust monotone and confidence-weighted residual readouts."""
from itertools import product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga314 import field, load
from .ga_force import build
from .ga_static_offset import VALID, estimates


def main():
    root=Path("runs")
    dest=root/"ga334"
    dest.mkdir(parents=True,exist_ok=True)
    a,y,b,p,_,g=load(VALID,root/"ga310")
    c,*_=load(VALID,root/"ga313")
    d=np.concatenate([build((n,1,1))[0] for n in VALID])
    heads=[]
    for file in (root/"ga318/all-8.ubj",root/"ga330/all-9.ubj"):
        m=XGBRegressor(device="cpu")
        m.load_model(file)
        heads.append(m.predict(d)+d[:,0]-b)
    static=.525*np.mean(heads,axis=0)
    u=field(a,10,"stable")-p
    v=c.mean((0,1))-p
    baseline=.2*(u+v)+static
    spread=np.hypot(a.mean(1).std(0),c.mean(1).std(0))
    error=np.abs(u-v)
    report={}
    for scale,power,distance,saturation in product(
            (0.,.25,.5,1.),(0.,.25,.5),(0.,.15,.35),(0.,.15,.35)):
        x=baseline
        if scale:
            x=x/(1+scale*(spread-np.median(spread))).clip(.5)
        if power:
            x=x/(1+power*(error-np.median(error))).clip(.5)
        if distance:
            x=x*(1+distance*(d[:,0]-4.75))
        if saturation:
            x=x/(1+saturation*np.abs(x))
        q=b+x
        o=estimates(q,y,b)
        o["macro"]=float(np.mean([estimates(q[g==n],y[g==n],b[g==n])["residual_correlation"]
                                  for n in VALID]))
        report[f"{scale}/{power}/{distance}/{saturation}"]=o
    ranked=sorted(report,key=lambda k:-report[k]["macro"])
    record(dest/"development.json",dict(selected=ranked[:8],trials=report))
    print([(k,report[k]) for k in ranked[:5]],flush=True)


if __name__=="__main__":
    main()
