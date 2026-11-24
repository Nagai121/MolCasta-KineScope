"""Compare thermal-ensemble signals at identical validation origins."""
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
    dest=root/"ga337"
    a,y,b,p,_,g=load(VALID,root/"ga310")
    c,*_=load(VALID,root/"ga313")
    low,*_=load(VALID,dest/"275K")
    high,*_=load(VALID,dest/"330K")
    x=np.concatenate([build((n,1,1))[0] for n in VALID])
    response=[]
    for path in (root/"ga318/all-8.ubj",root/"ga330/all-9.ubj"):
        m=XGBRegressor(device="cpu")
        m.load_model(path)
        response.append(m.predict(x)+x[:,0]-b)
    st=.525*np.mean(response,axis=0)
    goal=.2*(field(a,10,"stable")+c.mean((0,1))-2*p)
    lo=low.mean((0,1))-p
    hi=high.mean((0,1))-p
    score={}
    for source,amount,gradient in product(("low","high","mean"),(0.,.1,.25,.4),(-.1,0.,.1)):
        signal={"low":lo,"high":hi,"mean":.5*(lo+hi)}[source]
        forecast=b+st+(1-amount)*goal+.4*amount*signal+gradient*(hi-lo)
        score[f"{source}/{amount}/{gradient}"]=estimates(forecast,y,b)
    keys=sorted(score,key=lambda n:-score[n]["residual_correlation"])
    record(dest/"development.json",dict(selected=keys[:6],trials=score))
    print([(n,score[n]) for n in keys[:6]],flush=True)


if __name__=="__main__":
    main()
