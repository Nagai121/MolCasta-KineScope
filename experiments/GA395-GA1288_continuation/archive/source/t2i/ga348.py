"""GA348 selects an ensemble with scaled static and learned short contributions."""
from itertools import product
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST
from .explore import record
from .ga314 import field, load
from .ga344 import inputs
from .ga_native_offset import HOLD
from .ga_static_offset import VALID, estimates


ROOT=Path("runs")


def predict(path,x):
    model=XGBRegressor(device="cpu")
    model.load_model(path)
    return model.predict(x)+x[:,0]


def components(names,amber,charmm,short_keys,new_static):
    a,y,b,p,_,g=load(names,amber)
    c,y2,b2,p2,_,g2=load(names,charmm)
    for x,z in ((y,y2),(b,b2),(p,p2),(g,g2)):
        np.testing.assert_array_equal(x,z)
    x,d,t=inputs(names,amber,charmm)
    np.testing.assert_allclose(t,y,atol=3.e-6,rtol=0)
    s1=predict(ROOT/"ga318/all-8.ubj",x)-b
    s2=predict(ROOT/"ga330/all-9.ubj",x)-b
    s3=predict(ROOT/"ga347"/f"{new_static}.ubj",x)-b
    st=.5*(s1+s2)
    dynamic=.2*(field(a,10,"stable")+c.mean((0,1))-2*p)
    z={}
    for k in short_keys:
        use=np.c_[x[:,:3] if k.startswith("generated") else x,d]
        model=XGBRegressor(device="cpu")
        model.load_model(ROOT/"ga344"/f"{k}.ubj")
        z[k]=model.predict(use)+x[:,0]-b
    return y,b,g,st,s3,dynamic,z


def forecasts(data,rule):
    y,b,_,st,s3,dynamic,z=data
    tag,gn,gs=rule
    model=b+(1-gs)*(dynamic+.525*((1-gn)*st+gn*s3))+gs*z[tag]
    return model,estimates(model,y,b)


def main():
    dest=ROOT/"ga348"
    dest.mkdir(parents=True,exist_ok=True)
    short=json.loads((ROOT/"ga344/development.json").read_bytes())["trials"]
    scale=json.loads((ROOT/"ga347/development.json").read_bytes())["trials"]
    wide=max((k for k in scale if k.startswith("all-")),
             key=lambda k:scale[k]["residual_correlation"])
    names=tuple(short)
    dev=components(VALID,ROOT/"ga310",ROOT/"ga313",names,wide)
    y,b,_,s,_,dyn,_=dev
    reference=estimates(b+dyn+.525*s,y,b)
    rules=list(product(names,(0.,.25,.5,1.),(0.,.10,.20,.35)))
    rows=sorted(((r,forecasts(dev,r)[1]) for r in rules),
                key=lambda t:-t[1]["residual_correlation"])
    best,metric=rows[0]
    ledger=dict(selected=best,static_model=wide,validation=metric,reference=reference,trial_count=len(rows),
                trials=[dict(rule=list(r),metrics=s) for r,s in rows])
    record(dest/"development.json",ledger)
    print(best,metric,reference,flush=True)
    if metric["residual_correlation"]-reference["residual_correlation"]<.003:
        return
    other=tuple(json.loads((ROOT/"ga321/reserve.json").read_bytes())["proteins"])
    for case,n,p,q in (
        ("previous",TEST+HOLD,ROOT/"ga312",ROOT/"ga321"),
        ("reserve",other,ROOT/"ga321/reserve/amber",ROOT/"ga321/reserve")):
        data=components(n,p,q,(best[0],),wide)
        pred,metrics=forecasts(data,best)
        y,b,g,st,_,dyn,_=data
        out=dict(rule=best,pairs=len(y),metrics=metrics,reference=estimates(b+dyn+.525*st,y,b))
        record(dest/f"{case}.json",out)
        np.savez(dest/f"{case}.npz",predicted=pred,target=y,static=b,proteins=g)
        print(case,out,flush=True)


if __name__=="__main__":
    main()
