"""Measure complementary error between scaled static models and generated motion."""
import json
from pathlib import Path
import itertools

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga314 import field, load
from .ga_force import build
from .ga_static_offset import VALID, estimates


def main():
    dest=Path("runs/ga330")
    a,y,b,p,f,g=load(VALID,Path("runs/ga310"))
    c,*_=load(VALID,Path("runs/ga313"))
    motion=.2*(field(a,10,"stable")+c.mean((0,1))-2*p)
    inputs=np.concatenate([build((n,1,1))[0] for n in VALID])
    ledger=json.loads((dest/"development.json").read_bytes())
    stat={}
    old=XGBRegressor(device="cpu")
    old.load_model("runs/ga318/all-8.ubj")
    f=old.predict(inputs)+inputs[:,0]-b
    for name in ledger["trials"]:
        model=XGBRegressor(device="cpu")
        model.load_model(dest/f"{name}.ubj")
        cols=slice(None) if name.startswith("all") else np.r_[0:3,169:inputs.shape[1]]
        stat[name]=model.predict(inputs[:,cols])+inputs[:,0]-b
    result={}
    for key,w,amount in itertools.product(stat,(0.,.25,.5,.75,1.),(.25,.525,.75)):
        delta=w*stat[key]+(1-w)*f
        metrics=estimates(b+motion+amount*delta,y,b)
        result[f"{key}/{w}/{amount}"]=metrics
    best=sorted(result,key=lambda k:-result[k]["residual_correlation"])
    record(dest/"fusion.json",dict(selected=best[0],trials=result))
    print([(k,result[k]) for k in best[:5]],flush=True)


if __name__=="__main__":
    main()
