"""GA358 learns short-motion corrections against a protein-out static baseline."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga344 import inputs
from .ga348 import components, forecasts
from .ga_static_offset import VALID, estimates, ridge

R=Path("runs")

def static(x):
    return np.c_[x[:,:3],x[:,0]**2]

def main():
    dest=R/"ga358"
    dest.mkdir(exist_ok=True)
    names=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    batches=[inputs((n,), R/"ga333", R/"ga355") for n in names]
    group=np.concatenate([np.full(len(y),n) for n,(_,_,y) in zip(names,batches)])
    x,d,y=tuple(np.concatenate(v) for v in zip(*batches))
    st=static(x)
    oof=np.empty_like(y)
    for n in names:
        mask=group==n
        oof[mask]=ridge(st[~mask],y[~mask]).predict(st[mask])
    q,h,t=inputs(VALID,R/"ga310",R/"ga313")
    dev=components(VALID,R/"ga310",R/"ga313",("generated-6",),"all-9")
    actual,base,*_=dev
    np.testing.assert_allclose(actual,t,rtol=0,atol=3e-6)
    parent,pm=forecasts(dev,("generated-6",.25,.2))
    projection=ridge(st,d)
    train_res=d-projection.predict(st)
    val_res=h-projection.predict(static(q))
    first=ridge(st,y).predict(static(q))
    trials={}
    for layout,depth in product(("dynamic","conditioned"), (2,4,6)):
        tx=train_res if layout=="dynamic" else np.c_[st,train_res]
        tv=val_res if layout=="dynamic" else np.c_[static(q),val_res]
        m=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,max_bin=96,
                       n_estimators=300,learning_rate=.045,n_jobs=4)
        m.fit(tx,y-oof)
        if not json.loads(m.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
            raise RuntimeError("CUDA fit required")
        m.set_params(device="cpu")
        p=first+m.predict(tv)
        for w in (.1,.2,.35,.5):
            trials[f"{layout}/{depth}/{w}"]=estimates((1-w)*parent+w*p,t,base)
        m.save_model(dest/f"{layout}-{depth}.ubj")
    best=max(trials,key=lambda k:trials[k]["residual_correlation"])
    record(dest/"development.json",dict(best=best,trials=trials,parent=pm,train_proteins=names,pairs=len(y)))
    print(best,trials[best],pm,flush=True)

if __name__=="__main__":
    main()
