"""GA362 measures the matched structural control without the sequence embedding."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga348 import components, forecasts
from .ga_force import build
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga362"

def main():
    D.mkdir(exist_ok=True)
    train=tuple(json.loads((R/"ga318/development.json").read_bytes())["train"])
    assert not set(train)&set(VALID)
    jobs=[(n,r,4) for n in train for r in (1,2,3)]
    with ThreadPoolExecutor(4) as pool:
        groups=list(pool.map(build,jobs))
    x,y=tuple(np.concatenate(g) for g in zip(*groups))
    del groups
    v,t=tuple(np.concatenate(g) for g in zip(*(build((n,1,1)) for n in VALID)))
    dev=components(VALID,R/"ga310",R/"ga313",("generated-6",),"all-9")
    truth,base,*_=dev
    np.testing.assert_allclose(t,truth,rtol=0,atol=3e-6)
    anchor,pm=forecasts(dev,("generated-6",.25,.2))
    results={}
    for label,short in (("compact",True),("structure",False)):
        xx=x[:,:3] if short else x
        vv=v[:,:3] if short else v
        for depth in (4,7):
            head=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,max_bin=96,
                              learning_rate=.045,n_estimators=360,n_jobs=4,
                              subsample=.85,colsample_bytree=.85)
            head.fit(xx,y-x[:,0])
            if not json.loads(head.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("CUDA training required")
            head.set_params(device="cpu")
            p=head.predict(vv)+v[:,0]
            key=f"{label}-{depth}"
            results[key]={"unmixed":estimates(p,t,base)}
            for w in (.1,.2,.35,.5):
                results[key][str(w)]=estimates((1-w)*anchor+w*p,t,base)
            head.save_model(D/f"{key}.ubj")
            print(key,results[key],flush=True)
    record(D/"development.json",dict(trials=results,parent=pm,proteins=train,rows=len(y)))

if __name__=="__main__":
    main()


