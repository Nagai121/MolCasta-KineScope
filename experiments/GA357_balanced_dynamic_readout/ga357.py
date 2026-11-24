"""GA357 balances training proteins and starting distances in dynamic readout."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga344 import inputs
from .ga348 import components, forecasts
from .ga_static_offset import VALID, estimates

R=Path("runs")

def main():
    dest=R/"ga357"
    dest.mkdir(exist_ok=True)
    train=json.loads((R/"ga344/development.json").read_bytes())["successful"]
    batches=[inputs((p,),R/"ga333",R/"ga344") for p in train]
    x,d,y=tuple(np.concatenate(v) for v in zip(*batches))
    weights=np.concatenate([np.full(len(v[2]),1./len(v[2])) for v in batches])
    weights/=weights.mean()
    dx=np.clip((x[:,0]-3.5)*2,0,4.99).astype(int)
    db=np.bincount(dx,minlength=5)
    distance=len(dx)/(5*db[dx])
    q,h,t=inputs(VALID,R/"ga310",R/"ga313")
    ref=components(VALID,R/"ga310",R/"ga313",("generated-6",),"all-9")
    yv,base,*_=ref
    np.testing.assert_allclose(yv,t,rtol=0,atol=3e-6)
    parent,pm=forecasts(ref,("generated-6",.25,.2))
    strength={}
    gpu_verified=False
    for (label,weight),depth in product(
            (("protein",weights),("distance",distance),("both",weights*distance)),(3,6)):
        mod=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,
                         n_estimators=300,learning_rate=.045,max_bin=96,
                         subsample=.85,colsample_bytree=.86,n_jobs=4)
        mod.fit(np.c_[x[:,:3],d],y-x[:,0],sample_weight=weight)
        config=json.loads(mod.get_booster().save_config())
        gpu_verified=config["learner"]["generic_param"]["device"].startswith("cuda")
        if not gpu_verified:
            raise RuntimeError("CUDA fit required")
        mod.set_params(device="cpu")
        p=mod.predict(np.c_[q[:,:3],h])+q[:,0]
        name=f"{label}-{depth}"
        for w in (.10,.20,.35):
            strength[f"{name}/{w}"]=estimates((1-w)*parent+w*p,t,base)
        mod.save_model(dest/f"{name}.ubj")
    selected=max(strength,key=lambda k:strength[k]["residual_correlation"])
    record(dest/"development.json",dict(selected=selected,trials=strength,parent=pm,
                                        pairs=len(t),train_proteins=train,cuda=gpu_verified))
    print(selected,strength[selected],pm,flush=True)

if __name__=="__main__":
    main()
