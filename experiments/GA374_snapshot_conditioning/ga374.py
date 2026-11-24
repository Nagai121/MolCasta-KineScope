"""GA374 normalizes motion readouts per static snapshot and distance stratum."""
from itertools import product
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga373 import assemble
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga374"

def normalize(d,dist,protein,origin,divide):
    result=np.empty_like(d)
    for n in np.unique(protein):
        for t in np.unique(origin[protein==n]):
            members=(protein==n)&(origin==t)
            x=d[members]
            q=dist[members]
            group=(q-3.5).clip(0,2.499)*2.
            bin_id=group.astype(int)
            loc=np.empty_like(x)
            for b in np.unique(bin_id):
                use=bin_id==b
                context=x[use] if use.sum()>=15 else x
                loc[use]=x[use]-context.mean(0)
                if divide:
                    loc[use]/=np.maximum(context.std(0),.05)
            result[members]=loc
    return result

def main():
    D.mkdir(exist_ok=True)
    names=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    x,d,_,y,o,g=assemble(names,(49,),R/"ga333",R/"ga355")
    v,e,_,t,vo,vg=assemble(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        np.testing.assert_array_equal(vo,z["origin"])
        np.testing.assert_array_equal(vg,z["proteins"])
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":vo==49,"later":vo!=49}
    trials={}
    for mode in ("raw","center","scale","joint"):
        if mode=="raw":
            a,b=d,e
        else:
            a=normalize(d,x[:,0],g,o,mode=="scale")
            b=normalize(e,v[:,0],vg,vo,mode=="scale")
            if mode=="joint":
                a,b=np.c_[d,a],np.c_[e,b]
        for depth in (2,4,6):
            head=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,
                              max_bin=96,n_estimators=300,learning_rate=.045,n_jobs=4)
            head.fit(np.c_[x[:,:3],a],y-x[:,0])
            if not json.loads(head.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("CUDA required")
            head.set_params(device="cpu")
            p=head.predict(np.c_[v[:,:3],b])+v[:,0]
            tag=f"{mode}-{depth}"
            head.save_model(D/f"{tag}.ubj")
            for weight in (.1,.2,.35):
                q=(1-weight)*parent+weight*p
                trials[f"{tag}/{weight}"]={k:estimates(q[m],t[m],base[m]) for k,m in masks.items()}
    best=max(trials,key=lambda n:trials[n]["all"]["residual_correlation"])
    record(D/"development.json",dict(best=best,trials=trials,train=names))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()

