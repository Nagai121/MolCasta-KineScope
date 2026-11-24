"""GA373 learns the two-force-field ROD neighborhood response."""
from itertools import product
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import shard
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga308 import relations
from .ga333 import WINDOW
from .ga_force import build
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga373"

def assemble(names,origins,amber,charmm):
    blocks=[]
    for n in names:
        x,y=build((n,1,1+(max(origins)-49)//100))
        with np.load(shard(ROOT,n,1,100,8.)) as cache:
            pair=cache["pair"]
            keep=pair[:,2]<=6.
            r=pair[keep][:len(y)]
            time=cache["origin"][keep][:len(y)]
        np.testing.assert_array_equal(r[:,2],x[:,0])
        for origin in origins:
            mask=time==origin
            pair=r[mask]
            raw=[]
            graph=[]
            for folder in (amber,charmm):
                with np.load(folder/f"{n}-{origin}.npz") as z:
                    np.testing.assert_array_equal(z["pair"],pair)
                    h=z["tracks"][:2,:len(WINDOW)]-pair[:,2]
                raw.append(h)
                graph.append(relations([dict(pair=pair)],h.transpose(2,0,1)))
            a,b=raw
            u,v=a.mean(0).T,b.mean(0).T
            dyn=np.c_[u,v,u-v,a.std((0,1)),b.std((0,1)),a.mean(1).std(0),b.mean(1).std(0)]
            g=np.c_[graph[0],graph[1],graph[0]-graph[1]]
            blocks.append((x[mask],dyn,g,y[mask],time[mask],np.repeat(n,len(g))))
    return tuple(np.concatenate(s) for s in zip(*blocks))

def main():
    D.mkdir(exist_ok=True)
    names=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    x,d,g,y,_,_=assemble(names,(49,),R/"ga333",R/"ga355")
    v,e,h,t,o,p=assemble(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(p,z["proteins"])
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    results={}
    for view,depth in product(("own","relations","combined"),(2,4,6)):
        a,b={"own":(d,e),"relations":(g,h),"combined":(np.c_[d,g],np.c_[e,h])}[view]
        model=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,
                           n_estimators=300,learning_rate=.045,max_bin=96,n_jobs=4)
        model.fit(np.c_[x[:,:3],a],y-x[:,0])
        if not json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
            raise RuntimeError("CUDA required")
        model.set_params(device="cpu")
        predict=model.predict(np.c_[v[:,:3],b])+v[:,0]
        tag=f"{view}-{depth}"
        model.save_model(D/f"{tag}.ubj")
        for w in (.1,.2,.35,.5):
            q=(1-w)*parent+w*predict
            results[f"{tag}/{w}"]={n:estimates(q[m],t[m],base[m]) for n,m in masks.items()}
    best=max(results,key=lambda k:results[k]["all"]["residual_correlation"])
    ctrl={n:estimates(parent[m],t[m],base[m]) for n,m in masks.items()}
    record(D/"development.json",dict(best=best,trials=results,parent=ctrl,train=names,rows=len(y)))
    print(best,results[best],ctrl,flush=True)

if __name__=="__main__":
    main()

