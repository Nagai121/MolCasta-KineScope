"""GA383 uses direction-resolved motion to describe competing candidate edges."""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga377_score import load, minicalc
from .ga379 import moving
from .ga_static_offset import VALID, estimates

R=Path("runs")

def compare(initial,xyz,pair,residue,names,mc):
    a,b=pair[:,:2].astype(np.intp).T
    distance=pair[:,2]
    direction=(initial[b]-initial[a])/distance[:,None]
    actual=xyz[:,:,b]-xyz[:,:,a]
    radial=np.linalg.norm(actual,axis=-1)-distance
    shift=radial.mean((0,1))
    spread=radial.std((0,1))
    ward=(np.linalg.norm(actual,axis=-1)<=3.5).mean((0,1))
    x=np.c_[shift,spread,ward]
    weights=np.exp(-.5*((distance-3.5)/1.0)**2)
    edges=np.concatenate((a,b))
    direction=np.concatenate((direction,-direction))
    value=np.tile(x,(2,1))
    weight=np.tile(weights,2)
    m=np.bincount(edges,weights=weight,minlength=len(initial))
    sums=np.zeros((len(initial),x.shape[1]))
    tensor=np.zeros((len(initial),x.shape[1],3))
    np.add.at(sums,edges,weight[:,None]*value)
    np.add.at(tensor,edges,weight[:,None,None]*value[:,:,None]*direction[:,None,:])
    output=[]
    for ix,axis in ((a,direction[:len(a)]),(b,direction[len(a):])):
        denom=(m[ix]-weights).clip(.01)[:,None]
        scalar=(sums[ix]-weights[:,None]*x)/denom
        vector=(np.einsum("nfc,nc->nf",tensor[ix],axis)-weights[:,None]*x)/denom
        output.extend((scalar,vector))
    return np.c_[moving(initial,xyz,pair,residue,names,mc),*output]

def main():
    dest=R/"ga383"
    dest.mkdir(exist_ok=True)
    names=json.loads((R/"ga377/train/manifest.json").read_bytes())["proteins"]
    mc=minicalc(Path("D:/MiniCalc/local_frames.py"))
    x,d,g,y,_,_=load(names,"train",(49,149),mc,extra=compare)
    v,e,h,t,o,n=load(VALID,"validation",(49,149,249),mc,extra=compare)
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(n,z["proteins"])
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    trials={}
    for key,i in (("own",slice(0,67)),("graph",slice(None))):
        a,b=np.c_[x[:,:3],d,g[:,i]],np.c_[v[:,:3],e,h[:,i]]
        for depth in (3,6):
            m=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,max_bin=96,
                            n_estimators=300,learning_rate=.045,n_jobs=4)
            m.fit(a,y-x[:,0])
            if not json.loads(m.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("CUDA required")
            m.set_params(device="cpu")
            p=m.predict(b)+v[:,0]
            tag=f"{key}-{depth}"
            m.save_model(dest/f"{tag}.ubj")
            for w in (.1,.2,.35):
                q=parent*(1-w)+w*p
                trials[f"{tag}/{w}"]={k:estimates(q[j],t[j],base[j]) for k,j in masks.items()}
    best=max(trials,key=lambda k:trials[k]["all"]["residual_correlation"])
    record(dest/"development.json",dict(best=best,trials=trials,rows=len(y)))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()
