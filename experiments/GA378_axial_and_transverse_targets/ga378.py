"""GA378 splits the training distance change into axial and transverse terms."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import registry, trajectory
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga377_score import aligned, load, minicalc
from .ga_static_offset import VALID, estimates

R=Path("runs")

def observed(task):
    name,origin=task
    f=R/"ga377/train"/f"{name}-{origin}.npz"
    with np.load(f) as a:
        ref,pair=a["origin_xyz"],a["pair"]
    atom=registry(ROOT,name)
    tr=trajectory(name,1)
    it=next(atom.blocks(tr,start=origin-10,stop=origin+1,chunk_frames=11))
    if len(it.positions)!=11:
        raise RuntimeError("Expected the fixed 1 ns target history")
    ca=np.flatnonzero(np.asarray(atom.atoms(tr).to_pydict()["name"])=="CA")
    i,j=pair[:,:2].astype(int).T
    e=ref[j]-ref[i]
    e/=np.linalg.norm(e,axis=1,keepdims=True)
    motion=np.asarray([aligned(p,ref,ca) for p in it.positions])
    q=motion[:,j]-motion[:,i]
    a=np.einsum("fnd,nd->fn",q,e)
    longitudinal=a.mean(0)-pair[:,2]
    transverse=(np.linalg.norm(q,axis=2)-a).mean(0)
    return np.c_[longitudinal,transverse].astype(np.float32)

def model():
    return XGBRegressor(device="cuda",tree_method="hist",max_depth=4,max_bin=96,
                        n_estimators=300,learning_rate=.045,n_jobs=4)

def train(x,y):
    head=model()
    head.fit(x,y)
    if not json.loads(head.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
        raise RuntimeError("CUDA required")
    head.set_params(device="cpu")
    return head

def main():
    d=R/"ga378"
    d.mkdir(exist_ok=True)
    names=json.loads((R/"ga377/train/manifest.json").read_bytes())["proteins"]
    mc=minicalc(Path("D:/MiniCalc/local_frames.py"))
    x,a,g,y,_,_=load(names,"train",(49,149),mc)
    v,b,h,t,o,n=load(VALID,"validation",(49,149,249),mc)
    tasks=[(name,origin) for name in names for origin in (49,149)
           if (R/"ga377/train"/f"{name}-{origin}.npz").exists()]
    with ThreadPoolExecutor(4) as pool:
        truth=np.concatenate(list(pool.map(observed,tasks)))
    np.testing.assert_allclose(truth.sum(1),y-x[:,0],atol=.005,rtol=0)
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(n,z["proteins"])
        np.testing.assert_allclose(t,z["target"],atol=3e-6,rtol=0)
        base,parent=z["baseline"],z["parent"]
    a=np.c_[x[:,:3],a,g]
    b=np.c_[v[:,:3],b,h]
    direct=train(a,y-x[:,0])
    axial,perp=(train(a,truth[:,i]) for i in range(2))
    paths={"direct":direct.predict(b),"split":axial.predict(b)+perp.predict(b)}
    for label,head in (("direct",direct),("axial",axial),("transverse",perp)):
        head.save_model(d/f"{label}.ubj")
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    trials={}
    for k,p in paths.items():
        for amount in (.1,.2,.35):
            z=(1-amount)*parent+amount*(v[:,0]+p)
            trials[f"{k}/{amount}"]={name:estimates(z[m],t[m],base[m]) for name,m in masks.items()}
    best=max(trials,key=lambda k:trials[k]["all"]["residual_correlation"])
    record(d/"development.json",dict(trials=trials,best=best,rows=len(y),train=names,
                                     transverse_sd=float(truth[:,1].std())))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()
