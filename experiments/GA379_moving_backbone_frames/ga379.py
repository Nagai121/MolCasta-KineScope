"""GA379 uses moving backbone frames to describe generated pair orientation."""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga377_score import load, minicalc
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga379"

def skeleton(residue,names):
    groups={}
    for i,(r,n) in enumerate(zip(residue,names)):
        groups.setdefault(r,{})[n]=i
    triples=[]
    valid=[]
    for i,r in enumerate(residue):
        index=groups[r]
        ok={"N","CA","C"}<=index.keys()
        triples.append(tuple(index[k] for k in ("N","CA","C")) if ok else (i,i,i))
        valid.append(ok)
    return np.asarray(triples),np.asarray(valid)

def local(position,left,right,frames,good,mc):
    n,ca,c=position[frames[left]].transpose(1,0,2)
    x=c-ca
    e1=x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-6)
    y=n-ca
    y-=np.sum(y*e1,axis=1,keepdims=True)*e1
    e2=y/np.linalg.norm(y,axis=1,keepdims=True).clip(1e-6)
    basis=np.stack((e1,e2,np.cross(e1,e2)),axis=2)
    state=mc.FrameState(np.zeros((len(left),3)),basis,())
    displacement=position[right]-position[left]
    out=mc.to_local_displacement(state,displacement)
    return np.where(good[left,None],out,0.)

def moving(initial,xyz,pair,residue,names,mc):
    a,b=pair[:,:2].astype(int).T
    frames,good=skeleton(residue,names)
    def project(x):
        return np.c_[local(x,a,b,frames,good,mc),local(x,b,a,frames,good,mc)]
    reference=project(initial)
    path=np.asarray([[project(x)-reference for x in branch] for branch in xyz])
    return np.c_[reference,path.mean((0,1)),path[:,-1].mean(0),
                 path.std((0,1)),path[:,:3].mean((0,1))].astype(np.float32)

def main():
    D.mkdir(exist_ok=True)
    mc=minicalc(Path("D:/MiniCalc/local_frames.py"))
    train=json.loads((R/"ga377/train/manifest.json").read_bytes())["proteins"]
    x,d,g,y,_,_=load(train,"train",(49,149),mc,extra=moving)
    v,e,h,t,o,n=load(VALID,"validation",(49,149,249),mc,extra=moving)
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(n,z["proteins"])
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    results={}
    for label,q,r in (("moving",np.c_[d,g[:,37:]],np.c_[e,h[:,37:]]),
                      ("combined",np.c_[d,g],np.c_[e,h])):
        for depth in (2,4,6):
            head=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,max_bin=96,
                              n_estimators=300,learning_rate=.045,n_jobs=4)
            head.fit(np.c_[x[:,:3],q],y-x[:,0])
            if not json.loads(head.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("CUDA required")
            head.set_params(device="cpu")
            p=head.predict(np.c_[v[:,:3],r])+v[:,0]
            tag=f"{label}-{depth}"
            head.save_model(D/f"{tag}.ubj")
            for amount in (.1,.2,.35):
                qv=(1-amount)*parent+amount*p
                results[f"{tag}/{amount}"]={k:estimates(qv[m],t[m],base[m]) for k,m in masks.items()}
    best=max(results,key=lambda k:results[k]["all"]["residual_correlation"])
    record(D/"development.json",dict(best=best,results=results,rows=len(y),train=train))
    print(best,results[best],flush=True)

if __name__=="__main__":
    main()
