"""GA390 reads directional hydrogen-bond geometry from generated heavy coordinates."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from xgboost import XGBRegressor

from .explore import record
from .ga377_score import load,minicalc
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga390"

def unit(v):
    return v/np.linalg.norm(v,axis=-1,keepdims=True).clip(1e-6)

def chemistry(initial,xyz,pair,residue,names,mc):
    a,b=pair[:,:2].astype(int).T
    groups=np.r_[a,b]
    _,near=cKDTree(initial).query(initial[groups],k=5)
    offset=initial[near]-initial[groups,None]
    dist=np.linalg.norm(offset,axis=2)
    connected=(dist>1.)&(dist<=1.85)
    geom=[]
    for branch in xyz:
        path=[]
        for frame in branch:
            donor=frame[b]-frame[a]
            length=np.linalg.norm(donor,axis=1)
            direction=unit(donor)
            local=frame[near]-frame[groups,None]
            inferred=unit(-(unit(local)*connected[:,:,None]).sum(1))
            donor_alignment=(inferred[:len(a)]*direction).sum(1)
            acceptor_alignment=-(inferred[len(a):]*direction).sum(1)
            h_donor=donor_alignment.clip(0,1)
            h_acceptor=acceptor_alignment.clip(0,1)
            align=h_donor*h_acceptor
            suitable=align*np.exp(-.5*((length-3.)/.6)**2)
            path.append(np.c_[donor_alignment,acceptor_alignment,align,suitable,
                              (length<=3.5)&(align>=.45),h_donor*(length-pair[:,2])])
        geom.append(path)
    g=np.asarray(geom)
    return np.c_[g.mean((0,1)),g.std((0,1)),g[:,-1].mean(0),
                 connected.sum(1).reshape(2,-1).T].astype(np.float32)

def main():
    D.mkdir(exist_ok=True)
    names=json.loads((R/"ga377/train/manifest.json").read_bytes())["proteins"]
    mc=minicalc(Path("D:/MiniCalc/local_frames.py"))
    x,d,g,y,_,_=load(names,"train",(49,149),mc,extra=chemistry)
    v,e,h,t,o,n=load(VALID,"validation",(49,149,249),mc,extra=chemistry)
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(n,z["proteins"])
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    trials={}
    for name,u,vv in (("control",d,e),("angular",np.c_[d,g[:,37:]],np.c_[e,h[:,37:]]),
                       ("all",np.c_[d,g],np.c_[e,h])):
        for depth in (3,6):
            head=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,max_bin=96,
                              n_estimators=300,learning_rate=.045,n_jobs=4)
            head.fit(np.c_[x[:,:3],u],y-x[:,0])
            if not json.loads(head.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("CUDA required")
            head.set_params(device="cpu")
            q=head.predict(np.c_[v[:,:3],vv])+v[:,0]
            tag=f"{name}-{depth}"
            head.save_model(D/f"{tag}.ubj")
            for weight in (.1,.2,.35):
                p=(1-weight)*parent+weight*q
                trials[f"{tag}/{weight}"]={k:estimates(p[m],t[m],base[m]) for k,m in masks.items()}
    best=max(trials,key=lambda k:trials[k]["all"]["residual_correlation"])
    record(D/"development.json",dict(best=best,trials=trials,train=names,rows=len(y)))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()
