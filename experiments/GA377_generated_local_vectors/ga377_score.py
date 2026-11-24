"""GA377 compares generated distances and MiniCalc-frame generated motion."""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import registry, shard, trajectory
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga_force import build
from .ga_local_relations import minicalc
from .ga_static_offset import VALID, estimates

R=Path("runs")
D=R/"ga377"

def aligned(x,origin,anchors):
    """Use a proper rotation fitted on the generated backbone."""
    q=x[anchors]
    p=origin[anchors]
    xbar,ybar=q.mean(0),p.mean(0)
    u,_,v=np.linalg.svd((q-xbar).T@(p-ybar))
    u[:,-1]*=np.sign(np.linalg.det(u@v))
    return (x-xbar)@(u@v)+ybar

def vector_features(initial,frames,pair,residue,ca,mc):
    a,b=pair[:,:2].astype(int).T
    original=initial[b]-initial[a]
    np.testing.assert_allclose(np.linalg.norm(original,axis=1),pair[:,2],rtol=0,atol=.02)
    e1=original/np.linalg.norm(original,axis=1,keepdims=True)
    ca_at={residue[k]:k for k in ca}
    left_ca=np.array([ca_at.get(residue[k],k) for k in a])
    right_ca=np.array([ca_at.get(residue[k],k) for k in b])
    helpers=[initial[left_ca]-initial[a],initial[right_ca]-initial[b]]
    helpers.extend(np.broadcast_to(initial[k]-initial[ca[0]],e1.shape) for k in ca[1:5])
    helper=np.zeros_like(e1)
    selected=np.zeros(len(pair),bool)
    for q in helpers:
        projected=q-np.sum(q*e1,axis=1,keepdims=True)*e1
        choose=(~selected)&(np.linalg.norm(projected,axis=1)>.04)
        helper[choose]=projected[choose]
        selected|=choose
    if not np.all(selected):
        raise RuntimeError("Degenerate local coordinates")
    e2=helper/np.linalg.norm(helper,axis=1,keepdims=True)
    basis=np.stack((e1,e2,np.cross(e1,e2)),axis=2)
    state=mc.FrameState(np.zeros((len(pair),3)),basis,())
    signals=[]
    for branch in frames:
        path=[]
        for snap in branch:
            motion=aligned(snap,initial,ca)-initial
            rel=mc.to_local_displacement(state,motion[b]-motion[a])
            left=mc.to_local_displacement(state,motion[a])
            right=mc.to_local_displacement(state,motion[b])
            path.append(np.c_[rel,left,right])
        signals.append(path)
    x=np.asarray(signals)
    return np.c_[x.mean((0,1)),x[:,-1].mean(0),x.std((0,1)),x[:,:3].mean((0,1)),
                 np.linalg.norm(x[:,:,:,:3],axis=-1).mean((0,1))]

def load(names,category,origins,mc,extra=None):
    values=[]
    for name in names:
        x,y=build((name,1,len(origins)))
        atoms=registry(ROOT,name).atoms(trajectory(name,1)).to_pydict()
        residue=np.asarray(atoms["residue_index"])
        ca=np.flatnonzero(np.asarray(atoms["name"])=="CA")
        with np.load(shard(ROOT,name,1,100,8.)) as source:
            p=source["pair"]
            keep=p[:,2]<=6.
            pair,starts=p[keep][:len(y)],source["origin"][keep][:len(y)]
        np.testing.assert_array_equal(pair[:,2],x[:,0])
        for origin in origins:
            file=D/category/f"{name}-{origin}.npz"
            if not file.exists() and category=="train":
                continue
            mask=starts==origin
            with np.load(file) as z:
                np.testing.assert_array_equal(z["pair"],pair[mask])
                h=z["tracks"]-pair[mask,2]
                vec=vector_features(z["origin_xyz"],z["generated_xyz"],pair[mask],
                                    residue,ca,mc)
                if extra is not None:
                    vec=np.c_[vec,extra(z["origin_xyz"],z["generated_xyz"],pair[mask],residue,atoms["name"],mc)]
            scalar=np.c_[h.mean(0).T,h.mean(1).std(0),h.std((0,1)),
                         h.min((0,1)),h.max((0,1))]
            values.append((x[mask],scalar,vec,y[mask],starts[mask],np.repeat(name,len(vec))))
    return tuple(np.concatenate(z) for z in zip(*values))

def main():
    mc=minicalc(Path("D:/MiniCalc/local_frames.py"))
    plan=json.loads((D/"train/manifest.json").read_bytes())
    x,d,g,y,_,proteins=load(plan["proteins"],"train",(49,149),mc)
    v,e,h,t,o,n=load(VALID,"validation",(49,149,249),mc)
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        np.testing.assert_array_equal(o,z["origin"])
        np.testing.assert_array_equal(n,z["proteins"])
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":o==49,"later":o!=49}
    results={}
    for name,q,r in (("scalar",d,e),("oriented",np.c_[d,g],np.c_[e,h])):
        for depth in (2,4,6):
            head=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,max_bin=96,
                              n_estimators=300,learning_rate=.045,n_jobs=4)
            head.fit(np.c_[x[:,:3],q],y-x[:,0])
            if not json.loads(head.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
                raise RuntimeError("CUDA required")
            head.set_params(device="cpu")
            pred=head.predict(np.c_[v[:,:3],r])+v[:,0]
            tag=f"{name}-{depth}"
            head.save_model(D/f"{tag}.ubj")
            for weight in (.1,.2,.35):
                p=(1-weight)*parent+weight*pred
                results[f"{tag}/{weight}"]={k:estimates(p[m],t[m],base[m]) for k,m in masks.items()}
    best=max(results,key=lambda k:results[k]["all"]["residual_correlation"])
    record(D/"development.json",dict(trials=results,best=best,rows=len(y),
             train=np.unique(proteins).tolist(),parent={k:estimates(parent[m],t[m],base[m]) for k,m in masks.items()}))
    print(best,results[best],flush=True)

if __name__=="__main__":
    main()


