"""Learn a static past-mean correction with force and local orientation descriptors."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import REPLICAS,TEST,cell,registry,shard,trajectory
from tseries.features import _mic
from tseries.hb_preprocessing_force_projection import projected
from tseries.hb_heavy_dynamics import neighbor_pairs
from tseries.hb_single_structure import OUT,ROOT
from .explore import record
from .ga_scale import TAIL,rows
from .ga_static_offset import TRAIN_GROUPS,VALID,cohort,extract,estimates,ridge

DEST=Path("runs/ga_force")
LIMIT=6


def static(position,box,pair,atoms):
    charges=np.select([np.asarray(atoms["element"])==k for k in ("O","N","C","S")],
                      [-.30,-.10,.10,-.09])
    phys=projected(position,box,pair,atoms,charges)
    a,b=pair[:,:2].astype(int).T
    vec=_mic(position[b]-position[a],box)
    d=pair[:,2]
    axis=vec/d[:,None]
    names=np.asarray(atoms["name"])
    res=np.asarray(atoms["residue_index"])
    ca={r:i for i,(r,n) in enumerate(zip(res,names)) if n=="CA"}
    bb={n:{r:i for i,(r,k) in enumerate(zip(res,names)) if k==n} for n in ("N","C","CA","CB")}
    orient=[]
    for atoms_in,sign in ((a,1),(b,-1)):
        for refs in bb.values():
            idx=np.array([refs.get(res[i],i) for i in atoms_in])
            u=_mic(position[idx]-position[atoms_in],box)
            orient.append(sign*(u*axis).sum(1)/np.maximum(np.linalg.norm(u,axis=1),.01))
    edges=neighbor_pairs(position,box,7.)
    i,j=np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]]
    spacing=np.linalg.norm(_mic(position[j]-position[i],box),axis=1)
    polar=np.isin(atoms["element"],("N","O","S"))
    crowd=[]
    for cutoff in (3.,4.,5.5,7.):
        w=(spacing<cutoff).astype(float)
        for marker in (w,w*polar[j],w*(res[i]!=res[j])):
            z=np.bincount(i,weights=marker,minlength=len(position))
            crowd.extend((z[a],z[b]))
    return np.column_stack((phys,*orient,*crowd,d/pair[:,6].clip(1),
                            np.log1p(np.abs(res[a]-res[b])))).astype(np.float32)


def build(task):
    name,replica,limit=task
    path=DEST/f"{name}-R{replica}-{limit}.npz"
    if path.exists():
        with np.load(path) as z:
            return z["x"],z["y"]
    pp,xx,yy=rows(name,replica,limit)
    source=shard(ROOT,name,replica,100,8.)
    with np.load(source) as z:
        seq=z["origin"][z["pair"][:,2]<=6.][:len(pp)]
    lengths=np.r_[0,np.flatnonzero(np.diff(seq))+1,len(seq)]
    reg=registry(ROOT,name)
    tid=trajectory(name,replica)
    atoms=reg.atoms(tid).to_pydict()
    factors=[]
    for a,b in zip(lengths[:-1],lengths[1:]):
        origin=int(seq[a])
        block=next(reg.blocks(tid,start=origin,stop=origin+1))
        factors.append(static(block.positions[0],cell(block.box[0]),pp[a:b],atoms))
    features=np.c_[pp[:,2],pp[:,6:8],xx,np.concatenate(factors)].astype(np.float32)
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez(f,x=features,y=yy)
    path.with_suffix(".tmp").replace(path)
    return features,yy


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    original=[r for group in TRAIN_GROUPS for r in cohort(group,1)]
    yt,input_train=extract(original)
    v=cohort(VALID,3)
    h=cohort(TEST,3)+cohort(TAIL[:2],3)+cohort(TAIL[2:],2)
    yv,iv=extract(v)
    yh,ih=extract(h)
    r=ridge(input_train["compact"],yt)
    a,b=r.predict(iv["compact"]),r.predict(ih["compact"])
    p=tuple(json.loads((OUT/"volume_96_training_panel.json").read_bytes())["proteins"])
    names=tuple(n for n in p if n not in VALID+TEST+TAIL)
    roster=names[:64]
    tasks=[(n,rep,LIMIT) for n in roster for rep in REPLICAS]
    with ThreadPoolExecutor(5) as pool:
        data=list(pool.map(build,tasks))
    x,y=[np.concatenate(v) for v in zip(*data)]
    va=np.concatenate([build((n,1,3))[0] for n in VALID])
    test=np.concatenate([build((n,1,3 if n in TEST+TAIL[:2] else 2))[0] for n in TEST+TAIL])
    results=dict(train=roster,replicas=REPLICAS,origins_max=LIMIT,rows=len(y),trials={})
    for feature in ("force","all","core"):
        columns=slice(None) if feature=="all" else (np.r_[0:3,169:x.shape[1]] if feature=="force" else slice(0,169))
        u=x[:,columns]
        for depth in (5,8,10):
            learner=XGBRegressor(device="cuda",tree_method="hist",n_estimators=430,max_depth=depth,
                   max_bin=96,learning_rate=.055,n_jobs=4,subsample=.83,colsample_bytree=.82)
            learner.fit(u,y-u[:,0])
            learner.set_params(device="cpu")
            pred=learner.predict(va[:,columns])+va[:,0]
            key=f"{feature}-{depth}"
            results["trials"][key]=dict(validation=estimates(pred,yv,a))
            learner.save_model(DEST/f"{key}.ubj")
            record(DEST/"results.json",results)
            print(key,results["trials"][key],flush=True)
    best=max(results["trials"],key=lambda k:results["trials"][k]["validation"]["residual_correlation"])
    feature=best.split("-")[0]
    columns=slice(None) if feature=="all" else (np.r_[0:3,169:x.shape[1]] if feature=="force" else slice(0,169))
    learner=XGBRegressor(device="cpu")
    learner.load_model(DEST/f"{best}.ubj")
    pred=learner.predict(test[:,columns])+test[:,0]
    results["selected"]=best
    results["test"]=estimates(pred,yh,b)
    np.savez(DEST/"evaluation.npz",predicted=pred,target=yh,static=b)
    record(DEST/"results.json",results)
    print(best,results["test"],flush=True)


if __name__=="__main__":
    main()
