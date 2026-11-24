"""GA335 augments the static model with directional solvent accessibility."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from xgboost import XGBRegressor

from tseries.data import registry, trajectory, shard, cell
from tseries.features import _mic
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga309 import predict
from .ga_force import build
from .ga_static_offset import VALID, cohort, estimates, extract


DEST=Path("runs/ga335")
phi=np.pi*(3.-np.sqrt(5.))*np.arange(32)
z=1.-(np.arange(32)+.5)/16
RADIAL=np.column_stack((np.cos(phi)*np.sqrt(1-z*z),np.sin(phi)*np.sqrt(1-z*z),z))


def shell(pos,box,pair,types):
    endpoints=np.unique(pair[:,:2].astype(int))
    active=pos[endpoints]
    tree=cKDTree(pos)
    near=tree.query(active,k=min(45,len(pos)))[1]
    delta=_mic(pos[near]-active[:,None],box)
    radius=np.select([types=="O",types=="N",types=="S"],[1.52,1.55,1.8],default=1.7)
    result=[]
    a,b=np.searchsorted(endpoints,pair[:,:2].astype(int)).T
    axis=pair[:,3:6]/pair[:,2,None]
    relative=axis@RADIAL.T
    for water in (1.1,1.4,1.9):
        sample=(radius[endpoints]+water)[:,None,None]*RADIAL
        delta2=np.square(delta[:,None]-sample[:,:,None]).sum(-1)
        occluded=(delta2<(radius[near][:,None]+water)**2)&(near[:,None]!=endpoints[:,None,None])
        exposed=~occluded.any(-1)
        result.extend((exposed[a].mean(1),exposed[b].mean(1)))
        for cutoff in (0.,.4,.7):
            m=relative>=cutoff
            n=relative<=-cutoff
            result.extend(((exposed[a]&m).sum(1)/m.sum(1).clip(1),
                           (exposed[b]&n).sum(1)/n.sum(1).clip(1)))
        result.extend((np.sum(exposed[a]*relative,axis=1)/32,
                       -np.sum(exposed[b]*relative,axis=1)/32))
    return np.column_stack(result).astype(np.float32)


def pack(task):
    name,replica,n=task
    cache=DEST/f"{name}-{replica}-{n}.npz"
    x,y=build(task)
    if cache.exists():
        with np.load(cache) as a:
            return x,y,a["fields"]
    with np.load(shard(ROOT,name,replica,100,8.)) as data:
        include=data["pair"][:,2]<=6.
        pairs=data["pair"][include][:len(y)]
        times=data["origin"][include][:len(y)]
    np.testing.assert_array_equal(pairs[:,2],x[:,0])
    bounds=np.r_[0,np.flatnonzero(np.diff(times))+1,len(times)]
    reg=registry(ROOT,name)
    tid=trajectory(name,replica)
    atoms=np.asarray(reg.atoms(tid).to_pydict()["element"])
    f=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        t=int(times[lo])
        block=next(reg.blocks(tid,start=t,stop=t+1,chunk_frames=1))
        f.append(shell(block.positions[0],cell(block.box[0]),pairs[lo:hi],atoms))
    v=np.concatenate(f)
    with cache.with_suffix(".tmp").open("wb") as handle:
        np.savez(handle,fields=v)
    cache.with_suffix(".tmp").replace(cache)
    return x,y,v


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    names=json.loads(Path("runs/ga318/development.json").read_bytes())["train"]
    jobs=[(n,r,4) for n in names for r in (1,2,3)]
    batches=[]
    with ThreadPoolExecutor(4) as pool:
        for k,item in enumerate(pool.map(pack,jobs),1):
            batches.append(item)
            if k%30==0:
                print("Prepared",k,flush=True)
    x,y,h=tuple(np.concatenate(vals) for vals in zip(*batches))
    del batches
    v,t,u=tuple(np.concatenate(vals) for vals in zip(*(pack((n,1,3)) for n in VALID)))
    target,features=extract(cohort(VALID,3))
    np.testing.assert_allclose(target,t,atol=3.e-6,rtol=0)
    with np.load("runs/ga_native_offset/baselines.npz") as a:
        base=predict(a,"real_baseline_",features["compact"])
    report=dict(proteins=names,rows=len(y),columns=h.shape[1],trials={})
    for name,s,d in (("solvent",np.c_[x[:,:3],h],np.c_[v[:,:3],u]),
                     ("joint",np.c_[x,h],np.c_[v,u])):
        for depth in (5,8):
            started=time.perf_counter()
            model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=450,
                               max_depth=depth,max_bin=96,n_jobs=4,learning_rate=.045,
                               subsample=.85,colsample_bytree=.86)
            model.fit(s,y-x[:,0])
            model.set_params(device="cpu")
            output=model.predict(d)+v[:,0]
            key=f"{name}-{depth}"
            report["trials"][key]=dict(seconds=time.perf_counter()-started,**estimates(output,t,base))
            model.save_model(DEST/f"{key}.ubj")
            record(DEST/"development.json",report)
            print(key,report["trials"][key],flush=True)


if __name__=="__main__":
    main()
