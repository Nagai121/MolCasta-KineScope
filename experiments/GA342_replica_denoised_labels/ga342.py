"""GA342 denoises training labels with the same pair at nearby training poses."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from xgboost import XGBRegressor

from tseries.data import registry, trajectory, shard
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga309 import predict
from .ga_force import build
from .ga_static_offset import VALID, cohort, estimates, extract


def protein(name):
    features=[]
    labels=[]
    pairs=[]
    reg=registry(ROOT,name)
    topology=None
    for replica in (1,2,3):
        table=reg.atoms(trajectory(name,replica))
        identifiers=table.select(["residue_index","name"]).to_pydict()
        if topology is not None and topology!=identifiers:
            raise RuntimeError(f"Pair topology differs across replicas of {name}")
        topology=identifiers
        x,y=build((name,replica,4))
        with np.load(shard(ROOT,name,replica,100,8.)) as z:
            r=z["pair"][z["pair"][:,2]<=6.][:len(y)]
        np.testing.assert_array_equal(x[:,0],r[:,2])
        ids=r[:,:2].astype(np.int64)
        pairs.append(ids[:,0]*table.num_rows+ids[:,1])
        features.append(x)
        labels.append(y)
    x=np.concatenate(features)
    y=np.concatenate(labels)
    key=np.concatenate(pairs)
    d=x[:,0]
    order=np.argsort(key,kind="stable")
    sorted_key=key[order]
    boundary=np.r_[0,np.flatnonzero(np.diff(sorted_key))+1,len(d)]
    cut=np.arange(3.5,6.001,.125)
    group=np.repeat(np.arange(len(boundary)-1),np.diff(boundary))
    distance=d[order]
    target=y[order]
    out=[]
    for width in (.18,.35,.60):
        z=np.exp(-.5*((distance[:,None]-cut)/width)**2).astype(np.float32)
        weight=np.add.reduceat(z,boundary[:-1],axis=0)
        mass=np.add.reduceat(z*target[:,None],boundary[:-1],axis=0)
        prior=np.sum(z*mass[group],axis=1)/np.maximum(np.sum(z*weight[group],axis=1),1.e-7)
        back=np.empty_like(y)
        back[order]=prior
        out.append(back)
    return x,y,np.column_stack(out)


def main():
    dest=Path("runs/ga342")
    dest.mkdir(parents=True,exist_ok=True)
    names=json.loads(Path("runs/ga330/development.json").read_bytes())["train"]
    blocked=json.loads(Path("runs/ga321/reserve.json").read_bytes())["proteins"]
    assert not set(names)&set((*VALID,*blocked))
    items=[]
    with ThreadPoolExecutor(5) as pool:
        for n,r in enumerate(pool.map(protein,names),1):
            items.append(r)
            if n%25==0:
                print("Prepared",n,flush=True)
    x,t,smoothed=tuple(np.concatenate(z) for z in zip(*items))
    del items
    v=np.concatenate([build((p,1,3))[0] for p in VALID])
    y,features=extract(cohort(VALID,3))
    with np.load("runs/ga_native_offset/baselines.npz") as ref:
        base=predict(ref,"real_baseline_",features["compact"])
    ledger=dict(proteins=names,rows=len(x),label_protocol="training replicas only",trials={})
    for i,width in enumerate((.18,.35,.60)):
        for fraction in (.4,.8):
            started=time.perf_counter()
            target=(1.-fraction)*t+fraction*smoothed[:,i]-x[:,0]
            model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=450,
                               max_depth=8,max_bin=96,learning_rate=.045,n_jobs=4,
                               subsample=.85,colsample_bytree=.86)
            model.fit(x,target)
            model.set_params(device="cpu")
            p=model.predict(v)+v[:,0]
            key=f"{width}-{fraction}"
            ledger["trials"][key]=dict(seconds=time.perf_counter()-started,**estimates(p,y,base))
            model.save_model(dest/f"{key}.ubj")
            record(dest/"development.json",ledger)
            print(key,ledger["trials"][key],flush=True)


if __name__=="__main__":
    main()
