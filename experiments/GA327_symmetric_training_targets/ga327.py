"""GA327: symmetry-averaged structural training targets from observed data."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from xgboost import XGBRegressor

from tseries.data import registry, trajectory, shard, cell
from tseries.features import _mic
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga_force import build
from .ga309 import predict
from .ga_static_offset import VALID, cohort, estimates, extract


def sample(task):
    name,replica,count=task
    features,y=build(task)
    with np.load(shard(ROOT,name,replica,100,8.)) as z:
        mask=z["pair"][:,2]<=6.
        pair=z["pair"][mask][:len(y)]
        times=z["origin"][mask][:len(y)]
    np.testing.assert_array_equal(pair[:,2],features[:,0])
    reg=registry(ROOT,name)
    tid=trajectory(name,replica)
    bounds=np.r_[0,np.flatnonzero(np.diff(times))+1,len(times)]
    future=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        t=int(times[lo])
        frames=next(reg.blocks(tid,start=t,stop=t+11,chunk_frames=11))
        if len(frames.time)!=11 or abs(frames.time[-1]-frames.time[0]-1000)>1e-2:
            raise RuntimeError("Unexpected future grid for training")
        i,j=pair[lo:hi,:2].astype(int).T
        future.append(np.mean([np.linalg.norm(_mic(p[j]-p[i],cell(box)),axis=1)
                               for p,box in zip(frames.positions,frames.box)],axis=0))
    return features,y,np.concatenate(future)


def main():
    dest=Path("runs/ga327")
    dest.mkdir(parents=True,exist_ok=True)
    names=json.loads(Path("runs/ga318/development.json").read_bytes())["train"]
    assert not set(names)&set(VALID)
    jobs=[(name,rep,4) for name in names for rep in (1,2,3)]
    batches=[]
    with ThreadPoolExecutor(6) as pool:
        for n,item in enumerate(pool.map(sample,jobs),1):
            batches.append(item)
            if n%30==0:
                print("Prepared",n,flush=True)
    x,y,f=[np.concatenate(items) for items in zip(*batches)]
    del batches
    t,views=extract(cohort(VALID,3))
    v=np.concatenate([build((n,1,3))[0] for n in VALID])
    with np.load("runs/ga_native_offset/baselines.npz") as a:
        reference=predict(a,"real_baseline_",views["compact"])
    ledger=dict(proteins=names,rows=len(y),trials={})
    for ratio in (0.25,.5,.75):
        z=(1-ratio)*y+ratio*f-x[:,0]
        for depth in (6,8):
            now=time.perf_counter()
            key=f"{ratio}/{depth}"
            model=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,
                               max_bin=96,n_estimators=450,learning_rate=.045,n_jobs=4,
                               subsample=.85,colsample_bytree=.86)
            model.fit(x,z)
            model.set_params(device="cpu")
            pred=model.predict(v)+v[:,0]
            ledger["trials"][key]=dict(seconds=time.perf_counter()-now,**estimates(pred,t,reference))
            model.save_model(dest/f"future-{ratio}-{depth}.ubj")
            record(dest/"development.json",ledger)
            print(key,ledger["trials"][key],flush=True)


if __name__=="__main__":
    main()
