"""Scale GA static-only regression over archived proteins and replicas."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST, REPLICAS, cell, registry, shard, trajectory
from tseries.features import _mic
from tseries.hb_single_structure import OUT,ROOT
from .explore import record
from .ga_static_offset import VALID,TRAIN_GROUPS,cohort,extract,estimates,ridge

DEST=Path("runs/ga_scale")
TAIL=("4ftf_A","3msw_A","3wx4_A","4wlr_B")


def rows(name,replica,count=None):
    path=shard(ROOT,name,replica,100,8.)
    with np.load(path) as archive:
        pair,x,t=[archive[k] for k in ("pair","features","origin")]
    reg=registry(ROOT,name)
    tid=trajectory(name,replica)
    b=np.r_[0,np.flatnonzero(np.diff(t))+1,len(t)]
    output=[]
    for k,(a,z) in enumerate(zip(b[:-1],b[1:])):
        if count is not None and k>=count:
            break
        mask=pair[a:z,2]<=6
        block=pair[a:z][mask]
        i,j=block[:,:2].astype(int).T
        origin=int(t[a])
        frames=next(reg.blocks(tid,start=origin-10,stop=origin+1,chunk_frames=11))
        if len(frames.time)!=11 or abs(frames.time[-1]-frames.time[0]-1000)>.01:
            raise ValueError("Time grid differs")
        y=np.mean([np.linalg.norm(_mic(p[j]-p[i],cell(box)),axis=1)
                   for p,box in zip(frames.positions,frames.box)],axis=0)
        output.append((block,x[a:z][mask],y))
    return tuple(np.concatenate([r[k] for r in output]).astype(np.float32) for k in range(3))


def cache(task):
    p,r=task
    path=DEST/f"{p}-R{r}.npz"
    if not path.exists():
        pair,x,y=rows(p,r)
        temp=path.with_suffix(".tmp")
        with temp.open("wb") as f:
            np.savez(f,pair=pair,x=x,y=y)
        temp.replace(path)
    with np.load(path) as f:
        return tuple(f[k] for k in ("pair","x","y"))


def view(pair,struct,kind):
    d=pair[:,2]
    if kind=="short":
        return np.c_[d,pair[:,6:8],struct[:,:118]]
    if kind=="rich":
        return np.c_[d,pair[:,6:8],struct]
    return np.c_[d,pair[:,6:8],struct,struct[:,[1,3,5,6,7,39,40,56,57,118,119,134,135]]*d[:,None]]


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    train_reference=[r for group in TRAIN_GROUPS for r in cohort(group,1)]
    tv,tx=extract(train_reference)
    valid=cohort(VALID,3)
    test=cohort(TEST,3)+cohort(TAIL[:2],3)+cohort(TAIL[2:],2)
    v,vx=extract(valid)
    y,yx=extract(test)
    base=ridge(tx["compact"],tv)
    rv,ry=base.predict(vx["compact"]),base.predict(yx["compact"])
    vdata=[rows(n,1,3) for n in VALID]
    hdata=[rows(n,1,3 if n in TEST+TAIL[:2] else 2) for n in TEST+TAIL]
    def zip_data(items):
        return tuple(np.concatenate(c) for c in zip(*items))
    vp,vs,vt=zip_data(vdata)
    hp,hs,ht=zip_data(hdata)
    for data,ref,actual,reference_target in ((vp,valid,vt,v),(hp,test,ht,y)):
        if not np.array_equal(data,np.concatenate([r["pair"] for r in ref])) or not np.allclose(actual,reference_target,atol=3e-6):
            raise ValueError("Candidate/target alignment changed")
    pool=tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])
    train=tuple(dict.fromkeys((*[n for group in TRAIN_GROUPS for n in group],*pool)))
    train=tuple(n for n in train if n not in VALID+TEST+TAIL)
    stages=(12,24,len(train))
    log=dict(train=train,validation=VALID,test=TEST+TAIL,criterion=.5,trials={})
    start=time.perf_counter()
    for want in stages:
        names=train[:want]
        jobs=[(p,r) for p in names for r in REPLICAS]
        with ThreadPoolExecutor(5) as pool:
            items=list(pool.map(cache,jobs))
        ap,ax,ay=zip_data(items)
        print("Data",want,len(ay),"seconds",round(time.perf_counter()-start),flush=True)
        for kind in ("short","rich","conditioned"):
            xx=view(ap,ax,kind)
            for depth in (3,6,9):
                model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=350,
                      max_depth=depth,learning_rate=.06,subsample=.82,colsample_bytree=.78,
                      max_bin=96,n_jobs=4,random_state=704)
                model.fit(xx,ay)
                model.set_params(device="cpu")
                p=model.predict(view(vp,vs,kind))
                key=f"{want}-{kind}-{depth}"
                log["trials"][key]=dict(validation=estimates(p,v,rv),rows=len(ay))
                model.save_model(DEST/f"{key}.ubj")
                record(DEST/"results.json",log)
                print(key,log["trials"][key],flush=True)
        del ap,ax,ay,items
    winner=max(log["trials"],key=lambda k:log["trials"][k]["validation"]["residual_correlation"])
    _,kind,_=winner.split("-")
    model=XGBRegressor(device="cpu")
    model.load_model(DEST/f"{winner}.ubj")
    pred=model.predict(view(hp,hs,kind))
    log["selected"]=winner
    log["test_metrics"]=estimates(pred,y,ry)
    np.savez(DEST/"evaluation.npz",predicted=pred,target=y,static=ry)
    record(DEST/"results.json",log)
    print("Selected",winner,log["test_metrics"],flush=True)


if __name__=="__main__":
    main()
