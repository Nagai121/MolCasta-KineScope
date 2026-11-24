"""GA333 scales supervised learning from low-cost simulated motion."""
import json
from pathlib import Path
import time

import numpy as np
from xgboost import XGBRegressor

from tseries.data import shard
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga_force import build
from .ga309 import predict
from .ga_native240 import simulate, SAMPLES
from .ga_openmm_pilot import build as build_md
from .ga_static_offset import VALID, cohort, estimates, extract

WINDOW=tuple(t for t in SAMPLES if t<=25)


def candidates(names):
    parts=[]
    total=0
    for n in names:
        with np.load(shard(ROOT,n,1,100,8.)) as z:
            keep=(z["origin"]==49)&(z["pair"][:,2]<=6.)
            pairs=z["pair"][keep]
        if not len(pairs):
            raise RuntimeError(f"No candidate rows for {n}")
        count=len(pairs)
        total+=count
        parts.append(dict(replica=1,protein=n,pair=pairs,origin=np.full(count,49),
                          ends=np.array([count]),y=np.zeros(count)))
    return parts,{"combined":np.empty((total,0))},np.zeros(total)


def inputs(names,folder,nseeds,relational=False):
    result=[]
    for name in names:
        with np.load(folder/f"{name}-49.npz") as z:
            pair=z["pair"]
            h=z["tracks"][:nseeds]-pair[:,2]
        tr,y=build((name,1,1))
        np.testing.assert_array_equal(tr[:,0],pair[:,2])
        x=np.c_[h.mean(0).T,h.std((0,1)),h.mean(1).std(0),
                h.min((0,1)),h.max((0,1))]
        if relational:
            from .ga308 import relations
            extra=relations([dict(pair=pair)],h.transpose(2,0,1))
            x=np.c_[x,extra]
        result.append((x,tr,y))
    return tuple(np.concatenate(fields) for fields in zip(*result))


def main(relational=False):
    dest=Path("runs/ga333")
    dest.mkdir(parents=True,exist_ok=True)
    output=Path("runs/ga338") if relational else dest
    output.mkdir(parents=True,exist_ok=True)
    train=json.loads(Path("runs/ga330/development.json").read_bytes())["train"]
    roster=tuple(train[::2][:80])
    assert not set(roster)&set(VALID)
    report=dict(proteins=roster,window=WINDOW,seeds=2,trials={},failures={})
    for name in (() if relational else roster):
        try:
            simulate((name,),1,destination=dest,samples=WINDOW,branches=2,
                     setup_fn=build_md,parts_fn=candidates)
        except Exception as exc:
            report["failures"][name]=str(exc)
            record(output/"development.json",report)
    trained=[p for p in roster if (dest/f"{p}-49.npz").exists()]
    a,x,y=inputs(trained,dest,2,relational)
    q,z,t=inputs(VALID,Path("runs/ga325"),2,relational)
    _,v=extract(cohort(VALID,1))
    with np.load("runs/ga_native_offset/baselines.npz") as ref:
        base=predict(ref,"real_baseline_",v["compact"])
    prior=XGBRegressor(device="cpu")
    prior.load_model("runs/ga330/all-9.ubj")
    # Independent static reference for evaluation. Do not train against the
    # expanded prior's in-sample predictions.
    reference=z[:,0]+prior.predict(z)
    report["successful"]=trained
    report["rows"]=len(y)
    for name,data,valid in (("generated",np.c_[x[:,:3],a],np.c_[z[:,:3],q]),
                            ("joint",np.c_[x,a],np.c_[z,q]),("static",x,z)):
        for depth in (3,5,7):
            started=time.perf_counter()
            learner=XGBRegressor(device="cuda",tree_method="hist",n_estimators=360,
                                 max_depth=depth,learning_rate=.045,max_bin=96,
                                 subsample=.85,colsample_bytree=.85,n_jobs=4)
            learner.fit(data,y-x[:,0])
            learner.set_params(device="cpu")
            p=learner.predict(valid)+z[:,0]
            key=f"{name}-{depth}"
            report["trials"][key]=dict(seconds=time.perf_counter()-started,
                                       direct=estimates(p,t,base),
                                       blend=estimates(.5*(p+reference),t,base))
            learner.save_model(output/f"{key}.ubj")
            record(output/"development.json",report)
            print(key,report["trials"][key],flush=True)


if __name__=="__main__":
    main()
