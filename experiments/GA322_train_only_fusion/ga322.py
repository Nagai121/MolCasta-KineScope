"""GA322: learn fusion on training proteins, then evaluate fixed on validation."""
from itertools import product
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .explore import record
from .ga306 import cache_path, tracks
from .ga309 import predict
from .ga_force import build
from .ga_static_offset import TRAIN_GROUPS, VALID, cohort, estimates, extract


def static_predictions(rows):
    # GA304 was fitted on an independent roster; the loader is cached by group.
    head=XGBRegressor(device="cpu")
    head.load_model("runs/ga_force/all-8.ubj")
    grouped={}
    offsets={}
    for r in rows:
        name=r["protein"]
        rep=r.get("replica",1)
        key=name,rep
        if key not in grouped:
            features=build((name,rep,3))[0]
            values=head.predict(features)+features[:,0]
            grouped[key]=(features[:,0],values)
            offsets[key]=0
        d=r["pair"][:,2]
        f,p=grouped[key]
        # Matching crops skip origins without a stored generated trajectory.
        starts=np.flatnonzero(f==d[0])
        at=next(i for i in starts if np.array_equal(f[i:i+len(d)],d))
        yield p[at:at+len(d)]


def fields(rows,x,baseline,p0):
    t=tracks(rows)
    raw=t.mean((1,2))
    mu=t.mean(2)
    late=t[:,:,-5:].mean((1,2))
    return np.c_[raw-p0,late-p0,np.concatenate(list(static_predictions(rows)))-baseline,
                 t.std((1,2)),np.abs(mu[:,0]-mu[:,1]),x[:,0]-4.5]


def design(t,mode):
    b=t[:,:3]
    if mode=="base":
        return b
    if mode=="distance":
        return np.c_[b,b*t[:,5:6],b*t[:,5:6]**2]
    if mode=="dispersion":
        return np.c_[b,b/(.20+t[:,3:4]),b/(.20+t[:,4:5])]
    return np.c_[design(t,"distance"),design(t,"dispersion")[:,3:]]


def main():
    dest=Path("runs/ga322")
    dest.mkdir(parents=True,exist_ok=True)
    roster=json.loads(Path("runs/ga_force/results.json").read_bytes())["train"]
    names=tuple(p for g in TRAIN_GROUPS for p in g if p not in roster)
    train=[r for p in names for replica in (1,2) for r in cohort((p,),3,replica) if cache_path(r).exists()]
    val=cohort(VALID,3)
    y,x=extract(train)
    v,h=extract(val)
    t0,h0=x["compact"],h["compact"]
    with np.load("runs/ga_native_offset/baselines.npz") as pack:
        base_t=predict(pack,"real_baseline_",t0)
        base_v=predict(pack,"real_baseline_",h0)
        p_t=predict(pack,"native_baseline_",t0)[:,0]
        p_v=predict(pack,"native_baseline_",h0)[:,0]
    tr,va=fields(train,t0,base_t,p_t),fields(val,h0,base_v,p_v)
    result={}
    for mode,alpha,positive in product(("base","distance","dispersion","union"),(20.,100.,500.),(False,True)):
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,positive=positive))
        model.fit(design(tr,mode),y-base_t)
        m=estimates(base_v+model.predict(design(va,mode)),v,base_v)
        result[f"{mode}/{alpha}/{positive}"]=m
    ranked=sorted(result,key=lambda k:-result[k]["residual_correlation"])
    record(dest/"development.json",dict(proteins=names,origins=len(train),pairs=len(y),
                                         selected=ranked[:5],trials=result))
    print([(k,result[k]) for k in ranked[:5]],flush=True)


if __name__=="__main__":
    main()
