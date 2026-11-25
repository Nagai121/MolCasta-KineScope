"""GA1238+: learn use of explicit solvent instead of adding a fixed correction."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .ga333 import WINDOW
from .ga_explicit_solvent import simulate
from .ga_replica_teacher import trajectory_rows
from .ga_sequential50 import features,pool
from .ga_static_offset import VALID,estimates

R=Path("runs")
OUT=R/"ga1238-solvent-learned"


def descriptors(explicit,implicit):
    difference=explicit-implicit
    s=[]
    for data in (explicit,difference):
        s.extend((data.mean((0,1)),data.std((0,1)),data.mean(1).std(0),
                  data[:,3:].mean((0,1))-data[:,:3].mean((0,1)),
                  np.quantile(data,.25,axis=(0,1)),np.quantile(data,.75,axis=(0,1))))
    return np.column_stack(s).astype(np.float32)


def build(names,origins,x,d,where,caches,implicit_folder):
    values=[]
    index=[]
    for name in names:
        candidates,frames=trajectory_rows(name,1)
        for origin in origins:
            i=np.flatnonzero((where[0]==name)&(where[1]==origin))
            file=f"{name}-{origin}.npz"
            explicit_folder=caches[origin]
            with np.load(explicit_folder/file) as a,np.load(implicit_folder/file) as b:
                np.testing.assert_array_equal(a["pair"],b["pair"])
                np.testing.assert_array_equal(a["pair"],candidates[frames==origin])
                observed=a["tracks"][:2,:len(WINDOW)]
                comparison=b["tracks"][:2,:len(WINDOW)]
                np.testing.assert_array_equal(a["pair"][:,2],x[i,0])
                values.append(descriptors(observed-x[i,0],comparison-x[i,0]))
                index.append(i)
    return np.concatenate(values),np.concatenate(index)


def main():
    OUT.mkdir(exist_ok=True)
    folder=OUT/"tracks"
    folder.mkdir(exist_ok=True)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    names=sorted(set(p),key=lambda n:sha256(("GA1238/"+n).encode()).digest())[:24]
    assert not set(names)&set(vg)
    lock={"train_proteins":names,"train_origin":49,"validation_proteins":list(VALID),"length_ps":25}
    (OUT/"plan.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    for name in names:
        simulate(name,49,folder,seed_solvent=True)
    train,idx=build(names,(49,),x,d,(p,o),{49:folder},R/"ga333")
    valid,j=build(VALID,(49,149,249),v,e,(vg,vo),
                  {49:R/"ga1236-explicit-solvent",149:R/"ga1237-explicit-later",249:R/"ga1237-explicit-later"},
                  R/"ga360/amber")
    np.testing.assert_array_equal(j,np.arange(len(t)))
    xx=features(x,d,g,"level")[idx]
    vv=features(v,e,h,"level")
    rows=[]
    methods=[(view,depth) for view in ("implicit","solvent") for depth in (2,3,5,7)]
    for index,(view,dep) in enumerate(methods):
        a,b=(np.c_[xx,train],np.c_[vv,valid]) if view=="solvent" else (xx,vv)
        model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=420,max_depth=dep,
                           learning_rate=.035,max_bin=96,n_jobs=3,reg_lambda=14,subsample=.88)
        model.fit(a,y[idx]-x[idx,0])
        assert json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
        model.set_params(device="cpu")
        direct=model.predict(b)+v[:,0]
        prediction=.65*parent+.35*direct
        ms={k:estimates(prediction[m],t[m],base[m]) for k,m in
            (("all",vo>=0),("first",vo==49),("later",vo!=49))}
        tag=f"GA{1238+index}"
        model.save_model(OUT/f"{tag}.ubj")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=prediction,readout=direct)
        row={"id":tag,"view":view,"depth":dep,"metrics":ms}
        rows.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(row,flush=True)
    (OUT/"summary.json").write_text(json.dumps(max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"]),
                                              indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
