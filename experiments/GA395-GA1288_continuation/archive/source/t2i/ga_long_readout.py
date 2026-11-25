"""GA1176+: matched short and long response readouts on identical training poses."""
from functools import partial
import json
from pathlib import Path

import numpy as np
import openmm
from xgboost import XGBRegressor

from .ga365 import parts

from .ga_native240 import SAMPLES, simulate
from .ga_openmm_pilot import build
from .ga_sequential50 import pool
from .ga_static_offset import VALID, estimates

R=Path("runs")
OUT=R/"ga1176-long-readout"


def response(names, origins, amber, charmm):
    features, identity=[],[]
    for name in names:
        for origin in origins:
            key=f"{name}-{origin}.npz"
            with np.load(amber/key) as a,np.load(charmm/key) as b:
                np.testing.assert_array_equal(a["pair"],b["pair"])
                pairs=a["pair"]
                initial=pairs[:,2]
                tracks=[z["tracks"][:2]-initial for z in (a,b)]
                if any(z.shape[1]!=len(SAMPLES) for z in tracks):
                    raise RuntimeError(f"Missing full trajectory for {key}")
                s=[]
                for z in tracks:
                    for end in (6,10,13):
                        v=z[:,:end]
                        s.extend([v.mean((0,1)),v.std((0,1)),v.mean(1).std(0)])
                m=np.column_stack(s)
                features.append(m)
                identity.extend([(name,origin,float(x)) for x in initial])
    return np.concatenate(features),identity


def main():
    OUT.mkdir(exist_ok=True)
    (x,d,g,y,o,p),(v,e,h,t,vo,vg,base,parent)=pool()
    amber=R/"ga_native240"
    names=sorted(n for n in set(p) if (amber/f"{n}-49.npz").is_file())
    assert not set(names)&set(vg)
    (OUT/"training-plan.json").write_text(json.dumps(
        {"proteins":names,"branches":2,"sample_grid_ps":SAMPLES,"origin":49},indent=2),encoding="utf-8")
    force=partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    cache=OUT/"charmm"
    cache.mkdir(exist_ok=True)
    for n in names:
        opts=dict(destination=cache,samples=SAMPLES,branches=2,setup_fn=force,
                  parts_fn=partial(parts,origins=(49,)))
        try:
            simulate((n,),1,step_fs=2,minimize_iterations=60,**opts)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            (cache/f"{n}-recovery.json").write_text(json.dumps({"step_fs":1,"minimize":100}),encoding="utf-8")
            simulate((n,),1,step_fs=1,minimize_iterations=100,**opts)
    a,tag=response(names,(49,),amber,cache)
    b,valid_tag=response(VALID,(49,149,249),R/"ga360/amber",R/"ga360/charmm")
    train_index=np.concatenate([np.flatnonzero((p==n)&(o==49)) for n in names])
    for identities,dist,prot,origin in ((tag,x[train_index,0],p[train_index],o[train_index]),
                                        (valid_tag,v[:,0],vg,vo)):
        np.testing.assert_array_equal([q[0] for q in identities],prot)
        np.testing.assert_array_equal([q[1] for q in identities],origin)
        np.testing.assert_array_equal([q[2] for q in identities],dist)
    rows=[]
    short=[0,1,2,9,10,11]
    mid=short+[3,4,5,12,13,14]
    layout=[("short",short),("medium",mid),("long",list(range(18)))]
    for i,(window,columns,depth) in enumerate((name,col,dep) for name,col in layout for dep in (3,6)):
        train=np.c_[x[train_index],a[:,columns]]
        valid=np.c_[v,b[:,columns]]
        reg=XGBRegressor(device="cuda",tree_method="hist",n_estimators=350,max_depth=depth,
                           max_bin=96,learning_rate=.035,n_jobs=3,reg_lambda=12,subsample=.88)
        reg.fit(train,y[train_index]-x[train_index,0])
        assert json.loads(reg.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
        reg.set_params(device="cpu")
        direct=reg.predict(valid)+v[:,0]
        prediction=.65*parent+.35*direct
        metrics={k:estimates(prediction[m],t[m],base[m]) for k,m in
                 (("all",vo>=0),("first",vo==49),("later",vo!=49))}
        tag=f"GA{1176+i}"
        reg.save_model(OUT/f"{tag}.ubj")
        np.savez_compressed(OUT/f"{tag}.npz",prediction=prediction,readout=direct)
        row=dict(id=tag,window=window,depth=depth,columns=columns,metrics=metrics)
        rows.append(row)
        with (OUT/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(row,flush=True)
    (OUT/"summary.json").write_text(json.dumps(
        max(rows,key=lambda r:r["metrics"]["all"]["residual_correlation"]),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

