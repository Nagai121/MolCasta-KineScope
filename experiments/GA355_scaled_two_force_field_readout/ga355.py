"""GA355 scales a multi-force-field short-trajectory readout."""
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga309 import predict
from .ga333 import WINDOW, candidates
from .ga_force import build
from .ga_native240 import simulate
from .ga_openmm_pilot import build as build_md
from .ga_static_offset import VALID, estimates, cohort, extract


DEST=Path("runs/ga355")


def inputs(names,amber,charmm):
    packets=[]
    for name in names:
        with np.load(amber/f"{name}-49.npz") as a, np.load(charmm/f"{name}-49.npz") as b:
            np.testing.assert_array_equal(a["pair"],b["pair"])
            d=a["pair"][:,2]
            a=a["tracks"][:2,:len(WINDOW)]-d
            b=b["tracks"][:2,:len(WINDOW)]-d
        s,y=build((name,1,1))
        np.testing.assert_array_equal(d,s[:,0])
        am=a.mean(0).T
        bm=b.mean(0).T
        dyn=np.c_[am,bm,am-bm,a.std((0,1)),b.std((0,1)),a.mean(1).std(0),b.mean(1).std(0)]
        packets.append((s,dyn,y))
    return tuple(np.concatenate(v) for v in zip(*packets))


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    parent=Path("runs/ga333")
    trained=json.loads((parent/"development.json").read_bytes())["successful"]
    names=tuple(trained)
    report=dict(train=names,successful=[],failures={},trials={})
    fn=partial(build_md,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    for n in names:
        inherited=Path("runs/ga344")/f"{n}-49.npz"
        local=DEST/inherited.name
        if inherited.exists() and not local.exists():
            from shutil import copyfile
            copyfile(inherited,local)
        try:
            simulate((n,),1,destination=DEST,samples=WINDOW,branches=2,step_fs=2,
                     minimize_iterations=60,setup_fn=fn,parts_fn=candidates)
        except Exception as exc:
            report["failures"][n]=str(exc)
        else:
            report["successful"].append(n)
        record(DEST/"development.json",report)
    x,t,y=inputs(report["successful"],parent,DEST)
    v,z,h=inputs(VALID,Path("runs/ga325"),Path("runs/ga313"))
    j,u=extract(cohort(VALID,1))
    np.testing.assert_allclose(h,j,rtol=0,atol=3.e-6)
    with np.load("runs/ga_native_offset/baselines.npz") as a:
        base=predict(a,"real_baseline_",u["compact"])
    report["pairs"]=len(y)
    for key,p,q in (("generated",np.c_[x[:,:3],t],np.c_[v[:,:3],z]),
                    ("joint",np.c_[x,t],np.c_[v,z])):
        for depth in (2,4,6):
            tick=time.perf_counter()
            model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=300,
                               max_depth=depth,max_bin=96,learning_rate=.045,n_jobs=4,
                               subsample=.85,colsample_bytree=.86)
            model.fit(p,y-x[:,0])
            model.set_params(device="cpu")
            pred=model.predict(q)+v[:,0]
            name=f"{key}-{depth}"
            report["trials"][name]=dict(seconds=time.perf_counter()-tick,**estimates(pred,h,base))
            model.save_model(DEST/f"{name}.ubj")
            record(DEST/"development.json",report)
            print(name,report["trials"][name],flush=True)


if __name__=="__main__":
    main()

