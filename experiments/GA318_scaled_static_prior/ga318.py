"""GA318: train structural priors on a larger protein-disjoint panel."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST, REPLICAS
from tseries.hb_single_structure import OUT
from .explore import record
from .ga_force import build
from .ga_native_offset import HOLD
from .ga_static_offset import VALID, estimates, cohort, extract
from .ga309 import predict


def main():
    dest = Path("runs/ga318")
    dest.mkdir(parents=True, exist_ok=True)
    pools=[json.loads((OUT/n).read_bytes())["proteins"] for n in (
        "volume_128_training_panel.json","volume_96_training_panel.json","volume_training_panel.json")]
    train=tuple(n for n in dict.fromkeys(n for q in pools for n in q) if n not in TEST+HOLD+VALID)
    train=train[:112]
    report=dict(train=list(train),replicas=REPLICAS,maximum_origins=4,trials={})
    tasks=[(p,r,4) for p in train for r in REPLICAS]
    with ThreadPoolExecutor(5) as pool:
        data=[]
        for i, item in enumerate(pool.map(build,tasks),1):
            data.append(item)
            if i%18==0:
                print("Prepared",i,"/",len(tasks),flush=True)
    x,y=[np.concatenate(v) for v in zip(*data)]
    del data
    validation=cohort(VALID,3)
    t,v=extract(validation)
    h=np.concatenate([build((n,1,3))[0] for n in VALID])
    with np.load("runs/ga_native_offset/baselines.npz") as z:
        reference=predict(z,"real_baseline_",v["compact"])
    report["rows"]=len(y)
    for depth in (4,6,8):
        for kind, cols in (("all",slice(None)),("local",np.r_[0:3,169:x.shape[1]])):
            now=time.perf_counter()
            name=f"{kind}-{depth}"
            model=XGBRegressor(device="cuda",tree_method="hist",n_estimators=450,
                               max_depth=depth,max_bin=96,n_jobs=4,learning_rate=.045,
                               subsample=.85,colsample_bytree=.86)
            model.fit(x[:,cols],y-x[:,0])
            model.set_params(device="cpu")
            pred=model.predict(h[:,cols])+h[:,0]
            report["trials"][name]=dict(seconds=time.perf_counter()-now,
                                        **estimates(pred,t,reference))
            model.save_model(dest/f"{name}.ubj")
            record(dest/"development.json",report)
            print(name,report["trials"][name],flush=True)


if __name__=="__main__":
    main()
