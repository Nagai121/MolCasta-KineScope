"""GA364 calibrates Amber readout with training labels from multiple starts."""
from itertools import product
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import shard
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga333 import WINDOW
from .ga_force import build
from .ga_static_offset import VALID, TRAIN_GROUPS, estimates

R=Path("runs")
D=R/"ga364"

def data(names,folders):
    for n in names:
        x,y=build((n,1,3))
        with np.load(shard(ROOT,n,1,100,8.)) as z:
            keep=z["pair"][:,2]<=6.
            origins=z["origin"][keep][:len(y)]
            pairs=z["pair"][keep][:len(y)]
        for origin in np.unique(origins):
            files=[folder/f"{n}-{origin}.npz" for folder in folders]
            file=next((f for f in files if f.exists()),None)
            if file is None:
                continue
            keep=origins==origin
            with np.load(file) as a:
                np.testing.assert_array_equal(a["pair"],pairs[keep])
                h=a["tracks"][:2,:len(WINDOW)]-a["pair"][:,2]
            d=np.c_[h.mean(0).T,h.mean(1).std(0),h.std((0,1)),
                    h.min((0,1)),h.max((0,1))]
            yield x[keep],d,y[keep],origins[keep],np.repeat(n,len(d))

def main(expanded=False):
    dest=R/"ga365/readout" if expanded else D
    dest.mkdir(exist_ok=True)
    names=tuple(dict.fromkeys((*json.loads((R/"ga355/development.json").read_bytes())["successful"],
                              *(n for g in TRAIN_GROUPS for n in g))))
    assert not set(names)&set(VALID)
    folders=(R/"ga333",R/"ga_native240",R/"ga365") if expanded else (R/"ga333",R/"ga_native240")
    batches=list(data(names,folders))
    x,d,y,o,g=tuple(np.concatenate(z) for z in zip(*batches))
    v,e,t,vo,vg=tuple(np.concatenate(z) for z in zip(*data(VALID,(R/"ga360/amber",))))
    with np.load(R/"ga360/validation.npz") as z:
        np.testing.assert_allclose(t,z["target"],rtol=0,atol=3e-6)
        np.testing.assert_array_equal(vo,z["origin"])
        np.testing.assert_array_equal(vg,z["proteins"])
        base,parent=z["baseline"],z["parent"]
    masks={"all":np.ones(len(t),bool),"first":vo==49,"later":vo!=49}
    trials={}
    for static,depth,shift in product((3,x.shape[1]),(2,5),(0.,2.,5.)):
        weight=1.+shift*(o!=49)
        head=XGBRegressor(device="cuda",tree_method="hist",max_depth=depth,n_estimators=300,
                          max_bin=96,learning_rate=.045,n_jobs=4)
        head.fit(np.c_[x[:,:static],d],y-x[:,0],sample_weight=weight)
        if not json.loads(head.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda"):
            raise RuntimeError("CUDA required")
        head.set_params(device="cpu")
        p=head.predict(np.c_[v[:,:static],e])+v[:,0]
        name=f"{static}-{depth}-{shift}"
        for w in (.1,.2,.35):
            mix=(1-w)*parent+w*p
            trials[f"{name}/{w}"]={k:estimates(mix[i],t[i],base[i]) for k,i in masks.items()}
        head.save_model(dest/f"{name}.ubj")
    reference={k:estimates(parent[i],t[i],base[i]) for k,i in masks.items()}
    best=max(trials,key=lambda n:trials[n]["later"]["residual_correlation"])
    counts={str(n):int((o==n).sum()) for n in np.unique(o)}
    record(dest/"development.json",dict(best=best,trials=trials,parent=reference,
                                    proteins=np.unique(g).tolist(),rows_by_origin=counts))
    print(best,trials[best],reference,counts,flush=True)

if __name__=="__main__":
    main()

