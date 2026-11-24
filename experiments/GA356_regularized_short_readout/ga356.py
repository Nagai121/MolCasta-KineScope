"""GA356 tests regularized readout of matched two-force-field responses."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .explore import record
from .ga344 import inputs
from .ga348 import components, forecasts
from .ga_static_offset import VALID, estimates

R = Path("runs")

def matrix(x, dyn, mode):
    start = x[:, :3]
    a, b, d = np.split(dyn[:, :-4], 3, axis=1)
    raw = np.c_[a,b,dyn[:, -4:]]
    if mode == "summary":
        raw = np.c_[a.mean(1),b.mean(1),a[:,-1],b[:,-1],dyn[:, -4:]]
    elif mode == "conditioned":
        raw = np.c_[raw,raw*(start[:, :1]-4.75),raw*start[:,1:2],raw*start[:,2:3]]
    return np.c_[start, (start[:,0]-4.75)**2, raw]

def main():
    d = R/"ga356"
    d.mkdir(exist_ok=True)
    names = json.loads((R/"ga344/development.json").read_bytes())["successful"]
    x,m,y = inputs(names,R/"ga333",R/"ga344")
    v,n,t = inputs(VALID,R/"ga310",R/"ga313")
    ref = components(VALID,R/"ga310",R/"ga313",("generated-6",),"all-9")
    target,base,*_ = ref
    np.testing.assert_allclose(target,t,rtol=0,atol=3e-6)
    parent, pm = forecasts(ref,("generated-6",.25,.2))
    results = {}
    for kind,alpha in product(("summary","raw","conditioned"), (10.,100.,500.,2000.)):
        model = make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        model.fit(matrix(x,m,kind),y-x[:,0])
        pred = model.predict(matrix(v,n,kind))+v[:,0]
        for w in (.1,.25,.5,1.):
            results[f"{kind}/{alpha}/{w}"] = estimates((1-w)*parent+w*pred,t,base)
    best=max(results,key=lambda k:results[k]["residual_correlation"])
    record(d/"development.json",dict(selected=best,results=results,parent=pm,pairs=len(t),train_proteins=names))
    print(best,results[best],pm,flush=True)

if __name__=="__main__":
    main()
