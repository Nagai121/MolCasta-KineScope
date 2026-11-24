"""GA340 measures whether trained short generators complement the fixed blend."""
from itertools import product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga314 import field, load
from .ga333 import inputs
from .ga_static_offset import VALID, estimates


def main():
    root=Path("runs")
    dest=root/"ga340"
    dest.mkdir(parents=True,exist_ok=True)
    a,y,b,p,_,g=load(VALID,root/"ga310")
    c,*_=load(VALID,root/"ga313")
    pred=[]
    for path in (root/"ga318/all-8.ubj",root/"ga330/all-9.ubj"):
        _,x,_=inputs(VALID,root/"ga325",2)
        m=XGBRegressor(device="cpu")
        m.load_model(path)
        pred.append(m.predict(x)+x[:,0]-b)
    static=.525*np.mean(pred,axis=0)
    dynamic=.2*(field(a,10,"stable")+c.mean((0,1))-2*p)
    base=b+dynamic+static
    report={"reference":estimates(base,y,b)}
    for code in ("ga333","ga338"):
        x,s,truth=inputs(VALID,root/"ga325",2,relational=code=="ga338")
        np.testing.assert_allclose(truth,y,rtol=0,atol=3.e-6)
        for model,weight in product(("generated-3","joint-3","joint-5","joint-7"),(.1,.25,.5)):
            learner=XGBRegressor(device="cpu")
            learner.load_model(root/code/f"{model}.ubj")
            h=np.c_[s[:,:3] if model.startswith("generated") else s,x]
            v=learner.predict(h)+s[:,0]
            predicted=(1-weight)*base+weight*v
            report[f"{code}/{model}/{weight}"]=estimates(predicted,y,b)
    ordered=sorted(report,key=lambda k:-report[k]["residual_correlation"])
    record(dest/"development.json",dict(selected=ordered[:6],trials=report))
    print([(n,report[n]) for n in ordered[:6]],flush=True)


if __name__=="__main__":
    main()
