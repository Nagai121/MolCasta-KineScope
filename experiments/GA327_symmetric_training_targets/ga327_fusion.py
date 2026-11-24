"""Check whether symmetric supervision improves the fixed dynamic blend."""
from itertools import product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga314 import load, field
from .ga_force import build
from .ga_static_offset import VALID, estimates


def main():
    root=Path("runs")
    a,y,b,p,_,g=load(VALID,root/"ga310")
    c,*_=load(VALID,root/"ga313")
    r=.2*(field(a,10,"stable")+c.mean((0,1))-2*p)
    x=np.concatenate([build((n,1,1))[0] for n in VALID])
    f={}
    for path in [root/"ga318/all-8.ubj",*sorted((root/"ga327").glob("*.ubj"))]:
        m=XGBRegressor(device="cpu")
        m.load_model(path)
        f[path.stem]=m.predict(x)+x[:,0]-b
    scores={}
    for name,amount,mix in product(f,(.40,.525,.65,.75),(.5,.75,1.)):
        predicted=b+r+amount*(mix*f[name]+(1-mix)*f["all-8"])
        s=estimates(predicted,y,b)
        s["macro"]=float(np.mean([estimates(predicted[g==n],y[g==n],b[g==n])["residual_correlation"]
                                    for n in VALID]))
        scores[f"{name}/{amount}/{mix}"]=s
    ranked=sorted(scores,key=lambda t:-scores[t]["macro"])
    record(root/"ga327/fusion.json",dict(selected=ranked[:8],scores=scores))
    print([(k,scores[k]) for k in ranked[:5]],flush=True)


if __name__=="__main__":
    main()
