"""GA332 applies GA330's selected blend to a fixed confirmation panel."""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST
from .explore import record
from .ga314 import field, load
from .ga_force import build
from .ga_native_offset import HOLD
from .ga_static_offset import estimates


def main():
    root=Path("runs")
    dest=root/"ga332"
    dest.mkdir(parents=True,exist_ok=True)
    selected=json.loads((root/"ga330/fusion.json").read_bytes())["selected"]
    name,share,weight=selected.split("/")
    share,weight=float(share),float(weight)
    rule=dict(recipe=selected,proteins=TEST+HOLD,origins=1)
    record(dest/"locked.json",rule)
    names=TEST+HOLD
    a,y,b,p,_,g=load(names,root/"ga312")
    c,y2,b2,p2,_,g2=load(names,root/"ga321")
    for q,r in ((y,y2),(b,b2),(p,p2),(g,g2)):
        np.testing.assert_array_equal(q,r)
    x=np.concatenate([build((n,1,1))[0] for n in names])
    scores=[]
    for path in (root/"ga318/all-8.ubj",root/"ga330"/f"{name}.ubj"):
        head=XGBRegressor(device="cpu")
        head.load_model(path)
        score=head.predict(x)+x[:,0]-b
        scores.append(score)
    z=.2*(field(a,10,"stable")+c.mean((0,1))-2*p)
    predictions=b+z+weight*((1-share)*scores[0]+share*scores[1])
    parents=b+z+.525*scores[0]
    rng=np.random.default_rng(332)
    indexes=[np.flatnonzero(g==n) for n in names]
    bounds=[]
    for sample in rng.integers(len(names),size=(2000,len(names))):
        i=np.concatenate([indexes[j] for j in sample])
        prev=estimates(parents[i],y[i],b[i])["residual_correlation"]
        new=estimates(predictions[i],y[i],b[i])["residual_correlation"]
        bounds.append((new,new-prev))
    report=dict(rule=rule,pooled=estimates(predictions,y,b),parent=estimates(parents,y,b),
                protein_bootstrap_95=np.quantile(bounds,[.025,.975],axis=0).tolist(),
                pairs=len(y))
    record(dest/"confirmation.json",report)
    np.savez(dest/"evaluation.npz",predicted=predictions,target=y,static=b,proteins=g)
    print(report,flush=True)


if __name__=="__main__":
    main()
