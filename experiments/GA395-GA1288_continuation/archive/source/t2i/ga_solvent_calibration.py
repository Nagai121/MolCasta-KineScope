"""GA1254+: assess solvent correction reliability by leaving one development protein out."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from .ga333 import WINDOW
from .ga_static_offset import VALID,estimates

R=Path("runs")
OUT=R/"ga1254-solvent-calibration"


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(R/"ga360/validation.npz") as data:
        y,base,names,origin=[data[k] for k in ("target","baseline","proteins","origin")]
    with np.load(R/"ga1077-reliability-width/GA1086.npz") as data:
        parent=data["prediction"]
    row=np.empty((len(y),6),np.float32)
    for name in VALID:
        for time in (49,149,249):
            f=f"{name}-{time}.npz"
            directory="ga1236-explicit-solvent" if time==49 else "ga1237-explicit-later"
            with np.load(R/directory/f) as explicit,np.load(R/"ga360/amber"/f) as implicit:
                np.testing.assert_array_equal(explicit["pair"],implicit["pair"])
                a=explicit["tracks"][:2,:len(WINDOW)]
                b=implicit["tracks"][:2,:len(WINDOW)]
                offset=a.mean((0,1))-b.mean((0,1))
                occupancy_sd=a.mean(1).std(0)+b.mean(1).std(0)
                mask=(names==name)&(origin==time)
                row[mask]=np.c_[offset,offset-np.median(offset),occupancy_sd,
                                a.std((0,1)),b.std((0,1)),explicit["pair"][:,2]]
    methods={"raw":row[:,0:1],"centered":row[:,1:2],
             "clipped":row[:,1:2].clip(-.5,.5),
             "reliability":row[:,1:2]/(.20+row[:,2:3]),
             "context":np.c_[row[:,1:2],row[:,1:2]*(row[:,2:5]-.15),
                              row[:,1]*(row[:,5]-4.25)]}
    out={}
    for i,(tag,form) in enumerate(methods.items()):
        prediction=parent.copy()
        factors={}
        for name in VALID:
            train=names!=name
            model=Ridge(alpha=200,fit_intercept=False).fit(form[train],y[train]-parent[train])
            prediction[~train]+=model.predict(form[~train])
            factors[name]=model.coef_.tolist()
        scores={k:estimates(prediction[m],y[m],base[m]) for k,m in
                (("all",origin>=0),("first",origin==49),("later",origin!=49))}
        key=f"GA{1254+i}"
        out[key]=dict(method=tag,metrics=scores,factors=factors)
        np.savez_compressed(OUT/f"{key}.npz",prediction=prediction,parent=parent)
        print(key,tag,scores["all"]["residual_correlation"],flush=True)
    (OUT/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
