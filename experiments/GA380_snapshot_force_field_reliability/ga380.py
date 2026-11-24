"""Compare snapshot-level reliability weighting for generated force fields."""
from pathlib import Path

import numpy as np

from .explore import record
from .ga309 import predict
from .ga314 import field
from .ga_static_offset import VALID, cohort, estimates, extract

R=Path("runs")

def main():
    rows=cohort(VALID,3)
    y,z=extract(rows)
    with np.load(R/"ga_native_offset/baselines.npz") as f:
        prior=predict(f,"native_baseline_",z["compact"])[:,0]
    with np.load(R/"ga360/validation.npz") as f:
        np.testing.assert_allclose(y,f["target"],atol=3e-6,rtol=0)
        parent,base,origin,protein=(f[k] for k in ("parent","baseline","origin","proteins"))
    packets=[]
    for r in rows:
        tracks=[]
        for label in ("amber","charmm"):
            with np.load(R/"ga360"/label/f"{r['protein']}-{r['origin']}.npz") as f:
                np.testing.assert_array_equal(r["pair"],f["pair"])
                tracks.append(f["tracks"]-r["pair"][:,2])
        a,c=tracks
        u,v=field(a,10,"stable"),c.mean((0,1))
        va=np.mean(a.mean(1).var(0))
        vc=np.mean(c.mean(1).var(0))
        packets.append((u,v,np.full(len(u),va),np.full(len(u),vc)))
    a,c,va,vc=map(np.concatenate,zip(*packets))
    trials={}
    for floor in (.01,.04,.1):
        reliability=(vc+floor)/(va+vc+2*floor)-.5
        for strength in (.25,.5,1.):
            forecast=parent+.32*strength*reliability*((a-prior)-(c-prior))
            trials[f"{floor}/{strength}"]={
                "pooled":estimates(forecast,y,base),
                "first":estimates(forecast[origin==49],y[origin==49],base[origin==49]),
                "later":estimates(forecast[origin!=49],y[origin!=49],base[origin!=49]),
                "macro":float(np.mean([estimates(forecast[protein==n],y[protein==n],base[protein==n])["residual_correlation"] for n in VALID]))
            }
    best=max(trials,key=lambda k:trials[k]["pooled"]["residual_correlation"])
    d=R/"ga380"
    d.mkdir(exist_ok=True)
    record(d/"development.json",dict(parent=estimates(parent,y,base),trials=trials,best=best))
    print(best,trials[best],flush=True)

if __name__=="__main__":
    main()
