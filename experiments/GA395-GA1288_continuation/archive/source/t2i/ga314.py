"""GA314: infer a pair-level relaxation from an eight-seed ensemble."""
from itertools import product
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST
from .explore import record
from .ga_force import build
from .ga_native240 import SAMPLES
from .ga_native_offset import HOLD
from .ga309 import predict
from .ga_static_offset import VALID, cohort, estimates, extract


def load(names, cache):
    rows = cohort(names, 1)
    y, x = extract(rows)
    s = x["compact"]
    packets = []
    for r in rows:
        with np.load(cache / f"{r['protein']}-{r['origin']}.npz") as z:
            np.testing.assert_array_equal(r["pair"], z["pair"])
            packets.append(z["tracks"]-r["pair"][:,2])
    data = np.concatenate(packets,axis=2)
    with np.load("runs/ga_native_offset/baselines.npz") as z:
        base = predict(z,"real_baseline_",s)
        proj = predict(z,"native_baseline_",s)[:,0]
    f = np.concatenate([build((r["protein"],1,1))[0] for r in rows])
    np.testing.assert_allclose(f[:,0],s[:,0],rtol=0,atol=1e-6)
    force = XGBRegressor(device="cpu")
    force.load_model("runs/ga_force/all-8.ubj")
    g = np.concatenate([np.repeat(r["protein"],len(r["pair"])) for r in rows])
    return data, y, base, proj, force.predict(f)+f[:,0]-base, g


def field(t, time, aggregation):
    grid = np.asarray(SAMPLES)
    take = grid >= time
    s = t[:,take]
    mu = s.mean(1)
    if aggregation == "mean":
        return mu.mean(0)
    if aggregation == "median":
        return np.median(mu,axis=0)
    if aggregation == "trim2":
        return np.sort(mu,axis=0)[1:-1].mean(0)
    if aggregation == "stable":
        w = 1/(.10 + s.std(1))
        return np.sum(mu*w,axis=0)/np.sum(w,axis=0)
    if aggregation == "trapezoid":
        return np.trapezoid(s.mean(0),grid[take],axis=0)/(grid[take][-1]-grid[take][0])
    raise ValueError(aggregation)


def main():
    dest = Path("runs/ga314")
    dest.mkdir(parents=True,exist_ok=True)
    a,y,b,p,f,g = load(VALID, Path("runs/ga310"))
    trials=[]
    for time,method in product((0,10,25,40,60),("mean","median","trim2","stable","trapezoid")):
        x=field(a,time,method)-p
        for w in (0.,.5,1.,1.3125,1.75,2.5):
            v=b+.4*(x+w*f)
            local=[estimates(v[g==n],y[g==n],b[g==n])["residual_correlation"] for n in VALID]
            trials.append(dict(time=time,method=method,force_ratio=w,
                               macro=float(np.mean(local)),micro=estimates(v,y,b)))
    ordered=sorted(trials,key=lambda t:t["macro"],reverse=True)
    record(dest/"development.json",dict(selected=ordered[0],trials=ordered))
    print(ordered[:5],flush=True)


if __name__ == "__main__":
    main()
