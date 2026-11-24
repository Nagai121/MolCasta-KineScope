"""GA321 locks the scaled static and two-family ensemble for confirmation."""
from functools import partial
from pathlib import Path
import time

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST
from .explore import record
from .ga_force import build as static_features
from .ga_native240 import simulate, SAMPLES
from .ga_native_offset import HOLD
from .ga_openmm_pilot import build
from .ga314 import field, load
from .ga_static_offset import estimates


def main():
    dest=Path("runs/ga321")
    dest.mkdir(parents=True,exist_ok=True)
    names=TEST+HOLD
    rule=dict(proteins=names,origins=1,amber_seeds=8,charmm_seeds=2,
              window_ps=100,static_model="GA318/all-8",static_weight=.525,
              amber_weight=.2,charmm_weight=.2)
    record(dest/"locked.json",rule)
    builder=partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    simulate(names,1,destination=dest,samples=SAMPLES,branches=2,step_fs=2,
             minimize_iterations=60,setup_fn=builder)
    amber=Path("runs/ga312")
    while not (amber/"confirmation.json").exists():
        time.sleep(10)
    a,y,b,p,f,g=load(names,amber)
    c,y2,b2,p2,f2,g2=load(names,dest)
    for u,v in ((y,y2),(b,b2),(p,p2),(g,g2),(f,f2)):
        np.testing.assert_array_equal(u,v)
    x=np.concatenate([static_features((n,1,1))[0] for n in names])
    model=XGBRegressor(device="cpu")
    model.load_model("runs/ga318/all-8.ubj")
    s=model.predict(x)+x[:,0]-b
    r=.2*(field(a,10,"stable")+c.mean((0,1))-2*p)+.525*s
    pred=b+r
    summary=dict(pairs=len(y),rule=rule,pooled=estimates(pred,y,b),
                 proteins={n:estimates(pred[g==n],y[g==n],b[g==n]) for n in names})
    record(dest/"confirmation.json",summary)
    np.savez(dest/"evaluation.npz",predicted=pred,target=y,static=b,proteins=g)
    print(summary,flush=True)


if __name__=="__main__":
    main()
