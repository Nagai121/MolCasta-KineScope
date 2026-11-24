"""GA312 fixed eight-seed plus static-force confirmation on first origins."""
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from tseries.data import TEST
from .explore import record
from .ga_force import build
from .ga_native240 import SAMPLES, simulate
from .ga_native_offset import HOLD, native
from .ga309 import predict
from .ga_static_offset import cohort, estimates, extract


def main():
    dest = Path("runs/ga312")
    dest.mkdir(parents=True, exist_ok=True)
    proteins = TEST + HOLD
    rule = dict(proteins=proteins, origins=1, seeds=8, time_ps=100,
                terms=dict(projected_mean=.4, static_force=.525))
    record(dest/"locked.json", rule)
    simulate(proteins,1,destination=dest,samples=SAMPLES,branches=8)
    rows=cohort(proteins, 1)
    y,x=extract(rows)
    s=x["compact"]
    source=Path("runs/ga_native_offset")
    with np.load(source/"baselines.npz") as z, np.load(source/"residual-ridge.npz") as m:
        base=predict(z,"real_baseline_",s)
        proj=predict(z,"native_baseline_",s)
        r305=predict(m,"",native(rows)-proj)
    forces=np.concatenate([build((n,1,1))[0] for n in proteins])
    np.testing.assert_allclose(s[:,0],forces[:,0],rtol=0,atol=1.e-6)
    head=XGBRegressor(device="cpu")
    head.load_model("runs/ga_force/all-8.ubj")
    r_force=head.predict(forces)+forces[:,0]-base
    mean=[]
    times=[]
    for r in rows:
        with np.load(dest/f"{r['protein']}-{r['origin']}.npz") as z:
            np.testing.assert_array_equal(z["pair"],r["pair"])
            mean.append(z["tracks"].mean((0,1))-r["pair"][:,2])
            times.append(float(z["prep"])+z["contexts"].sum()+z["costs"][:,-1].sum())
    r312=.4*(np.concatenate(mean)-proj[:,0])+.525*r_force
    metrics={"ga305":estimates(base+r305,y,base),
             "ga309":estimates(base+r305+.525*r_force,y,base),
             "ga312":estimates(base+r312,y,base)}
    record(dest/"confirmation.json",dict(rule=rule,pairs=len(y),metrics=metrics,
                                        seconds_by_origin=times))
    np.savez(dest/"evaluation.npz",target=y,static=base,ga305=base+r305,
             ga309=base+r305+.525*r_force,ga312=base+r312)
    print(metrics,flush=True)


if __name__=="__main__":
    main()
