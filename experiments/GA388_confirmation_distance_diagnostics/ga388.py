"""GA388 diagnoses panel shift by origin distance, without changing the predictor."""
import json
from pathlib import Path
import numpy as np

from tseries.data import TEST
from .explore import record
from .ga_native_offset import HOLD
from .ga_static_offset import cohort, estimates

R=Path("runs")

def stats(mask,p,y,b):
    return dict(pairs=int(mask.sum()),metrics=estimates(p[mask],y[mask],b[mask]))

def main():
    plan=json.loads((R/"ga351/reserved.json").read_bytes())
    blind=tuple(q["id"] for q in plan["proteins"])
    reserve=tuple(json.loads((R/"ga321/reserve.json").read_bytes())["proteins"])
    inputs=(("previous",TEST+HOLD),("reserve",reserve),("blind-ga351",blind))
    panels={}
    for key,names in inputs:
        rows=cohort(names,1)
        distance=np.concatenate([r["pair"][:,2] for r in rows])
        roster=np.concatenate([np.repeat(r["protein"],len(r["pair"])) for r in rows])
        with np.load(R/"ga384"/f"{key}.npz") as z:
            np.testing.assert_array_equal(roster,z["proteins"])
            pred,target,base=(z[f] for f in ("predicted","target","static"))
        groups={}
        for lo,hi in ((3.5,4.),(4.,4.5),(4.5,5.),(5.,5.5),(5.5,6.001)):
            mask=(distance>=lo)&(distance<hi)
            groups[f"{lo:.1f}-{hi:.1f}"]=stats(mask,pred,target,base)
        panels[key]=dict(by_distance=groups,by_protein={n:stats(roster==n,pred,target,base) for n in names})
    dest=R/"ga388"
    dest.mkdir(exist_ok=True)
    record(dest/"diagnostics.json",dict(scope="diagnosis only; no subset is used as the primary result",panels=panels))
    print({k:{s:round(r["metrics"]["residual_correlation"],4) for s,r in d["by_distance"].items()}
           for k,d in panels.items()},flush=True)

if __name__=="__main__":
    main()
