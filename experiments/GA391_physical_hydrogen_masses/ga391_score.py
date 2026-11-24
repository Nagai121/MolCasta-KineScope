"""Compare physical hydrogen masses against a repartitioned matched control."""
from pathlib import Path
import numpy as np

from .explore import record
from .ga_static_offset import VALID, cohort, estimates

R=Path("runs")
D=R/"ga391"

def main():
    rows=cohort(VALID,3)
    series={}
    for surface in ("ACE","physical_mass"):
        groups=[]
        for row in rows:
            folder=R/"ga372/ACE" if surface=="ACE" else D
            with np.load(folder/f"{row['protein']}-{row['origin']}.npz") as z:
                np.testing.assert_array_equal(row["pair"],z["pair"])
                groups.append(z["tracks"].mean((0,1))-row["pair"][:,2])
        series[surface]=np.concatenate(groups)
    with np.load(R/"ga360/validation.npz") as z:
        y,b,parent,o=z["target"],z["baseline"],z["parent"],z["origin"]
    origin=np.concatenate([np.repeat(row["origin"],len(row["pair"])) for row in rows])
    np.testing.assert_array_equal(origin,o)
    delta=series["physical_mass"]-series["ACE"]
    masks={"all":np.ones(len(o),bool),"first":o==49,"later":o!=49}
    trials={}
    for w in (-.5,-.25,0.,.1,.25,.5):
        p=parent+w*delta
        trials[str(w)]={k:estimates(p[m],y[m],b[m]) for k,m in masks.items()}
    record(D/"development.json",dict(trials=trials,delta_sd=float(delta.std()),
                                     delta_correlation=float(np.corrcoef(delta,y-b)[0,1])))
    print({k:{g:v["residual_correlation"] for g,v in q.items()} for k,q in trials.items()},flush=True)

if __name__=="__main__":
    main()


