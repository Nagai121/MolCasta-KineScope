"""GA339 fixes the thermal gradient learned in GA337 and confirms it."""
from functools import partial
from pathlib import Path
import json

import numpy as np

from tseries.data import TEST
from .explore import record
from .ga314 import field, load
from .ga_native240 import simulate, SAMPLES
from .ga_native_offset import HOLD
from .ga_openmm_pilot import build
from .ga_static_offset import estimates


def main():
    dest=Path("runs/ga339")
    dest.mkdir(parents=True,exist_ok=True)
    names=TEST+HOLD
    key=json.loads(Path("runs/ga337/development.json").read_bytes())["selected"][0]
    source,amount,gradient=key.split("/")
    amount,gradient=float(amount),float(gradient)
    rule=dict(key=key,proteins=names,origins=1,temperature_K=(275,330),seeds=4,ps=25)
    record(dest/"locked.json",rule)
    for temperature in (275,330):
        directory=dest/f"{temperature}K"
        directory.mkdir(exist_ok=True)
        simulate(names,1,destination=directory,samples=tuple(t for t in SAMPLES if t<=25),
                 branches=4,step_fs=2,minimize_iterations=60,
                 setup_fn=partial(build,temperature=temperature))
    a,y,b,p,_,g=load(names,Path("runs/ga312"))
    c,*_=load(names,Path("runs/ga321"))
    lo,*_=load(names,dest/"275K")
    hi,*_=load(names,dest/"330K")
    lo=lo.mean((0,1))-p
    hi=hi.mean((0,1))-p
    selected={"low":lo,"high":hi,"mean":.5*(lo+hi)}[source]
    with np.load("runs/ga332/evaluation.npz") as z:
        np.testing.assert_array_equal(y,z["target"])
        np.testing.assert_array_equal(b,z["static"])
        np.testing.assert_array_equal(g,z["proteins"])
        parent=z["predicted"]
    dynamic=.2*(field(a,10,"stable")+c.mean((0,1))-2*p)
    predicted=parent-amount*dynamic+.4*amount*selected+gradient*(hi-lo)
    out=dict(rule=rule,pairs=len(y),metrics=estimates(predicted,y,b),parent=estimates(parent,y,b))
    record(dest/"confirmation.json",out)
    np.savez(dest/"evaluation.npz",predicted=predicted,target=y,static=b,proteins=g)
    print(out,flush=True)


if __name__=="__main__":
    main()
