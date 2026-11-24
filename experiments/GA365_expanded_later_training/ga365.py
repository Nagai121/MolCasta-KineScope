"""GA365 expands later-origin motion training without changing validation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import openmm

from tseries.data import shard
from tseries.hb_single_structure import ROOT
from .explore import record
from .ga333 import WINDOW
from .ga_native240 import simulate
from .ga_static_offset import VALID

R=Path("runs")
D=R/"ga365"

def parts(names, origins=(149,249)):
    items=[]
    count=0
    for name in names:
        with np.load(shard(ROOT,name,1,100,8.)) as z:
            keep=(z["pair"][:,2]<=6.)&np.isin(z["origin"],origins)
            pair,origin=z["pair"][keep],z["origin"][keep]
        if not len(pair):
            raise RuntimeError(f"No later candidates for {name}")
        ends=np.r_[np.flatnonzero(np.diff(origin))+1,len(origin)]
        items.append(dict(protein=name,replica=1,pair=pair,origin=origin,
                          ends=ends,y=np.zeros(len(origin))))
        count+=len(origin)
    return items,{"combined":np.empty((count,0))},np.zeros(count)

def main():
    D.mkdir(exist_ok=True)
    train=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    assert not set(train)&set(VALID)
    available=sorted(train,key=lambda n:hashlib.sha256(("GA365/"+n).encode()).digest())
    names=[n for n in available if not all((R/"ga_native240"/f"{n}-{i}.npz").exists()
                                           for i in (149,249))][:24]
    protocol=dict(proteins=names,samples=WINDOW,branches=2,origins=(149,249),
                  default_step_fs=4,minimization=25,failures={},recovery={})
    for n in names:
        try:
            simulate((n,),2,destination=D,samples=WINDOW,branches=2,parts_fn=parts)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            completed=sorted(p.name for p in D.glob(f"{n}-*.npz"))
            protocol["recovery"][n]=dict(completed=completed,step_fs=2,minimization=100)
            record(D/"manifest.json",protocol)
            try:
                simulate((n,),2,destination=D,samples=WINDOW,branches=2,parts_fn=parts,
                         step_fs=2,minimize_iterations=100)
            except openmm.OpenMMException as second:
                protocol["failures"][n]=str(second)
        record(D/"manifest.json",protocol)

if __name__=="__main__":
    main()

