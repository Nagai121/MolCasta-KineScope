"""GA377 produces local coordinate histories rather than scalar-distance histories alone."""
import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import openmm

from .explore import record
from .ga333 import WINDOW
from .ga365 import parts
from .ga_native240 import simulate
from .ga_static_offset import VALID

R=Path("runs")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("split",choices=("train","validation"))
    p.add_argument("--lane",type=int,choices=(0,1))
    args=p.parse_args()
    split=args.split
    d=R/"ga377"/split
    d.mkdir(parents=True,exist_ok=True)
    roster=json.loads((R/"ga355/development.json").read_bytes())["successful"]
    train=tuple(sorted(roster,key=lambda n:hashlib.sha256(("GA377/"+n).encode()).digest())[:24])
    assert not set(train)&set(VALID)
    names=train if split=="train" else VALID
    starts=(49,149) if split=="train" else (49,149,249)
    meta=dict(proteins=names,origins=starts,samples=WINDOW,
              branches=2,step_fs=2,minimization=100,recoveries={},failures={})
    run_names=names if args.lane is None else names[args.lane::2]
    manifest=d/("manifest.json" if args.lane is None else f"manifest-lane{args.lane}.json")
    meta["assigned"]=run_names
    for n in run_names:
        kw=dict(destination=d,samples=WINDOW,branches=2,coordinate_cache=True,
                minimize_iterations=100,parts_fn=partial(parts,origins=starts))
        try:
            simulate((n,),len(starts),step_fs=2,**kw)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            meta["recoveries"][n]=dict(step_fs=1,completed=sorted(f.name for f in d.glob(f"{n}-*.npz")))
            record(manifest,meta)
            try:
                simulate((n,),len(starts),step_fs=1,**kw)
            except openmm.OpenMMException as exc2:
                meta["failures"][n]=str(exc2)
        record(manifest,meta)

if __name__=="__main__":
    main()

