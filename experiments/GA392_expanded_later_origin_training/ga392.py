"""GA392 extends matched later-origin dynamics to the remaining training proteins."""
import argparse
import json
from functools import partial
from pathlib import Path
import openmm

from .explore import record
from .ga333 import WINDOW
from .ga365 import parts
from .ga_native240 import simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID

R=Path("runs")
D=R/"ga392"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("lane",type=int,choices=(0,1,2))
    rank=p.parse_args().lane
    first=tuple(json.loads((R/"ga355/development.json").read_bytes())["successful"])
    prev=set(json.loads((R/"ga365/manifest.json").read_bytes())["proteins"])
    candidates=tuple(n for n in first if n not in prev)
    assert not set(candidates)&set(VALID)
    fields=("amber","charmm")
    by_field={name:D/name for name in fields}
    for directory in by_field.values():
        directory.mkdir(parents=True,exist_ok=True)
    meta=dict(proteins=candidates,assigned=candidates[rank::3],origins=(149,249),
              samples=WINDOW,branches=2,failures={},recovery={})
    for n in meta["assigned"]:
        for label in fields:
            folder=by_field[label]
            is_amber=label=="amber"
            step,iterations=(4,25) if is_amber else (2,60)
            fn=build if is_amber else partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
            kwargs=dict(destination=folder,samples=WINDOW,branches=2,setup_fn=fn,parts_fn=parts)
            key=f"{label}/{n}"
            try:
                simulate((n,),2,step_fs=step,minimize_iterations=iterations,**kwargs)
            except openmm.OpenMMException as exc:
                if "NaN" not in str(exc):
                    raise
                meta["recovery"][key]=dict(step_fs=step//2,minimization=100,
                                           already=sorted(q.name for q in folder.glob(f"{n}-*.npz")))
                record(D/f"lane-{rank}.json",meta)
                try:
                    simulate((n,),2,step_fs=step//2,minimize_iterations=100,**kwargs)
                except openmm.OpenMMException as second:
                    meta["failures"][key]=str(second)
            record(D/f"lane-{rank}.json",meta)

if __name__=="__main__":
    main()
