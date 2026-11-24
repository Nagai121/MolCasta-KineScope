"""GA385 completes two-force-field matching for training at later origins."""
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
D=R/"ga385"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("lane",type=int,choices=(0,1))
    rank=p.parse_args().lane
    plan=json.loads((R/"ga365/manifest.json").read_bytes())
    names=tuple(plan["proteins"])
    assert not set(names)&set(VALID)
    D.mkdir(exist_ok=True)
    meta=dict(proteins=names,assigned=names[rank::2],origins=(149,249),
              seeds=2,window=WINDOW,recovery={},failures={})
    fn=partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    for n in meta["assigned"]:
        args=dict(destination=D,samples=WINDOW,branches=2,
                  setup_fn=fn,parts_fn=parts,minimize_iterations=60)
        try:
            simulate((n,),2,step_fs=2,**args)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            meta["recovery"][n]=dict(step_fs=1,complete=sorted(x.name for x in D.glob(f"{n}-*.npz")))
            record(D/f"lane-{rank}.json",meta)
            try:
                simulate((n,),2,step_fs=1,**args)
            except openmm.OpenMMException as second:
                meta["failures"][n]=str(second)
        record(D/f"lane-{rank}.json",meta)

if __name__=="__main__":
    main()
