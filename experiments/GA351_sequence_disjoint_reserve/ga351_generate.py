"""Prepare blind GA351 generation without evaluating labels."""
import argparse
import json
from functools import partial
from pathlib import Path

import openmm

from .explore import record
from .ga365 import parts
from .ga_native240 import SAMPLES, simulate
from .ga_openmm_pilot import build

D=Path("runs/ga351/blind")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("field",choices=("amber","charmm"))
    args=parser.parse_args()
    roster=json.loads(Path("runs/ga351/reserved.json").read_bytes())["proteins"]
    names=tuple(p["id"] for p in roster)
    d=D/args.field
    d.mkdir(parents=True,exist_ok=True)
    is_amber=args.field=="amber"
    step,branches,iterations=(4,8,25) if is_amber else (2,2,60)
    setup=build if is_amber else partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    record(d/"locked.json",dict(proteins=names,origin=49,window=100,
                                branches=branches,step_fs=step,minimize=iterations))
    for n in names:
        options=dict(destination=d,samples=SAMPLES,branches=branches,
                     setup_fn=setup,parts_fn=partial(parts,origins=(49,)))
        try:
            simulate((n,),1,step_fs=step,minimize_iterations=iterations,**options)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            record(d/f"{n}-recovery.json",dict(reason=str(exc),step_fs=step//2,minimize=100))
            simulate((n,),1,step_fs=step//2,minimize_iterations=100,**options)

if __name__=="__main__":
    main()

