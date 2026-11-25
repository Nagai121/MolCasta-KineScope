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


def main(panel=Path("runs/ga351"), field=None, resolve_terminal_histidine=False):
    parser=argparse.ArgumentParser()
    parser.add_argument("field",choices=("amber","charmm"))
    field=field or parser.parse_args().field
    roster=json.loads((panel/"reserved.json").read_bytes())["proteins"]
    names=tuple(p["id"] for p in roster)
    d=panel/"blind"/field
    d.mkdir(parents=True,exist_ok=True)
    is_amber=field=="amber"
    step,branches,iterations=(4,8,25) if is_amber else (2,2,60)
    setup=build if is_amber else partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    if resolve_terminal_histidine and not is_amber:
        from .ga_terminal_histidine import NeutralTerminalHistidine
        setup=partial(setup,forcefield_class=NeutralTerminalHistidine)
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




