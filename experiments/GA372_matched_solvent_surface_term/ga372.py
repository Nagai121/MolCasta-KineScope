"""GA372 changes only the nonpolar implicit-solvent surface model."""
import argparse
from functools import partial
from pathlib import Path

from .explore import record
from .ga333 import WINDOW
from .ga365 import parts
from .ga_native240 import simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID

def main():
    p=argparse.ArgumentParser()
    p.add_argument("sasa",choices=("ACE","LCPO"))
    m=p.parse_args().sasa
    base=Path("runs/ga372")/m
    base.mkdir(parents=True,exist_ok=True)
    record(base/"protocol.json",dict(surface=m,origins=(49,149,249),samples=WINDOW,
                                     branches=2,step_fs=2,minimization=100))
    simulate(VALID,3,destination=base,samples=WINDOW,branches=2,step_fs=2,
             minimize_iterations=100,setup_fn=partial(build,sasa_method=m),
             parts_fn=partial(parts,origins=(49,149,249)))

if __name__=="__main__":
    main()
