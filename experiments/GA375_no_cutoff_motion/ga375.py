"""GA375 compares all-range implicit electrostatics with a matched cutoff control."""
from functools import partial
from pathlib import Path

from .explore import record
from .ga333 import WINDOW
from .ga365 import parts
from .ga_native240 import simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID

def main():
    d=Path("runs/ga375")
    d.mkdir(exist_ok=True)
    record(d/"protocol.json",dict(samples=WINDOW,branches=2,step_fs=2,minimization=100,
                                  cutoff=False,control="GA372/ACE",origins=(49,149,249)))
    simulate(VALID,3,destination=d,samples=WINDOW,branches=2,step_fs=2,
             minimize_iterations=100,setup_fn=partial(build,cutoff_nm=None),
             parts_fn=partial(parts,origins=(49,149,249)))

if __name__=="__main__":
    main()
