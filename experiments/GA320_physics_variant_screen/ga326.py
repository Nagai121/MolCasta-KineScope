"""GA326 samples local motion under weak restraints on the origin backbone."""
from functools import partial
from pathlib import Path

from .ga_native240 import SAMPLES, simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID


if __name__=="__main__":
    dest=Path("runs/ga326")
    dest.mkdir(parents=True,exist_ok=True)
    simulate(VALID,1,destination=dest,samples=tuple(t for t in SAMPLES if t<=50),
             branches=2,step_fs=2,minimize_iterations=60,
             setup_fn=partial(build,ca_restraint=125.))
