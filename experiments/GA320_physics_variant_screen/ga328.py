"""GA328 third implicit MD parameterization at a low branch budget."""
from functools import partial
from pathlib import Path

from .ga_native240 import SAMPLES, simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID


if __name__=="__main__":
    dest=Path("runs/ga328")
    dest.mkdir(parents=True,exist_ok=True)
    fn=partial(build,force_files=("amber19-all.xml","implicit/gbn2.xml"))
    simulate(VALID,1,destination=dest,samples=tuple(t for t in SAMPLES if t<=50),
             branches=4,step_fs=2,minimize_iterations=60,setup_fn=fn)
