"""GA350 compares many short CHARMM branches against the two long branches."""
from functools import partial
from pathlib import Path

from .ga_native240 import SAMPLES, simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID


if __name__=="__main__":
    dest=Path("runs/ga350")
    dest.mkdir(parents=True,exist_ok=True)
    fn=partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    simulate(VALID,1,destination=dest,samples=tuple(t for t in SAMPLES if t<=25),
             branches=16,step_fs=2,minimize_iterations=60,setup_fn=fn)
