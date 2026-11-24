"""GA341 perturbs solvent dielectric at matched initial coordinates and seeds."""
from functools import partial
from pathlib import Path

from .ga_native240 import SAMPLES, simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID


if __name__=="__main__":
    for dielectric in (40,100):
        dest=Path(f"runs/ga341/{dielectric}")
        dest.mkdir(parents=True,exist_ok=True)
        simulate(VALID,1,destination=dest,samples=tuple(t for t in SAMPLES if t<=25),
                 branches=4,step_fs=2,minimize_iterations=60,
                 setup_fn=partial(build,solvent_dielectric=dielectric))
