"""GA323: overdamped short branches as an alternate static-origin readout."""
from functools import partial
from pathlib import Path

from .ga_native240 import SAMPLES, simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID


if __name__=="__main__":
    dest=Path("runs/ga323")
    dest.mkdir(parents=True,exist_ok=True)
    fast_window=tuple(t for t in SAMPLES if t<=50)
    simulate(VALID,1,destination=dest,samples=fast_window,branches=4,
             step_fs=2,minimize_iterations=60,setup_fn=partial(build,friction=10.))
