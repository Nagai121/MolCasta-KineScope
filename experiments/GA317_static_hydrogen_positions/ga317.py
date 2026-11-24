"""GA317 adds hydrogen positions from the same static frame, if available."""
from functools import partial
from pathlib import Path

from .ga_native240 import simulate, SAMPLES
from .ga_openmm_pilot import build
from .ga_static_offset import VALID


if __name__ == "__main__":
    dest = Path("runs/ga317")
    dest.mkdir(parents=True, exist_ok=True)
    fn = partial(build, force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    simulate(VALID,1,destination=dest,samples=SAMPLES,branches=2,step_fs=2,
             minimize_iterations=60,setup_fn=fn,preserve_hydrogens=True)
