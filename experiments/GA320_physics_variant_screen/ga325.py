"""GA325 tests more independent short trajectories at lower total integration."""
from pathlib import Path

from .ga_native240 import SAMPLES, simulate
from .ga_static_offset import VALID


if __name__=="__main__":
    dest=Path("runs/ga325")
    dest.mkdir(parents=True,exist_ok=True)
    simulate(VALID,1,destination=dest,samples=tuple(t for t in SAMPLES if t<=25),
             branches=16)
