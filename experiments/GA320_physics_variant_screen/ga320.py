"""GA320: same-origin comparison of refined implicit MD settings."""
from pathlib import Path

from .ga_native240 import SAMPLES, simulate
from .ga_static_offset import VALID


if __name__ == "__main__":
    dest=Path("runs/ga320")
    dest.mkdir(parents=True,exist_ok=True)
    simulate(VALID,1,destination=dest,samples=SAMPLES,branches=4,
             step_fs=2,minimize_iterations=100)
