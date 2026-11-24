"""Collect longer trajectories from the same static origins on validation only."""
from pathlib import Path

from .ga_native240 import simulate, SAMPLES
from .ga_static_offset import VALID


TIMES = (*SAMPLES, 125, 150, 200, 250, 300, 400, 500)


if __name__ == "__main__":
    destination = Path("runs/ga307/2fs")
    destination.mkdir(parents=True, exist_ok=True)
    simulate(VALID, 3, destination=destination,
             samples=TIMES, branches=2, step_fs=2, minimize_iterations=100)
