"""Estimate the value of independent seeds at fixed generated time."""
from pathlib import Path

from .ga_native240 import simulate, SAMPLES
from .ga_static_offset import VALID


if __name__ == "__main__":
    destination = Path("runs/ga310")
    destination.mkdir(parents=True, exist_ok=True)
    simulate(VALID, 1, destination=destination, samples=SAMPLES, branches=8)
