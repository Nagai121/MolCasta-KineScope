"""Reserve new confirmation proteins and stage only their input data."""
import argparse
from pathlib import Path

from .ga351 import main as reserve

OUT = Path("runs/ga841-fresh-panel")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire", action="store_true")
    options = parser.parse_args()
    design = OUT / "reserved.json"
    if not design.exists():
        reserve(design, "GA-841/")
    if options.acquire:
        from .ga351_acquire import main as acquire
        acquire(OUT)


if __name__=="__main__":
    main()
