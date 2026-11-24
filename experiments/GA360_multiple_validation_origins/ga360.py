"""GA360 generates matched force-field ensembles at later validation origins."""
from functools import partial
from pathlib import Path
from shutil import copyfile
import argparse
import openmm
from .explore import record

from .ga_native240 import SAMPLES, simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("field", choices=("amber", "charmm"))
    selected = parser.parse_args().field
    root=Path("runs")
    for label,existing,branches,step,minimum,fn in (
        ("amber","ga310",8,4,25,build),
        ("charmm","ga313",2,2,60,partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml")))):
        if label != selected:
            continue
        directory=root/"ga360"/label
        directory.mkdir(parents=True,exist_ok=True)
        for protein in VALID:
            name=f"{protein}-49.npz"
            if not (directory/name).exists():
                copyfile(root/existing/name,directory/name)
        for protein in VALID:
            before = {p.name for p in directory.glob(f"{protein}-*.npz")}
            try:
                simulate((protein,),3,destination=directory,samples=SAMPLES,branches=branches,
                         step_fs=step,minimize_iterations=minimum,setup_fn=fn)
            except openmm.OpenMMException as exc:
                if "NaN" not in str(exc):
                    raise
                completed = {p.name for p in directory.glob(f"{protein}-*.npz")}
                fallback = step//2
                record(directory/f"{protein}-recovery.json",
                       dict(reason=str(exc),original_step_fs=step,fallback_step_fs=fallback,
                            completed=sorted(completed),inherited=sorted(before)))
                simulate((protein,),3,destination=directory,samples=SAMPLES,branches=branches,
                         step_fs=fallback,minimize_iterations=100,setup_fn=fn)

if __name__=="__main__":
    main()

