"""Help the GA963 job prepare its last six CHARMM proteins independently."""
from functools import partial
import json
import os
from pathlib import Path

import openmm

from .ga333 import WINDOW
from .ga365 import parts
from .ga963_protein_expansion import OUT
from .ga_native240 import simulate
from .ga_openmm_pilot import build


def main(later=False, middle=False):
    out=Path("runs/ga1029-later-expansion") if later else OUT
    frames=(149,249) if later else (49,)
    all_names=json.loads((out/"plan.json").read_bytes())["proteins"]
    names=all_names[6:12] if middle else all_names[-(12 if later else 6):]
    original=out/"charmm"
    original.mkdir(exist_ok=True)
    stage=out/("charmm-middle" if middle else "charmm-helper")
    stage.mkdir(exist_ok=True)
    fn=partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    for name in reversed(names):
        targets=[original/f"{name}-{origin}.npz" for origin in frames]
        if all(target.exists() for target in targets):
            continue
        argv=dict(destination=stage,samples=WINDOW,branches=2,setup_fn=fn,
                  parts_fn=partial(parts,origins=frames))
        try:
            simulate((name,),len(frames),step_fs=2,minimize_iterations=60,**argv)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            (stage/f"{name}-recovery.json").write_text(
                json.dumps({"step_fs":1,"minimize":100}),encoding="utf-8")
            simulate((name,),len(frames),step_fs=1,minimize_iterations=100,**argv)
        for target in targets:
            try:
                os.link(stage/target.name,target)
                print("published",target.name,flush=True)
            except FileExistsError:
                print("main job already completed",target.name,flush=True)


if __name__=="__main__":
    import sys
    main(later="--later" in sys.argv,middle="--middle" in sys.argv)


