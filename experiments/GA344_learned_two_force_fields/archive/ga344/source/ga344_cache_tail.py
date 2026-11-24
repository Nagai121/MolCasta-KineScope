"""Prepare disjoint late caches while GA344 processes its earlier training proteins."""
from functools import partial
import json
from pathlib import Path

from .ga333 import candidates, WINDOW
from .ga_native240 import simulate
from .ga_openmm_pilot import build


if __name__=="__main__":
    chosen=json.loads(Path("runs/ga333/development.json").read_bytes())["successful"][::2]
    fn=partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    dest=Path("runs/ga344")
    for name in reversed(chosen[20:]):
        try:
            simulate((name,),1,destination=dest,samples=WINDOW,branches=2,step_fs=2,
                     minimize_iterations=60,setup_fn=fn,parts_fn=candidates)
        except Exception as exc:
            print(name,exc,flush=True)
