"""GA909+ stages an independent training replica and extends the mixture readout."""
from functools import partial
import json
from pathlib import Path

import numpy as np
import openmm

from .ga333 import WINDOW
from .ga365 import parts
from .ga373 import assemble
from .ga_multiwindow import targets
from .ga_native240 import simulate
from .ga_openmm_pilot import build

RUNS=Path("runs")
OUT=RUNS/"ga909-r2-augmentation"


def main():
    OUT.mkdir(exist_ok=True)
    amber=RUNS/"ga_native240"
    charmm=OUT/"charmm"
    charmm.mkdir(exist_ok=True)
    base=set(json.loads((RUNS/"ga330/development.json").read_bytes())["train"])
    names=sorted({f.name.removesuffix("-R2-49.npz") for f in amber.glob("*-R2-49.npz")}&base)
    if not names:
        raise RuntimeError("No existing matched-training Amber replica")

    for path in (RUNS/"ga351/reserved.json",RUNS/"ga841-fresh-panel/reserved.json"):
        reserved={r["id"] for r in json.loads(path.read_bytes())["proteins"]}
        assert not set(names)&reserved
    (OUT/"training-plan.json").write_text(json.dumps({"proteins":names,"replica":2,"origin":49},indent=2),encoding="utf-8")
    for name in names:
        setup=partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
        opts=dict(replica=2,destination=charmm,samples=WINDOW,branches=2,setup_fn=setup,
                  parts_fn=partial(parts,origins=(49,),replica=2))
        try:
            simulate((name,),1,step_fs=2,minimize_iterations=60,**opts)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            (OUT/f"{name}-recovery.json").write_text(json.dumps({"step_fs":1,"minimize":100}),encoding="utf-8")
            simulate((name,),1,step_fs=1,minimize_iterations=100,**opts)
    x,d,g,y,origins,proteins=assemble(names,(49,),amber,charmm,replica=2)
    label=targets(x,y,origins,proteins,OUT/"history-r2.npz",replica=2)
    np.savez_compressed(OUT/"train-r2.npz",x=x,d=d,g=g,y=y,origins=origins,proteins=proteins,label=label)
    print("assembled replica 2",len(names),len(y),flush=True)
    from .ga_mixture import main as train
    train(stability=True,replica2=True)


if __name__=="__main__":
    main()

