"""GA646 uses antithetic initial velocities to estimate the generated pair level."""
from functools import partial
import json
from pathlib import Path

import numpy as np
import openmm

from .ga333 import WINDOW
from .ga365 import parts
from .ga_native240 import simulate
from .ga_openmm_pilot import build
from .ga_static_offset import VALID, estimates

R=Path("runs")
OUT=R/"ga646-antithetic"


def antithetic(topology, positions):
    saved=None
    def paired(velocity, branch):
        nonlocal saved
        if branch % 2 == 0:
            saved=velocity
            return velocity
        return -saved
    return paired


def main():
    OUT.mkdir(exist_ok=True)
    setup=partial(build,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    for name in VALID:
        options=dict(destination=OUT,samples=WINDOW,branches=4,setup_fn=setup,
                     parts_fn=partial(parts,origins=(49,149,249)),velocity_hook=antithetic)
        try:
            simulate((name,),3,step_fs=2,minimize_iterations=60,**options)
        except openmm.OpenMMException as exc:
            if "NaN" not in str(exc):
                raise
            (OUT/f"{name}-recovery.json").write_text(json.dumps({"step_fs":1,"minimize":100}),encoding="utf-8")
            simulate((name,),3,step_fs=1,minimize_iterations=100,**options)
    deviation=[]
    expected=[]
    for name in VALID:
        for o in (49,149,249):
            namefile=f"{name}-{o}.npz"
            with np.load(OUT/namefile) as test, np.load(R/"ga507"/namefile) as control:
                np.testing.assert_array_equal(test["pair"],control["pair"])
                delta=test["tracks"].mean((0,1))-control["tracks"][:4].mean((0,1))
            expected.extend([(name,o)]*len(delta))
            deviation.append(delta)
    delta=np.concatenate(deviation)
    with np.load(R/"ga360/validation.npz") as z:
        y,b,g,o=[z[k] for k in ("target","baseline","proteins","origin")]
    np.testing.assert_array_equal(g,[v[0] for v in expected])
    np.testing.assert_array_equal(o,[v[1] for v in expected])
    with np.load(R/"ga588-transfer/GA594.npz") as z:
        parent=z["prediction"]
    results={}
    for w in (0.,.1,.2,.35,.5):
        pred=parent+w*delta
        results[str(w)]={k:estimates(pred[m],y[m],b[m]) for k,m in
                         (("all",o>=0),("first",o==49),("later",o!=49))}
    (OUT/"development.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print({k:v["all"]["residual_correlation"] for k,v in results.items()},flush=True)


if __name__=="__main__":
    main()
