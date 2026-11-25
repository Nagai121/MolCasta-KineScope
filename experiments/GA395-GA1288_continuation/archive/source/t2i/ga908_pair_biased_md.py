"""GA908: bias candidate pairs toward a shallow hydrogen-bond distance during generation."""
from functools import partial
import json
from pathlib import Path

import numpy as np
import openmm

from .ga333 import WINDOW
from .ga365 import parts
from .ga_native240 import simulate
from .ga_openmm_pilot import build
from .ga_replica_teacher import trajectory_rows
from .ga_static_offset import VALID, estimates

RUNS=Path("runs")
OUT=RUNS/"ga908-pair-biased-md"


def setup(protein, replica, origin_frame):
    state=build(protein,replica,force_files=("charmm36_2024.xml","implicit/gbn2.xml"))
    raw,heavy,source,topology,indices,field,simulation,reg,tid=state
    pair,origin=trajectory_rows(protein,replica)
    eligible=origin==origin_frame
    edges=np.unique(pair[eligible,:2].astype(np.intp),axis=0)
    well=openmm.CustomBondForce("-amplitude*exp(-0.5*((r-center)/width)^2)")
    well.addGlobalParameter("amplitude",2.5)
    well.addGlobalParameter("center",.32)
    well.addGlobalParameter("width",.065)
    well.setName("GA908 shallow candidate-pair bias")
    for a,b in indices[edges]:
        well.addBond(int(a),int(b),[])
    simulation.system.addForce(well)
    simulation.context.reinitialize(preserveState=False)
    return state


def main():
    OUT.mkdir(exist_ok=True)
    for name in VALID:
        for origin in (49,149,249):
            kwargs=dict(destination=OUT,samples=WINDOW,branches=4,
                        setup_fn=partial(setup,origin_frame=origin),
                        parts_fn=partial(parts,origins=(origin,)))
            try:
                simulate((name,),1,step_fs=2,minimize_iterations=60,**kwargs)
            except openmm.OpenMMException as exc:
                if "NaN" not in str(exc):
                    raise
                (OUT/f"{name}-{origin}-recovery.json").write_text(
                    json.dumps({"step_fs":1,"minimize":100}),encoding="utf-8")
                simulate((name,),1,step_fs=1,minimize_iterations=100,**kwargs)
    groups=[]
    offsets=[]
    diff=[]
    for n in VALID:
        for t in (49,149,249):
            tag=f"{n}-{t}.npz"
            with np.load(OUT/tag) as a,np.load(RUNS/"ga507"/tag) as b:
                np.testing.assert_array_equal(a["pair"],b["pair"])
                mean=a["tracks"].mean((0,1))
                diff.append(mean-b["tracks"][:4].mean((0,1)))
                groups.extend([n]*len(mean))
                offsets.extend([t]*len(mean))
    delta=np.concatenate(diff)
    with np.load(RUNS/"ga360/validation.npz") as z:
        target,base,g,o=[z[k] for k in ("target","baseline","proteins","origin")]
    np.testing.assert_array_equal(groups,g)
    np.testing.assert_array_equal(offsets,o)
    with np.load(RUNS/"ga890-robust-mixture/GA898.npz") as z:
        parent=z["prediction"]
    result={}
    for w in (0.,.1,.25,.5,1.):
        q=parent+w*delta
        result[str(w)]={k:estimates(q[m],target[m],base[m]) for k,m in
                        (("all",o>=0),("first",o==49),("later",o!=49))}
    (OUT/"development.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print({k:v["all"]["residual_correlation"] for k,v in result.items()},flush=True)


if __name__=="__main__":
    main()

