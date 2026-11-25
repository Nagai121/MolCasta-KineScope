"""GA1237: freeze the GA1236 mixture and test on later origins."""
import json
from pathlib import Path

import numpy as np

from .ga333 import WINDOW
from .ga_explicit_solvent import R,simulate
from .ga_static_offset import VALID,estimates

OUT=Path("runs/ga1237-explicit-later")


def main():
    OUT.mkdir(exist_ok=True)
    lock={"selector":"GA1236 first-origin pilot","coef":.1,"origins":[149,249],"model":"GA1086"}
    (OUT/"locked.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    by_group={}
    for name in VALID:
        for origin in lock["origins"]:
            simulate(name,origin,OUT)
            fname=f"{name}-{origin}.npz"
            with np.load(OUT/fname) as a,np.load(R/"ga360/amber"/fname) as b:
                np.testing.assert_array_equal(a["pair"],b["pair"])
                by_group[name,origin]=a["tracks"].mean((0,1))-b["tracks"][:2,:len(WINDOW)].mean((0,1))
    with np.load(R/"ga360/validation.npz") as z:
        y,base,names,origin=[z[k] for k in ("target","baseline","proteins","origin")]
    with np.load(R/"ga1077-reliability-width/GA1086.npz") as z:
        parent=z["prediction"]
    diff=np.zeros_like(y)
    for (name,frame),values in by_group.items():
        mask=(names==name)&(origin==frame)
        diff[mask]=values
    prediction=parent+.1*diff
    metrics={key:{"parent":estimates(parent[m],y[m],base[m]),"test":estimates(prediction[m],y[m],base[m])}
             for key,m in (("later",origin!=49),("all",origin>=0),("149",origin==149),("249",origin==249))}
    np.savez_compressed(OUT/"predictions.npz",prediction=prediction,parent=parent,target=y,base=base)
    (OUT/"development.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(metrics,flush=True)


if __name__=="__main__":
    main()
