"""GA1253: add GA386 as a retrospective comparator on GA841 without model selection."""
import json
from pathlib import Path

import numpy as np

from .ga373 import assemble
from .ga_frozen_adapter import load_anchor,readout
from .ga_static_offset import estimates

R=Path("runs")
OUT=R/"ga1253-ga841-legacy"


def main():
    panel=R/"ga841-fresh-panel"
    lock=json.loads((panel/"model-lock.json").read_bytes())
    roster=lock["proteins"]
    a,b=(panel/"blind"/k for k in ("amber","charmm"))
    x,d,_,y,_,p=assemble(roster,(49,),a,b)
    with np.load(panel/"confirmation.npz") as data:
        np.testing.assert_array_equal(data["proteins"],p)
        np.testing.assert_allclose(data["target"],y,rtol=0,atol=3e-6)
        parent,base,selected=data["parent"],data["baseline"],data["prediction"]
    anchor,state,w=load_anchor()
    output=(1-w)*parent+w*readout(anchor,state,x,d)
    result={"protocol":"retrospective on GA841 after completion; no model or hyperparameter choice",
            "n_pairs":len(y),"GA348":estimates(parent,y,base),
            "GA386":estimates(output,y,base),lock["id"]:estimates(selected,y,base)}
    OUT.mkdir(exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(OUT/"comparison.npz",GA386=output,GA821=selected,parent=parent,target=y,
                        baseline=base,proteins=p)
    print(result,flush=True)


if __name__=="__main__":
    main()
