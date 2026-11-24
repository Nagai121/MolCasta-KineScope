"""Apply the locked GA314 readout to the same first-origin confirmation as GA312."""
import json
from pathlib import Path

import numpy as np

from tseries.data import TEST
from .explore import record
from .ga_native_offset import HOLD
from .ga314 import field, load
from .ga_static_offset import estimates


def main():
    dest=Path("runs/ga314")
    locked=json.loads((dest/"development.json").read_bytes())["selected"]
    names=TEST+HOLD
    a,y,b,p,f,g=load(names,Path("runs/ga312"))
    prediction=b+.4*(field(a,locked["time"],locked["method"])-p+locked["force_ratio"]*f)
    out=dict(rule=locked,pairs=len(y),pooled=estimates(prediction,y,b),
             individual={n:estimates(prediction[g==n],y[g==n],b[g==n]) for n in names})
    record(dest/"confirmation.json",out)
    np.savez(dest/"evaluation.npz",target=y,static=b,predicted=prediction,proteins=g)
    print(out,flush=True)


if __name__=="__main__":
    main()
