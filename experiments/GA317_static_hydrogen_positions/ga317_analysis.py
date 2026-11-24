"""Compare keeping origin hydrogens with rebuilding them, on matched pairs."""
from pathlib import Path

import numpy as np

from .explore import record
from .ga314 import load, field
from .ga_static_offset import VALID, estimates


def main():
    parent=Path("runs")
    outputs=[load(VALID,parent/f"ga{n}") for n in (310,313,317)]
    a,y,b,p,f,g=outputs[0]
    c=outputs[1][0]
    h=outputs[2][0]
    for pack in outputs[1:]:
        for left,right in zip((y,b,p,f,g),pack[1:]):
            np.testing.assert_array_equal(left,right)
    a=field(a,10,"stable")-p
    c,h=[v.mean((0,1))-p for v in (c,h)]
    formulas={
        "rebuilt_charmm":c,
        "observed_static_H_charmm":h,
        "both_charmm":.5*(c+h),
        "amber_rebuilt_charmm":.5*(a+c),
        "amber_static_H_charmm":.5*(a+h),
        "three_sources":(a+c+h)/3.,
    }
    scores={n:estimates(b+.4*x+.525*f,y,b) for n,x in formulas.items()}
    record(parent/"ga317/development.json",scores)
    print(scores,flush=True)


if __name__=="__main__":
    main()
