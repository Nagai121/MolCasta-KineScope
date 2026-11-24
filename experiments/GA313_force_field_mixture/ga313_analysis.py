"""GA313 force-field transfer diagnostic on the matched first validation origins."""
from pathlib import Path

import numpy as np

from .explore import record
from .ga314 import load
from .ga_static_offset import VALID, estimates


def main():
    dest = Path("runs/ga313")
    a,y,b,p,f,g=load(VALID,dest)
    old,yy,bb,pp,ff,gg=load(VALID,Path("runs/ga310"))
    np.testing.assert_array_equal(y,yy)
    np.testing.assert_array_equal(g,gg)
    assert np.array_equal(b,bb) and np.array_equal(p,pp) and np.array_equal(f,ff)
    amber=old.mean((0,1))-p
    charmm=a.mean((0,1))-p
    out={}
    for weight in (0.,.125,.25,.375,.5,.625,.75,.875,1.):
        z=amber*(1-weight)+charmm*weight
        for strength in (0.,.525):
            out[f"{weight}/{strength}"]=estimates(b+.4*z+strength*f,y,b)
    record(dest/"development.json",dict(proteins=VALID,pairs=len(y),conditions=out))
    print(sorted(out.items(),key=lambda item:-item[1]["residual_correlation"])[:5],flush=True)


if __name__ == "__main__":
    main()
