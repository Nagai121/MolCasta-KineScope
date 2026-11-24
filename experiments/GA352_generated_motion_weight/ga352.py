"""Compare modest increases in the generated-motion contribution."""
from pathlib import Path

import numpy as np

from .explore import record
from .ga314 import load, field
from .ga348 import predict
from .ga_force import build
from .ga_static_offset import VALID, estimates

def main():
    root=Path("runs")
    a,y,b,p,_,g=load(VALID,root/"ga310")
    c,*_=load(VALID,root/"ga313")
    x=np.concatenate([build((n,1,1))[0] for n in VALID])
    s=.525*(.5*(predict(root/"ga318/all-8.ubj",x)+predict(root/"ga330/all-9.ubj",x))-b)
    amber=.2*(field(a,10,"stable")-p)
    charmm=.2*(c.mean((0,1))-p)
    out={}
    for label,aa,cc in (("baseline",1.,1.),("both+10%",1.1,1.1),
                        ("both+20%",1.2,1.2),("both+30%",1.3,1.3),
                        ("Amber+20%",1.2,1.),("CHARMM+20%",1.,1.2)):
        forecast=b+s+aa*amber+cc*charmm
        out[label]=dict(pooled=estimates(forecast,y,b),
                        individual={n:estimates(forecast[g==n],y[g==n],b[g==n])["residual_correlation"] for n in VALID})
    dest=root/"ga352"
    dest.mkdir(exist_ok=True)
    record(dest/"development.json",out)
    print({k:v["pooled"]["residual_correlation"] for k,v in out.items()})
if __name__=="__main__":
    main()
