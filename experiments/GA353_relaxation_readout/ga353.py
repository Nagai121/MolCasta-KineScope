"""GA353 regularized relaxation readout from generated trajectories."""
from pathlib import Path
import numpy as np

from .explore import record
from .ga314 import load, field
from .ga348 import predict
from .ga_force import build
from .ga_static_offset import VALID, estimates

def relaxation(h):
    # Estimate the one-step response on the evenly spaced 10 ps grid.
    track=h[:,[2,4,6,7,8,9,12],:]
    grid=np.array([10,20,30,40,50,60,100])
    select=np.diff(grid)==10
    x=track[:,:-1][:,select]
    y=track[:,1:][:,select]
    m=x.mean((0,1))
    n=y.mean((0,1))
    cross=((x-m)*(y-n)).mean((0,1))
    variance=((x-m)**2).mean((0,1))
    rho=((cross+.025*.65)/(variance+.025)).clip(0,.90)
    long=(n-rho*m)/(1-rho)
    return long.clip(-2.,2.)

def main():
    root=Path("runs")
    a,y,b,p,_,g=load(VALID,root/"ga310")
    c,*_=load(VALID,root/"ga313")
    x=np.concatenate([build((n,1,1))[0] for n in VALID])
    static=.525*(.5*(predict(root/"ga318/all-8.ubj",x)+predict(root/"ga330/all-9.ubj",x))-b)
    old_a=field(a,10,"stable")
    old_c=c.mean((0,1))
    new_a,new_c=relaxation(a),relaxation(c)
    out={}
    for name,u,v in (("amber",new_a,old_c),("charmm",old_a,new_c),("both",new_a,new_c)):
        for w in (0.,.1,.25,.5,1.):
            h=(1-w)*(old_a+old_c)+w*(u+v)
            out[f"{name}/{w}"]=estimates(b+static+.2*(h-2*p),y,b)
    dest=root/"ga353"
    dest.mkdir(exist_ok=True)
    record(dest/"development.json",out)
    print(sorted(((k,v["residual_correlation"]) for k,v in out.items()),key=lambda x:-x[1])[:7])
if __name__=="__main__":
    main()
