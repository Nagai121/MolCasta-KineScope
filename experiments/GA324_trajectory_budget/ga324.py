"""GA324: distinguish time and branch budget at matched validation origins."""
from itertools import combinations
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from .explore import record
from .ga_force import build
from .ga314 import load
from .ga_native240 import SAMPLES
from .ga_static_offset import VALID, estimates


def main():
    dest=Path("runs/ga324")
    dest.mkdir(parents=True,exist_ok=True)
    data,y,b,p,f,g=load(VALID,Path("runs/ga310"))
    x=np.concatenate([build((n,1,1))[0] for n in VALID])
    model=XGBRegressor(device="cpu")
    model.load_model("runs/ga318/all-8.ubj")
    f2=model.predict(x)+x[:,0]-b
    chrom,cy,cb,cp,cf,cg=load(VALID,Path("runs/ga313"))
    chrom=chrom.mean((0,1))-p
    out={}
    for time in (10,25,50,75,100):
        h=data[:,np.array(SAMPLES)<=time].mean(1)
        for seeds in (1,2,4,6,8):
            metrics=[]
            metrics_c=[]
            for c in combinations(range(len(data)),seeds):
                generated=h[list(c)].mean(0)-p
                metrics.append(estimates(b+.4*generated+.525*f2,y,b)["residual_correlation"])
                metrics_c.append(estimates(b+.2*(generated+chrom)+.525*f2,y,b)["residual_correlation"])
            out[f"{time}/{seeds}"]=dict(amber=float(np.mean(metrics)),
                                        two_force_fields=float(np.mean(metrics_c)),
                                        amber_total_ps=time*seeds)
    record(dest/"budget_curve.json",out)
    print(out,flush=True)


if __name__=="__main__":
    main()
