"""GA331: infer compatible atomic displacement from pairwise ensemble means."""
from itertools import product
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from xgboost import XGBRegressor

from .explore import record
from .ga314 import field, load
from .ga_force import build
from .ga_static_offset import VALID, cohort, estimates


def project(row, signal, weight, damping):
    pair=row["pair"]
    nodes=pair[:,:2].astype(np.intp)
    count=len(pair)
    axes=pair[:,3:6]/pair[:,2,None]
    xyz=np.arange(3)
    cols=(nodes[:,:,None]*3+xyz).reshape(-1)
    values=np.stack((-axes,axes),axis=1).astype(np.float64).reshape(-1).astype(np.float64)
    matrix=sparse.csr_matrix((values,(np.repeat(np.arange(count),6),cols)))
    w=np.sqrt(np.asarray(weight,dtype=np.float64))
    weighted=matrix.multiply(w[:,None]).tocsr()
    coords=lsmr(weighted,w*signal,damp=damping,atol=1e-5,btol=1e-5,maxiter=250)[0]
    return matrix@coords


def main():
    root=Path("runs")
    dest=root/"ga331"
    dest.mkdir(parents=True,exist_ok=True)
    a,y,b,p,_,g=load(VALID,root/"ga310")
    c,*_=load(VALID,root/"ga313")
    f_in=np.concatenate([build((n,1,1))[0] for n in VALID])
    stat=[]
    for name,model in (("ga318","all-8"),("ga330","all-9")):
        head=XGBRegressor(device="cpu")
        head.load_model(root/name/f"{model}.ubj")
        stat.append(head.predict(f_in)+f_in[:,0]-b)
    force=np.mean(stat,axis=0)
    raw=.5*(field(a,10,"stable")+c.mean((0,1)))
    spread=np.hypot(a.mean(1).std(0),c.mean(1).std(0))
    report={}
    rows=cohort(VALID,1)
    for method,damping,weighting in product(("raw","residual"),(.04,.15,.45,.9),("constant","dispersion")):
        q=raw-p if method=="residual" else raw
        features=[]
        for row in rows:
            use=g==row["protein"]
            w=np.ones(use.sum()) if weighting=="constant" else 1./(.1+spread[use])
            features.append(project(row,q[use],w,damping))
        smooth=np.concatenate(features)
        if method=="raw":
            smooth-=p
        for mix in (.25,.5,.75,1.):
            pred=b+.4*(mix*smooth+(1-mix)*(raw-p))+.525*force
            report[f"{method}/{damping}/{weighting}/{mix}"]=estimates(pred,y,b)
    report["reference"]=estimates(b+.4*(raw-p)+.525*force,y,b)
    ranked=sorted(report,key=lambda k:-report[k]["residual_correlation"])
    record(dest/"development.json",dict(selected=ranked[:5],trials=report))
    print([(k,report[k]) for k in ranked[:5]],flush=True)


if __name__=="__main__":
    main()
