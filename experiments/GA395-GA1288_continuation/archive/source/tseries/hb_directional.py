"""Directional elastic-network descriptors from one conformation."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from .data import TRAIN, VALIDATION, cell, registry, topology, trajectory
from .features import _mic
from .hb_single_structure import OUT, ROOT, explore, load
from .search_events import EXTRA_TRAINING


def directional_features(position, box, pair, maps):
    alpha = np.asarray([m["CA"] for m in maps])
    n = len(alpha)
    ordinal = np.full(len(position),-1,dtype=np.int32)
    for i,m in enumerate(maps):
        ordinal[list(m.values())] = i
    a,b = pair[:,:2].astype(np.int64).T
    i,j = ordinal[a],ordinal[b]
    if np.any(i<0) or np.any(j<0):
        raise ValueError("Missing residue mapping")
    ca = position[alpha]
    separation = _mic(ca[None,:]-ca[:,None],box)
    r2 = np.einsum("ijk,ijk->ij",separation,separation)
    left,right = np.where(np.triu((r2<=144)&(r2>1e-5),1))
    unit = separation[left,right]/np.sqrt(r2[left,right,None])
    strength = (16/np.maximum(1,r2[left,right]))[:,None,None]
    spring = strength*unit[:,:,None]*unit[:,None,:]
    h = np.zeros((n,n,3,3),dtype=np.float64)
    h[left,right] = -spring
    h[right,left] = -spring
    diag = -h.sum(axis=1) + .15*np.eye(3)
    h[np.arange(n),np.arange(n)] = diag
    m = h.transpose(0,2,1,3).reshape(3*n,3*n)
    inv = cho_solve(cho_factor(m,check_finite=False),np.eye(3*n),check_finite=False)
    cov = inv.reshape(n,3,n,3).transpose(0,2,1,3)
    relative = cov[i,i]+cov[j,j]-cov[i,j]-cov[j,i]
    ab = _mic(position[b]-position[a],box)
    direction = ab/pair[:,2,None]
    parallel = np.maximum(np.einsum("bi,bij,bj->b",direction,relative,direction),1e-6)
    trace = np.trace(relative,axis1=1,axis2=2)
    transverse = np.maximum(trace-parallel,1e-6)
    local_a = np.einsum("bi,bij,bj->b",direction,diag[i],direction)
    local_b = np.einsum("bi,bij,bj->b",direction,diag[j],direction)
    normalized_gap = (pair[:,2]-3.5)/np.sqrt(parallel)
    result = np.column_stack((parallel,transverse,parallel/transverse,normalized_gap,
                              local_a,local_b,np.trace(cov[i,i],axis1=1,axis2=2),
                              np.trace(cov[j,j],axis1=1,axis2=2))).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Invalid elastic features")
    return result


def add_directional(part,stride):
    protein, replica = part["protein"],part["replica"]
    reg = registry(ROOT,protein)
    tid=trajectory(protein,replica)
    path=ROOT/protein/"trajectories"/tid/"hb-directional"
    path.mkdir(exist_ok=True)
    version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache=path/f"{stride}-{version}.npz"
    if cache.is_file():
        with np.load(cache) as z:
            if np.array_equal(z["pair"],part["pair"]) and np.array_equal(z["origin"],part["origin"]):
                part["directional"]=z["features"]
                return
    maps=topology(reg.atoms(tid).to_pydict())[0]
    out=[]
    with threadpool_limits(limits=1,user_api="blas"):
        for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
            t=int(part["origin"][start])
            frame=next(reg.blocks(tid,start=t,stop=t+1))
            out.append(directional_features(frame.positions[0],cell(frame.box[0]),part["pair"][start:end],maps))
    features=np.concatenate(out)
    part["directional"]=features
    tmp=NamedTemporaryFile(dir=path,delete=False)
    try:
        with tmp:
            np.savez(tmp,features=features,pair=part["pair"],origin=part["origin"])
        os.replace(tmp.name,cache)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def main():
    train=load(TRAIN,25,True,True)
    extra=load(EXTRA_TRAINING,50,True,True)
    validation=load(VALIDATION,100,geometry=True)
    for parts,stride in ((train,25),(extra,50),(validation,100)):
        for part in parts:
            add_directional(part,stride)
    configurations=(
        ("xgb",118,"hb-origin-directional",False,4,1.0),
        ("xgb",166,"hb-origin-directional",False,6,1.0),
        ("xgb",118,"hb-origin-directional",True,6,1.0),
        ("xgb",166,"hb-origin-directional",True,5,1.0),
        ("xgb-soft-local",118,"hb-origin-directional",False,3,1.0),
        ("xgb-soft-tune-0.15-0.25-320-local",142,"hb-origin-directional",True,5,1.0),
        ("xgb-soft-tune-0.25-0.4-240-local",166,"hb-origin-directional",True,4,1.0),
        ("xgb",166,"hb-origin-directional",False,4,.75),
    )
    explore("directional",configurations,661,train+extra,validation)


if __name__=="__main__":
    main()
